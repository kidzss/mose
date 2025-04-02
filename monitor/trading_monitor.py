import sys
import os
import logging
import time
import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy import create_engine
from backtest.strategy_evaluator import StrategyEvaluator
from backtest.parameter_optimizer import ParameterOptimizer
from backtest.market_analyzer import MarketAnalyzer
from backtest.risk_manager import RiskManager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.trading_config import TradingConfig, default_config
from strategy.uss_cpgw import calculate_indicators, check_last_day_signal

class DataFetcher:
    def __init__(self, config: TradingConfig):
        self.config = config
        self._engine = None
        self.logger = logging.getLogger(__name__)

    @property
    def engine(self):
        if self._engine is None:
            db = self.config.database
            self._engine = create_engine(
                f"mysql+pymysql://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
            )
        return self._engine

    def get_stock_data(self, stock: str, days: int = 100) -> pd.DataFrame:
        """获取指定股票的历史数据"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                Date AS `date`, 
                Open AS `Open`, 
                High AS `High`, 
                Low AS `Low`, 
                Close AS `Close`, 
                Volume AS `Volume`, 
                AdjClose AS `Adj Close`
            FROM stock_code_time
            WHERE Code = %s
            AND Date BETWEEN %s AND %s
            ORDER BY Date ASC;
            """
            
            data = pd.read_sql_query(
                query,
                self.engine,
                params=(stock, start_date, end_date)
            )
            
            if data.empty:
                self.logger.warning(f"无法获取股票 {stock} 的数据")
                return None
                
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            return data
            
        except Exception as e:
            self.logger.error(f"获取股票 {stock} 数据时出错: {str(e)}")
            return None

    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        try:
            if self.config.stock_pool:
                return self.config.stock_pool
                
            query = "SELECT DISTINCT Code FROM stock_code_time ORDER BY Code"
            df = pd.read_sql_query(query, self.engine)
            return df['Code'].tolist()
        except Exception as e:
            self.logger.error(f"获取股票列表时出错: {str(e)}")
            return []

class SignalGenerator:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_signals(self, df: pd.DataFrame) -> Dict:
        """生成交易信号"""
        try:
            if df is None or df.empty:
                return None
                
            # 计算指标
            df = calculate_indicators(df)
            if df is None:
                return None
                
            # 检查信号
            signal = check_last_day_signal(df)
            
            return {
                'date': df.index[-1],
                'signal': signal,
                'close': df['Close'].iloc[-1],
                'indicators': {
                    'Long Line': df['Long Line'].iloc[-1],
                    'Hot Money Line': df['Hot Money Line'].iloc[-1],
                    'Main Force Line': df['Main Force Line'].iloc[-1]
                }
            }
        except Exception as e:
            self.logger.error(f"生成信号时出错: {str(e)}")
            return None

class AlertSystem:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.last_alert_time = {}  # 记录每个股票的最后提醒时间

    def _can_send_alert(self, stock: str) -> bool:
        """检查是否可以发送提醒"""
        now = datetime.now()
        if stock in self.last_alert_time:
            time_diff = (now - self.last_alert_time[stock]).total_seconds()
            return time_diff >= self.config.monitoring.alert_cooldown
        return True

    def send_email(self, subject: str, body: str):
        """发送邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email.sender_email
            msg['To'] = ', '.join(self.config.email.receiver_emails)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(self.config.email.smtp_server, self.config.email.smtp_port) as server:
                server.starttls()
                server.login(
                    self.config.email.sender_email,
                    self.config.email.sender_password
                )
                server.send_message(msg)

            self.logger.info(f"成功发送邮件: {subject}")
        except Exception as e:
            self.logger.error(f"发送邮件时出错: {str(e)}")

    def format_signal_message(self, stock: str, signal_data: Dict) -> str:
        """格式化信号消息"""
        return f"""
        <h2>股票交易信号提醒</h2>
        <p>股票代码: {stock}</p>
        <p>信号时间: {signal_data['date']}</p>
        <p>当前价格: {signal_data['close']:.2f}</p>
        <p>信号类型: {signal_data['signal']}</p>
        <h3>指标数据:</h3>
        <ul>
            <li>长庄线: {signal_data['indicators']['Long Line']:.2f}</li>
            <li>游资线: {signal_data['indicators']['Hot Money Line']:.2f}</li>
            <li>主力线: {signal_data['indicators']['Main Force Line']:.2f}</li>
        </ul>
        """

    def process_signal(self, stock: str, signal_data: Dict):
        """处理交易信号"""
        if not signal_data or 'signal' not in signal_data:
            return

        if not self._can_send_alert(stock):
            return

        signal = signal_data['signal']
        if "Buy signal" in signal or "Sell signal" in signal:
            subject = f"交易信号提醒 - {stock}"
            body = self.format_signal_message(stock, signal_data)
            self.send_email(subject, body)
            self.last_alert_time[stock] = datetime.now()

class TradingMonitor:
    def __init__(self, config: TradingConfig = None):
        self.config = config or default_config
        self.setup_logging()
        
        self.data_fetcher = DataFetcher(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.alert_system = AlertSystem(self.config)
        
        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )

    def monitor_stock(self, stock: str):
        """监控单个股票"""
        try:
            # 获取数据
            df = self.data_fetcher.get_stock_data(stock)
            if df is None:
                return

            # 生成信号
            signal_data = self.signal_generator.generate_signals(df)
            if signal_data is None:
                return

            # 处理信号
            self.alert_system.process_signal(stock, signal_data)

        except Exception as e:
            self.logger.error(f"监控股票 {stock} 时出错: {str(e)}")

    def run(self):
        """运行监控系统"""
        self.logger.info("启动交易监控系统...")
        
        while True:
            try:
                # 获取股票列表
                stocks = self.data_fetcher.get_stock_list()
                if not stocks:
                    self.logger.error("无法获取股票列表")
                    time.sleep(self.config.monitoring.retry_delay)
                    continue

                # 监控每个股票
                for stock in stocks:
                    self.monitor_stock(stock)

                # 等待下一次检查
                time.sleep(self.config.monitoring.check_interval)

            except Exception as e:
                self.logger.error(f"监控系统运行出错: {str(e)}")
                time.sleep(self.config.monitoring.retry_delay)

def main():
    """主函数"""
    monitor = TradingMonitor()
    monitor.run()

if __name__ == "__main__":
    main() 