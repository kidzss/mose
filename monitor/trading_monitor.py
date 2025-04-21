import sys
import os
import logging
import time
import smtplib
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy import create_engine

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.trading_config import TradingConfig, default_config

class AlertSystem:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_email(self, subject: str, body: str):
        """发送邮件通知"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email.sender
            msg['To'] = self.config.email.recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP_SSL(self.config.email.smtp_server, self.config.email.smtp_port) as server:
                server.login(self.config.email.sender, self.config.email.password)
                server.send_message(msg)
                
            self.logger.info(f"邮件发送成功: {subject}")
            
        except Exception as e:
            self.logger.error(f"发送邮件时出错: {str(e)}")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for trading signals"""
    if df is None or df.empty:
        return None
        
    try:
        # Calculate moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Calculate volume indicators
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
        
        # Calculate momentum indicators
        df['RSI'] = calculate_rsi(df['Close'], period=14)
        df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df['Close'])
        
        # Calculate custom indicators
        df['Long Line'] = (df['MA20'] + df['MA10']) / 2
        df['Hot Money Line'] = df['MA5']
        df['Main Force Line'] = df['MA10']
        
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {str(e)}")
        return None

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD, Signal line, and Histogram"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def check_last_day_signal(df: pd.DataFrame) -> str:
    """Check for trading signals based on the latest data"""
    if df is None or len(df) < 20:
        return "Insufficient data"
        
    try:
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Check for buy signals
        buy_signal = (
            last_row['Hot Money Line'] > last_row['Main Force Line'] and
            last_row['Main Force Line'] > last_row['Long Line'] and
            last_row['Volume'] > last_row['Volume_MA5'] * 1.5
        )
        
        # Check for sell signals
        sell_signal = (
            last_row['Hot Money Line'] < last_row['Main Force Line'] and
            last_row['Main Force Line'] < last_row['Long Line'] and
            last_row['Volume'] > last_row['Volume_MA5']
        )
        
        if buy_signal:
            return "Buy signal detected"
        elif sell_signal:
            return "Sell signal detected"
        else:
            return "No clear signal"
            
    except Exception as e:
        logging.error(f"Error checking signals: {str(e)}")
        return "Error in signal detection"

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

class TradingMonitor:
    def __init__(self, config: TradingConfig = None):
        self.config = config or default_config
        self.data_fetcher = DataFetcher(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.alert_system = AlertSystem(self.config)
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        log_file = 'trading_monitor.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def monitor_stock(self, stock: str):
        """监控单个股票"""
        try:
            # 获取数据
            data = self.data_fetcher.get_stock_data(stock)
            if data is None:
                return
                
            # 生成信号
            signal_data = self.signal_generator.generate_signals(data)
            if signal_data is None:
                return
                
            # 处理信号
            self.process_signal(stock, signal_data)
            
        except Exception as e:
            self.logger.error(f"监控股票 {stock} 时出错: {str(e)}")

    def run(self):
        """运行监控"""
        try:
            # 获取股票列表
            stocks = self.data_fetcher.get_stock_list()
            if not stocks:
                self.logger.error("无法获取股票列表")
                return
                
            # 监控每个股票
            for stock in stocks:
                self.monitor_stock(stock)
                time.sleep(1)  # 避免请求过于频繁
                
        except Exception as e:
            self.logger.error(f"运行监控时出错: {str(e)}")

def main():
    """主函数"""
    monitor = TradingMonitor()
    monitor.run()

if __name__ == "__main__":
    main() 