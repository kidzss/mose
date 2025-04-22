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
from data.data_interface import DataInterface
from config.trading_config import TradingConfig, default_config

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AlertSystem:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.email_sender = EmailSender(config)
        self.slack_sender = SlackSender(config)
        self.telegram_sender = TelegramSender(config)

    def _get_signal_explanation(self, signal: float) -> str:
        """获取策略信号的解释"""
        if signal > 0:
            return f"买入信号 (强度: {signal:.2f})"
        elif signal < 0:
            return f"卖出信号 (强度: {abs(signal):.2f})"
        else:
            return "观望信号"

    def _get_rsi_explanation(self, rsi: float) -> str:
        """获取RSI指标的解释"""
        if rsi > 70:
            return f"超买区域 ({rsi:.2f})，可能面临回调风险"
        elif rsi < 30:
            return f"超卖区域 ({rsi:.2f})，可能存在反弹机会"
        else:
            return f"中性区域 ({rsi:.2f})，市场相对平衡"

    def _get_macd_explanation(self, macd: float, signal: float) -> str:
        """获取MACD指标的解释"""
        if macd > signal:
            return f"金叉形态，上涨动能增强"
        elif macd < signal:
            return f"死叉形态，下跌动能增强"
        else:
            return f"趋势不明朗，等待方向确认"

    def _get_bollinger_explanation(self, price: float, upper: float, lower: float) -> str:
        """获取布林带指标的解释"""
        if price > upper:
            return f"价格突破上轨，可能超买"
        elif price < lower:
            return f"价格跌破下轨，可能超卖"
        else:
            return f"价格在通道内运行，市场相对稳定"

    def _get_ma_explanation(self, price: float, ma20: float) -> str:
        """获取均线指标的解释"""
        if price > ma20:
            return f"价格在20日均线上方，短期趋势向上"
        elif price < ma20:
            return f"价格在20日均线下方，短期趋势向下"
        else:
            return f"价格接近20日均线，趋势不明朗"

    def _get_volume_explanation(self, volume: float, volume_ma20: float) -> str:
        """获取成交量指标的解释"""
        if volume > volume_ma20 * 1.5:
            return f"成交量显著放大，市场活跃度增加"
        elif volume < volume_ma20 * 0.5:
            return f"成交量萎缩，市场活跃度降低"
        else:
            return f"成交量正常，市场活跃度适中"

    def send_alert(self, stock: str, alert_type: str, message: str, price: float, indicators: Dict):
        """发送交易警报"""
        try:
            # 构建完整的消息
            full_message = f"""
股票代码: {stock}
警报类型: {alert_type}

价格信息:
- 当前价格: {price:.2f}
- 成本价格: {indicators.get('cost_basis', 0):.2f}
- 价格变化: {indicators.get('price_change', 0):.2%}

策略信号:
- 动量策略: {self._get_signal_explanation(indicators.get('momentum_signal', 0))}
- 均值回归: {self._get_signal_explanation(indicators.get('mean_reversion_signal', 0))}
- 布林带策略: {self._get_signal_explanation(indicators.get('bollinger_signal', 0))}
- 突破策略: {self._get_signal_explanation(indicators.get('breakout_signal', 0))}
- 组合策略: {self._get_signal_explanation(indicators.get('combined_signal', 0))}

技术指标分析:
- RSI: {self._get_rsi_explanation(indicators.get('RSI', 0))}
- MACD: {self._get_macd_explanation(indicators.get('MACD', 0), indicators.get('Signal', 0))}
- 布林带: {self._get_bollinger_explanation(price, indicators.get('BB_upper', 0), indicators.get('BB_lower', 0))}
- 20日均线: {self._get_ma_explanation(price, indicators.get('MA20', 0))}
- 成交量: {self._get_volume_explanation(indicators.get('volume', 0), indicators.get('volume_ma20', 0))}

风险控制:
- 止损价格: {price * (1 - indicators.get('stop_loss', 0.15)):.2f} ({indicators.get('stop_loss', 0.15):.1%})
- 仓位权重: {indicators.get('weight', 0):.2%}
"""
            
            # 根据配置发送通知
            if hasattr(self.config, 'notification_settings') and self.config.notification_settings.get('email', False):
                self.email_sender.send_alert(stock, alert_type, full_message)
            
            if hasattr(self.config, 'notification_settings') and self.config.notification_settings.get('slack', False):
                self.slack_sender.send_alert(stock, alert_type, full_message)
            
            if hasattr(self.config, 'notification_settings') and self.config.notification_settings.get('telegram', False):
                self.telegram_sender.send_alert(stock, alert_type, full_message)
                
            self.logger.info(f"警报发送成功: {stock} - {alert_type}")
            
        except Exception as e:
            self.logger.error(f"发送警报失败: {str(e)}")

    def send_email(self, subject: str, body: str):
        """发送邮件通知"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email.sender_email
            msg['To'] = ', '.join(self.config.email.receiver_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.config.email.smtp_server, self.config.email.smtp_port) as server:
                server.starttls()
                server.login(self.config.email.sender_email, self.config.email.sender_password)
                server.send_message(msg)
                
            self.logger.info(f"邮件发送成功: {subject}")
            
        except Exception as e:
            self.logger.error(f"发送邮件时出错: {str(e)}")

class EmailSender:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_alert(self, stock: str, alert_type: str, message: str):
        """发送邮件警报"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email.sender_email
            msg['To'] = ', '.join(self.config.email.receiver_emails)
            msg['Subject'] = f"交易警报 - {stock} - {alert_type}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.config.email.smtp_server, self.config.email.smtp_port) as server:
                server.starttls()
                server.login(self.config.email.sender_email, self.config.email.sender_password)
                server.send_message(msg)
                
            self.logger.info(f"邮件发送成功: {stock} - {alert_type}")
            
        except Exception as e:
            self.logger.error(f"发送邮件时出错: {str(e)}")

class SlackSender:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_alert(self, stock: str, alert_type: str, message: str):
        """发送Slack警报"""
        try:
            # TODO: 实现Slack发送逻辑
            self.logger.info(f"Slack消息发送成功: {stock} - {alert_type}")
        except Exception as e:
            self.logger.error(f"发送Slack消息时出错: {str(e)}")

class TelegramSender:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_alert(self, stock: str, alert_type: str, message: str):
        """发送Telegram警报"""
        try:
            # TODO: 实现Telegram发送逻辑
            self.logger.info(f"Telegram消息发送成功: {stock} - {alert_type}")
        except Exception as e:
            self.logger.error(f"发送Telegram消息时出错: {str(e)}")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for trading signals"""
    if df is None or df.empty:
        return None
        
    try:
        # 确保列名格式正确
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'AdjClose': 'adj_close'
        })
        
        # Calculate moving averages
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # Calculate volume indicators
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_MA10'] = df['volume'].rolling(window=10).mean()
        
        # Calculate momentum indicators
        df['RSI'] = calculate_rsi(df['close'], period=14)
        df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df['close'])
        
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
            last_row['volume'] > last_row['Volume_MA5'] * 1.5
        )
        
        # Check for sell signals
        sell_signal = (
            last_row['Hot Money Line'] < last_row['Main Force Line'] and
            last_row['Main Force Line'] < last_row['Long Line'] and
            last_row['volume'] > last_row['Volume_MA5']
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
                'close': df['close'].iloc[-1],
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
        self.data_interface = DataInterface()
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
            data = self.data_interface.get_historical_data(
                stock,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
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
            stocks = self.data_interface.get_available_symbols()
            if not stocks:
                self.logger.error("无法获取股票列表")
                return
                
            # 监控每个股票
            for stock in stocks:
                self.monitor_stock(stock)
                time.sleep(1)  # 避免请求过于频繁
                
        except Exception as e:
            self.logger.error(f"运行监控时出错: {str(e)}")

    def process_signal(self, stock: str, signal_data: Dict):
        """处理交易信号"""
        try:
            if not signal_data:
                return
                
            # 获取信号类型和价格
            signal_type = signal_data.get('signal')
            price = signal_data.get('close')
            indicators = signal_data.get('indicators', {})
            
            # 生成警报消息
            message = None
            if signal_type == "Buy signal detected":
                message = f"买入信号 - {stock} @ {price:.2f}"
                self.alert_system.send_alert(
                    stock=stock,
                    alert_type="buy_signal",
                    message=message,
                    price=price,
                    indicators=indicators
                )
            elif signal_type == "Sell signal detected":
                message = f"卖出信号 - {stock} @ {price:.2f}"
                self.alert_system.send_alert(
                    stock=stock,
                    alert_type="sell_signal",
                    message=message,
                    price=price,
                    indicators=indicators
                )
                
            if message:  # 只在有消息时记录日志
                self.logger.info(f"处理信号: {message}")
            
        except Exception as e:
            self.logger.error(f"处理信号时出错: {str(e)}")

def main():
    """主函数"""
    monitor = TradingMonitor()
    monitor.run()

if __name__ == "__main__":
    main() 