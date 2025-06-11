<<<<<<< HEAD
import pandas as pd
import numpy as np
from openbb_terminal.sdk import openbb
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('amd_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AMDStrategy:
    def __init__(self):
        load_dotenv()
        self.symbol = "AMD"
        self.price_threshold = 95.0
        self.ma_period = 13
        self.email_sender = os.getenv('EMAIL_SENDER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_receiver = os.getenv('EMAIL_RECEIVER')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        
    def get_stock_data(self):
        """获取AMD的股票数据"""
        try:
            # 获取过去30天的数据以确保有足够的数据计算均线
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            df = openbb.stocks.load(
                symbol=self.symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # 计算13日均线
            df['MA13'] = df['Close'].rolling(window=self.ma_period).mean()
            return df
        except Exception as e:
            logger.error(f"获取股票数据时出错: {str(e)}")
            return None

    def send_email(self, subject, message):
        """发送邮件通知"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_receiver
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_sender, self.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"邮件发送成功: {subject}")
        except Exception as e:
            logger.error(f"发送邮件时出错: {str(e)}")

    def check_signals(self):
        """检查交易信号"""
        df = self.get_stock_data()
        if df is None or df.empty:
            logger.error("无法获取股票数据")
            return

        # 获取最新的数据点
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        current_price = latest['Close']
        current_ma = latest['MA13']
        previous_price = previous['Close']
        previous_ma = previous['MA13']

        # 检查卖出信号
        if current_price < self.price_threshold or current_price < current_ma:
            if previous_price >= self.price_threshold and previous_price >= previous_ma:
                message = f"""
                AMD卖出信号触发:
                当前价格: ${current_price:.2f}
                13日均线: ${current_ma:.2f}
                价格阈值: ${self.price_threshold:.2f}
                时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                self.send_email("AMD卖出信号", message)
                logger.info("卖出信号已触发")

        # 检查买入信号
        elif (current_price >= self.price_threshold or current_price >= current_ma) and \
             (previous_price < self.price_threshold or previous_price < previous_ma):
            message = f"""
            AMD买入信号触发:
            当前价格: ${current_price:.2f}
            13日均线: ${current_ma:.2f}
            价格阈值: ${self.price_threshold:.2f}
            时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            self.send_email("AMD买入信号", message)
            logger.info("买入信号已触发")

def main():
    strategy = AMDStrategy()
    strategy.check_signals()

if __name__ == "__main__":
=======
import pandas as pd
import numpy as np
from openbb_terminal.sdk import openbb
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('amd_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AMDStrategy:
    def __init__(self):
        load_dotenv()
        self.symbol = "AMD"
        self.price_threshold = 95.0
        self.ma_period = 13
        self.email_sender = os.getenv('EMAIL_SENDER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_receiver = os.getenv('EMAIL_RECEIVER')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        
    def get_stock_data(self):
        """获取AMD的股票数据"""
        try:
            # 获取过去30天的数据以确保有足够的数据计算均线
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            df = openbb.stocks.load(
                symbol=self.symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # 计算13日均线
            df['MA13'] = df['Close'].rolling(window=self.ma_period).mean()
            return df
        except Exception as e:
            logger.error(f"获取股票数据时出错: {str(e)}")
            return None

    def send_email(self, subject, message):
        """发送邮件通知"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_receiver
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_sender, self.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"邮件发送成功: {subject}")
        except Exception as e:
            logger.error(f"发送邮件时出错: {str(e)}")

    def check_signals(self):
        """检查交易信号"""
        df = self.get_stock_data()
        if df is None or df.empty:
            logger.error("无法获取股票数据")
            return

        # 获取最新的数据点
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        current_price = latest['Close']
        current_ma = latest['MA13']
        previous_price = previous['Close']
        previous_ma = previous['MA13']

        # 检查卖出信号
        if current_price < self.price_threshold or current_price < current_ma:
            if previous_price >= self.price_threshold and previous_price >= previous_ma:
                message = f"""
                AMD卖出信号触发:
                当前价格: ${current_price:.2f}
                13日均线: ${current_ma:.2f}
                价格阈值: ${self.price_threshold:.2f}
                时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                self.send_email("AMD卖出信号", message)
                logger.info("卖出信号已触发")

        # 检查买入信号
        elif (current_price >= self.price_threshold or current_price >= current_ma) and \
             (previous_price < self.price_threshold or previous_price < previous_ma):
            message = f"""
            AMD买入信号触发:
            当前价格: ${current_price:.2f}
            13日均线: ${current_ma:.2f}
            价格阈值: ${self.price_threshold:.2f}
            时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            self.send_email("AMD买入信号", message)
            logger.info("买入信号已触发")

def main():
    strategy = AMDStrategy()
    strategy.check_signals()

if __name__ == "__main__":
>>>>>>> 3d7330be7ea0ecb409ac485e1c8391bc6d56a2de
    main() 