import logging
from typing import List, Dict, Optional
from datetime import datetime
import os

from config.trading_config import default_config
from monitor.trading_monitor import AlertSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NotificationManager")

class NotificationManager:
    """通知管理器，处理各种提醒功能"""
    
    def __init__(self):
        """初始化通知管理器"""
        self.alert_system = AlertSystem(default_config)
        self.thresholds = default_config.notification_threshold
        
    def send_trade_signal(self, stock: str, action: str, price: float, reason: str):
        """
        发送交易信号提醒
        
        参数:
            stock: 股票代码
            action: 操作（买入/卖出）
            price: 当前价格
            reason: 交易原因
        """
        subject = f"交易信号提醒 - {stock} {action}"
        body = f"""
        <h2>交易信号提醒</h2>
        <p><strong>股票代码:</strong> {stock}</p>
        <p><strong>操作:</strong> {action}</p>
        <p><strong>价格:</strong> {price:.2f}</p>
        <p><strong>原因:</strong> {reason}</p>
        <p><strong>时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        self.alert_system.send_email(subject, body)
        
    def send_market_alert(self, market_status: Dict):
        """
        发送市场状况提醒
        
        参数:
            market_status: 市场状况信息字典
        """
        subject = f"市场状况提醒 - {market_status.get('market_condition', 'Unknown')}"
        body = f"""
        <h2>市场状况提醒</h2>
        <p><strong>市场状况:</strong> {market_status.get('market_condition')}</p>
        <p><strong>风险等级:</strong> {market_status.get('risk_level')}</p>
        <p><strong>机会板块:</strong> {', '.join(market_status.get('opportunity_sectors', []))}</p>
        <p><strong>时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        self.alert_system.send_email(subject, body)
        
    def send_volatility_alert(self, stock: str, volatility: float, avg_volatility: float):
        """
        发送波动性提醒
        
        参数:
            stock: 股票代码
            volatility: 当前波动率
            avg_volatility: 平均波动率
        """
        if volatility > avg_volatility * (1 + self.thresholds.market_volatility):
            subject = f"波动性提醒 - {stock}"
            body = f"""
            <h2>波动性提醒</h2>
            <p><strong>股票代码:</strong> {stock}</p>
            <p><strong>当前波动率:</strong> {volatility:.2%}</p>
            <p><strong>平均波动率:</strong> {avg_volatility:.2%}</p>
            <p><strong>时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            self.alert_system.send_email(subject, body)
            
    def send_price_alert(self, stock: str, current_price: float, prev_price: float):
        """
        发送价格变动提醒
        
        参数:
            stock: 股票代码
            current_price: 当前价格
            prev_price: 之前价格
        """
        price_change = (current_price - prev_price) / prev_price
        if abs(price_change) > self.thresholds.price_change:
            subject = f"价格变动提醒 - {stock}"
            body = f"""
            <h2>价格变动提醒</h2>
            <p><strong>股票代码:</strong> {stock}</p>
            <p><strong>当前价格:</strong> {current_price:.2f}</p>
            <p><strong>价格变动:</strong> {price_change:.2%}</p>
            <p><strong>时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            self.alert_system.send_email(subject, body) 