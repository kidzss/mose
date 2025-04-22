import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class EmailConfig:
    sender_password: str  # 必需参数放在前面
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = "kidzss@gmail.com"
    receiver_emails: List[str] = field(default_factory=lambda: ["kidzss@gmail.com"])

@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""  # 根据实际情况设置密码
    database: str = "mose"

@dataclass
class StrategyConfig:
    cpgw_params: Dict = field(default_factory=lambda: {
        "long_period": 34,
        "hot_money_period": 14,
        "main_force_period": 34
    })
    
    # 策略阈值
    thresholds = {
        "buy": {
            "long_line_threshold": 12,
            "main_force_threshold": 8,
            "hot_money_threshold": 7.2
        },
        "sell": {
            "main_force_threshold": 80,
            "hot_money_threshold": 95,
            "long_line_threshold": 60
        }
    }

@dataclass
class MonitoringConfig:
    check_interval: int = 300  # 检查间隔（秒）
    retry_delay: int = 60  # 出错后重试延迟（秒）
    alert_cooldown: int = 3600  # 提醒冷却时间（秒）

@dataclass
class NotificationThreshold:
    price_change: float = 0.01  # 价格变动阈值（1%）
    volume_change: float = 1.2  # 成交量变动阈值（120%）
    market_volatility: float = 0.01  # 市场波动阈值（1%）

@dataclass
class TradingConfig:
    # 必需参数
    price_alert_threshold: float
    loss_alert_threshold: float
    profit_target: float
    stop_loss: float
    check_interval: int
    email_notifications: bool
    update_interval: int
    risk_thresholds: Dict
    notification_settings: Dict
    sector_specific_settings: Dict
    email: EmailConfig
    database: DatabaseConfig
    
    # 可选参数
    log_level: str = "INFO"
    log_file: str = "trading_monitor.log"
    stock_pool: Optional[List[str]] = None  # 可选的股票池
    
    def __post_init__(self):
        if self.stock_pool is None:
            self.stock_pool = []
        if self.email.receiver_emails is None:
            self.email.receiver_emails = []

# 默认配置
default_config = TradingConfig(
    price_alert_threshold=0.05,
    loss_alert_threshold=0.05,
    profit_target=0.25,
    stop_loss=0.15,
    check_interval=60,
    email_notifications=True,
    update_interval=60,
    risk_thresholds={
        'volatility': 0.02,
        'concentration': 0.3,
        'var': 0.1
    },
    notification_settings={
        'email': True,
        'slack': False,
        'telegram': False
    },
    sector_specific_settings={
        'semiconductor': {
            'stop_loss': 0.08,
            'price_alert_threshold': 0.03,
            'volatility_threshold': 0.03
        },
        'tech': {
            'stop_loss': 0.12,
            'price_alert_threshold': 0.04,
            'volatility_threshold': 0.025
        },
        'healthcare': {
            'stop_loss': 0.1,
            'price_alert_threshold': 0.04,
            'volatility_threshold': 0.02
        },
        'automotive': {
            'stop_loss': 0.15,
            'price_alert_threshold': 0.05,
            'volatility_threshold': 0.03
        }
    },
    email=EmailConfig(
        sender_password="wlkp dbbz xpgk rkhy"  # 替换为从Google生成的应用专用密码
    ),
    database=DatabaseConfig(
        password=""  # 设置正确的数据库密码
    )
) 