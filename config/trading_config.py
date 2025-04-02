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
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""  # MySQL没有密码
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
    price_change: float = 0.05  # 价格变动阈值（5%）
    volume_change: float = 2.0  # 成交量变动阈值（200%）
    market_volatility: float = 0.03  # 市场波动阈值（3%）

@dataclass
class TradingConfig:
    email: EmailConfig
    database: DatabaseConfig
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    notification_threshold: NotificationThreshold = field(default_factory=NotificationThreshold)
    stock_pool: Optional[List[str]] = None  # 可选的股票池
    log_level: str = "INFO"
    log_file: str = "trading_monitor.log"
    
    def __post_init__(self):
        if self.stock_pool is None:
            self.stock_pool = []
        if self.email.receiver_emails is None:
            self.email.receiver_emails = []

# 默认配置
default_config = TradingConfig(
    email=EmailConfig(
        sender_password="wlkp dbbz xpgk rkhy"  # 替换为从Google生成的应用专用密码
    ),
    database=DatabaseConfig(
        password=""
    )
) 