"""
数据源配置文件
包含了各种数据源的配置信息
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from config.trading_config import DatabaseConfig

@dataclass
class MySQLConfig:
    """MySQL数据库配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "mose"


@dataclass
class FutuConfig:
    """富途API配置"""
    ip: str = "127.0.0.1"
    port: int = 11111
    password: str = ""
    skip_auth: bool = True  # 默认跳过认证，适用于仅需要行情数据的场景


@dataclass
class YahooFinanceConfig:
    """Yahoo Finance API配置"""
    proxy: Optional[str] = None  # 代理服务器地址，如 "http://127.0.0.1:7890"
    timeout: int = 30  # 请求超时时间(秒)
    max_retries: int = 3  # 最大重试次数


@dataclass
class DataConfig:
    """数据配置总类"""
    # 默认数据源，可以是 'mysql', 'futu', 'yahoo' 等
    default_source: str = "mysql"
    
    # 各数据源配置
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    futu: FutuConfig = field(default_factory=FutuConfig)
    yahoo: YahooFinanceConfig = field(default_factory=YahooFinanceConfig)
    
    # 数据处理配置
    cache_dir: str = "data/cache"  # 数据缓存目录
    use_cache: bool = True  # 是否使用本地缓存
    cache_expiry_days: int = 7  # 缓存过期天数
    
    # 基本数据配置 
    default_lookback_days: int = 252  # 默认回溯天数（约1年交易日）
    
    def get_mysql_dict(self) -> Dict:
        """获取MySQL配置字典"""
        return {
            'host': self.mysql.host,
            'port': self.mysql.port,
            'user': self.mysql.user,
            'password': self.mysql.password,
            'database': self.mysql.database
        }
    
    def get_futu_dict(self) -> Dict:
        """获取富途配置字典"""
        return {
            'ip': self.futu.ip,
            'port': self.futu.port,
            'password': self.futu.password,
            'skip_auth': self.futu.skip_auth
        }
    
    def get_yahoo_dict(self) -> Dict:
        """获取Yahoo Finance配置字典"""
        return {
            'proxy': self.yahoo.proxy,
            'timeout': self.yahoo.timeout,
            'max_retries': self.yahoo.max_retries
        }
    
    def get_source_config(self, source_name: str) -> Dict:
        """
        获取指定数据源的配置
        
        参数:
            source_name: 数据源名称 ('mysql', 'futu', 'yahoo')
            
        返回:
            对应数据源的配置字典
        """
        return {
            'mysql': self.get_mysql_dict(),
            'futu': self.get_futu_dict(),
            'yahoo': self.get_yahoo_dict()
        }.get(source_name, {})
    
    def get_all_configs(self) -> Dict:
        """获取所有数据源配置"""
        return {
            'mysql': self.get_mysql_dict(),
            'futu': self.get_futu_dict(),
            'yahoo': self.get_yahoo_dict()
        }


# 默认配置实例
default_data_config = DataConfig() 