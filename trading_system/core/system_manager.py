import logging
import datetime as dt
from pathlib import Path
import json
import threading
import queue
from typing import Dict, List, Optional
from config.trading_config import DatabaseConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingSystem")

class SystemManager:
    """交易系统管理器 - 整个系统的核心控制器"""
    
    def __init__(self, config_path: str = "config/system_config.json"):
        self.config = self._load_config(config_path)
        # 使用DatabaseConfig覆盖数据库配置
        self.config['database'] = {
            'host': DatabaseConfig.host,
            'port': DatabaseConfig.port,
            'user': DatabaseConfig.user,
            'password': DatabaseConfig.password,
            'database': DatabaseConfig.database
        }
        self.is_running = False
        self.components = {}
        self.message_queue = queue.Queue()
        
        # 初始化系统组件
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> dict:
        """加载系统配置"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            "system": {
                "name": "TradingSystem",
                "version": "1.0.0",
                "mode": "development"
            },
            "monitoring": {
                "market_indices": ["SPY", "QQQ", "IWM"],
                "check_interval": 60,
                "alert_thresholds": {
                    "price_change": 0.02,
                    "volume_ratio": 2.0,
                    "volatility_ratio": 1.5
                }
            },
            "trading": {
                "max_positions": 10,
                "position_size": 0.1,
                "risk_per_trade": 0.02
            },
            "database": {
                "host": "localhost",
                "port": 3306,
                "database": "trading_system"
            }
        }
        
    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # 初始化数据管理器
            from .data_manager import DataManager
            db_config = {
                'host': DatabaseConfig.host,
                'port': DatabaseConfig.port,
                'user': DatabaseConfig.user,
                'password': DatabaseConfig.password,
                'database': DatabaseConfig.database
            }
            self.components['data_manager'] = DataManager(db_config)
            
            # 初始化市场监控器
            from .market_monitor import MarketMonitor
            self.components['market_monitor'] = MarketMonitor(
                market_indices=self.config['monitoring']['market_indices'],
                check_interval=self.config['monitoring']['check_interval'],
                alert_thresholds=self.config['monitoring']['alert_thresholds']
            )
            
            # 初始化交易管理器
            from .trade_manager import TradeManager
            self.components['trade_manager'] = TradeManager(
                self.config['trading']
            )
            
            # 初始化风险管理器
            from .risk_manager import RiskManager
            self.components['risk_manager'] = RiskManager(
                self.config['trading']['risk_per_trade']
            )
            
            logger.info("系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化系统组件失败: {e}")
            raise
            
    def start(self):
        """启动系统"""
        if self.is_running:
            logger.warning("系统已经在运行")
            return
            
        try:
            logger.info("启动交易系统...")
            self.is_running = True
            
            # 启动消息处理线程
            self.message_thread = threading.Thread(
                target=self._process_messages,
                daemon=True
            )
            self.message_thread.start()
            
            # 启动市场监控
            self.components['market_monitor'].start_monitoring()
            
            # 启动交易管理器
            self.components['trade_manager'].start()
            
            logger.info("交易系统启动成功")
            
        except Exception as e:
            logger.error(f"启动系统失败: {e}")
            self.is_running = False
            raise
            
    def stop(self):
        """停止系统"""
        if not self.is_running:
            logger.warning("系统未在运行")
            return
            
        try:
            logger.info("停止交易系统...")
            self.is_running = False
            
            # 停止所有组件
            for name, component in self.components.items():
                if hasattr(component, 'stop'):
                    component.stop()
                    
            logger.info("交易系统已停止")
            
        except Exception as e:
            logger.error(f"停止系统时出错: {e}")
            raise
            
    def _process_messages(self):
        """处理系统消息"""
        while self.is_running:
            try:
                message = self.message_queue.get(timeout=1.0)
                self._handle_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
                
    def _handle_message(self, message: Dict):
        """处理单个消息"""
        try:
            msg_type = message.get('type')
            if msg_type == 'market_alert':
                self.components['risk_manager'].handle_market_alert(message)
            elif msg_type == 'trade_signal':
                self.components['trade_manager'].handle_trade_signal(message)
            elif msg_type == 'system_error':
                self._handle_system_error(message)
        except Exception as e:
            logger.error(f"处理消息 {message} 时出错: {e}")
            
    def _handle_system_error(self, error_message: Dict):
        """处理系统错误"""
        logger.error(f"系统错误: {error_message}")
        # 根据错误级别采取相应措施
        severity = error_message.get('severity', 'low')
        if severity == 'high':
            self.stop()
            
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "is_running": self.is_running,
            "components_status": {
                name: component.get_status() 
                for name, component in self.components.items()
                if hasattr(component, 'get_status')
            },
            "last_update": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def update_config(self, new_config: Dict):
        """更新系统配置"""
        try:
            self.config.update(new_config)
            # 更新各组件的配置
            for name, component in self.components.items():
                if hasattr(component, 'update_config'):
                    component.update_config(new_config.get(name, {}))
            logger.info("系统配置更新成功")
        except Exception as e:
            logger.error(f"更新系统配置失败: {e}")
            raise 