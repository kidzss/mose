import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import datetime as dt
import threading
import time

logger = logging.getLogger("MarketMonitor")

class MarketMonitor:
    """市场监控器 - 负责监控市场状况和生成警报"""
    
    def __init__(
        self,
        market_indices: List[str],
        check_interval: int = 60,
        alert_thresholds: Dict = None
    ):
        self.market_indices = market_indices
        self.check_interval = check_interval
        self.alert_thresholds = alert_thresholds or {
            "price_change": 0.02,
            "volume_ratio": 2.0,
            "volatility_ratio": 1.5
        }
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.market_conditions = {}
        self.alerts = []
        
    def start_monitoring(self):
        """启动市场监控"""
        if self.is_monitoring:
            logger.warning("市场监控已在运行")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("市场监控已启动")
        
    def stop_monitoring(self):
        """停止市场监控"""
        if not self.is_monitoring:
            logger.warning("市场监控未在运行")
            return
            
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("市场监控已停止")
        
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 分析市场状况
                self._analyze_market_conditions()
                
                # 检查是否需要生成警报
                self._check_alerts()
                
                # 等待下一次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"市场监控循环出错: {e}")
                time.sleep(60)  # 出错后等待一分钟再重试
                
    def _analyze_market_conditions(self):
        """分析市场状况"""
        try:
            # 获取市场指数数据
            from .data_manager import DataManager
            from config.trading_config import DatabaseConfig
            
            # 使用正确的数据库配置
            db_config = {
                'host': DatabaseConfig.host,
                'port': DatabaseConfig.port,
                'user': DatabaseConfig.user,
                'password': DatabaseConfig.password,
                'database': DatabaseConfig.database
            }
            data_manager = DataManager(db_config)
            
            # 获取最近30天的数据
            market_data = data_manager.get_market_data(
                self.market_indices,
                start_date=(dt.datetime.now() - dt.timedelta(days=30)).strftime('%Y-%m-%d')
            )
            
            # 分析每个指数
            for symbol, data in market_data.items():
                if data.empty:
                    continue
                    
                # 计算技术指标
                analysis = self._calculate_technical_indicators(data)
                
                # 更新市场状况
                self.market_conditions[symbol] = {
                    "last_price": data['Close'].iloc[-1],
                    "price_change": data['Close'].pct_change().iloc[-1],
                    "volume_ratio": data['Volume'].iloc[-1] / data['Volume'].iloc[-20:].mean(),
                    "volatility": data['Close'].pct_change().std() * np.sqrt(252),
                    "trend": analysis['trend'],
                    "support_resistance": analysis['support_resistance'],
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
        except Exception as e:
            logger.error(f"分析市场状况时出错: {e}")
            
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """计算技术指标"""
        try:
            # 计算移动平均
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # 判断趋势
            current_price = data['Close'].iloc[-1]
            ma20 = data['MA20'].iloc[-1]
            ma50 = data['MA50'].iloc[-1]
            
            if current_price > ma20 > ma50:
                trend = "uptrend"
            elif current_price < ma20 < ma50:
                trend = "downtrend"
            else:
                trend = "sideways"
                
            # 计算支撑位和阻力位
            price_range = data['Close'].iloc[-20:]
            support = price_range.min()
            resistance = price_range.max()
            
            return {
                "trend": trend,
                "support_resistance": {
                    "support": support,
                    "resistance": resistance
                }
            }
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            return {"trend": "unknown", "support_resistance": {}}
            
    def _check_alerts(self):
        """检查是否需要生成警报"""
        try:
            for symbol, conditions in self.market_conditions.items():
                # 检查价格变动
                if abs(conditions['price_change']) > self.alert_thresholds['price_change']:
                    self._generate_alert(
                        symbol,
                        "price_change",
                        f"价格变动 {conditions['price_change']:.2%}"
                    )
                    
                # 检查成交量异常
                if conditions['volume_ratio'] > self.alert_thresholds['volume_ratio']:
                    self._generate_alert(
                        symbol,
                        "volume_spike",
                        f"成交量放大 {conditions['volume_ratio']:.2f} 倍"
                    )
                    
                # 检查趋势变化
                if conditions['trend'] != self._get_previous_trend(symbol):
                    self._generate_alert(
                        symbol,
                        "trend_change",
                        f"趋势改变为 {conditions['trend']}"
                    )
                    
        except Exception as e:
            logger.error(f"检查警报时出错: {e}")
            
    def _generate_alert(self, symbol: str, alert_type: str, message: str):
        """生成警报"""
        alert = {
            "symbol": symbol,
            "type": alert_type,
            "message": message,
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.alerts.append(alert)
        logger.info(f"生成警报: {alert}")
        
    def _get_previous_trend(self, symbol: str) -> str:
        """获取之前的趋势"""
        # 这里应该从历史数据中获取，现在简化处理
        return "unknown"
        
    def get_market_status(self) -> Dict:
        """获取市场状态"""
        return {
            "market_conditions": self.market_conditions,
            "latest_alerts": self.alerts[-10:],  # 最近10条警报
            "monitoring_status": {
                "is_monitoring": self.is_monitoring,
                "last_update": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
    def get_status(self) -> Dict:
        """获取监控器状态"""
        return {
            "is_monitoring": self.is_monitoring,
            "indices_count": len(self.market_indices),
            "alerts_count": len(self.alerts),
            "last_update": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def update_config(self, new_config: Dict):
        """更新配置"""
        if 'market_indices' in new_config:
            self.market_indices = new_config['market_indices']
        if 'check_interval' in new_config:
            self.check_interval = new_config['check_interval']
        if 'alert_thresholds' in new_config:
            self.alert_thresholds.update(new_config['alert_thresholds'])
        logger.info("市场监控配置已更新") 