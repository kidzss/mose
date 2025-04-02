import pandas as pd
import numpy as np
import datetime as dt
import time
import logging
from typing import List, Dict, Optional, Union, Tuple, Set
import threading
import json
import os
from pathlib import Path

from .data_fetcher import DataFetcher
from backtest.risk_manager import RiskManager
from backtest.volatility_manager import VolatilityManager
from backtest.strategy_factory import StrategyFactory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketMonitor")


class Alert:
    """警报类，用于存储和管理警报信息"""
    def __init__(
        self,
        symbol: str,
        alert_type: str,
        message: str,
        level: str = "info",
        timestamp: Optional[dt.datetime] = None,
        data: Optional[Dict] = None
    ):
        """
        初始化警报
        
        参数:
            symbol: 股票代码
            alert_type: 警报类型（如'price_alert', 'volatility_alert', 'risk_alert'等）
            message: 警报消息
            level: 警报级别（'info', 'warning', 'danger'）
            timestamp: 时间戳，默认为当前时间
            data: 附加数据
        """
        self.symbol = symbol
        self.alert_type = alert_type
        self.message = message
        self.level = level
        self.timestamp = timestamp or dt.datetime.now()
        self.data = data or {}
        self.id = f"{self.symbol}_{self.alert_type}_{int(self.timestamp.timestamp())}"
        self.is_read = False
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "alert_type": self.alert_type,
            "message": self.message,
            "level": self.level,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "is_read": self.is_read
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Alert':
        """从字典创建警报"""
        alert = cls(
            symbol=data["symbol"],
            alert_type=data["alert_type"],
            message=data["message"],
            level=data["level"],
            timestamp=dt.datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {})
        )
        alert.id = data["id"]
        alert.is_read = data.get("is_read", False)
        return alert


class MarketMonitor:
    """
    市场监控类，负责监控市场数据和生成警报
    
    功能：
    1. 监控市场指数（SPY、QQQ等）
    2. 判断市场环境（多头、空头、震荡）
    3. 触发交易策略执行
    """
    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        risk_manager: Optional[RiskManager] = None,
        strategy_factory: Optional[StrategyFactory] = None,
        config_file: str = "monitor_config.json",
        alert_history_file: str = "alert_history.json",
        check_interval: int = 300,  # 检查间隔（秒）
        max_alerts: int = 100,      # 最大警报数量
        mode: str = 'prod'          # 运行模式：'prod' 或 'dev'
    ):
        # 初始化组件
        self.data_fetcher = data_fetcher or DataFetcher()
        self.risk_manager = risk_manager
        self.strategy_factory = strategy_factory or StrategyFactory()
        
        # 配置
        self.config_file = config_file
        self.alert_history_file = alert_history_file
        self.check_interval = check_interval
        self.max_alerts = max_alerts
        self.mode = mode
        
        # 市场指数配置
        self.market_indices = {
            'SPY': '^GSPC',    # S&P 500
            'QQQ': '^NDX',     # NASDAQ 100
            'IWM': '^RUT'      # Russell 2000
        }
        
        # 市场状态
        self.market_state = {
            'trend': 'neutral',      # 市场趋势：bullish/bearish/neutral
            'volatility': 'normal',  # 波动性：high/normal/low
            'risk_level': 'medium'   # 风险水平：high/medium/low
        }
        
        # 初始化警报列表
        self.alerts = []
        
        # 初始化监控的股票集合
        self.monitored_symbols = set()
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化策略
        self.strategies = self.strategy_factory.create_all_strategies()
        
        # 初始化 stock_manager
        from monitor.stock_manager import StockManager
        self.stock_manager = StockManager(self.data_fetcher)
        
        logger.info(f"MarketMonitor初始化完成，运行模式: {mode}")
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"从 {self.config_file} 加载配置成功")
                return config
            else:
                # 创建默认配置
                default_config = {
                    "monitored_symbols": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
                    "price_alerts": {
                        "enabled": True,
                        "threshold_percent": 2.0  # 价格变动阈值（百分比）
                    },
                    "volatility_alerts": {
                        "enabled": True,
                        "threshold_ratio": 1.5  # 波动率比率阈值
                    },
                    "volume_alerts": {
                        "enabled": True,
                        "threshold_ratio": 2.0  # 成交量比率阈值
                    },
                    "risk_alerts": {
                        "enabled": True,
                        "max_drawdown": 0.1,  # 最大回撤阈值
                        "var_threshold": 0.03  # VaR阈值
                    }
                }
                
                # 保存默认配置
                self._save_config(default_config)
                logger.info(f"创建默认配置并保存到 {self.config_file}")
                return default_config
        except Exception as e:
            logger.error(f"加载配置文件时出错: {e}")
            return {}
            
    def _save_config(self, config: Dict = None) -> None:
        """保存配置文件"""
        try:
            config_to_save = config or self.config
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            logger.info(f"配置保存到 {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置文件时出错: {e}")
            
    def _load_alerts(self) -> None:
        """加载警报历史"""
        try:
            if os.path.exists(self.alert_history_file):
                with open(self.alert_history_file, 'r') as f:
                    alerts_data = json.load(f)
                
                self.alerts = [Alert.from_dict(alert_data) for alert_data in alerts_data]
                logger.info(f"从 {self.alert_history_file} 加载 {len(self.alerts)} 条警报")
        except Exception as e:
            logger.error(f"加载警报历史时出错: {e}")
            self.alerts = []
            
    def _save_alerts(self) -> None:
        """保存警报历史"""
        try:
            # 限制警报数量
            if len(self.alerts) > self.max_alerts:
                # 保留最新的警报
                self.alerts = sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:self.max_alerts]
                
            alerts_data = [alert.to_dict() for alert in self.alerts]
            
            # 确保目录存在
            Path(os.path.dirname(self.alert_history_file)).mkdir(parents=True, exist_ok=True)
            
            with open(self.alert_history_file, 'w') as f:
                json.dump(alerts_data, f, indent=4)
                
            logger.info(f"保存 {len(self.alerts)} 条警报到 {self.alert_history_file}")
        except Exception as e:
            logger.error(f"保存警报历史时出错: {e}")
            
    def add_alert(self, alert: Alert) -> None:
        """添加警报"""
        self.alerts.append(alert)
        logger.info(f"添加警报: {alert.symbol} - {alert.alert_type} - {alert.message}")
        
        # 保存警报历史
        self._save_alerts()
        
    def get_alerts(
        self,
        symbol: Optional[str] = None,
        alert_type: Optional[str] = None,
        level: Optional[str] = None,
        start_time: Optional[dt.datetime] = None,
        end_time: Optional[dt.datetime] = None,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Alert]:
        """获取警报"""
        filtered_alerts = self.alerts
        
        # 按条件筛选
        if symbol:
            filtered_alerts = [a for a in filtered_alerts if a.symbol == symbol]
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        if start_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= start_time]
        if end_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp <= end_time]
        if unread_only:
            filtered_alerts = [a for a in filtered_alerts if not a.is_read]
            
        # 按时间排序并限制数量
        filtered_alerts = sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return filtered_alerts
        
    def mark_alert_as_read(self, alert_id: str) -> bool:
        """将警报标记为已读"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.is_read = True
                self._save_alerts()
                return True
        return False
        
    def clear_alerts(self, older_than: Optional[dt.datetime] = None) -> int:
        """清除警报"""
        if older_than:
            old_count = len(self.alerts)
            self.alerts = [a for a in self.alerts if a.timestamp >= older_than]
            cleared_count = old_count - len(self.alerts)
        else:
            cleared_count = len(self.alerts)
            self.alerts = []
            
        self._save_alerts()
        return cleared_count
        
    def add_symbol(self, symbol: str) -> None:
        """添加监控的股票"""
        if symbol not in self.monitored_symbols:
            self.monitored_symbols.add(symbol)
            
            # 更新配置
            self.config["monitored_symbols"] = list(self.monitored_symbols)
            self._save_config()
            
            logger.info(f"添加监控股票: {symbol}")
            
    def remove_symbol(self, symbol: str) -> None:
        """移除监控的股票"""
        if symbol in self.monitored_symbols:
            self.monitored_symbols.remove(symbol)
            
            # 更新配置
            self.config["monitored_symbols"] = list(self.monitored_symbols)
            self._save_config()
            
            logger.info(f"移除监控股票: {symbol}")
            
    def start_monitoring(self) -> None:
        """启动监控"""
        if self.is_running:
            logger.warning("监控已经在运行中")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("开始市场监控")
        
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self.is_running:
            logger.warning("监控未在运行")
            return
            
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        logger.info("停止市场监控")
        
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_running:
            try:
                self.last_check_time = dt.datetime.now()
                logger.info(f"执行市场检查，监控 {len(self.monitored_symbols)} 只股票")
                
                # 检查市场数据
                self._check_market_data()
                
                # 检查风险指标
                if self.risk_manager:
                    self._check_risk_metrics()
                    
                # 等待下一次检查
                logger.info(f"检查完成，等待 {self.check_interval} 秒后进行下一次检查")
                for _ in range(self.check_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"监控循环中出错: {e}")
                time.sleep(60)  # 出错后等待一分钟再重试
                
    def _check_market_data(self) -> None:
        """检查市场数据"""
        try:
            # 获取最新数据
            symbols = list(self.monitored_symbols)
            latest_data = self.data_fetcher.get_latest_data(symbols, days=5)
            
            for symbol, data in latest_data.items():
                if data.empty:
                    logger.warning(f"未获取到 {symbol} 的数据")
                    continue
                    
                # 检查价格变动
                self._check_price_change(symbol, data)
                
                # 检查波动率
                self._check_volatility(symbol, data)
                
                # 检查成交量
                self._check_volume(symbol, data)
        except Exception as e:
            logger.error(f"检查市场数据时出错: {e}")
            
    def _check_price_change(self, symbol: str, data: pd.DataFrame) -> None:
        """检查价格变动"""
        if not self.config.get("price_alerts", {}).get("enabled", True):
            return
            
        try:
            # 获取最新价格和前一天价格
            if len(data) < 2:
                return
                
            latest_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            
            # 计算价格变动百分比
            price_change_pct = (latest_price - prev_price) / prev_price * 100
            
            # 检查是否超过阈值
            threshold = self.config.get("price_alerts", {}).get("threshold_percent", 2.0)
            
            if abs(price_change_pct) >= threshold:
                direction = "上涨" if price_change_pct > 0 else "下跌"
                message = f"{symbol} 价格{direction} {abs(price_change_pct):.2f}%，超过阈值 {threshold}%"
                level = "info" if price_change_pct > 0 else "warning"
                
                # 创建警报
                alert = Alert(
                    symbol=symbol,
                    alert_type="price_alert",
                    message=message,
                    level=level,
                    data={
                        "price": latest_price,
                        "prev_price": prev_price,
                        "change_percent": price_change_pct,
                        "threshold": threshold
                    }
                )
                
                self.add_alert(alert)
        except Exception as e:
            logger.error(f"检查 {symbol} 的价格变动时出错: {e}")
            
    def _check_volatility(self, symbol: str, data: pd.DataFrame) -> None:
        """检查波动率"""
        if not self.config.get("volatility_alerts", {}).get("enabled", True):
            return
            
        try:
            # 计算波动率
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 10:
                return
                
            # 计算当前波动率（最近5天）和历史波动率（全部数据）
            current_vol = returns.tail(5).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            
            # 计算波动率比率
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            # 检查是否超过阈值
            threshold = self.config.get("volatility_alerts", {}).get("threshold_ratio", 1.5)
            
            if vol_ratio >= threshold:
                message = f"{symbol} 波动率比率为 {vol_ratio:.2f}，超过阈值 {threshold}"
                
                # 创建警报
                alert = Alert(
                    symbol=symbol,
                    alert_type="volatility_alert",
                    message=message,
                    level="warning",
                    data={
                        "current_volatility": current_vol,
                        "historical_volatility": historical_vol,
                        "volatility_ratio": vol_ratio,
                        "threshold": threshold
                    }
                )
                
                self.add_alert(alert)
        except Exception as e:
            logger.error(f"检查 {symbol} 的波动率时出错: {e}")
            
    def _check_volume(self, symbol: str, data: pd.DataFrame) -> None:
        """检查成交量"""
        if not self.config.get("volume_alerts", {}).get("enabled", True):
            return
            
        try:
            # 获取最新成交量和平均成交量
            if len(data) < 5:
                return
                
            latest_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].iloc[:-1].mean()
            
            # 计算成交量比率
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 检查是否超过阈值
            threshold = self.config.get("volume_alerts", {}).get("threshold_ratio", 2.0)
            
            if volume_ratio >= threshold:
                message = f"{symbol} 成交量为平均值的 {volume_ratio:.2f} 倍，超过阈值 {threshold}"
                
                # 创建警报
                alert = Alert(
                    symbol=symbol,
                    alert_type="volume_alert",
                    message=message,
                    level="info",
                    data={
                        "volume": latest_volume,
                        "avg_volume": avg_volume,
                        "volume_ratio": volume_ratio,
                        "threshold": threshold
                    }
                )
                
                self.add_alert(alert)
        except Exception as e:
            logger.error(f"检查 {symbol} 的成交量时出错: {e}")
            
    def _check_risk_metrics(self) -> None:
        """检查风险指标"""
        if not self.risk_manager or not self.config.get("risk_alerts", {}).get("enabled", True):
            return
            
        try:
            # 获取最新数据
            symbols = list(self.monitored_symbols)
            end_date = dt.datetime.now().strftime('%Y-%m-%d')
            start_date = (dt.datetime.now() - dt.timedelta(days=30)).strftime('%Y-%m-%d')
            
            latest_data = {}
            for symbol in symbols:
                data = self.data_fetcher.get_historical_data(symbol, start_date=start_date, end_date=end_date)
                if not data.empty:
                    latest_data[symbol] = data
            
            for symbol, data in latest_data.items():
                if data.empty or len(data) < 20:
                    continue
                    
                # 计算收益率
                returns = data['Close'].pct_change().dropna()
                
                # 计算持仓
                positions = pd.Series(1, index=returns.index)  # 假设持有1单位
                
                # 计算风险指标
                risk_metrics = self.risk_manager.evaluate_risk(returns, positions)
                
                # 检查最大回撤
                max_drawdown_threshold = self.config.get("risk_alerts", {}).get("max_drawdown", 0.1)
                if risk_metrics.max_drawdown >= max_drawdown_threshold:
                    message = f"{symbol} 最大回撤为 {risk_metrics.max_drawdown:.2%}，超过阈值 {max_drawdown_threshold:.2%}"
                    
                    # 创建警报
                    alert = Alert(
                        symbol=symbol,
                        alert_type="drawdown_alert",
                        message=message,
                        level="danger",
                        data={
                            "max_drawdown": risk_metrics.max_drawdown,
                            "threshold": max_drawdown_threshold
                        }
                    )
                    
                    self.add_alert(alert)
                    
                # 检查VaR
                var_threshold = self.config.get("risk_alerts", {}).get("var_threshold", 0.03)
                if risk_metrics.value_at_risk >= var_threshold:
                    message = f"{symbol} VaR为 {risk_metrics.value_at_risk:.2%}，超过阈值 {var_threshold:.2%}"
                    
                    # 创建警报
                    alert = Alert(
                        symbol=symbol,
                        alert_type="var_alert",
                        message=message,
                        level="warning",
                        data={
                            "var": risk_metrics.value_at_risk,
                            "threshold": var_threshold
                        }
                    )
                    
                    self.add_alert(alert)
        except Exception as e:
            logger.error(f"检查风险指标时出错: {e}")
            
    def generate_market_report(self) -> Dict:
        """生成市场报告"""
        try:
            report = {
                "timestamp": dt.datetime.now().isoformat(),
                "monitored_symbols": list(self.monitored_symbols),
                "alerts": {
                    "total": len(self.alerts),
                    "unread": len([a for a in self.alerts if not a.is_read]),
                    "by_level": {
                        "info": len([a for a in self.alerts if a.level == "info"]),
                        "warning": len([a for a in self.alerts if a.level == "warning"]),
                        "danger": len([a for a in self.alerts if a.level == "danger"])
                    },
                    "recent": [a.to_dict() for a in self.get_alerts(limit=5)]
                },
                "market_data": {}
            }
            
            # 获取市场数据
            for symbol in self.monitored_symbols:
                try:
                    data = self.data_fetcher.get_latest_data([symbol], days=1)[symbol]
                    if not data.empty:
                        latest = data.iloc[-1]
                        report["market_data"][symbol] = {
                            "price": latest["Close"],
                            "change": latest.get("Change", 0),
                            "change_percent": latest.get("ChangePercent", 0),
                            "volume": latest["Volume"]
                        }
                except Exception as e:
                    logger.error(f"获取 {symbol} 的市场数据时出错: {e}")
                    
            return report
        except Exception as e:
            logger.error(f"生成市场报告时出错: {e}")
            return {
                "error": str(e),
                "timestamp": dt.datetime.now().isoformat()
            }

    def analyze_market_condition(self) -> Dict[str, str]:
        """
        分析市场状况
        
        返回:
            市场状态字典
        """
        try:
            # 获取市场指数数据
            end_date = dt.datetime.now().strftime('%Y-%m-%d')
            start_date = (dt.datetime.now() - dt.timedelta(days=20)).strftime('%Y-%m-%d')
            
            index_data = {}
            for name, symbol in self.market_indices.items():
                data = self.data_fetcher.get_historical_data(symbol, start_date, end_date)
                if not data.empty:
                    index_data[name] = data
                    
            if not index_data:
                logger.error("未获取到任何市场指数数据")
                return self.market_state
                
            # 分析趋势
            trend = self._analyze_trend(index_data)
            
            # 分析波动性
            volatility = self._analyze_volatility(index_data)
            
            # 分析风险水平
            risk_level = self._analyze_risk_level(index_data)
            
            # 更新市场状态
            self.market_state = {
                'trend': trend,
                'volatility': volatility,
                'risk_level': risk_level
            }
            
            logger.info(f"市场状态更新:\n趋势: {trend}\n波动性: {volatility}\n风险水平: {risk_level}")
            
            return self.market_state
            
        except Exception as e:
            logger.error(f"分析市场状况时出错: {e}")
            return self.market_state
            
    def check_risk_metrics(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        检查风险指标
        
        参数:
            data: 股票数据字典，键为股票代码，值为DataFrame
        """
        try:
            for symbol, df in data.items():
                if df.empty:
                    continue
                    
                # 计算最大回撤
                if 'Close' in df.columns:
                    price_series = df['Close']
                    rolling_max = price_series.expanding().max()
                    drawdown = (price_series - rolling_max) / rolling_max
                    max_drawdown = abs(drawdown.min())
                    
                    # 如果最大回撤超过阈值，生成警报
                    if max_drawdown > self.config.get('risk_alerts', {}).get('max_drawdown', 0.1):
                        alert = Alert(
                            symbol=symbol,
                            alert_type='risk_alert',
                            message=f"最大回撤 ({max_drawdown:.2%}) 超过阈值",
                            level='warning',
                            data={'max_drawdown': max_drawdown}
                        )
                        self.add_alert(alert)
                        
        except Exception as e:
            logger.error(f"检查风险指标时出错: {e}")
            
    def _analyze_trend(self, index_data: Dict[str, pd.DataFrame]) -> str:
        """分析市场趋势"""
        try:
            # 使用S&P 500作为主要参考
            spy_data = index_data.get('SPY')
            if spy_data is None or spy_data.empty:
                return 'neutral'
                
            # 计算20日移动平均线
            if 'Close' in spy_data.columns:
                ma20 = spy_data['Close'].rolling(window=20).mean()
                current_price = spy_data['Close'].iloc[-1]
                
                # 计算趋势强度
                price_change = (current_price - spy_data['Close'].iloc[0]) / spy_data['Close'].iloc[0]
                
                if current_price > ma20.iloc[-1] and price_change > 0.05:
                    return 'bull'
                elif current_price < ma20.iloc[-1] and price_change < -0.05:
                    return 'bear'
                    
            return 'neutral'
            
        except Exception as e:
            logger.error(f"分析趋势时出错: {e}")
            return 'neutral'
            
    def _analyze_volatility(self, index_data: Dict[str, pd.DataFrame]) -> str:
        """分析市场波动性"""
        try:
            # 使用VIX指数或计算波动率
            volatilities = []
            
            for data in index_data.values():
                if 'Close' in data.columns:
                    # 计算日收益率的标准差
                    returns = data['Close'].pct_change().dropna()
                    vol = returns.std() * np.sqrt(252)  # 年化波动率
                    volatilities.append(vol)
                    
            if volatilities:
                avg_vol = np.mean(volatilities)
                if avg_vol > 0.25:  # 25%年化波动率
                    return 'high'
                elif avg_vol < 0.15:  # 15%年化波动率
                    return 'low'
                    
            return 'normal'
            
        except Exception as e:
            logger.error(f"分析波动性时出错: {e}")
            return 'normal'
            
    def _analyze_risk_level(self, index_data: Dict[str, pd.DataFrame]) -> str:
        """分析风险水平"""
        try:
            risk_scores = []
            
            for data in index_data.values():
                if 'Close' in data.columns:
                    # 计算风险指标
                    returns = data['Close'].pct_change().dropna()
                    
                    # 计算夏普比率
                    sharpe = np.sqrt(252) * returns.mean() / returns.std()
                    
                    # 计算最大回撤
                    cum_returns = (1 + returns).cumprod()
                    rolling_max = cum_returns.expanding().max()
                    drawdowns = (cum_returns - rolling_max) / rolling_max
                    max_drawdown = abs(drawdowns.min())
                    
                    # 综合风险分数
                    risk_score = max_drawdown - sharpe * 0.1
                    risk_scores.append(risk_score)
                    
            if risk_scores:
                avg_risk = np.mean(risk_scores)
                if avg_risk > 0.2:
                    return 'high'
                elif avg_risk < 0.1:
                    return 'low'
                    
            return 'medium'
            
        except Exception as e:
            logger.error(f"分析风险水平时出错: {e}")
            return 'medium' 