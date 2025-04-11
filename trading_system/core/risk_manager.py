import logging
import datetime as dt
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("RiskManager")

class RiskManager:
    """风险管理器 - 负责风险控制和监控"""
    
    def __init__(self, risk_per_trade: float = 0.02):
        self.risk_per_trade = risk_per_trade
        self.risk_metrics = {}
        self.alerts = []
        
    def handle_market_alert(self, alert: Dict):
        """处理市场警报"""
        try:
            # 记录警报
            self.alerts.append({
                'type': alert.get('type'),
                'message': alert.get('message'),
                'timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 根据警报类型更新风险指标
            alert_type = alert.get('type')
            if alert_type == 'price_change':
                self._update_price_risk(alert)
            elif alert_type == 'volume_spike':
                self._update_volume_risk(alert)
            elif alert_type == 'volatility':
                self._update_volatility_risk(alert)
                
            # 检查是否需要采取风险控制措施
            self._check_risk_thresholds()
            
        except Exception as e:
            logger.error(f"处理市场警报时出错: {e}")
            
    def _update_price_risk(self, alert: Dict):
        """更新价格风险指标"""
        symbol = alert.get('symbol')
        if symbol not in self.risk_metrics:
            self.risk_metrics[symbol] = {}
            
        self.risk_metrics[symbol]['price_change'] = alert.get('price_change', 0)
        self.risk_metrics[symbol]['last_update'] = dt.datetime.now()
        
    def _update_volume_risk(self, alert: Dict):
        """更新成交量风险指标"""
        symbol = alert.get('symbol')
        if symbol not in self.risk_metrics:
            self.risk_metrics[symbol] = {}
            
        self.risk_metrics[symbol]['volume_ratio'] = alert.get('volume_ratio', 1)
        self.risk_metrics[symbol]['last_update'] = dt.datetime.now()
        
    def _update_volatility_risk(self, alert: Dict):
        """更新波动率风险指标"""
        symbol = alert.get('symbol')
        if symbol not in self.risk_metrics:
            self.risk_metrics[symbol] = {}
            
        self.risk_metrics[symbol]['volatility'] = alert.get('volatility', 0)
        self.risk_metrics[symbol]['last_update'] = dt.datetime.now()
        
    def _check_risk_thresholds(self):
        """检查风险阈值"""
        try:
            for symbol, metrics in self.risk_metrics.items():
                # 检查价格变动风险
                if abs(metrics.get('price_change', 0)) > self.risk_per_trade:
                    self._generate_risk_alert(
                        symbol,
                        'high_price_risk',
                        f"价格变动超过风险阈值: {metrics['price_change']:.2%}"
                    )
                    
                # 检查成交量风险
                if metrics.get('volume_ratio', 1) > 3.0:
                    self._generate_risk_alert(
                        symbol,
                        'high_volume_risk',
                        f"成交量异常: {metrics['volume_ratio']:.2f}倍"
                    )
                    
                # 检查波动率风险
                if metrics.get('volatility', 0) > 0.4:  # 40%年化波动率
                    self._generate_risk_alert(
                        symbol,
                        'high_volatility_risk',
                        f"波动率过高: {metrics['volatility']:.2%}"
                    )
                    
        except Exception as e:
            logger.error(f"检查风险阈值时出错: {e}")
            
    def _generate_risk_alert(self, symbol: str, risk_type: str, message: str):
        """生成风险警报"""
        alert = {
            'symbol': symbol,
            'type': risk_type,
            'message': message,
            'timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.alerts.append(alert)
        logger.warning(f"风险警报: {alert}")
        
    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """计算建仓数量"""
        try:
            # 计算每股风险
            risk_per_share = abs(price - stop_loss)
            if risk_per_share <= 0:
                logger.warning("无效的止损价格")
                return 0
                
            # 计算可承受的最大损失
            max_loss = self._get_total_capital() * self.risk_per_trade
            
            # 计算建仓数量
            position_size = int(max_loss / risk_per_share)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"计算建仓数量时出错: {e}")
            return 0
            
    def _get_total_capital(self) -> float:
        """获取总资本"""
        # 这里应该从资金管理模块获取实际资金，现在返回模拟值
        return 1000000.0
        
    def get_risk_metrics(self) -> Dict:
        """获取风险指标"""
        return self.risk_metrics
        
    def get_alerts(self) -> List[Dict]:
        """获取警报历史"""
        return self.alerts
        
    def get_status(self) -> Dict:
        """获取风险管理器状态"""
        return {
            "monitored_symbols": len(self.risk_metrics),
            "alerts_count": len(self.alerts),
            "risk_per_trade": self.risk_per_trade,
            "last_update": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def update_config(self, risk_per_trade: float):
        """更新配置"""
        self.risk_per_trade = risk_per_trade
        logger.info(f"风险管理器配置已更新: risk_per_trade = {risk_per_trade}") 