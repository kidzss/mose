import logging
import datetime as dt
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("TradeManager")

class TradeManager:
    """交易管理器 - 负责执行交易操作和管理持仓"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_positions = config.get('max_positions', 10)
        self.position_size = config.get('position_size', 0.1)
        self.positions = {}  # 当前持仓
        self.orders = []     # 订单历史
        self.is_trading = False
        
    def start(self):
        """启动交易管理器"""
        if self.is_trading:
            logger.warning("交易管理器已经在运行")
            return
            
        self.is_trading = True
        logger.info("交易管理器启动成功")
        
    def stop(self):
        """停止交易管理器"""
        if not self.is_trading:
            logger.warning("交易管理器未在运行")
            return
            
        self.is_trading = False
        logger.info("交易管理器已停止")
        
    def handle_trade_signal(self, signal: Dict):
        """处理交易信号"""
        try:
            if not self.is_trading:
                logger.warning("交易管理器未启动，忽略交易信号")
                return
                
            symbol = signal.get('symbol')
            action = signal.get('action')  # 'buy' or 'sell'
            price = signal.get('price')
            
            if action == 'buy':
                self._process_buy_signal(symbol, price)
            elif action == 'sell':
                self._process_sell_signal(symbol, price)
                
        except Exception as e:
            logger.error(f"处理交易信号时出错: {e}")
            
    def _process_buy_signal(self, symbol: str, price: float):
        """处理买入信号"""
        try:
            # 检查是否达到最大持仓数
            if len(self.positions) >= self.max_positions:
                logger.warning(f"已达到最大持仓数 {self.max_positions}，忽略买入信号")
                return
                
            # 计算购买数量
            position_value = self.position_size * self._get_total_capital()
            quantity = int(position_value / price)
            
            if quantity <= 0:
                logger.warning("计算的购买数量为0，忽略买入信号")
                return
                
            # 记录交易
            order = {
                'symbol': symbol,
                'action': 'buy',
                'quantity': quantity,
                'price': price,
                'timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.orders.append(order)
            
            # 更新持仓
            self.positions[symbol] = {
                'quantity': quantity,
                'price': price,
                'timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"买入订单执行成功: {order}")
            
        except Exception as e:
            logger.error(f"处理买入信号时出错: {e}")
            
    def _process_sell_signal(self, symbol: str, price: float):
        """处理卖出信号"""
        try:
            if symbol not in self.positions:
                logger.warning(f"未持有 {symbol}，忽略卖出信号")
                return
                
            position = self.positions[symbol]
            quantity = position['quantity']
            
            # 记录交易
            order = {
                'symbol': symbol,
                'action': 'sell',
                'quantity': quantity,
                'price': price,
                'timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.orders.append(order)
            
            # 移除持仓
            del self.positions[symbol]
            
            logger.info(f"卖出订单执行成功: {order}")
            
        except Exception as e:
            logger.error(f"处理卖出信号时出错: {e}")
            
    def _get_total_capital(self) -> float:
        """获取总资本"""
        # 这里应该从资金管理模块获取实际资金，现在返回模拟值
        return 1000000.0
        
    def get_positions(self) -> Dict:
        """获取当前持仓"""
        return self.positions
        
    def get_orders(self) -> List[Dict]:
        """获取订单历史"""
        return self.orders
        
    def get_status(self) -> Dict:
        """获取交易管理器状态"""
        return {
            "is_trading": self.is_trading,
            "positions_count": len(self.positions),
            "orders_count": len(self.orders),
            "last_update": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def update_config(self, new_config: Dict):
        """更新配置"""
        self.config.update(new_config)
        self.max_positions = new_config.get('max_positions', self.max_positions)
        self.position_size = new_config.get('position_size', self.position_size)
        logger.info("交易管理器配置已更新") 