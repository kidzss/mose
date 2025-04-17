import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PositionManager:
    """仓位管理类，用于管理不同策略的仓位"""
    
    def __init__(self):
        """初始化仓位管理器"""
        # 策略分类
        self.strategy_types = {
            'short_term': ['intraday', 'breakout'],  # 短期策略
            'medium_term': ['trend', 'momentum', 'mean_reversion', 'bollinger'],  # 中期策略
            'long_term': ['cpgw', 'niuniu']  # 长期策略
        }
        
        # 仓位限制
        self.position_limits = {
            'short_term': 0.08,  # 短期策略仓位上限
            'medium_term': 0.12,  # 中期策略仓位上限
            'long_term': 0.08,  # 长期策略仓位上限
            'total': 0.20  # 总仓位上限
        }
        
        # 当前仓位
        self.current_positions = {
            'short_term': {},
            'medium_term': {},
            'long_term': {}
        }
        
        # 止损设置
        self.stop_loss = {
            'short_term': 0.03,  # 短期策略止损
            'medium_term': 0.07,  # 中期策略止损
            'long_term': 0.10  # 长期策略止损
        }
        
        # 止盈设置
        self.take_profit = {
            'short_term': 0.05,  # 短期策略止盈
            'medium_term': 0.12,  # 中期策略止盈
            'long_term': 0.20  # 长期策略止盈
        }
    
    def get_strategy_type(self, strategy_name: str) -> str:
        """获取策略类型"""
        for stype, strategies in self.strategy_types.items():
            if strategy_name in strategies:
                return stype
        return None
    
    def can_open_position(self, strategy_name: str, symbol: str, position_size: float) -> bool:
        """检查是否可以开仓"""
        stype = self.get_strategy_type(strategy_name)
        if not stype:
            return False
            
        # 检查策略类型仓位限制
        current_type_position = sum(self.current_positions[stype].values())
        if current_type_position + position_size > self.position_limits[stype]:
            return False
            
        # 检查总仓位限制
        total_position = sum(sum(positions.values()) for positions in self.current_positions.values())
        if total_position + position_size > self.position_limits['total']:
            return False
            
        return True
    
    def open_position(self, strategy_name: str, symbol: str, position_size: float, entry_price: float):
        """开仓"""
        stype = self.get_strategy_type(strategy_name)
        if not stype:
            return False
            
        if not self.can_open_position(strategy_name, symbol, position_size):
            return False
            
        # 记录仓位
        if symbol not in self.current_positions[stype]:
            self.current_positions[stype][symbol] = {
                'size': position_size,
                'entry_price': entry_price,
                'entry_time': datetime.now()
            }
        else:
            # 更新现有仓位
            self.current_positions[stype][symbol]['size'] += position_size
            # 更新入场价格（加权平均）
            total_size = self.current_positions[stype][symbol]['size']
            old_value = self.current_positions[stype][symbol]['entry_price'] * (total_size - position_size)
            new_value = entry_price * position_size
            self.current_positions[stype][symbol]['entry_price'] = (old_value + new_value) / total_size
            
        return True
    
    def close_position(self, strategy_name: str, symbol: str, exit_price: float) -> float:
        """平仓"""
        stype = self.get_strategy_type(strategy_name)
        if not stype:
            return 0.0
            
        if symbol not in self.current_positions[stype]:
            return 0.0
            
        # 计算盈亏
        position = self.current_positions[stype][symbol]
        pnl = (exit_price - position['entry_price']) * position['size']
        
        # 移除仓位
        del self.current_positions[stype][symbol]
        
        return pnl
    
    def check_stop_loss(self, symbol: str, current_price: float) -> Dict[str, float]:
        """检查止损"""
        closed_positions = {}
        
        for stype in self.current_positions:
            if symbol in self.current_positions[stype]:
                position = self.current_positions[stype][symbol]
                loss_pct = (current_price - position['entry_price']) / position['entry_price']
                
                if loss_pct <= -self.stop_loss[stype]:
                    pnl = self.close_position(next(s for s in self.strategy_types[stype] 
                                                 if s in self.current_positions[stype]), 
                                           symbol, current_price)
                    closed_positions[stype] = pnl
                    
        return closed_positions
    
    def check_take_profit(self, symbol: str, current_price: float) -> Dict[str, float]:
        """检查止盈"""
        closed_positions = {}
        
        for stype in self.current_positions:
            if symbol in self.current_positions[stype]:
                position = self.current_positions[stype][symbol]
                profit_pct = (current_price - position['entry_price']) / position['entry_price']
                
                if profit_pct >= self.take_profit[stype]:
                    pnl = self.close_position(next(s for s in self.strategy_types[stype] 
                                                 if s in self.current_positions[stype]), 
                                           symbol, current_price)
                    closed_positions[stype] = pnl
                    
        return closed_positions
    
    def get_current_positions(self) -> Dict[str, Dict[str, float]]:
        """获取当前仓位"""
        return self.current_positions
    
    def get_position_size(self, strategy_name: str, symbol: str) -> float:
        """获取特定策略的仓位大小"""
        stype = self.get_strategy_type(strategy_name)
        if not stype:
            return 0.0
            
        return self.current_positions[stype].get(symbol, {}).get('size', 0.0)
    
    def adjust_position_limits(self, market_volatility: float):
        """根据市场波动率调整仓位限制"""
        if market_volatility > 0.3:  # 高波动率
            # 降低短期策略仓位
            self.position_limits['short_term'] = 0.05
            # 增加中期策略仓位
            self.position_limits['medium_term'] = 0.15
            # 保持长期策略仓位不变
        elif market_volatility < 0.1:  # 低波动率
            # 增加短期策略仓位
            self.position_limits['short_term'] = 0.10
            # 降低中期策略仓位
            self.position_limits['medium_term'] = 0.10
            # 保持长期策略仓位不变
            
    def __str__(self) -> str:
        """返回仓位管理器的状态信息"""
        position_info = []
        for stype, positions in self.current_positions.items():
            if positions:
                position_info.append(f"{stype} positions:")
                for symbol, pos in positions.items():
                    position_info.append(f"  {symbol}: size={pos['size']:.4f}, entry_price={pos['entry_price']:.2f}")
        
        limits_info = [f"{k}: {v:.4f}" for k, v in self.position_limits.items()]
        
        return "\n".join([
            "PositionManager Status:",
            "Current Positions:",
            *position_info,
            "Position Limits:",
            *limits_info
        ]) 