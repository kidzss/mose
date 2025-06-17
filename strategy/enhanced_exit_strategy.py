#!/usr/bin/env python3
"""
增强退出策略模块
结合基本面和技术面指标的智能止损止盈系统
实现第一个和第二个专家建议的退出策略
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ExitSignalType:
    """退出信号类型"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    FUNDAMENTAL_EXIT = "fundamental_exit"
    TECHNICAL_EXIT = "technical_exit"
    TRAILING_STOP = "trailing_stop"
    DYNAMIC_EXIT = "dynamic_exit"

class ExitReason:
    """退出原因"""
    MAX_LOSS_REACHED = "最大亏损止损"
    PROFIT_TARGET_HIT = "获利目标达成"
    FUNDAMENTAL_DETERIORATION = "基本面恶化"
    TECHNICAL_BREAKDOWN = "技术面突破"
    VALUATION_OVEREXTENDED = "估值过度延伸"
    DRAWDOWN_LIMIT = "回撤限制"
    TIME_DECAY = "时间衰减"

class EnhancedExitStrategy:
    """增强退出策略"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化退出策略
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        self.position_tracker = {}  # 跟踪持仓信息
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            # 基础止损止盈参数
            'max_loss_pct': 0.20,          # 最大亏损20%
            'profit_target_pct': 0.50,     # 获利目标50%
            'trailing_stop_pct': 0.10,     # 追踪止损10%
            
            # 动态调整参数
            'volatility_adjustment': True,   # 根据波动率调整
            'fundamental_weight': 0.3,      # 基本面权重
            'technical_weight': 0.7,        # 技术面权重
            
            # 基本面退出参数
            'pe_threshold_multiplier': 1.5, # PE超过历史均值1.5倍
            'roe_decline_threshold': 0.3,   # ROE下降30%
            'fundamental_check_days': 90,   # 基本面检查周期
            
            # 技术面退出参数
            'ma_crossover_exit': True,      # 均线死叉退出
            'rsi_overbought': 80,          # RSI超买
            'volume_spike_multiplier': 2.0, # 成交量异常倍数
            
            # 时间管理
            'max_holding_days': 365,       # 最大持仓天数
            'profit_decay_days': 180,      # 利润衰减天数
        }
    
    def calculate_exit_signals(self, symbol: str, current_price: float, 
                             entry_data: Dict, market_data: pd.DataFrame,
                             fundamental_data: Optional[Dict] = None) -> Dict:
        """
        计算退出信号
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            entry_data: 入场数据
            market_data: 市场数据
            fundamental_data: 基本面数据
            
        Returns:
            退出信号字典
        """
        try:
            exit_signals = {
                'should_exit': False,
                'exit_type': None,
                'exit_reason': None,
                'exit_price': current_price,
                'urgency': 'low',  # low, medium, high
                'confidence': 0.0,
                'details': {}
            }
            
            # 计算基础盈亏
            entry_price = entry_data['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 1. 基础止损检查
            stop_loss_signal = self._check_stop_loss(pnl_pct, entry_data, market_data)
            if stop_loss_signal['should_exit']:
                return stop_loss_signal
            
            # 2. 基础止盈检查
            take_profit_signal = self._check_take_profit(pnl_pct, entry_data, market_data)
            if take_profit_signal['should_exit']:
                return take_profit_signal
            
            # 3. 基本面退出检查
            if fundamental_data:
                fundamental_signal = self._check_fundamental_exit(
                    symbol, fundamental_data, entry_data, pnl_pct
                )
                if fundamental_signal['should_exit']:
                    return fundamental_signal
            
            # 4. 技术面退出检查
            technical_signal = self._check_technical_exit(
                market_data, entry_data, current_price
            )
            if technical_signal['should_exit']:
                return technical_signal
            
            # 5. 动态止损检查
            dynamic_signal = self._check_dynamic_exit(
                symbol, current_price, entry_data, market_data, pnl_pct
            )
            if dynamic_signal['should_exit']:
                return dynamic_signal
            
            return exit_signals
            
        except Exception as e:
            logger.error(f"计算退出信号失败 {symbol}: {e}")
            return {'should_exit': False, 'error': str(e)}
    
    def _check_stop_loss(self, pnl_pct: float, entry_data: Dict, 
                        market_data: pd.DataFrame) -> Dict:
        """检查基础止损"""
        max_loss = -abs(self.config['max_loss_pct'])
        
        if pnl_pct <= max_loss:
            return {
                'should_exit': True,
                'exit_type': ExitSignalType.STOP_LOSS,
                'exit_reason': ExitReason.MAX_LOSS_REACHED,
                'urgency': 'high',
                'confidence': 0.95,
                'details': {
                    'pnl_pct': pnl_pct,
                    'max_loss_threshold': max_loss,
                    'message': f"触发止损：亏损{pnl_pct:.1%}超过最大容忍度{max_loss:.1%}"
                }
            }
        
        return {'should_exit': False}
    
    def _check_take_profit(self, pnl_pct: float, entry_data: Dict,
                          market_data: pd.DataFrame) -> Dict:
        """检查基础止盈"""
        profit_target = self.config['profit_target_pct']
        
        # 动态调整止盈目标
        if len(market_data) >= 20:
            volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            # 高波动时提高止盈目标，低波动时降低
            volatility_factor = max(0.5, min(2.0, volatility / 0.02))
            adjusted_target = profit_target * volatility_factor
        else:
            adjusted_target = profit_target
        
        if pnl_pct >= adjusted_target:
            return {
                'should_exit': True,
                'exit_type': ExitSignalType.TAKE_PROFIT,
                'exit_reason': ExitReason.PROFIT_TARGET_HIT,
                'urgency': 'medium',
                'confidence': 0.85,
                'details': {
                    'pnl_pct': pnl_pct,
                    'profit_target': adjusted_target,
                    'message': f"达成止盈：获利{pnl_pct:.1%}达到目标{adjusted_target:.1%}"
                }
            }
        
        return {'should_exit': False}
    
    def _check_fundamental_exit(self, symbol: str, fundamental_data: Dict,
                               entry_data: Dict, pnl_pct: float) -> Dict:
        """检查基本面退出信号"""
        exit_signals = []
        
        # PE估值检查
        current_pe = fundamental_data.get('trailingPE', 0)
        entry_pe = entry_data.get('entry_pe', current_pe)
        
        if current_pe > 0 and entry_pe > 0:
            pe_increase = (current_pe - entry_pe) / entry_pe
            threshold = self.config['pe_threshold_multiplier'] - 1
            
            if pe_increase > threshold and pnl_pct > 0.2:  # 获利超过20%且PE大幅上涨
                exit_signals.append({
                    'type': 'pe_overvaluation',
                    'severity': 'high',
                    'message': f"PE估值过高：从{entry_pe:.1f}上涨至{current_pe:.1f}（+{pe_increase:.1%}）"
                })
        
        # ROE恶化检查
        current_roe = fundamental_data.get('returnOnEquity', 0)
        entry_roe = entry_data.get('entry_roe', current_roe)
        
        if current_roe > 0 and entry_roe > 0:
            roe_decline = (entry_roe - current_roe) / entry_roe
            
            if roe_decline > self.config['roe_decline_threshold']:
                exit_signals.append({
                    'type': 'roe_deterioration',
                    'severity': 'high',
                    'message': f"ROE恶化：从{entry_roe:.1%}下降至{current_roe:.1%}（-{roe_decline:.1%}）"
                })
        
        # 债务风险检查
        debt_ratio = fundamental_data.get('debtToEquity', 0) / 100
        if debt_ratio > 3.0:  # 债务权益比超过300%
            exit_signals.append({
                'type': 'debt_risk',
                'severity': 'medium',
                'message': f"债务风险升高：债务权益比{debt_ratio:.1f}"
            })
        
        # 如果有高严重级别的信号，建议退出
        high_severity_signals = [s for s in exit_signals if s['severity'] == 'high']
        
        if high_severity_signals:
            return {
                'should_exit': True,
                'exit_type': ExitSignalType.FUNDAMENTAL_EXIT,
                'exit_reason': ExitReason.FUNDAMENTAL_DETERIORATION,
                'urgency': 'high',
                'confidence': 0.8,
                'details': {
                    'signals': exit_signals,
                    'message': f"基本面恶化：{len(high_severity_signals)}个高风险信号"
                }
            }
        
        return {'should_exit': False}
    
    def _check_technical_exit(self, market_data: pd.DataFrame, entry_data: Dict,
                             current_price: float) -> Dict:
        """检查技术面退出信号"""
        if len(market_data) < 50:
            return {'should_exit': False}
        
        exit_signals = []
        
        # 均线死叉检查
        if self.config['ma_crossover_exit']:
            ma20 = market_data['close'].rolling(20).mean().iloc[-1]
            ma50 = market_data['close'].rolling(50).mean().iloc[-1]
            
            if current_price < ma20 < ma50:
                exit_signals.append({
                    'type': 'ma_crossover',
                    'severity': 'medium',
                    'message': f"均线死叉：价格{current_price:.2f} < MA20({ma20:.2f}) < MA50({ma50:.2f})"
                })
        
        # RSI超买检查
        if 'rsi' in market_data.columns:
            current_rsi = market_data['rsi'].iloc[-1]
            if current_rsi > self.config['rsi_overbought']:
                exit_signals.append({
                    'type': 'rsi_overbought',
                    'severity': 'low',
                    'message': f"RSI超买：{current_rsi:.1f}"
                })
        
        # 成交量异常检查
        if 'volume' in market_data.columns:
            recent_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = recent_volume / avg_volume
            
            if volume_ratio > self.config['volume_spike_multiplier']:
                exit_signals.append({
                    'type': 'volume_spike',
                    'severity': 'medium',
                    'message': f"成交量异常：{volume_ratio:.1f}倍于平均水平"
                })
        
        # 综合技术信号评估
        medium_signals = [s for s in exit_signals if s['severity'] == 'medium']
        
        if len(medium_signals) >= 2:  # 至少2个中等信号
            return {
                'should_exit': True,
                'exit_type': ExitSignalType.TECHNICAL_EXIT,
                'exit_reason': ExitReason.TECHNICAL_BREAKDOWN,
                'urgency': 'medium',
                'confidence': 0.7,
                'details': {
                    'signals': exit_signals,
                    'message': f"技术面转弱：{len(medium_signals)}个技术信号"
                }
            }
        
        return {'should_exit': False}
    
    def _check_dynamic_exit(self, symbol: str, current_price: float,
                           entry_data: Dict, market_data: pd.DataFrame,
                           pnl_pct: float) -> Dict:
        """检查动态退出信号"""
        # 追踪止损
        if pnl_pct > 0.1:  # 盈利超过10%开始追踪止损
            if symbol not in self.position_tracker:
                self.position_tracker[symbol] = {
                    'highest_price': current_price,
                    'trailing_stop': current_price * (1 - self.config['trailing_stop_pct'])
                }
            
            tracker = self.position_tracker[symbol]
            
            # 更新最高价和追踪止损
            if current_price > tracker['highest_price']:
                tracker['highest_price'] = current_price
                tracker['trailing_stop'] = current_price * (1 - self.config['trailing_stop_pct'])
            
            # 检查是否触发追踪止损
            if current_price <= tracker['trailing_stop']:
                return {
                    'should_exit': True,
                    'exit_type': ExitSignalType.TRAILING_STOP,
                    'exit_reason': ExitReason.DRAWDOWN_LIMIT,
                    'urgency': 'high',
                    'confidence': 0.9,
                    'details': {
                        'current_price': current_price,
                        'trailing_stop': tracker['trailing_stop'],
                        'highest_price': tracker['highest_price'],
                        'message': f"触发追踪止损：{current_price:.2f} <= {tracker['trailing_stop']:.2f}"
                    }
                }
        
        # 时间衰减检查
        entry_date = entry_data.get('entry_date')
        if entry_date:
            holding_days = (datetime.now() - entry_date).days
            
            if holding_days > self.config['max_holding_days']:
                return {
                    'should_exit': True,
                    'exit_type': ExitSignalType.DYNAMIC_EXIT,
                    'exit_reason': ExitReason.TIME_DECAY,
                    'urgency': 'low',
                    'confidence': 0.6,
                    'details': {
                        'holding_days': holding_days,
                        'max_days': self.config['max_holding_days'],
                        'message': f"持仓时间过长：{holding_days}天 > {self.config['max_holding_days']}天"
                    }
                }
        
        return {'should_exit': False}
    
    def update_position_tracker(self, symbol: str, action: str):
        """更新持仓跟踪器"""
        if action == 'exit' and symbol in self.position_tracker:
            del self.position_tracker[symbol]
    
    def get_exit_summary(self, symbol: str) -> Dict:
        """获取退出策略摘要"""
        return {
            'config': self.config,
            'position_tracker': self.position_tracker.get(symbol, {}),
            'strategy_name': 'EnhancedExitStrategy',
            'version': '1.0'
        }


# 示例使用
if __name__ == "__main__":
    # 创建退出策略
    exit_strategy = EnhancedExitStrategy()
    
    # 模拟数据
    entry_data = {
        'entry_price': 100.0,
        'entry_date': datetime.now() - timedelta(days=30),
        'entry_pe': 20.0,
        'entry_roe': 0.15
    }
    
    fundamental_data = {
        'trailingPE': 25.0,
        'returnOnEquity': 0.12,
        'debtToEquity': 50.0
    }
    
    # 创建示例市场数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    market_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 5000000, 100),
        'rsi': np.random.uniform(30, 70, 100)
    }, index=dates)
    
    # 测试退出信号
    current_price = 120.0  # 20%盈利
    signals = exit_strategy.calculate_exit_signals(
        'TEST', current_price, entry_data, market_data, fundamental_data
    )
    
    print("退出信号分析结果:")
    print(f"应该退出: {signals.get('should_exit', False)}")
    if signals.get('should_exit'):
        print(f"退出类型: {signals.get('exit_type')}")
        print(f"退出原因: {signals.get('exit_reason')}")
        print(f"紧急程度: {signals.get('urgency')}")
        print(f"置信度: {signals.get('confidence', 0):.1%}") 