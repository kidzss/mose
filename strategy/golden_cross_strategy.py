import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from .strategy_base import Strategy

class GoldenCrossStrategy(Strategy):
    """
    金叉死叉交易策略
    
    策略说明:
    1. 买入条件: 
       - 短期均线上穿长期均线（金叉）
       - 如果risk_filter=True，则还需要价格大于中期均线
    
    2. 卖出条件:
       - 短期均线下穿长期均线（死叉）
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化金叉死叉策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - short_sma: 短期均线周期，默认5
                - long_sma: 长期均线周期，默认20
                - mid_sma: 中期均线周期，默认10
                - risk_filter: 是否使用风险过滤，默认True
        """
        default_params = {
            'short_sma': 5,
            'long_sma': 20,
            'mid_sma': 10,
            'risk_filter': True
        }
        
        # 合并参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('GoldenCrossStrategy', default_params)
        self.logger.info(f"初始化金叉死叉策略，参数: {default_params}")
        self.version = '1.0.0'
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        try:
            if data is None or data.empty:
                self.logger.warning("数据为空，无法计算指标")
                return pd.DataFrame()
            
            # 复制数据以避免修改原始数据
            df = data.copy()
            
            # 计算移动平均线
            short_sma = self.parameters['short_sma']
            long_sma = self.parameters['long_sma']
            mid_sma = self.parameters['mid_sma']
            
            df['short_sma'] = df['close'].rolling(window=short_sma).mean()
            df['long_sma'] = df['close'].rolling(window=long_sma).mean()
            df['mid_sma'] = df['close'].rolling(window=mid_sma).mean()
            
            # 计算金叉死叉条件
            df['golden_cross'] = (df['short_sma'] > df['long_sma']) & (df['short_sma'].shift(1) <= df['long_sma'].shift(1))
            df['death_cross'] = (df['short_sma'] < df['long_sma']) & (df['short_sma'].shift(1) >= df['long_sma'].shift(1))
            
            # 风险过滤条件
            df['above_mid_sma'] = df['close'] > df['mid_sma']
            
            # 填充NaN值
            df = df.bfill().ffill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return data
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame，其中:
            1 = 买入信号
            0 = 持有/无信号
            -1 = 卖出信号
        """
        try:
            # 计算技术指标
            df = self.calculate_indicators(data)
            
            # 初始化信号列
            df['signal'] = 0
            
            if self.parameters['risk_filter']:
                # 使用风险过滤条件（必须在中期均线之上）
                df.loc[(df['golden_cross']) & (df['above_mid_sma']), 'signal'] = 1
            else:
                # 不使用风险过滤
                df.loc[df['golden_cross'], 'signal'] = 1
            
            # 死叉卖出信号
            df.loc[df['death_cross'], 'signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return data.assign(signal=0)
            
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        df = self.calculate_indicators(data)
        
        # 短期/长期均线差值，标准化到[-1, 1]
        sma_diff = (df['short_sma'] - df['long_sma']) / df['close']
        # 限制在 [-1, 1] 范围内
        sma_diff = sma_diff.clip(-0.1, 0.1) / 0.1
        
        # 金叉信号组件
        golden_cross_component = df['golden_cross'].astype(float)
        
        # 死叉信号组件
        death_cross_component = -df['death_cross'].astype(float)
        
        # 价格相对于中期均线的位置
        price_position = (df['close'] - df['mid_sma']) / df['mid_sma']
        # 标准化到 [-1, 1]
        price_position = price_position.clip(-0.1, 0.1) / 0.1
        
        return {
            "sma_diff": sma_diff,
            "golden_cross": golden_cross_component,
            "death_cross": death_cross_component,
            "price_position": price_position,
            "composite": df['signal']
        }
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "sma_diff": {
                "type": "trend",
                "time_scale": "medium",
                "description": "短期均线与长期均线之差",
                "min_value": -1,
                "max_value": 1
            },
            "golden_cross": {
                "type": "trend",
                "time_scale": "medium",
                "description": "金叉信号（短期均线上穿长期均线）",
                "min_value": 0,
                "max_value": 1
            },
            "death_cross": {
                "type": "trend",
                "time_scale": "medium",
                "description": "死叉信号（短期均线下穿长期均线）",
                "min_value": -1,
                "max_value": 0
            },
            "price_position": {
                "type": "trend",
                "time_scale": "medium",
                "description": "价格相对于中期均线的位置",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "medium",
                "description": "金叉死叉策略综合信号",
                "min_value": -1,
                "max_value": 1
            }
        } 