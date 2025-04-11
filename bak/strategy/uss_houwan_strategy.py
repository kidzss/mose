import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from .strategy_base import Strategy

class HouWangStrategy(Strategy):
    """
    猴王交易策略
    
    策略说明:
    1. 买入条件: 
       - 价格突破前N日最高点后回调至支撑位
       - 成交量在突破时放大
       - 回调时成交量减小
    
    2. 卖出条件:
       - 价格跌破M日低点
       - 或者价格达到预设的获利目标
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化猴王策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - high_period: 最高点参考周期，默认20
                - low_period: 最低点参考周期，默认10
                - pullback_threshold: 回调阈值，默认0.02 (2%)
                - volume_increase: 成交量放大阈值，默认1.5
                - volume_decrease: 成交量减小阈值，默认0.7
                - profit_target: 获利目标，默认0.1 (10%)
        """
        default_params = {
            'high_period': 20,
            'low_period': 10,
            'pullback_threshold': 0.02,
            'volume_increase': 1.5,
            'volume_decrease': 0.7,
            'profit_target': 0.1
        }
        
        # 合并参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('HouWangStrategy', default_params)
        self.logger.info(f"初始化猴王策略，参数: {default_params}")
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
            
            # 计算N日最高点和最低点
            high_period = self.parameters['high_period']
            low_period = self.parameters['low_period']
            
            df['highest_high'] = df['high'].rolling(window=high_period).max()
            df['lowest_low'] = df['low'].rolling(window=low_period).min()
            
            # 计算相对于N日最高点的回调幅度
            df['pullback'] = (df['highest_high'] - df['close']) / df['highest_high']
            
            # 计算成交量变化
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
            
            # 判断价格是否突破前N日最高点
            df['breakout'] = (df['close'] > df['highest_high'].shift(1)) & (df['volume_ratio'] > self.parameters['volume_increase'])
            
            # 判断价格是否回调至支撑位
            df['pullback_to_support'] = (df['pullback'] > self.parameters['pullback_threshold']) & (df['volume_ratio'] < self.parameters['volume_decrease'])
            
            # 标记支撑位回调的买入条件
            df['buy_condition'] = False
            
            # 计算买入条件：先突破，然后回调
            for i in range(1, len(df)):
                if df['breakout'].iloc[i-1] and df['pullback_to_support'].iloc[i]:
                    df['buy_condition'].iloc[i] = True
            
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
            
            # 生成买入信号
            df.loc[df['buy_condition'], 'signal'] = 1
            
            # 生成卖出信号
            # 1. 跌破最低点
            df.loc[df['close'] < df['lowest_low'].shift(1), 'signal'] = -1
            
            # 2. 达到获利目标
            for i in range(1, len(df)):
                if df['signal'].iloc[i-1] == 1:  # 前一天是买入信号
                    entry_price = df['close'].iloc[i-1]
                    profit_target = entry_price * (1 + self.parameters['profit_target'])
                    if df['high'].iloc[i] >= profit_target:
                        df['signal'].iloc[i] = -1
            
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
        
        # 突破信号组件
        breakout_component = df['breakout'].astype(float)
        
        # 回调信号组件
        pullback_component = 1 - df['pullback'] / df['pullback'].max()  # 标准化到0-1
        
        # 成交量信号组件
        volume_component = (df['volume_ratio'] - 1) / (df['volume_ratio'].max() - 1)  # 标准化到0-1
        volume_component = volume_component.clip(0, 1)  # 限制在0-1范围内
        
        return {
            "breakout": breakout_component,
            "pullback": pullback_component,
            "volume": volume_component,
            "composite": df['signal']
        }
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "breakout": {
                "type": "momentum",
                "time_scale": "short",
                "description": "价格突破前N日最高点",
                "min_value": 0,
                "max_value": 1
            },
            "pullback": {
                "type": "mean_reversion",
                "time_scale": "short",
                "description": "价格回调至支撑位",
                "min_value": 0,
                "max_value": 1
            },
            "volume": {
                "type": "volume",
                "time_scale": "short",
                "description": "成交量变化特征",
                "min_value": 0,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "short",
                "description": "猴王策略综合信号",
                "min_value": -1,
                "max_value": 1
            }
        } 