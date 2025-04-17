import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, Any
from data.data_interface import DataInterface
from strategy.strategy_base import Strategy, MarketRegime


class USSMarketForecast(Strategy):
    """
    USS市场预测策略
    使用动量和聚类指标进行市场预测
    """
    
    def __init__(self, name: str = "USS Market Forecast", parameters: dict = None):
        """初始化策略"""
        default_params = {
            'med_len': 31,
            'mom_len': 5,
            'near_len': 3,
            'sma_periods': [5, 10, 20, 100]
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(name, default_params)
        self.data_interface = DataInterface()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算SMA
        for period in self.parameters['sma_periods']:
            df[f"SMA_{period}"] = round(df["close"].rolling(window=period).mean(), 3)
        
        # 计算动量和聚类指标
        med_len = self.parameters['med_len']
        near_len = self.parameters['near_len']
        
        df['lowest_low_med'] = df['low'].rolling(window=med_len).min()
        df['highest_high_med'] = df['high'].rolling(window=med_len).max()
        df['fastK_I'] = (df['close'] - df['lowest_low_med']) / (df['highest_high_med'] - df['lowest_low_med']) * 100

        df['lowest_low_near'] = df['low'].rolling(window=near_len).min()
        df['highest_high_near'] = df['high'].rolling(window=near_len).max()
        df['fastK_N'] = (df['close'] - df['lowest_low_near']) / (df['highest_high_near'] - df['lowest_low_near']) * 100

        min1 = df['low'].rolling(window=4).min()
        max1 = df['high'].rolling(window=4).max()
        df['momentum'] = ((df['close'] - min1) / (max1 - min1)) * 100

        df['bull_cluster'] = (df['momentum'] <= 20) & (df['fastK_I'] <= 20) & (df['fastK_N'] <= 20)
        df['bear_cluster'] = (df['momentum'] >= 80) & (df['fastK_I'] >= 80) & (df['fastK_N'] >= 80)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = data.copy()
        df['signal'] = 0  # 默认无信号
        
        for i in range(len(df) - 1):
            # 买入信号
            if (df['bull_cluster'].iloc[i] and 
                df['SMA_5'].iloc[i] > df['SMA_10'].iloc[i]):
                df.iloc[i, df.columns.get_loc('signal')] = 1
                
            # 卖出信号
            elif (df['bear_cluster'].iloc[i] or 
                  df['SMA_5'].iloc[i] < df['SMA_10'].iloc[i]):
                df.iloc[i, df.columns.get_loc('signal')] = -1
                
        return df

    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """提取信号组件"""
        return {
            'bull_cluster': data['bull_cluster'],
            'bear_cluster': data['bear_cluster'],
            'momentum': data['momentum'],
            'fastK_I': data['fastK_I'],
            'fastK_N': data['fastK_N'],
            'sma_5': data['SMA_5'],
            'sma_10': data['SMA_10']
        }

    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """获取信号元数据"""
        return {
            'bull_cluster': {
                'description': '看多聚类信号',
                'type': 'boolean',
                'range': [False, True]
            },
            'bear_cluster': {
                'description': '看空聚类信号',
                'type': 'boolean',
                'range': [False, True]
            },
            'momentum': {
                'description': '动量指标',
                'type': 'float',
                'range': [0, 100]
            },
            'fastK_I': {
                'description': '中期随机指标',
                'type': 'float',
                'range': [0, 100]
            },
            'fastK_N': {
                'description': '短期随机指标',
                'type': 'float',
                'range': [0, 100]
            }
        }

    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """判断市场状态"""
        if len(data) < 20:
            return MarketRegime.UNKNOWN
            
        latest = data.iloc[-1]
        
        # 根据聚类信号判断市场状态
        if latest['bull_cluster']:
            return MarketRegime.BULLISH
        elif latest['bear_cluster']:
            return MarketRegime.BEARISH
        
        # 根据动量判断
        if latest['momentum'] > 80:
            return MarketRegime.VOLATILE
        elif latest['momentum'] < 20:
            return MarketRegime.LOW_VOLATILITY
            
        return MarketRegime.RANGING
