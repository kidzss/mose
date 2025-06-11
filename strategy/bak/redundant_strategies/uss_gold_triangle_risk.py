import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple
from data.data_interface import DataInterface
from strategy.strategy_base import Strategy, MarketRegime


class USSGoldTriangleRisk(Strategy):
    """
    USS黄金三角风险策略
    使用多个移动平均线的交叉来判断趋势和风险
    """
    
    def __init__(self, name: str = "USS Gold Triangle Risk", parameters: dict = None):
        """初始化策略"""
        default_params = {
            'sma_periods': [5, 10, 20, 100],
            'no_risk': True  # 对于一些杠杆产品例如tqqq，可以将risk设置为False，普通的股票可以使用默认值
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
            
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = data.copy()
        df['signal'] = 0  # 默认无信号
        
        for i in range(len(df) - 1):
            moving_average_5 = df["SMA_5"].iloc[i + 1]
            moving_average_10 = df["SMA_10"].iloc[i + 1]
            moving_average_20 = df["SMA_20"].iloc[i + 1]
            moving_average_100 = df["SMA_100"].iloc[i + 1]
            close = df["close"].iloc[i]

            # 定义买入和卖出条件
            cond1 = moving_average_5 > moving_average_10
            cond2 = moving_average_10 > moving_average_20
            cond3 = df["SMA_10"].iloc[i] < df["SMA_20"].iloc[i]
            cond4 = close > moving_average_100  # 股价大于SMA_100

            # 产生买入信号
            if cond1 and cond2 and cond3 and (cond4 or self.parameters['no_risk']):
                df.iloc[i, df.columns.get_loc('signal')] = 1
            # 产生卖出信号
            elif not cond2 and not cond3:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                
        return df

    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """提取信号组件"""
        return {
            'sma_5': data['SMA_5'],
            'sma_10': data['SMA_10'],
            'sma_20': data['SMA_20'],
            'sma_100': data['SMA_100'],
            'price': data['close']
        }

    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """获取信号元数据"""
        return {
            'sma_5': {
                'description': '5日简单移动平均线',
                'type': 'float',
                'range': [0, float('inf')]
            },
            'sma_10': {
                'description': '10日简单移动平均线',
                'type': 'float',
                'range': [0, float('inf')]
            },
            'sma_20': {
                'description': '20日简单移动平均线',
                'type': 'float',
                'range': [0, float('inf')]
            },
            'sma_100': {
                'description': '100日简单移动平均线',
                'type': 'float',
                'range': [0, float('inf')]
            }
        }

    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """判断市场状态"""
        if len(data) < 20:
            return MarketRegime.UNKNOWN
            
        latest = data.iloc[-1]
        
        # 根据均线关系判断市场状态
        if (latest['SMA_5'] > latest['SMA_10'] > latest['SMA_20'] > latest['SMA_100']):
            return MarketRegime.BULLISH
        elif (latest['SMA_5'] < latest['SMA_10'] < latest['SMA_20'] < latest['SMA_100']):
            return MarketRegime.BEARISH
            
        # 根据价格与长期均线的关系判断
        if latest['close'] > latest['SMA_100']:
            if latest['SMA_5'] > latest['SMA_20']:
                return MarketRegime.BULLISH
            else:
                return MarketRegime.RANGING
        else:
            if latest['SMA_5'] < latest['SMA_20']:
                return MarketRegime.BEARISH
            else:
                return MarketRegime.RANGING
