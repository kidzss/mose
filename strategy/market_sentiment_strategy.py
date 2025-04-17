from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from strategy_base import Strategy
from data.market_sentiment import MarketSentimentData
import logging

logger = logging.getLogger(__name__)

class MarketSentimentStrategy(Strategy):
    """
    市场情绪策略，结合VIX指数、Put/Call Ratio和市场宽度指标
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化市场情绪策略
        
        参数:
            parameters: 策略参数
        """
        # 默认参数
        default_params = {
            # VIX参数
            'vix_threshold_high': 30,    # VIX高位阈值
            'vix_threshold_low': 20,     # VIX低位阈值
            'vix_smoothing': 5,          # VIX平滑周期
            
            # Put/Call Ratio参数
            'pcr_threshold_high': 1.2,   # PCR高位阈值
            'pcr_threshold_low': 0.8,    # PCR低位阈值
            'pcr_smoothing': 5,          # PCR平滑周期
            
            # 市场宽度参数
            'breadth_period': 20,        # 市场宽度计算周期
            'breadth_threshold_high': 0.7,  # 市场宽度高位阈值
            'breadth_threshold_low': 0.3,   # 市场宽度低位阈值
            
            # 综合参数
            'sentiment_weight': 0.3,     # 情绪指标权重
            'trend_weight': 0.4,         # 趋势指标权重
            'volatility_weight': 0.3,    # 波动率指标权重
        }
        
        # 更新参数
        if parameters:
            default_params.update(parameters)
            
        # 调用父类初始化
        super().__init__(name="MarketSentimentStrategy", parameters=default_params)
        
        # 初始化市场情绪数据
        self.market_sentiment = MarketSentimentData()
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场情绪指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了市场情绪指标的DataFrame
        """
        df = data.copy()
        
        # 1. VIX指标
        # 获取VIX数据
        vix_data = []
        for date in df.index:
            vix = self.market_sentiment.get_vix(date.strftime('%Y-%m-%d'))
            vix_data.append(vix)
        df['vix'] = vix_data
        
        # 计算VIX变化率
        df['vix_change'] = df['vix'].pct_change()
        
        # 计算平滑后的VIX
        df['vix_smooth'] = df['vix'].rolling(window=self.params['vix_smoothing']).mean()
        
        # 2. Put/Call Ratio指标
        # 获取PCR数据
        pcr_data = []
        for date in df.index:
            pcr = self.market_sentiment.get_put_call_ratio(date.strftime('%Y-%m-%d'))
            pcr_data.append(pcr)
        df['pcr'] = pcr_data
        
        # 计算PCR变化率
        df['pcr_change'] = df['pcr'].pct_change()
        
        # 计算平滑后的PCR
        df['pcr_smooth'] = df['pcr'].rolling(window=self.params['pcr_smoothing']).mean()
        
        # 3. 市场宽度指标
        # 计算上涨股票比例
        df['advance_ratio'] = df['close'].rolling(window=self.params['breadth_period']).apply(
            lambda x: len(x[x > x.shift(1)]) / len(x)
        )
        
        # 计算新高新低比例
        df['new_high_ratio'] = df['high'].rolling(window=self.params['breadth_period']).apply(
            lambda x: len(x[x == x.max()]) / len(x)
        )
        df['new_low_ratio'] = df['low'].rolling(window=self.params['breadth_period']).apply(
            lambda x: len(x[x == x.min()]) / len(x)
        )
        
        # 计算综合市场宽度
        df['market_breadth'] = (
            df['advance_ratio'] * 0.4 +
            df['new_high_ratio'] * 0.3 +
            (1 - df['new_low_ratio']) * 0.3
        )
        
        # 4. 计算综合情绪得分
        # VIX得分 (0-1)
        df['vix_score'] = np.where(
            df['vix'] > self.params['vix_threshold_high'],
            0,
            np.where(
                df['vix'] < self.params['vix_threshold_low'],
                1,
                (self.params['vix_threshold_high'] - df['vix']) /
                (self.params['vix_threshold_high'] - self.params['vix_threshold_low'])
            )
        )
        
        # PCR得分 (0-1)
        df['pcr_score'] = np.where(
            df['pcr'] > self.params['pcr_threshold_high'],
            0,
            np.where(
                df['pcr'] < self.params['pcr_threshold_low'],
                1,
                (self.params['pcr_threshold_high'] - df['pcr']) /
                (self.params['pcr_threshold_high'] - self.params['pcr_threshold_low'])
            )
        )
        
        # 市场宽度得分 (0-1)
        df['breadth_score'] = np.where(
            df['market_breadth'] > self.params['breadth_threshold_high'],
            1,
            np.where(
                df['market_breadth'] < self.params['breadth_threshold_low'],
                0,
                (df['market_breadth'] - self.params['breadth_threshold_low']) /
                (self.params['breadth_threshold_high'] - self.params['breadth_threshold_low'])
            )
        )
        
        # 计算综合情绪得分
        df['sentiment_score'] = (
            df['vix_score'] * self.params['sentiment_weight'] +
            df['pcr_score'] * self.params['sentiment_weight'] +
            df['breadth_score'] * self.params['sentiment_weight']
        )
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 计算好指标的DataFrame
            
        返回:
            添加了信号的DataFrame
        """
        df = data.copy()
        
        # 初始化信号列
        df['signal'] = 0
        
        # 生成信号
        for i in range(1, len(df)):
            current_data = df.iloc[i]
            
            # 1. 情绪指标
            sentiment_bullish = current_data['sentiment_score'] > 0.7
            sentiment_bearish = current_data['sentiment_score'] < 0.3
            
            # 2. 趋势指标
            trend_up = (
                current_data['market_breadth'] > self.params['breadth_threshold_high'] and
                current_data['vix'] < self.params['vix_threshold_low']
            )
            trend_down = (
                current_data['market_breadth'] < self.params['breadth_threshold_low'] and
                current_data['vix'] > self.params['vix_threshold_high']
            )
            
            # 3. 波动率指标
            volatility_high = current_data['vix'] > self.params['vix_threshold_high']
            volatility_low = current_data['vix'] < self.params['vix_threshold_low']
            
            # 生成买入信号
            if sentiment_bullish and trend_up and volatility_low:
                df.loc[df.index[i], 'signal'] = 1
                
            # 生成卖出信号
            elif sentiment_bearish and trend_down and volatility_high:
                df.loc[df.index[i], 'signal'] = -1
                
        return df
        
    def get_market_regime(self, data: pd.DataFrame) -> str:
        """
        判断当前市场环境
        
        参数:
            data: 计算好指标的DataFrame
            
        返回:
            市场环境类型
        """
        current_data = data.iloc[-1]
        
        # 计算综合得分
        sentiment_score = current_data['sentiment_score']
        trend_score = (
            current_data['market_breadth'] * 0.5 +
            (1 - current_data['vix_score']) * 0.3 +
            (1 - current_data['pcr_score']) * 0.2
        )
        volatility_score = 1 - current_data['vix_score']
        
        # 判断市场环境
        if sentiment_score > 0.7 and trend_score > 0.7 and volatility_score > 0.7:
            return 'bullish'
        elif sentiment_score < 0.3 and trend_score < 0.3 and volatility_score < 0.3:
            return 'bearish'
        elif current_data['vix'] > self.params['vix_threshold_high']:
            return 'volatile'
        else:
            return 'ranging'
            
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号元数据
        
        返回:
            信号元数据字典
        """
        return {
            'sentiment_score': {
                'description': '综合情绪得分',
                'type': 'float',
                'range': [0, 1]
            },
            'vix_score': {
                'description': 'VIX得分',
                'type': 'float',
                'range': [0, 1]
            },
            'pcr_score': {
                'description': 'Put/Call Ratio得分',
                'type': 'float',
                'range': [0, 1]
            },
            'breadth_score': {
                'description': '市场宽度得分',
                'type': 'float',
                'range': [0, 1]
            },
            'market_regime': {
                'description': '市场环境',
                'type': 'string',
                'values': ['bullish', 'bearish', 'volatile', 'ranging']
            }
        } 