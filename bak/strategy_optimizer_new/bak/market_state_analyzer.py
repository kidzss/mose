# strategy_optimizer/data_processors/market_state_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class MarketStateAnalyzer:
    """
    市场状态分析器
    
    分析市场状态并提供相关特征
    """
    
    def __init__(self, price_data: pd.DataFrame = None):
        """初始化市场状态分析器"""
        self.price_data = price_data
        self.market_state = None
    
    def set_price_data(self, price_data: pd.DataFrame):
        """设置价格数据"""
        self.price_data = price_data
        
    def analyze_market_state(self, 
                            window_short: int = 50, 
                            window_long: int = 200,
                            volatility_window: int = 20) -> pd.DataFrame:
        """
        分析市场状态
        
        参数:
            window_short: 短期均线窗口
            window_long: 长期均线窗口
            volatility_window: 波动率计算窗口
            
        返回:
            包含市场状态分类的DataFrame
        """
        if self.price_data is None:
            raise ValueError("请先设置价格数据")
        
        price = self.price_data["close"].copy()
        
        # 计算趋势相关指标
        sma_short = price.rolling(window=window_short).mean()
        sma_long = price.rolling(window=window_long).mean()
        
        # 计算波动率
        returns = price.pct_change()
        volatility = returns.rolling(window=volatility_window).std() * np.sqrt(252)
        
        # 计算市场状态
        market_state = pd.DataFrame(index=price.index)
        
        # 趋势状态
        market_state["trend"] = 0  # 中性
        market_state.loc[sma_short > sma_long, "trend"] = 1  # 上升趋势
        market_state.loc[sma_short < sma_long, "trend"] = -1  # 下降趋势
        
        # 波动性状态
        median_volatility = volatility.median()
        market_state["volatility"] = 0  # 中等波动性
        market_state.loc[volatility > median_volatility * 1.5, "volatility"] = 1  # 高波动性
        market_state.loc[volatility < median_volatility * 0.5, "volatility"] = -1  # 低波动性
        
        # 综合市场状态
        # 1: 强势上涨 (上升趋势 + 低/中等波动)
        # 2: 波动上涨 (上升趋势 + 高波动)
        # 3: 震荡市场 (中性趋势)
        # 4: 波动下跌 (下降趋势 + 高波动)
        # 5: 稳定下跌 (下降趋势 + 低/中等波动)
        
        market_state["market_state"] = 3  # 默认为震荡市场
        
        # 强势上涨
        mask_strong_up = (market_state["trend"] == 1) & (market_state["volatility"].isin([0, -1]))
        market_state.loc[mask_strong_up, "market_state"] = 1
        
        # 波动上涨
        mask_volatile_up = (market_state["trend"] == 1) & (market_state["volatility"] == 1)
        market_state.loc[mask_volatile_up, "market_state"] = 2
        
        # 波动下跌
        mask_volatile_down = (market_state["trend"] == -1) & (market_state["volatility"] == 1)
        market_state.loc[mask_volatile_down, "market_state"] = 4
        
        # 稳定下跌
        mask_strong_down = (market_state["trend"] == -1) & (market_state["volatility"].isin([0, -1]))
        market_state.loc[mask_strong_down, "market_state"] = 5
        
        self.market_state = market_state
        return market_state
    
    def get_market_regime_features(self) -> pd.DataFrame:
        """
        获取市场状态特征
        
        返回:
            包含市场状态特征的DataFrame
        """
        if self.market_state is None:
            self.analyze_market_state()
            
        # 将市场状态转换为One-Hot编码
        market_state_dummies = pd.get_dummies(self.market_state["market_state"], 
                                             prefix="market_state",
                                             dtype=float)
        
        return market_state_dummies