#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场状态分析模块

该模块提供了用于分析和识别不同市场状态的工具和功能。市场状态分析对于交易策略的优化和调整非常重要，
因为不同的策略在不同的市场环境（如牛市、熊市、震荡市）中表现各异。

主要功能:
1. 从数据库获取市场数据
2. 计算市场趋势指标（如各种时间窗口的移动平均线）
3. 识别市场状态（牛市、熊市、震荡市等）
4. 分析不同市场状态下的波动性和流动性特征
5. 为策略优化提供市场状态信息，辅助动态调整策略参数

使用示例:
```python
# 初始化市场状态分析器
engine = create_engine('mysql://user:password@localhost/stockdb')
market_analyzer = MarketState(engine)

# 获取市场数据
market_data = market_analyzer.get_market_data(
    symbols=['SPY', 'QQQ'], 
    start_date='2020-01-01', 
    end_date='2023-01-01'
)

# 计算市场趋势指标
trend_indicators = market_analyzer.calculate_market_trend(market_data['SPY'])

# 识别市场状态
market_states = market_analyzer.identify_market_state(market_data['SPY'])

# 获取市场特征
market_features = market_analyzer.get_market_features(market_data['SPY'])
```
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

class MarketState:
    """
    市场状态分析类
    
    该类提供了一系列方法用于分析市场数据，识别市场状态，并计算各种市场特征指标。
    它能够帮助交易策略根据不同的市场环境调整参数和行为，提高策略的稳健性和适应能力。
    
    主要功能:
    - 从数据库获取股票历史数据
    - 计算技术指标和市场特征
    - 基于历史数据识别市场状态（趋势、震荡、牛市、熊市等）
    - 提供市场状态分类器，用于实时市场状态判断
    - 为信号组合和策略优化提供市场状态信息
    
    属性:
        engine: SQLAlchemy数据库引擎，用于从数据库获取市场数据
    """
    
    def __init__(self, engine):
        """
        初始化市场状态分析器
        
        Args:
            engine: SQLAlchemy数据库引擎，用于连接到存储市场数据的数据库
        """
        self.engine = engine
        
    def get_market_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """获取市场数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            字典，键为股票代码，值为对应的数据框
        """
        market_data = {}
        for symbol in symbols:
            query = """
            SELECT *
            FROM stock_time_code
            WHERE Code = %s
            AND Date BETWEEN %s AND %s
            ORDER BY Date
            """
            df = pd.read_sql_query(query, self.engine, params=(symbol, start_date, end_date))
            if not df.empty:
                market_data[symbol] = df
        return market_data
        
    def calculate_market_trend(self, df: pd.DataFrame, windows: List[int] = [20, 50, 200]) -> Dict[str, float]:
        """计算市场趋势指标
        
        Args:
            df: 价格数据框
            windows: 移动平均窗口列表
            
        Returns:
            趋势指标字典
        """
        trends = {}
        current_price = df['Close'].iloc[-1]
        
        # 计算各周期移动平均线趋势
        for window in windows:
            ma = df['Close'].rolling(window=window).mean()
            trends[f'MA{window}_Trend'] = 1 if current_price > ma.iloc[-1] else -1
            
        # 计算趋势强度（基于斜率）
        for window in windows:
            ma = df['Close'].rolling(window=window).mean()
            slope = (ma.iloc[-1] - ma.iloc[-window]) / window
            trends[f'MA{window}_Strength'] = slope / ma.iloc[-1]  # 归一化斜率
            
        return trends
        
    def calculate_volatility_state(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算波动率状态
        
        Args:
            df: VIX指数数据框
            
        Returns:
            波动率状态字典
        """
        vix_current = df['Close'].iloc[-1]
        vix_ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        vix_percentile = df['Close'].rank(pct=True).iloc[-1]
        
        return {
            'VIX_Level': vix_current,
            'VIX_MA20_Ratio': vix_current / vix_ma20,
            'VIX_Percentile': vix_percentile
        }
        
    def calculate_sector_rotation(self, sector_data: Dict[str, pd.DataFrame], 
                                lookback: int = 20) -> Dict[str, float]:
        """计算行业轮动指标
        
        Args:
            sector_data: 行业ETF数据字典
            lookback: 回看期限
            
        Returns:
            行业轮动指标字典
        """
        rotation = {}
        # 计算各行业相对强弱
        for sector, df in sector_data.items():
            returns = df['Close'].pct_change(periods=lookback).iloc[-1]
            rotation[f'{sector}_RelStr'] = returns
            
        # 计算防御性行业与周期性行业的比值
        defensive = ['XLP', 'XLU', 'XLV']  # 必需消费品、公用事业、医疗保健
        cyclical = ['XLY', 'XLI', 'XLB']   # 可选消费品、工业、材料
        
        def_returns = np.mean([rotation[f'{sector}_RelStr'] for sector in defensive])
        cyc_returns = np.mean([rotation[f'{sector}_RelStr'] for sector in cyclical])
        
        rotation['Defensive_Cyclical_Ratio'] = def_returns / cyc_returns if cyc_returns != 0 else 1.0
        
        return rotation
        
    def calculate_breadth_indicators(self, df_spy: pd.DataFrame, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """计算市场宽度指标
        
        Args:
            df_spy: SPY数据框
            sector_data: 行业ETF数据字典
            
        Returns:
            市场宽度指标字典
        """
        breadth = {}
        
        # 计算行业ETF中处于上升趋势的比例
        sectors_above_ma50 = 0
        total_sectors = len(sector_data)
        
        for df in sector_data.values():
            ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            if df['Close'].iloc[-1] > ma50:
                sectors_above_ma50 += 1
                
        breadth['Sectors_Above_MA50'] = sectors_above_ma50 / total_sectors
        
        # 计算SPY的趋势确认指标
        spy_close = df_spy['Close'].iloc[-1]
        spy_ma20 = df_spy['Close'].rolling(window=20).mean().iloc[-1]
        spy_ma50 = df_spy['Close'].rolling(window=50).mean().iloc[-1]
        
        breadth['SPY_Trend_Confirm'] = 1.0 if spy_close > spy_ma20 > spy_ma50 else 0.0
        
        return breadth
        
    def calculate_market_state(self, date: str) -> Dict[str, float]:
        """计算特定日期的市场状态
        
        Args:
            date: 日期字符串
            
        Returns:
            市场状态指标字典
        """
        # 设置回溯期限
        lookback_days = 200
        start_date = pd.to_datetime(date) - pd.Timedelta(days=lookback_days)
        
        # 获取各类市场数据
        core_symbols = ['^VIX', 'SPY', 'QQQ', 'TLT']
        sector_symbols = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU']
        
        market_data = self.get_market_data(core_symbols, start_date.strftime('%Y-%m-%d'), date)
        sector_data = self.get_market_data(sector_symbols, start_date.strftime('%Y-%m-%d'), date)
        
        if not all(symbol in market_data for symbol in core_symbols):
            logger.warning(f"部分核心市场数据缺失: {date}")
            return {}
            
        # 计算各类指标
        market_state = {}
        
        # 1. 市场趋势
        spy_trends = self.calculate_market_trend(market_data['SPY'])
        market_state.update(spy_trends)
        
        # 2. 波动率状态
        vix_state = self.calculate_volatility_state(market_data['^VIX'])
        market_state.update(vix_state)
        
        # 3. 行业轮动
        sector_rotation = self.calculate_sector_rotation(sector_data)
        market_state.update(sector_rotation)
        
        # 4. 市场宽度
        breadth = self.calculate_breadth_indicators(market_data['SPY'], sector_data)
        market_state.update(breadth)
        
        # 5. 债券市场状态
        tlt_trends = self.calculate_market_trend(market_data['TLT'], windows=[20, 50])
        market_state.update({f'Bond_{k}': v for k, v in tlt_trends.items()})
        
        # 6. 科技股相对强弱
        qqq_spy_ratio = (market_data['QQQ']['Close'].pct_change(20).iloc[-1] / 
                        market_data['SPY']['Close'].pct_change(20).iloc[-1])
        market_state['Tech_RelStr'] = qqq_spy_ratio
        
        return market_state
        
    def get_market_regime(self, market_state: Dict[str, float]) -> str:
        """基于市场状态判断市场阶段
        
        Args:
            market_state: 市场状态指标字典
            
        Returns:
            市场阶段描述
        """
        if not market_state:
            return 'Unknown'
            
        # 定义市场阶段的判断规则
        if (market_state['MA200_Trend'] > 0 and 
            market_state['VIX_Percentile'] < 0.3 and 
            market_state['Sectors_Above_MA50'] > 0.7):
            return 'Bull_Strong'  # 强势牛市
            
        elif (market_state['MA50_Trend'] > 0 and 
              market_state['VIX_Percentile'] < 0.5 and 
              market_state['Sectors_Above_MA50'] > 0.5):
            return 'Bull_Normal'  # 普通牛市
            
        elif (market_state['MA20_Trend'] < 0 and 
              market_state['VIX_Percentile'] > 0.7 and 
              market_state['Sectors_Above_MA50'] < 0.3):
            return 'Bear_Strong'  # 强势熊市
            
        elif (market_state['MA50_Trend'] < 0 and 
              market_state['VIX_Percentile'] > 0.5 and 
              market_state['Sectors_Above_MA50'] < 0.5):
            return 'Bear_Normal'  # 普通熊市
            
        else:
            return 'Neutral'  # 盘整市场

def create_market_features(market_state: Dict[str, float]) -> pd.Series:
    """将市场状态转换为特征向量
    
    Args:
        market_state: 市场状态指标字典
        
    Returns:
        特征向量
    """
    features = pd.Series(market_state)
    
    # 添加市场阶段的独热编码
    regime_map = {
        'Bull_Strong': [1, 0, 0, 0, 0],
        'Bull_Normal': [0, 1, 0, 0, 0],
        'Neutral': [0, 0, 1, 0, 0],
        'Bear_Normal': [0, 0, 0, 1, 0],
        'Bear_Strong': [0, 0, 0, 0, 1],
        'Unknown': [0, 0, 0, 0, 0]
    }
    
    regime = MarketState(None).get_market_regime(market_state)
    regime_features = pd.Series(regime_map[regime], 
                              index=['Regime_' + k for k in regime_map.keys()])
    
    return pd.concat([features, regime_features]) 