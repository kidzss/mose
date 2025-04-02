import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketRelativeAnalyzer:
    def __init__(self, stock_data: pd.DataFrame, spy_data: pd.DataFrame):
        """
        初始化市场相对分析器
        
        参数:
            stock_data: 个股数据，包含OHLCV数据
            spy_data: SPY数据，作为市场基准
        """
        self.stock_data = stock_data
        self.spy_data = spy_data
        self._prepare_data()
        
    def _prepare_data(self):
        """准备分析所需的数据"""
        # 确保两个数据框使用相同的日期索引
        common_dates = self.stock_data.index.intersection(self.spy_data.index)
        self.stock_data = self.stock_data.loc[common_dates]
        self.spy_data = self.spy_data.loc[common_dates]
        
        # 计算收益率
        self.stock_returns = self.stock_data['Close'].pct_change()
        self.spy_returns = self.spy_data['Close'].pct_change()
        
        # 移除首行的NaN
        self.stock_returns = self.stock_returns.dropna()
        self.spy_returns = self.spy_returns.dropna()
        
    def calculate_beta(self, window: int = None) -> float:
        """
        计算Beta系数
        
        参数:
            window: 计算窗口，如果为None则使用全部数据
            
        返回:
            beta: Beta系数
        """
        if window:
            stock_returns = self.stock_returns[-window:]
            spy_returns = self.spy_returns[-window:]
        else:
            stock_returns = self.stock_returns
            spy_returns = self.spy_returns
            
        # 计算协方差和方差
        covariance = np.cov(stock_returns, spy_returns)[0][1]
        variance = np.var(spy_returns)
        
        # 计算beta
        beta = covariance / variance if variance != 0 else 1.0
        return beta
        
    def calculate_correlation(self, window: int = None) -> float:
        """
        计算与SPY的相关系数
        
        参数:
            window: 计算窗口，如果为None则使用全部数据
            
        返回:
            correlation: 相关系数
        """
        if window:
            stock_returns = self.stock_returns[-window:]
            spy_returns = self.spy_returns[-window:]
        else:
            stock_returns = self.stock_returns
            spy_returns = self.spy_returns
            
        return stock_returns.corr(spy_returns)
        
    def calculate_relative_strength(self, window: int = 252) -> float:
        """
        计算相对强弱指标
        
        参数:
            window: 计算窗口，默认使用一年的交易日数据
            
        返回:
            rs: 相对强弱指标
        """
        # 计算累积收益
        stock_cum_returns = (1 + self.stock_returns[-window:]).cumprod()
        spy_cum_returns = (1 + self.spy_returns[-window:]).cumprod()
        
        # 计算相对强弱
        rs = (stock_cum_returns[-1] / stock_cum_returns[0]) / (spy_cum_returns[-1] / spy_cum_returns[0])
        return rs
        
    def calculate_alpha(self, risk_free_rate: float = 0.0, window: int = None) -> float:
        """
        计算Alpha（超额收益）
        
        参数:
            risk_free_rate: 无风险利率，年化
            window: 计算窗口，如果为None则使用全部数据
            
        返回:
            alpha: Alpha值（年化）
        """
        if window:
            stock_returns = self.stock_returns[-window:]
            spy_returns = self.spy_returns[-window:]
        else:
            stock_returns = self.stock_returns
            spy_returns = self.spy_returns
            
        # 计算日化无风险利率
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # 计算beta
        beta = self.calculate_beta(window)
        
        # 计算平均超额收益
        stock_excess_return = stock_returns - daily_rf
        market_excess_return = spy_returns - daily_rf
        
        # 计算alpha（年化）
        alpha = (stock_excess_return.mean() - beta * market_excess_return.mean()) * 252
        return alpha
        
    def analyze_market_relative(self, window: int = 252) -> Dict:
        """
        综合分析相对于市场的表现
        
        参数:
            window: 分析窗口，默认使用一年的交易日数据
            
        返回:
            分析结果字典
        """
        beta = self.calculate_beta(window)
        correlation = self.calculate_correlation(window)
        rs = self.calculate_relative_strength(window)
        alpha = self.calculate_alpha(window=window)
        
        # 计算跟踪误差
        tracking_error = np.std(self.stock_returns[-window:] - self.spy_returns[-window:]) * np.sqrt(252)
        
        # 计算信息比率
        active_return = self.stock_returns[-window:].mean() - self.spy_returns[-window:].mean()
        information_ratio = (active_return * 252) / tracking_error if tracking_error != 0 else 0
        
        return {
            'beta': beta,
            'correlation': correlation,
            'relative_strength': rs,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
        
    def generate_report(self) -> str:
        """生成分析报告"""
        # 获取不同时间窗口的分析结果
        analysis_1y = self.analyze_market_relative(252)  # 一年
        analysis_6m = self.analyze_market_relative(126)  # 六个月
        analysis_3m = self.analyze_market_relative(63)   # 三个月
        
        report = f"""
市场相对分析报告
==============

一年期分析:
- Beta: {analysis_1y['beta']:.2f}
- 相关系数: {analysis_1y['correlation']:.2f}
- 相对强弱: {analysis_1y['relative_strength']:.2f}
- Alpha(年化): {analysis_1y['alpha']*100:.2f}%
- 跟踪误差: {analysis_1y['tracking_error']*100:.2f}%
- 信息比率: {analysis_1y['information_ratio']:.2f}

六个月分析:
- Beta: {analysis_6m['beta']:.2f}
- 相关系数: {analysis_6m['correlation']:.2f}
- 相对强弱: {analysis_6m['relative_strength']:.2f}
- Alpha(年化): {analysis_6m['alpha']*100:.2f}%
- 跟踪误差: {analysis_6m['tracking_error']*100:.2f}%
- 信息比率: {analysis_6m['information_ratio']:.2f}

三个月分析:
- Beta: {analysis_3m['beta']:.2f}
- 相关系数: {analysis_3m['correlation']:.2f}
- 相对强弱: {analysis_3m['relative_strength']:.2f}
- Alpha(年化): {analysis_3m['alpha']*100:.2f}%
- 跟踪误差: {analysis_3m['tracking_error']*100:.2f}%
- 信息比率: {analysis_3m['information_ratio']:.2f}

分析解读:
1. Beta {'>' if analysis_1y['beta'] > 1 else '<'} 1 表示股票相对于市场{'波动更大' if analysis_1y['beta'] > 1 else '波动更小'}
2. 相对强弱 {'>' if analysis_1y['relative_strength'] > 1 else '<'} 1 表示股票{'跑赢' if analysis_1y['relative_strength'] > 1 else '跑输'}大盘
3. Alpha {'为正' if analysis_1y['alpha'] > 0 else '为负'}说明{'有' if analysis_1y['alpha'] > 0 else '无'}超额收益
4. 信息比率反映了主动管理的效果，{'较好' if abs(analysis_1y['information_ratio']) > 0.5 else '一般'}
"""
        return report 