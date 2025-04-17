import pandas as pd
import numpy as np
import datetime as dt
import time
import logging
from typing import List, Dict, Optional, Union, Tuple, Set, Any
import threading
import json
import os
from pathlib import Path
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import talib
import traceback
import redis
from sqlalchemy import text

from .data_fetcher import DataFetcher
from backtest.risk_manager import RiskManager
from backtest.volatility_manager import VolatilityManager
from backtest.strategy_factory import StrategyFactory
from backtest.strategy import CombinedStrategy
from utils.data_loader import DataLoader

class Alert:
    """警报类，用于生成和管理警报"""
    def __init__(self, alert_type: str, message: str, level: str = 'info'):
        self.type = alert_type
        self.message = message
        self.level = level
        self.timestamp = pd.Timestamp.now()

class MarketMonitor:
    """市场监控类，用于分析市场状态和生成警报"""
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化市场监控器
        
        参数:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path) if config_path else {}
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载配置文件时出错: {str(e)}")
            return {}
            
    def _analyze_market_state(self, market_data: Dict) -> Dict:
        """
        分析市场状态
        
        参数:
            market_data: 市场数据字典
            
        返回:
            Dict: 市场状态字典
        """
        try:
            if not market_data:
                return {'market_state': 'unknown'}
                
            # 分析市场趋势
            trend = self._analyze_trend(market_data)
            
            # 分析市场动量
            momentum = self._analyze_momentum(market_data)
            
            # 分析市场宽度
            breadth = self._analyze_market_breadth(market_data)
            
            # 分析风险水平
            risk = self._analyze_risk_level(market_data)
            
            # 综合分析
            market_state = 'neutral'  # 默认状态
            if trend > 0.6 and momentum > 0.6 and breadth > 0.6 and risk < 0.4:
                market_state = 'bullish'
            elif trend < 0.4 and momentum < 0.4 and breadth < 0.4 and risk > 0.6:
                market_state = 'bearish'
                
            return {
                'market_state': market_state,
                'trend': trend,
                'momentum': momentum,
                'breadth': breadth,
                'risk': risk
            }
            
        except Exception as e:
            self.logger.error(f"分析市场状态时出错: {str(e)}")
            return {'market_state': 'unknown'}
            
    def _analyze_trend(self, market_data: Dict) -> float:
        """分析市场趋势"""
        try:
            if not isinstance(market_data, dict):
                self.logger.error("市场数据格式错误")
                return 0.5
                
            # 初始化趋势得分
            trend_score = 0.5
            
            # 计算移动平均线
            for symbol, data in market_data.items():
                close_prices = pd.Series(data['close'])
                ma5 = close_prices.rolling(window=5).mean()
                ma20 = close_prices.rolling(window=20).mean()
                ma60 = close_prices.rolling(window=60).mean()
                
                # 短期趋势 (40%)
                if ma5.iloc[-1] > ma20.iloc[-1]:
                    trend_score += 0.2
                else:
                    trend_score -= 0.2
                    
                # 中期趋势 (30%)
                if ma20.iloc[-1] > ma60.iloc[-1]:
                    trend_score += 0.15
                else:
                    trend_score -= 0.15
                    
                # 价格位置 (30%)
                current_price = close_prices.iloc[-1]
                if current_price > ma5.iloc[-1]:
                    trend_score += 0.1
                if current_price > ma20.iloc[-1]:
                    trend_score += 0.1
                if current_price > ma60.iloc[-1]:
                    trend_score += 0.1
            
            # 确保分数在0-1范围内
            trend_score = max(0.0, min(1.0, trend_score))
            
            return float(trend_score)
            
        except Exception as e:
            self.logger.error(f"分析趋势时出错: {str(e)}")
            return 0.5
            
    def _analyze_momentum(self, market_data: Dict) -> float:
        """分析市场动量"""
        try:
            if not isinstance(market_data, dict):
                self.logger.error("市场数据格式错误")
                return 0.5
                
            # 初始化动量得分
            momentum_score = 0.5
            
            for symbol, data in market_data.items():
                close_prices = pd.Series(data['close'])
                
                # 计算RSI (40%)
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                if rsi.iloc[-1] > 70:
                    momentum_score += 0.2
                elif rsi.iloc[-1] < 30:
                    momentum_score -= 0.2
                    
                # 计算MACD (30%)
                exp1 = close_prices.ewm(span=12, adjust=False).mean()
                exp2 = close_prices.ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                
                if macd.iloc[-1] > signal.iloc[-1]:
                    momentum_score += 0.15
                else:
                    momentum_score -= 0.15
                    
                # 计算价格动量 (30%)
                returns_5 = (close_prices.iloc[-1] / close_prices.iloc[-5] - 1)
                returns_20 = (close_prices.iloc[-1] / close_prices.iloc[-20] - 1)
                
                if returns_5 > 0:
                    momentum_score += 0.15
                else:
                    momentum_score -= 0.15
                    
                if returns_20 > 0:
                    momentum_score += 0.15
                else:
                    momentum_score -= 0.15
            
            # 确保分数在0-1范围内
            momentum_score = max(0.0, min(1.0, momentum_score))
            
            return float(momentum_score)
            
        except Exception as e:
            self.logger.error(f"分析动量时出错: {str(e)}")
            return 0.5
        
    def _analyze_market_breadth(self, market_data: Dict) -> float:
        """分析市场宽度"""
        try:
            if not isinstance(market_data, dict):
                self.logger.error("市场数据格式错误")
                return 0.5
                
            # 初始化市场宽度得分
            breadth_score = 0.5
            
            # 计算上涨和下跌的股票数量
            advances = 0
            declines = 0
            
            for symbol, data in market_data.items():
                close_prices = pd.Series(data['close'])
                if close_prices.iloc[-1] > close_prices.iloc[-2]:
                    advances += 1
                else:
                    declines += 1
                    
            total_stocks = advances + declines
            if total_stocks > 0:
                advance_ratio = advances / total_stocks
                
                # 根据上涨比例调整得分
                if advance_ratio >= 0.7:
                    breadth_score = 0.9
                elif advance_ratio >= 0.6:
                    breadth_score = 0.7
                elif advance_ratio >= 0.4:
                    breadth_score = 0.5
                elif advance_ratio >= 0.3:
                    breadth_score = 0.3
                else:
                    breadth_score = 0.1
                    
            return float(breadth_score)
            
        except Exception as e:
            self.logger.error(f"分析市场宽度时出错: {str(e)}")
            return 0.5

    def _analyze_risk_level(self, market_data: Dict) -> float:
        """分析风险水平"""
        try:
            if not isinstance(market_data, dict):
                self.logger.error("市场数据格式错误")
                return 0.5

            # 初始化风险得分
            risk_score = 0.5
            
            for symbol, data in market_data.items():
                close_prices = pd.Series(data['close'])
                
                # 计算波动率 (40%)
                returns = close_prices.pct_change()
                volatility = returns.std() * np.sqrt(252)
                
                if volatility > 0.4:
                    risk_score += 0.2
                elif volatility > 0.3:
                    risk_score += 0.1
                elif volatility < 0.1:
                    risk_score -= 0.2
                
                # 计算最大回撤 (30%)
                cummax = close_prices.cummax()
                drawdown = (close_prices - cummax) / cummax
                max_drawdown = abs(drawdown.min())
            
                if max_drawdown > 0.2:
                    risk_score += 0.15
                elif max_drawdown > 0.1:
                    risk_score += 0.1
                elif max_drawdown < 0.05:
                    risk_score -= 0.15
                    
                # 计算夏普比率 (30%)
                avg_return = returns.mean() * 252
                sharpe = avg_return / volatility if volatility != 0 else 0
                
                if sharpe < 0:
                    risk_score += 0.15
                elif sharpe < 1:
                    risk_score += 0.1
                elif sharpe > 2:
                    risk_score -= 0.15
                    
            # 确保分数在0-1范围内
            risk_score = max(0.0, min(1.0, risk_score))
            
            return float(risk_score)
            
        except Exception as e:
            self.logger.error(f"分析风险水平时出错: {str(e)}")
            return 0.5