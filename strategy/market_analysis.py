from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketAnalysis:
    """
    市场分析类，整合技术指标和市场状态分析
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        # 默认参数
        self.default_params = {
            # RSI参数
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # MACD参数
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # 移动平均线参数
            'ma_short': 20,
            'ma_medium': 50,
            'ma_long': 200,
            
            # 波动率参数
            'atr_period': 14,
            'volatility_period': 20,
            
            # 趋势强度参数
            'adx_period': 14,
            'adx_threshold': 25,
            
            # 风险等级参数
            'risk_levels': {
                'low': 0.2,
                'medium': 0.4,
                'high': 0.6
            }
        }
        
        # 更新参数
        if params:
            self.default_params.update(params)
            
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        
        # 1. RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.default_params['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.default_params['rsi_period']).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 2. MACD
        exp1 = df['close'].ewm(span=self.default_params['macd_fast'], adjust=False).mean()
        exp2 = df['close'].ewm(span=self.default_params['macd_slow'], adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.default_params['macd_signal'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 3. 移动平均线
        df['ma_short'] = df['close'].rolling(window=self.default_params['ma_short']).mean()
        df['ma_medium'] = df['close'].rolling(window=self.default_params['ma_medium']).mean()
        df['ma_long'] = df['close'].rolling(window=self.default_params['ma_long']).mean()
        
        # 4. 波动率
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=self.default_params['atr_period']).mean()
        
        # 价格波动率
        df['volatility'] = df['close'].pct_change().rolling(window=self.default_params['volatility_period']).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=60).mean()
        
        return df
        
    def analyze_market_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析市场状态
        
        参数:
            data: 计算好指标的DataFrame
            
        返回:
            市场状态分析结果
        """
        current_data = data.iloc[-1]
        
        # 1. 趋势强度分析
        trend_strength = self._analyze_trend_strength(data)
        
        # 2. 波动率水平分析
        volatility_level = self._analyze_volatility(data)
        
        # 3. 风险等级分析
        risk_level = self._analyze_risk_level(data)
        
        # 4. 行业机会分析
        sector_opportunities = self._analyze_sector_opportunities(data)
        
        return {
            'trend_strength': trend_strength,
            'volatility_level': volatility_level,
            'risk_level': risk_level,
            'sector_opportunities': sector_opportunities,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def _analyze_trend_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析趋势强度
        
        参数:
            data: 计算好指标的DataFrame
            
        返回:
            趋势强度分析结果
        """
        current_data = data.iloc[-1]
        
        # 计算ADX
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # 计算方向移动
        up_move = data['high'] - data['high'].shift()
        down_move = data['low'].shift() - data['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # 计算平滑后的TR和DM
        tr_smooth = true_range.rolling(window=self.default_params['adx_period']).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=self.default_params['adx_period']).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=self.default_params['adx_period']).mean()
        
        # 计算方向指标
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.default_params['adx_period']).mean()
        
        # 判断趋势方向
        trend_direction = 'up' if current_data['ma_short'] > current_data['ma_long'] else 'down'
        
        # 判断趋势强度
        if adx.iloc[-1] > self.default_params['adx_threshold']:
            trend_strength = 'strong'
        elif adx.iloc[-1] > self.default_params['adx_threshold'] * 0.5:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
            
        return {
            'direction': trend_direction,
            'strength': trend_strength,
            'adx': float(adx.iloc[-1]),
            'plus_di': float(plus_di.iloc[-1]),
            'minus_di': float(minus_di.iloc[-1])
        }
        
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析波动率水平
        
        参数:
            data: 计算好指标的DataFrame
            
        返回:
            波动率分析结果
        """
        current_data = data.iloc[-1]
        
        # 判断波动率水平
        if current_data['volatility_ratio'] > 2.0:
            volatility_level = 'high'
        elif current_data['volatility_ratio'] > 1.5:
            volatility_level = 'elevated'
        elif current_data['volatility_ratio'] < 0.5:
            volatility_level = 'low'
        else:
            volatility_level = 'normal'
            
        return {
            'level': volatility_level,
            'atr': float(current_data['atr']),
            'volatility': float(current_data['volatility']),
            'volatility_ratio': float(current_data['volatility_ratio'])
        }
        
    def _analyze_risk_level(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析风险等级
        
        参数:
            data: 计算好指标的DataFrame
            
        返回:
            风险等级分析结果
        """
        current_data = data.iloc[-1]
        
        # 计算风险得分
        risk_score = (
            current_data['volatility_ratio'] * 0.4 +
            (1 - current_data['rsi'] / 100) * 0.3 +
            abs(current_data['macd_hist']) * 0.3
        )
        
        # 判断风险等级
        if risk_score > self.default_params['risk_levels']['high']:
            risk_level = 'high'
        elif risk_score > self.default_params['risk_levels']['medium']:
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        return {
            'level': risk_level,
            'score': float(risk_score),
            'components': {
                'volatility_ratio': float(current_data['volatility_ratio']),
                'rsi': float(current_data['rsi']),
                'macd_hist': float(current_data['macd_hist'])
            }
        }
        
    def _analyze_sector_opportunities(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析行业机会
        
        参数:
            data: 计算好指标的DataFrame
            
        返回:
            行业机会分析结果
        """
        # 这里需要行业数据，暂时返回空结果
        return {
            'sectors': [],
            'opportunities': []
        }
        
    def get_analysis_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取分析元数据
        
        返回:
            分析元数据字典
        """
        return {
            'trend_strength': {
                'description': '趋势强度分析',
                'components': ['adx', 'plus_di', 'minus_di'],
                'levels': ['strong', 'moderate', 'weak']
            },
            'volatility_level': {
                'description': '波动率水平分析',
                'components': ['atr', 'volatility', 'volatility_ratio'],
                'levels': ['high', 'elevated', 'normal', 'low']
            },
            'risk_level': {
                'description': '风险等级分析',
                'components': ['volatility_ratio', 'rsi', 'macd_hist'],
                'levels': ['high', 'medium', 'low']
            },
            'sector_opportunities': {
                'description': '行业机会分析',
                'components': ['sectors', 'opportunities']
            }
        } 