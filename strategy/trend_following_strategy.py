from typing import Dict, Any
import pandas as pd
import numpy as np
from .strategy_base import Strategy
from .market_analysis import MarketAnalysis

class TrendFollowingStrategy(Strategy):
    """
    趋势跟踪策略，结合多时间框架趋势确认、趋势强度指标和波动率管理
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        
        # 默认参数
        self.default_params = {
            # 趋势确认参数
            'short_term_period': 20,  # 短期趋势周期
            'medium_term_period': 50,  # 中期趋势周期
            'long_term_period': 200,   # 长期趋势周期
            
            # 趋势强度参数
            'adx_period': 14,         # ADX计算周期
            'adx_threshold': 25,      # ADX趋势强度阈值
            
            # 趋势确认参数
            'rsi_period': 14,         # RSI计算周期
            'rsi_overbought': 70,     # RSI超买阈值
            'rsi_oversold': 30,       # RSI超卖阈值
            
            # 波动率管理参数
            'atr_period': 14,         # ATR计算周期
            'volatility_threshold': 0.02,  # 波动率阈值
            'position_size': 0.1,     # 基础仓位大小
            'max_position_size': 0.3,  # 最大仓位大小
        }
        
        # 更新参数
        if params:
            self.default_params.update(params)
            
        # 初始化状态变量
        self.current_position = 0
        self.current_losing_streak = 0
        self.total_pnl = 0
        
        # 初始化市场分析器
        self.market_analyzer = MarketAnalysis()
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        
        # 1. 多时间框架趋势确认
        # 计算移动平均线
        df['ma_short'] = df['close'].rolling(window=self.default_params['short_term_period']).mean()
        df['ma_medium'] = df['close'].rolling(window=self.default_params['medium_term_period']).mean()
        df['ma_long'] = df['close'].rolling(window=self.default_params['long_term_period']).mean()
        
        # 计算趋势方向
        df['trend_short'] = df['close'] > df['ma_short']
        df['trend_medium'] = df['close'] > df['ma_medium']
        df['trend_long'] = df['close'] > df['ma_long']
        
        # 计算趋势得分 (0-3)
        df['trend_score'] = (
            df['trend_short'].astype(int) * 0.5 +
            df['trend_medium'].astype(int) * 0.8 +
            df['trend_long'].astype(int) * 1.0
        )
        
        # 2. 趋势强度指标
        # 计算ADX
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # 计算方向移动
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
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
        df['adx'] = dx.rolling(window=self.default_params['adx_period']).mean()
        
        # 3. 趋势确认指标
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.default_params['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.default_params['rsi_period']).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 4. 波动率管理
        # 计算ATR
        df['atr'] = true_range.rolling(window=self.default_params['atr_period']).mean()
        
        # 计算波动率
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=60).mean()
        
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
            
            # 1. 趋势确认
            trend_aligned = (
                current_data['trend_short'] and
                current_data['trend_medium'] and
                current_data['trend_long']
            )
            
            # 2. 趋势强度
            trend_strong = current_data['adx'] > self.default_params['adx_threshold']
            
            # 3. 趋势确认指标
            rsi_ok = (
                current_data['rsi'] < self.default_params['rsi_overbought'] and
                current_data['rsi'] > self.default_params['rsi_oversold']
            )
            macd_bullish = current_data['macd'] > current_data['macd_signal']
            
            # 4. 波动率管理
            volatility_ok = current_data['volatility_ratio'] < 2.0
            
            # 生成买入信号
            if (trend_aligned and trend_strong and rsi_ok and macd_bullish and volatility_ok):
                df.loc[df.index[i], 'signal'] = 1
                
            # 生成卖出信号
            elif (not trend_aligned and trend_strong and not rsi_ok and not macd_bullish and volatility_ok):
                df.loc[df.index[i], 'signal'] = -1
                
        return df
        
    def calculate_position_size(self, data: pd.DataFrame) -> float:
        """
        计算仓位大小
        
        参数:
            data: 计算好指标的DataFrame
            
        返回:
            仓位大小 (0-1)
        """
        current_data = data.iloc[-1]
        
        # 基础仓位
        position_size = self.default_params['position_size']
        
        # 根据趋势强度调整仓位
        if current_data['trend_score'] > 2.0:
            position_size *= 1.5
        elif current_data['trend_score'] < 1.0:
            position_size *= 0.5
            
        # 根据波动率调整仓位
        if current_data['volatility_ratio'] > 1.5:
            position_size *= 0.5
            
        # 根据当前亏损情况调整仓位
        if self.current_losing_streak > 0:
            position_size *= (1 - 0.1 * self.current_losing_streak)
            
        # 确保仓位大小在合理范围内
        position_size = max(0.01, min(position_size, self.default_params['max_position_size']))
        
        return position_size
        
    def _check_risk_limits(self, current_price: float) -> bool:
        """
        检查风险限制
        
        参数:
            current_price: 当前价格
            
        返回:
            是否达到风险限制
        """
        # 检查每日最大亏损
        if self.total_pnl < -self.default_params['max_daily_loss'] * current_price:
            self.logger.warning("Daily loss limit reached")
            return True
            
        # 检查总最大亏损
        if self.total_pnl < -self.default_params['max_total_loss'] * current_price:
            self.logger.warning("Total loss limit reached")
            return True
            
        return False 