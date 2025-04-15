import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TDIStrategy:
    """TDI策略类"""
    
    def __init__(self, rsi_period=14, adx_period=14):
        """
        初始化TDI策略
        
        参数:
            rsi_period: RSI周期
            adx_period: ADX周期
        """
        self.rsi_period = rsi_period
        self.adx_period = adx_period
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # 创建副本避免警告
            result = data.copy()
            
            # 计算RSI
            delta = result['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            rs = avg_gain / avg_loss
            result.loc[:, 'rsi'] = 100 - (100 / (1 + rs))
            
            # 计算ADX
            high_low = result['high'] - result['low']
            high_close = abs(result['high'] - result['close'].shift())
            low_close = abs(result['low'] - result['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=self.adx_period).mean()
            
            plus_dm = result['high'].diff()
            minus_dm = result['low'].diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0)
            minus_dm = minus_dm.where(minus_dm < 0, 0).abs()
            
            tr14 = true_range.rolling(window=self.adx_period).sum()
            plus_di14 = 100 * plus_dm.rolling(window=self.adx_period).sum() / tr14
            minus_di14 = 100 * minus_dm.rolling(window=self.adx_period).sum() / tr14
            dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
            result.loc[:, 'adx'] = dx.rolling(window=self.adx_period).mean()
            
            return result
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            return data
    
    def generate_signal(self, data: pd.DataFrame) -> dict:
        """生成交易信号"""
        try:
            current_data = data.iloc[-1]
            signal = {'action': 'hold', 'price': current_data['close']}
            
            # TDI策略信号生成逻辑
            if current_data['rsi'] < 30 and current_data['adx'] > 25:
                signal['action'] = 'buy'
            elif current_data['rsi'] > 70 and current_data['adx'] > 25:
                signal['action'] = 'sell'
            
            return signal
            
        except Exception as e:
            logger.error(f"生成交易信号时出错: {str(e)}")
            return {'action': 'hold', 'price': data['close'].iloc[-1]} 