import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MAStrategy:
    """移动平均线交叉策略"""
    
    def __init__(self, short_window=20, long_window=50):
        """
        初始化移动平均线策略
        
        参数:
            short_window: 短期移动平均线窗口
            long_window: 长期移动平均线窗口
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # 创建副本避免警告
            result = data.copy()
            
            # 计算移动平均线
            result.loc[:, 'ma_short'] = result['close'].rolling(window=self.short_window).mean()
            result.loc[:, 'ma_long'] = result['close'].rolling(window=self.long_window).mean()
            
            # 计算移动平均线交叉信号
            result.loc[:, 'signal'] = np.where(
                result['ma_short'] > result['ma_long'], 1, 0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            return data
    
    def generate_signal(self, data: pd.DataFrame) -> dict:
        """生成交易信号"""
        try:
            current_data = data.iloc[-1]
            signal = {'action': 'hold', 'price': current_data['close']}
            
            # 移动平均线交叉策略信号生成逻辑
            if current_data['signal'] == 1 and data.iloc[-2]['signal'] == 0:
                signal['action'] = 'buy'
            elif current_data['signal'] == 0 and data.iloc[-2]['signal'] == 1:
                signal['action'] = 'sell'
            
            return signal
            
        except Exception as e:
            logger.error(f"生成交易信号时出错: {str(e)}")
            return {'action': 'hold', 'price': data['close'].iloc[-1]} 