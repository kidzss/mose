import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class MarketStateAnalyzer:
    def __init__(self):
        self.states = {
            'strong_uptrend': {'confidence': 0.9, 'description': '强势上涨'},
            'uptrend': {'confidence': 0.7, 'description': '上涨趋势'},
            'weak_uptrend': {'confidence': 0.5, 'description': '弱势上涨'},
            'sideways': {'confidence': 0.3, 'description': '横盘整理'},
            'weak_downtrend': {'confidence': -0.5, 'description': '弱势下跌'},
            'downtrend': {'confidence': -0.7, 'description': '下跌趋势'},
            'strong_downtrend': {'confidence': -0.9, 'description': '强势下跌'}
        }
        
    async def analyze_market_states(self, df: pd.DataFrame) -> List[Dict]:
        """分析市场状态"""
        try:
            # 计算技术指标
            df = self._calculate_indicators(df)
            
            # 分析每个时间点的市场状态
            states = []
            for i in range(len(df)):
                state = self._analyze_single_state(df.iloc[i])
                states.append(state)
                
            return states
        except Exception as e:
            print(f"分析市场状态时出错: {str(e)}")
            return []
            
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()
        
        # 计算成交量变化
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
        
        return df
        
    def _analyze_single_state(self, row: pd.Series) -> Dict:
        """分析单个时间点的市场状态"""
        # 初始化分数
        score = 0
        confidence = 0
        reasons = []
        
        # 趋势分析
        if row['close'] > row['MA20']:
            score += 1
            if row['close'] > row['MA50']:
                score += 1
                reasons.append("价格位于长期均线上方")
        else:
            score -= 1
            if row['close'] < row['MA50']:
                score -= 1
                reasons.append("价格位于长期均线下方")
                
        # MACD分析
        if row['MACD_Hist'] > 0:
            score += 1
            if row['MACD'] > 0:
                score += 1
                reasons.append("MACD显示上涨动能")
        else:
            score -= 1
            if row['MACD'] < 0:
                score -= 1
                reasons.append("MACD显示下跌动能")
                
        # RSI分析
        if row['RSI'] > 70:
            score += 1
            reasons.append("RSI显示超买")
        elif row['RSI'] < 30:
            score -= 1
            reasons.append("RSI显示超卖")
            
        # 成交量分析
        if row['volume'] > row['Volume_MA20']:
            if score > 0:
                score += 1
                reasons.append("放量上涨")
            else:
                score -= 1
                reasons.append("放量下跌")
                
        # 布林带分析
        if row['close'] > row['BB_Upper']:
            score += 1
            reasons.append("价格突破布林带上轨")
        elif row['close'] < row['BB_Lower']:
            score -= 1
            reasons.append("价格跌破布林带下轨")
            
        # 根据分数确定市场状态
        if score >= 4:
            state_type = 'strong_uptrend'
        elif score >= 2:
            state_type = 'uptrend'
        elif score >= 0:
            state_type = 'weak_uptrend'
        elif score == 0:
            state_type = 'sideways'
        elif score >= -2:
            state_type = 'weak_downtrend'
        elif score >= -4:
            state_type = 'downtrend'
        else:
            state_type = 'strong_downtrend'
            
        # 计算置信度
        base_confidence = self.states[state_type]['confidence']
        confidence = min(abs(score) / 6, 1) * base_confidence
        
        return {
            'state_type': state_type,
            'description': self.states[state_type]['description'],
            'confidence': confidence,
            'score': score,
            'reasons': reasons
        } 