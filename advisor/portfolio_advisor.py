import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

class PortfolioAdvisor:
    def __init__(self, data_source):
        self.data_source = data_source

    async def analyze_market_trend(self, symbol: str, days: int) -> Dict[str, Any]:
        """
        分析市场趋势，返回趋势指标、置信度、成交量变化和价格变化。
        
        Args:
            symbol: 股票代码
            days: 分析的天数
            
        Returns:
            包含趋势分析结果的字典，包括：
            - trend: 趋势指标 (-100 到 100)
            - confidence: 置信度 (0 到 100)
            - volume_change: 成交量变化百分比
            - price_change: 价格变化百分比
        """
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        
        # 获取历史数据
        df = await self.data_source.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # 计算价格变化
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        
        # 计算成交量变化
        avg_volume_first_5 = df['volume'].iloc[:5].mean()
        avg_volume_last_5 = df['volume'].iloc[-5:].mean()
        volume_change = ((avg_volume_last_5 - avg_volume_first_5) / avg_volume_first_5) * 100
        
        # 计算趋势
        # 使用简单的移动平均线判断趋势
        ma5 = df['close'].rolling(window=5).mean()
        ma20 = df['close'].rolling(window=20).mean()
        
        # 如果5日线在20日线上方，趋势为正
        trend = 100 if ma5.iloc[-1] > ma20.iloc[-1] else -100
        
        # 计算置信度
        # 基于价格与移动平均线的距离计算置信度
        price_distance = abs(df['close'].iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100
        confidence = min(100, price_distance * 2)  # 将距离转换为0-100的置信度
        
        return {
            'trend': trend,
            'confidence': float(confidence),
            'volume_change': float(volume_change),
            'price_change': float(price_change)
        } 