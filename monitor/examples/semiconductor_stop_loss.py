import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.data_interface import YahooFinanceRealTimeSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiconductorStopLossAnalyzer:
    def __init__(self):
        self.data_source = YahooFinanceRealTimeSource()
        
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime):
        """获取历史数据"""
        return await self.data_source.get_historical_data(symbol, start_date, end_date, timeframe='1d')
        
    def calculate_correlation(self, stock_df: pd.DataFrame, sp500_df: pd.DataFrame):
        """计算股票与标普500的相关性"""
        stock_returns = stock_df['close'].pct_change()
        sp500_returns = sp500_df['close'].pct_change()
        return stock_returns.corr(sp500_returns)
        
    def calculate_relative_strength(self, stock_df: pd.DataFrame, sp500_df: pd.DataFrame):
        """计算相对强度"""
        stock_returns = stock_df['close'].pct_change()
        sp500_returns = sp500_df['close'].pct_change()
        relative_strength = (1 + stock_returns) / (1 + sp500_returns)
        return relative_strength.cumprod()
        
    async def analyze_semiconductor_stocks(self, start_date: datetime, end_date: datetime):
        """分析半导体股票"""
        # 获取数据
        amd_df = await self.get_historical_data("AMD", start_date, end_date)
        nvda_df = await self.get_historical_data("NVDA", start_date, end_date)
        sp500_df = await self.get_historical_data("^GSPC", start_date, end_date)
        
        if amd_df.empty or nvda_df.empty or sp500_df.empty:
            logger.warning("无法获取完整的历史数据")
            return None
            
        # 计算相关性
        amd_correlation = self.calculate_correlation(amd_df, sp500_df)
        nvda_correlation = self.calculate_correlation(nvda_df, sp500_df)
        
        # 计算相对强度
        amd_relative_strength = self.calculate_relative_strength(amd_df, sp500_df)
        nvda_relative_strength = self.calculate_relative_strength(nvda_df, sp500_df)
        
        # 计算关键指标
        amd_volatility = amd_df['close'].pct_change().std()
        nvda_volatility = nvda_df['close'].pct_change().std()
        
        # 计算建议止损点
        amd_stop_loss = min(0.15, amd_volatility * 2)
        nvda_stop_loss = min(0.15, nvda_volatility * 2)
        
        return {
            'AMD': {
                'correlation': amd_correlation,
                'relative_strength': amd_relative_strength,
                'volatility': amd_volatility,
                'suggested_stop_loss': amd_stop_loss
            },
            'NVDA': {
                'correlation': nvda_correlation,
                'relative_strength': nvda_relative_strength,
                'volatility': nvda_volatility,
                'suggested_stop_loss': nvda_stop_loss
            }
        }
        
    def generate_trading_signals(self, analysis: dict, current_prices: dict):
        """生成交易信号"""
        signals = {}
        
        for symbol, data in analysis.items():
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
                
            # 计算相对强度趋势
            relative_strength = data['relative_strength']
            recent_trend = relative_strength.iloc[-5:].mean() / relative_strength.iloc[-10:-5].mean()
            
            # 生成信号
            if recent_trend < 0.95:  # 相对强度下降
                signals[symbol] = {
                    'action': 'reduce',
                    'reason': '相对强度下降',
                    'stop_loss': data['suggested_stop_loss']
                }
            elif recent_trend < 0.9:  # 显著下降
                signals[symbol] = {
                    'action': 'sell',
                    'reason': '显著相对强度下降',
                    'stop_loss': data['suggested_stop_loss']
                }
            else:
                signals[symbol] = {
                    'action': 'hold',
                    'reason': '相对强度稳定',
                    'stop_loss': data['suggested_stop_loss']
                }
                
        return signals

async def main():
    # 初始化分析器
    analyzer = SemiconductorStopLossAnalyzer()
    
    # 设置分析时间范围（过去一年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 运行分析
    logger.info("开始分析半导体股票...")
    analysis = await analyzer.analyze_semiconductor_stocks(start_date, end_date)
    
    if analysis:
        # 获取当前价格
        current_prices = {
            'AMD': (await analyzer.get_historical_data("AMD", end_date - timedelta(days=1), end_date))['close'].iloc[-1],
            'NVDA': (await analyzer.get_historical_data("NVDA", end_date - timedelta(days=1), end_date))['close'].iloc[-1]
        }
        
        # 生成交易信号
        signals = analyzer.generate_trading_signals(analysis, current_prices)
        
        # 输出分析结果
        logger.info("\n分析结果:")
        for symbol, data in analysis.items():
            logger.info(f"\n{symbol}:")
            logger.info(f"与标普500相关性: {data['correlation']:.2f}")
            logger.info(f"波动率: {data['volatility']:.2%}")
            logger.info(f"建议止损点: {data['suggested_stop_loss']:.2%}")
            
            signal = signals[symbol]
            logger.info(f"交易建议: {signal['action']}")
            logger.info(f"原因: {signal['reason']}")

if __name__ == "__main__":
    asyncio.run(main()) 