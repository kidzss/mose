import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.data_interface import YahooFinanceRealTimeSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopLossBacktester:
    def __init__(self):
        self.data_source = YahooFinanceRealTimeSource()
        
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime):
        """获取历史数据"""
        return await self.data_source.get_historical_data(symbol, start_date, end_date, timeframe='1d')
        
    def calculate_returns(self, df: pd.DataFrame, stop_loss: float):
        """计算在给定止损点下的收益率"""
        if df.empty:
            return 0.0
            
        # 获取初始价格
        initial_price = df['close'].iloc[0]
        stop_loss_price = initial_price * (1 - stop_loss)
        
        # 计算每日收益率
        returns = []
        current_price = initial_price
        
        for _, row in df.iterrows():
            price = row['close']
            
            # 检查是否触发止损
            if price <= stop_loss_price:
                returns.append((stop_loss_price - initial_price) / initial_price)
                break
                
            # 计算当日收益率
            daily_return = (price - current_price) / current_price
            returns.append(daily_return)
            current_price = price
            
        # 计算总收益率
        total_return = (1 + pd.Series(returns)).prod() - 1
        return total_return
        
    async def backtest_stop_loss(self, symbol: str, start_date: datetime, 
                               end_date: datetime, stop_loss_levels: list):
        """回测不同止损点"""
        # 获取历史数据
        df = await self.get_historical_data(symbol, start_date, end_date)
        if df.empty:
            logger.warning(f"无法获取 {symbol} 的历史数据")
            return None
            
        results = {}
        for stop_loss in stop_loss_levels:
            returns = self.calculate_returns(df, stop_loss)
            results[stop_loss] = returns
            
        return results
        
    async def backtest_portfolio(self, positions: dict, start_date: datetime, 
                               end_date: datetime, stop_loss_levels: list):
        """回测整个投资组合"""
        portfolio_results = {}
        
        for symbol, data in positions.items():
            logger.info(f"回测 {symbol}...")
            results = await self.backtest_stop_loss(symbol, start_date, end_date, stop_loss_levels)
            if results:
                portfolio_results[symbol] = results
                
        return portfolio_results
        
    def analyze_results(self, results: dict):
        """分析回测结果"""
        analysis = {}
        
        for symbol, stop_loss_returns in results.items():
            # 找到最佳止损点
            best_stop_loss = max(stop_loss_returns.items(), key=lambda x: x[1])[0]
            worst_stop_loss = min(stop_loss_returns.items(), key=lambda x: x[1])[0]
            
            analysis[symbol] = {
                'best_stop_loss': best_stop_loss,
                'best_return': stop_loss_returns[best_stop_loss],
                'worst_stop_loss': worst_stop_loss,
                'worst_return': stop_loss_returns[worst_stop_loss],
                'all_returns': stop_loss_returns
            }
            
        return analysis

async def main():
    # 初始化回测器
    backtester = StopLossBacktester()
    
    # 设置回测参数
    positions = {
        "GOOG": {"cost_basis": 170.478, "shares": 59},
        "TSLA": {"cost_basis": 289.434, "shares": 28},
        "AMD": {"cost_basis": 123.737, "shares": 58},
        "NVDA": {"cost_basis": 138.843, "shares": 40},
        "PFE": {"cost_basis": 25.899, "shares": 80},
        "MSFT": {"cost_basis": 370.95, "shares": 3},
        "TMDX": {"cost_basis": 101.75, "shares": 13}
    }
    
    # 设置回测时间范围（过去一年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 测试不同的止损点
    stop_loss_levels = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    
    # 运行回测
    logger.info("开始回测...")
    results = await backtester.backtest_portfolio(positions, start_date, end_date, stop_loss_levels)
    
    # 分析结果
    analysis = backtester.analyze_results(results)
    
    # 输出结果
    logger.info("\n回测结果分析:")
    for symbol, data in analysis.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"最佳止损点: {data['best_stop_loss']:.2%} (收益率: {data['best_return']:.2%})")
        logger.info(f"最差止损点: {data['worst_stop_loss']:.2%} (收益率: {data['worst_return']:.2%})")
        logger.info("所有止损点的收益率:")
        for stop_loss, returns in data['all_returns'].items():
            logger.info(f"  {stop_loss:.2%}: {returns:.2%}")

if __name__ == "__main__":
    asyncio.run(main()) 