import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from data.data_interface import YahooFinanceRealTimeSource
from monitor.portfolio_monitor import PortfolioMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiconductorTradingStrategy:
    def __init__(self):
        self.data_source = YahooFinanceRealTimeSource()
        self.positions = {
            'AMD': {'price': 72.0, 'action': 'sell_all', 'support': 60.0},
            'NVDA': {'price': 84.0, 'action': 'reduce_80', 'support': None},
            'SPY': {'price': 4900.0, 'action': 'put_protection', 'support': None}
        }
        
    async def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        data = await self.data_source.get_historical_data(symbol, start_date, end_date, timeframe='1d')
        if not data.empty:
            return data['close'].iloc[-1]
        return None
        
    async def check_amd_strategy(self, current_price: float) -> dict:
        """检查AMD策略"""
        if current_price < self.positions['AMD']['price']:
            return {
                'action': 'sell_all',
                'reason': f'AMD跌破止损价{self.positions["AMD"]["price"]}',
                'details': f'当前价格: {current_price}, 强支撑位: {self.positions["AMD"]["support"]}'
            }
        return None
        
    async def check_nvda_strategy(self, current_price: float) -> dict:
        """检查NVDA策略"""
        if current_price < self.positions['NVDA']['price']:
            return {
                'action': 'reduce_80',
                'reason': f'NVDA跌破空头趋势线{self.positions["NVDA"]["price"]}',
                'details': f'当前价格: {current_price}, 建议减仓80%'
            }
        return None
        
    async def check_spy_strategy(self, current_price: float) -> dict:
        """检查标普500策略"""
        if current_price < self.positions['SPY']['price']:
            return {
                'action': 'put_protection',
                'reason': f'标普500跌破{self.positions["SPY"]["price"]}',
                'details': '建议在反弹时使用put期权进行保护'
            }
        return None
        
    async def analyze_market_conditions(self) -> dict:
        """分析市场条件并生成交易建议"""
        recommendations = {}
        
        # 获取当前价格
        current_prices = {}
        for symbol in self.positions.keys():
            price = await self.get_current_price(symbol)
            if price:
                current_prices[symbol] = price
                
        # 检查各个策略
        if 'AMD' in current_prices:
            amd_recommendation = await self.check_amd_strategy(current_prices['AMD'])
            if amd_recommendation:
                recommendations['AMD'] = amd_recommendation
                
        if 'NVDA' in current_prices:
            nvda_recommendation = await self.check_nvda_strategy(current_prices['NVDA'])
            if nvda_recommendation:
                recommendations['NVDA'] = nvda_recommendation
                
        if 'SPY' in current_prices:
            spy_recommendation = await self.check_spy_strategy(current_prices['SPY'])
            if spy_recommendation:
                recommendations['SPY'] = spy_recommendation
                
        return recommendations
        
    async def monitor_strategy(self, interval: int = 300):
        """持续监控策略"""
        while True:
            try:
                recommendations = await self.analyze_market_conditions()
                if recommendations:
                    logger.info("\n交易建议:")
                    for symbol, recommendation in recommendations.items():
                        logger.info(f"\n{symbol}:")
                        logger.info(f"操作: {recommendation['action']}")
                        logger.info(f"原因: {recommendation['reason']}")
                        logger.info(f"详情: {recommendation['details']}")
                else:
                    logger.info("当前无交易信号")
                    
            except Exception as e:
                logger.error(f"监控过程中发生错误: {str(e)}")
                
            await asyncio.sleep(interval)

async def main():
    strategy = SemiconductorTradingStrategy()
    await strategy.monitor_strategy()

if __name__ == "__main__":
    asyncio.run(main()) 