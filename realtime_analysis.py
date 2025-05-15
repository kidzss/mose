import asyncio
import logging
from datetime import datetime
import pandas as pd
from data.data_interface import YahooFinanceRealTimeSource
from strategy.market_analysis import MarketAnalysis

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealtimeAnalysis:
    def __init__(self):
        self.data_source = YahooFinanceRealTimeSource()
        self.market_analysis = MarketAnalysis()
        self.monitored_stocks = ['AMD', 'NVDA', 'PFE', 'MSFT', 'TMDX']
        
    async def analyze_stock(self, symbol: str):
        """分析单个股票的实时数据"""
        try:
            # 获取实时数据
            data = await self.data_source.get_realtime_data([symbol])
            if symbol not in data:
                logger.warning(f"未获取到 {symbol} 的实时数据")
                return
                
            # 获取最新数据
            latest_data = data[symbol].iloc[-1]
            
            # 计算技术指标
            indicators = self.market_analysis.calculate_technical_indicators(data[symbol])
            
            # 生成分析报告
            report = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': latest_data.get('close', 0),
                'change': latest_data.get('change', 0),
                'volume': latest_data.get('volume', 0),
                'indicators': {
                    'rsi': float(indicators['rsi'].iloc[-1]),
                    'macd': float(indicators['macd'].iloc[-1]),
                    'macd_signal': float(indicators['macd_signal'].iloc[-1]),
                    'ma_short': float(indicators['ma_short'].iloc[-1]),
                    'ma_medium': float(indicators['ma_medium'].iloc[-1]),
                    'ma_long': float(indicators['ma_long'].iloc[-1]),
                    'volatility': float(indicators['volatility'].iloc[-1])
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"分析 {symbol} 时出错: {str(e)}")
            return None
            
    async def run_analysis(self):
        """运行实时分析"""
        logger.info("开始实时分析...")
        
        while True:
            try:
                for symbol in self.monitored_stocks:
                    report = await self.analyze_stock(symbol)
                    if report:
                        logger.info(f"\n{symbol} 分析报告:")
                        logger.info(f"价格: {report['price']:.2f}")
                        logger.info(f"涨跌幅: {report['change']:.2%}")
                        logger.info(f"成交量: {report['volume']:,}")
                        logger.info("技术指标:")
                        logger.info(f"RSI: {report['indicators']['rsi']:.2f}")
                        logger.info(f"MACD: {report['indicators']['macd']:.2f}")
                        logger.info(f"MACD Signal: {report['indicators']['macd_signal']:.2f}")
                        logger.info(f"MA20: {report['indicators']['ma_short']:.2f}")
                        logger.info(f"MA50: {report['indicators']['ma_medium']:.2f}")
                        logger.info(f"MA200: {report['indicators']['ma_long']:.2f}")
                        logger.info(f"波动率: {report['indicators']['volatility']:.2%}")
                        
                # 等待1分钟再次分析
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"运行分析时出错: {str(e)}")
                await asyncio.sleep(60)  # 出错后等待1分钟再试

async def main():
    analyzer = RealtimeAnalysis()
    await analyzer.run_analysis()

if __name__ == "__main__":
    asyncio.run(main()) 