from typing import List, Dict
import pandas as pd
from data.data_updater import MarketDataUpdater
from log import logger
from monitor.stock_monitor import StockMonitor
from monitor.strategy_manager import StrategyManager
from monitor.notification_manager import NotificationManager

class Monitor:
    def __init__(self):
        self.data_updater = MarketDataUpdater()
        self.stock_monitor = StockMonitor()
        self.strategy_manager = StrategyManager()
        self.notification_manager = NotificationManager()
        
    def get_stock_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """获取股票数据"""
        try:
            # 直接从data_updater获取数据
            data = {}
            for symbol in symbols:
                try:
                    # 确保symbol是字符串
                    if not isinstance(symbol, str):
                        symbol = str(symbol)
                        
                    # 获取最新数据
                    end_date = pd.Timestamp.now()
                    start_date = end_date - pd.Timedelta(days=120)
                    
                    # 使用data_fetcher获取数据
                    df = self.data_updater.data_fetcher.get_stock_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if df is not None and not df.empty:
                        data[symbol] = df
                    else:
                        logger.warning(f"股票 {symbol} 没有获取到数据")
                except Exception as e:
                    logger.error(f"获取 {symbol} 的数据时出错: {str(e)}")
                    continue
                    
            if not data:
                logger.error("没有获取到任何股票数据")
            else:
                logger.info(f"成功获取 {len(data)} 只股票的数据")
                
            return data
                
        except Exception as e:
            logger.error(f"获取股票数据时出错: {str(e)}")
            return {}
            
    def analyze_and_notify(self, symbols: List[str]):
        """分析股票并发送通知"""
        try:
            # 获取股票数据
            stock_data = self.get_stock_data(symbols)
            
            # 分析每只股票
            analysis_results = []
            for symbol, data in stock_data.items():
                try:
                    # 使用策略管理器分析
                    strategy_analysis = self.strategy_manager.analyze_stock(data, symbol)
                    
                    # 使用股票监控器生成详细分析
                    analysis_result = self.stock_monitor.get_stock_analysis(symbol)
                    
                    # 合并分析结果
                    full_analysis = {
                        **analysis_result,
                        'strategy_signals': strategy_analysis['signals'],
                        'market_sentiment': strategy_analysis['market_sentiment'],
                        'technical_indicators': strategy_analysis['technical_indicators']
                    }
                    
                    analysis_results.append(full_analysis)
                    
                except Exception as e:
                    logger.error(f"分析股票 {symbol} 时出错: {str(e)}")
                    continue
                    
            # 生成报告并发送通知
            if analysis_results:
                self.notification_manager.send_batch_alerts([{
                    "type": "daily_summary",
                    "message": self._generate_html_report(analysis_results)
                }])
                
        except Exception as e:
            logger.error(f"分析并发送通知时出错: {str(e)}")
            
    def _generate_html_report(self, analysis_results: List[Dict]) -> str:
        """生成HTML格式的报告"""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .stock { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }
                .positive { color: green; }
                .negative { color: red; }
                .neutral { color: gray; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>股票分析报告</h1>
        """
        
        for result in analysis_results:
            html += f"""
            <div class="stock">
                <h2>{result['symbol']}</h2>
                <table>
                    <tr><th>当前价格</th><td>${result['current_price']}</td></tr>
                    <tr><th>价格变化</th><td class="{'positive' if float(result['price_change'].strip('%')) > 0 else 'negative'}">{result['price_change']}</td></tr>
                    <tr><th>交易量变化</th><td>{result['volume_change']}</td></tr>
                    <tr><th>趋势</th><td>{result['trend']}</td></tr>
                    <tr><th>RSI</th><td>{result['rsi']} ({result['rsi_status']})</td></tr>
                    <tr><th>MACD状态</th><td>{result['macd_status']}</td></tr>
                    <tr><th>风险等级</th><td>{result['risk_level']}</td></tr>
                    <tr><th>止损价格</th><td>${result['stop_loss_price']}</td></tr>
                    <tr><th>止盈价格</th><td>${result['take_profit_price']}</td></tr>
                </table>
                <h3>策略建议</h3>
                <ul>
            """
            
            for recommendation in result['recommendations']:
                html += f"<li>{recommendation}</li>"
                
            html += """
                </ul>
                <h3>市场情绪</h3>
                <p>整体情绪: {result['market_sentiment']['overall']}</p>
                <p>波动率: {result['market_sentiment']['market_volatility']}</p>
            </div>
            """
            
        html += """
        </body>
        </html>
        """
        
        return html 