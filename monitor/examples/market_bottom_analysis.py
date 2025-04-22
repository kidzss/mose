import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from monitor.stock_monitor import StockMonitor
from monitor.semiconductor_trading_strategy import SemiconductorTradingStrategy
from monitor.notification_system import NotificationSystem
from data.data_loader import DataLoader
from monitor.data_fetcher import DataFetcher
from monitor.strategy_manager import StrategyManager
from monitor.report_generator import ReportGenerator
from monitor.market_monitor import MarketMonitor
from monitor.stock_manager import StockManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketBottomAnalyzer:
    def __init__(self):
        # 创建必要的组件
        data_fetcher = DataFetcher()
        strategy_manager = StrategyManager()
        report_generator = ReportGenerator()
        market_monitor = MarketMonitor()
        stock_manager = StockManager(data_fetcher=data_fetcher)
        
        # 初始化StockMonitor
        self.stock_monitor = StockMonitor(
            data_fetcher=data_fetcher,
            strategy_manager=strategy_manager,
            report_generator=report_generator,
            market_monitor=market_monitor,
            stock_manager=stock_manager
        )
        
        self.semiconductor_strategy = SemiconductorTradingStrategy()
        self.data_loader = DataLoader()
        self.notification_system = NotificationSystem()
        
    async def analyze_market_bottom(self):
        """分析市场底部信号"""
        try:
            # 获取SPY数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 获取一年的数据
            spy_data = self.data_loader.load_historical_data('SPY', start_date, end_date)
            
            # 计算技术指标
            indicators = self._calculate_technical_indicators(spy_data)
            
            # 分析市场状态
            market_state = self._analyze_market_state(indicators)
            
            # 分析半导体板块
            semiconductor_state = await self._analyze_semiconductor_sector()
            
            # 生成分析报告
            report = self._generate_analysis_report(market_state, indicators, semiconductor_state)
            
            # 发送邮件通知
            await self._send_notification(report)
            
            return report
            
        except Exception as e:
            logger.error(f"市场底部分析失败: {e}")
            return None
            
    def _calculate_technical_indicators(self, data):
        """计算技术指标"""
        indicators = {}
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        indicators['macd'] = exp1 - exp2
        indicators['signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        
        # 移动平均线
        indicators['sma20'] = data['close'].rolling(window=20).mean()
        indicators['sma50'] = data['close'].rolling(window=50).mean()
        indicators['sma200'] = data['close'].rolling(window=200).mean()
        
        # 成交量变化
        indicators['volume_change'] = data['volume'].pct_change()
        
        return indicators
        
    def _analyze_market_state(self, indicators):
        """分析市场状态"""
        state = {
            'is_bottom': False,
            'signals': [],
            'risk_level': 'info'
        }
        
        # 检查RSI超卖
        if indicators['rsi'].iloc[-1] < 30:
            state['signals'].append('RSI超卖')
            state['risk_level'] = 'warning'
            
        # 检查MACD金叉
        if indicators['macd'].iloc[-2] < indicators['signal'].iloc[-2] and \
           indicators['macd'].iloc[-1] > indicators['signal'].iloc[-1]:
            state['signals'].append('MACD金叉')
            state['risk_level'] = 'opportunity'
            
        # 检查价格与移动平均线的关系
        if indicators['sma20'].iloc[-1] > indicators['sma50'].iloc[-1] and \
           indicators['sma50'].iloc[-1] > indicators['sma200'].iloc[-1]:
            state['signals'].append('均线多头排列')
            state['risk_level'] = 'opportunity'
            
        # 检查成交量放大
        if indicators['volume_change'].iloc[-1] > 1.5:
            state['signals'].append('成交量放大')
            state['risk_level'] = 'warning'
            
        # 综合判断市场底部
        if len(state['signals']) >= 2 and 'RSI超卖' in state['signals']:
            state['is_bottom'] = True
            state['risk_level'] = 'opportunity'
            
        return state
        
    async def _analyze_semiconductor_sector(self):
        """分析半导体板块"""
        state = {}
        
        # 分析AMD
        amd_data = await self.stock_monitor.get_stock_analysis('AMD')
        if "error" not in amd_data:
            state['AMD状态'] = self._analyze_stock_state(amd_data)
            
        # 分析NVDA
        nvda_data = await self.stock_monitor.get_stock_analysis('NVDA')
        if "error" not in nvda_data:
            state['NVDA状态'] = self._analyze_stock_state(nvda_data)
            
        # 板块整体判断
        if 'AMD状态' in state and 'NVDA状态' in state:
            if state['AMD状态']['趋势'] == '上升' and state['NVDA状态']['趋势'] == '上升':
                state['板块趋势'] = "强势"
            elif state['AMD状态']['趋势'] == '上升' or state['NVDA状态']['趋势'] == '上升':
                state['板块趋势'] = "分化"
            else:
                state['板块趋势'] = "弱势"
                
        return state
        
    def _analyze_stock_state(self, data):
        """分析个股状态"""
        state = {}
        
        # 趋势判断
        try:
            price_change = float(data.get('price_change', '0%').strip('%')) / 100
        except (ValueError, TypeError, AttributeError):
            price_change = 0
        state['趋势'] = "上升" if price_change > 0 else "下降"
        
        # 强度判断
        try:
            rsi = float(data.get('rsi', 50))
        except (ValueError, TypeError):
            rsi = 50
            
        if rsi > 70:
            state['强度'] = "超买"
        elif rsi < 30:
            state['强度'] = "超卖"
        else:
            state['强度'] = "中性"
            
        # 支撑位判断
        current_price = data.get('current_price')
        sma20 = data.get('sma20')
        
        if current_price is not None and sma20 is not None:
            state['支撑'] = "20日均线支撑" if current_price > sma20 else "跌破20日均线"
        else:
            state['支撑'] = "数据不足"
            
        return state
        
    def _generate_analysis_report(self, market_state, indicators, semiconductor_state):
        """生成分析报告"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_state': market_state,
            'semiconductor_state': semiconductor_state,
            'indicators': {
                'rsi': round(indicators['rsi'].iloc[-1], 2),
                'macd': round(indicators['macd'].iloc[-1], 2),
                'signal': round(indicators['signal'].iloc[-1], 2),
                'sma20': round(indicators['sma20'].iloc[-1], 2),
                'sma50': round(indicators['sma50'].iloc[-1], 2),
                'sma200': round(indicators['sma200'].iloc[-1], 2),
                'volume_change': round(indicators['volume_change'].iloc[-1] * 100, 2)
            }
        }
        
        return report
        
    async def _send_notification(self, report):
        """发送邮件通知"""
        try:
            # 构建HTML格式的邮件内容
            html_content = f"""
            <h2>市场底部分析报告</h2>
            <p>时间: {report['timestamp']}</p>
            <h3>市场状态</h3>
            <ul>
                <li>是否底部: {'是' if report['market_state']['is_bottom'] else '否'}</li>
                <li>风险等级: {report['market_state']['risk_level']}</li>
                <li>信号: {', '.join(report['market_state']['signals'])}</li>
            </ul>
            <h3>技术指标</h3>
            <ul>
                <li>RSI: {report['indicators']['rsi']}</li>
                <li>MACD: {report['indicators']['macd']}</li>
                <li>Signal: {report['indicators']['signal']}</li>
                <li>SMA20: {report['indicators']['sma20']}</li>
                <li>SMA50: {report['indicators']['sma50']}</li>
                <li>SMA200: {report['indicators']['sma200']}</li>
                <li>成交量变化: {report['indicators']['volume_change']}%</li>
            </ul>
            <h3>半导体板块分析</h3>
            <ul>
                <li>板块趋势: {report['semiconductor_state']['板块趋势']}</li>
                <li>AMD状态: {report['semiconductor_state']['AMD状态']['趋势']}</li>
                <li>NVDA状态: {report['semiconductor_state']['NVDA状态']['趋势']}</li>
            </ul>
            """
            
            # 发送邮件
            await self.notification_system.send_email(
                subject="市场底部分析报告",
                body=html_content,
                is_html=True
            )
            
            logger.info("市场底部分析报告已发送")
            
        except Exception as e:
            logger.error(f"发送邮件通知失败: {e}")

async def main():
    analyzer = MarketBottomAnalyzer()
    report = await analyzer.analyze_market_bottom()
    if report:
        logger.info(f"市场底部分析完成: {report}")

if __name__ == "__main__":
    asyncio.run(main()) 