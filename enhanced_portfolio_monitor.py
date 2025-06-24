#!/usr/bin/env python3
"""
增强版投资组合监控器
同时监控当前持仓和观察仓股票
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPortfolioMonitor:
    """增强版投资组合监控器"""
    
    def __init__(self, config_file='config/portfolio_config.json'):
        """初始化监控器"""
        self.config_file = config_file
        self.load_config()
        
        # 实际持仓数据 (基于最新更新)
        self.actual_holdings = {
            'NVDA': {'shares': 75, 'cost_basis': 126.79},
            'GOOG': {'shares': 47, 'cost_basis': 150.16}, 
            'AMD': {'shares': 40, 'cost_basis': 126.21},
            'TSLA': {'shares': 12, 'cost_basis': 227.62},
            'MSFT': {'shares': 6, 'cost_basis': 359.60},
            'PFE': {'shares': 87, 'cost_basis': 22.08},
            'TMDX': {'shares': 23, 'cost_basis': 87.78}
        }
        
        logger.info("📊 增强版投资组合监控器初始化完成")
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            self.positions = self.config.get('positions', {})
            self.watchlist = self.config.get('watchlist', {})
            self.monitor_config = self.config.get('monitor_config', {})
            
            logger.info(f"✅ 配置加载成功: {len(self.positions)}个持仓, {len(self.watchlist)}个观察股")
            
        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            self.positions = {}
            self.watchlist = {}
            self.monitor_config = {}
    
    def get_stock_data(self, symbol, period="1mo"):
        """获取股票数据"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # 技术指标
            ma5 = hist['Close'].rolling(5).mean().iloc[-1]
            ma20 = hist['Close'].rolling(20).mean().iloc[-1]
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 成交量
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # 波动率
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # 52周高低点
            high_52w = hist['High'].rolling(252).max().iloc[-1] if len(hist) >= 252 else hist['High'].max()
            low_52w = hist['Low'].rolling(252).min().iloc[-1] if len(hist) >= 252 else hist['Low'].min()
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'ma5': ma5,
                'ma20': ma20,
                'rsi': current_rsi,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'price_change_1d': (current_price / hist['Close'].iloc[-2] - 1) * 100 if len(hist) > 1 else 0,
                'price_vs_ma20': (current_price / ma20 - 1) * 100 if ma20 > 0 else 0,
                'distance_from_high': (current_price / high_52w - 1) * 100,
                'distance_from_low': (current_price / low_52w - 1) * 100,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            }
            
        except Exception as e:
            logger.warning(f"获取{symbol}数据失败: {e}")
            return None
    
    def analyze_current_portfolio(self):
        """分析当前投资组合"""
        portfolio_data = []
        total_value = 0
        
        for symbol, holding in self.actual_holdings.items():
            stock_data = self.get_stock_data(symbol)
            
            if stock_data:
                shares = holding['shares']
                cost_basis = holding['cost_basis']
                current_price = stock_data['current_price']
                
                market_value = shares * current_price
                cost_value = shares * cost_basis
                unrealized_pnl = market_value - cost_value
                unrealized_pnl_pct = (unrealized_pnl / cost_value) * 100
                
                total_value += market_value
                
                portfolio_data.append({
                    'symbol': symbol,
                    'shares': shares,
                    'cost_basis': cost_basis,
                    'current_price': current_price,
                    'market_value': market_value,
                    'cost_value': cost_value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'weight': 0,  # 稍后计算
                    'rsi': stock_data['rsi'],
                    'price_change_1d': stock_data['price_change_1d'],
                    'price_vs_ma20': stock_data['price_vs_ma20'],
                    'volatility': stock_data['volatility'],
                    'distance_from_high': stock_data['distance_from_high']
                })
        
        # 计算权重
        for item in portfolio_data:
            item['weight'] = (item['market_value'] / total_value) * 100
        
        return portfolio_data, total_value
    
    def analyze_watchlist(self):
        """分析观察仓股票"""
        watchlist_data = []
        
        for symbol, watch_config in self.watchlist.items():
            stock_data = self.get_stock_data(symbol)
            
            if stock_data:
                current_price = stock_data['current_price']
                target_price = watch_config['target_price']
                conservative_price = watch_config['conservative_price']
                
                # 计算距离目标价位的差距
                distance_to_target = (current_price / target_price - 1) * 100
                distance_to_conservative = (current_price / conservative_price - 1) * 100
                
                # 买入信号判断
                buy_signal = "WAIT"
                if current_price <= conservative_price:
                    buy_signal = "STRONG_BUY"
                elif current_price <= target_price:
                    buy_signal = "BUY"
                elif stock_data['rsi'] < 30:
                    buy_signal = "OVERSOLD"
                
                # 技术面评分
                tech_score = 0
                if stock_data['rsi'] < 30:
                    tech_score += 3
                elif stock_data['rsi'] < 40:
                    tech_score += 2
                elif stock_data['rsi'] > 70:
                    tech_score -= 2
                
                if stock_data['price_vs_ma20'] < -5:
                    tech_score += 2
                elif stock_data['price_vs_ma20'] > 5:
                    tech_score -= 1
                
                if stock_data['distance_from_high'] < -15:
                    tech_score += 2
                
                watchlist_data.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'target_price': target_price,
                    'conservative_price': conservative_price,
                    'distance_to_target': distance_to_target,
                    'distance_to_conservative': distance_to_conservative,
                    'buy_signal': buy_signal,
                    'tech_score': tech_score,
                    'priority': watch_config['priority'],
                    'batch': watch_config['batch'],
                    'category': watch_config['category'],
                    'entry_strategy': watch_config['entry_strategy'],
                    'target_investment': watch_config['target_investment'],
                    'rsi': stock_data['rsi'],
                    'price_change_1d': stock_data['price_change_1d'],
                    'price_vs_ma20': stock_data['price_vs_ma20'],
                    'distance_from_high': stock_data['distance_from_high'],
                    'notes': watch_config.get('notes', '')
                })
        
        return watchlist_data
    
    def get_market_context(self):
        """获取市场环境"""
        spy_data = self.get_stock_data('SPY')
        vix_data = self.get_stock_data('^VIX')
        
        market_context = {
            'spy_price': spy_data['current_price'] if spy_data else 0,
            'spy_change': spy_data['price_change_1d'] if spy_data else 0,
            'spy_vs_ma20': spy_data['price_vs_ma20'] if spy_data else 0,
            'spy_rsi': spy_data['rsi'] if spy_data else 50,
            'vix_level': vix_data['current_price'] if vix_data else 20,
            'vix_change': vix_data['price_change_1d'] if vix_data else 0
        }
        
        # 市场情绪判断
        if market_context['spy_rsi'] > 70 and market_context['vix_level'] < 15:
            market_sentiment = "GREEDY"
        elif market_context['spy_rsi'] < 30 or market_context['vix_level'] > 30:
            market_sentiment = "FEARFUL"
        else:
            market_sentiment = "NEUTRAL"
        
        market_context['sentiment'] = market_sentiment
        
        return market_context
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        # 分析数据
        portfolio_data, total_value = self.analyze_current_portfolio()
        watchlist_data = self.analyze_watchlist()
        market_context = self.get_market_context()
        
        # 生成报告
        report = []
        report.append("=" * 120)
        report.append("📊 增强版投资组合监控报告")
        report.append(f"📅 报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 120)
        
        # 市场环境
        report.append(f"\n🌍 市场环境概览:")
        report.append("-" * 100)
        report.append(f"• SPY: ${market_context['spy_price']:.2f} ({market_context['spy_change']:+.2f}%)")
        report.append(f"• SPY vs MA20: {market_context['spy_vs_ma20']:+.1f}%")
        report.append(f"• SPY RSI: {market_context['spy_rsi']:.1f}")
        report.append(f"• VIX: {market_context['vix_level']:.1f} ({market_context['vix_change']:+.2f}%)")
        report.append(f"• 市场情绪: {market_context['sentiment']}")
        
        # 当前投资组合
        report.append(f"\n💼 当前投资组合 (总价值: ${total_value:,.2f}):")
        report.append("-" * 100)
        report.append(f"{'股票':<8} {'股数':<6} {'成本':<8} {'现价':<8} {'市值':<10} {'盈亏':<10} {'盈亏%':<8} {'权重':<6} {'RSI':<5} {'日涨跌':<7}")
        report.append("-" * 90)
        
        for item in sorted(portfolio_data, key=lambda x: x['weight'], reverse=True):
            report.append(f"{item['symbol']:<8} "
                         f"{item['shares']:<6} "
                         f"${item['cost_basis']:<7.2f} "
                         f"${item['current_price']:<7.2f} "
                         f"${item['market_value']:<9.0f} "
                         f"${item['unrealized_pnl']:<9.0f} "
                         f"{item['unrealized_pnl_pct']:<7.1f}% "
                         f"{item['weight']:<5.1f}% "
                         f"{item['rsi']:<4.0f} "
                         f"{item['price_change_1d']:+.2f}%")
        
        # 投资组合统计
        total_pnl = sum(item['unrealized_pnl'] for item in portfolio_data)
        total_cost = sum(item['cost_value'] for item in portfolio_data)
        total_pnl_pct = (total_pnl / total_cost) * 100
        
        report.append(f"\n📈 投资组合统计:")
        report.append("-" * 100)
        report.append(f"• 总成本: ${total_cost:,.2f}")
        report.append(f"• 总市值: ${total_value:,.2f}")
        report.append(f"• 总盈亏: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        
        # 观察仓分析
        report.append(f"\n👀 观察仓股票分析:")
        report.append("=" * 120)
        
        # 按批次分组
        batches = {}
        for item in watchlist_data:
            batch = item['batch']
            if batch not in batches:
                batches[batch] = []
            batches[batch].append(item)
        
        batch_names = {1: "第1批 - 防御性资产", 2: "第2批 - 科技股", 3: "第3批 - 消费+ETF"}
        
        for batch_num in sorted(batches.keys()):
            batch_items = batches[batch_num]
            report.append(f"\n🎯 {batch_names.get(batch_num, f'第{batch_num}批')}:")
            report.append("-" * 100)
            report.append(f"{'股票':<8} {'现价':<10} {'目标价':<10} {'保守价':<10} {'距目标':<8} {'信号':<12} {'RSI':<5} {'日涨跌':<7} {'优先级':<8}")
            report.append("-" * 85)
            
            for item in sorted(batch_items, key=lambda x: (x['priority'] == 'HIGH', x['tech_score']), reverse=True):
                signal_color = ""
                if item['buy_signal'] == 'STRONG_BUY':
                    signal_color = "🟢"
                elif item['buy_signal'] == 'BUY':
                    signal_color = "🟡"
                elif item['buy_signal'] == 'OVERSOLD':
                    signal_color = "🔵"
                else:
                    signal_color = "🔴"
                
                report.append(f"{item['symbol']:<8} "
                             f"${item['current_price']:<9.2f} "
                             f"${item['target_price']:<9.2f} "
                             f"${item['conservative_price']:<9.2f} "
                             f"{item['distance_to_target']:+.1f}% "
                             f"{signal_color}{item['buy_signal']:<11} "
                             f"{item['rsi']:<4.0f} "
                             f"{item['price_change_1d']:+.2f}% "
                             f"{item['priority']:<8}")
                
                if item['notes']:
                    report.append(f"         备注: {item['notes']}")
        
        # 买入建议
        report.append(f"\n💡 今日买入建议:")
        report.append("-" * 100)
        
        immediate_buys = [item for item in watchlist_data if item['buy_signal'] in ['STRONG_BUY', 'BUY']]
        oversold_buys = [item for item in watchlist_data if item['buy_signal'] == 'OVERSOLD']
        
        if immediate_buys:
            report.append(f"🟢 立即买入推荐:")
            for item in sorted(immediate_buys, key=lambda x: x['tech_score'], reverse=True):
                report.append(f"  • {item['symbol']}: ${item['current_price']:.2f} "
                             f"(距目标价{item['distance_to_target']:+.1f}%, RSI:{item['rsi']:.0f})")
        
        if oversold_buys:
            report.append(f"🔵 超卖机会:")
            for item in oversold_buys:
                report.append(f"  • {item['symbol']}: ${item['current_price']:.2f} "
                             f"(RSI:{item['rsi']:.0f}, 技术性超卖)")
        
        wait_stocks = [item for item in watchlist_data if item['buy_signal'] == 'WAIT']
        if wait_stocks:
            report.append(f"🔴 继续等待:")
            for item in wait_stocks:
                report.append(f"  • {item['symbol']}: ${item['current_price']:.2f} "
                             f"(距目标价{item['distance_to_target']:+.1f}%, 等待回调)")
        
        # 风险提醒
        report.append(f"\n⚠️ 风险提醒:")
        report.append("-" * 100)
        
        high_rsi_holdings = [item for item in portfolio_data if item['rsi'] > 70]
        if high_rsi_holdings:
            report.append(f"• 持仓中RSI超买股票: {', '.join([item['symbol'] for item in high_rsi_holdings])}")
        
        concentrated_holdings = [item for item in portfolio_data if item['weight'] > 25]
        if concentrated_holdings:
            concentrated_str = ', '.join([f"{item['symbol']}({item['weight']:.1f}%)" for item in concentrated_holdings])
            report.append(f"• 权重过高股票: {concentrated_str}")
        
        if market_context['sentiment'] == 'GREEDY':
            report.append(f"• 市场情绪贪婪，建议谨慎追涨")
        elif market_context['sentiment'] == 'FEARFUL':
            report.append(f"• 市场情绪恐慌，可考虑逢低买入")
        
        # 下周计划
        report.append(f"\n📅 下周执行计划:")
        report.append("-" * 100)
        
        batch1_ready = [item for item in watchlist_data if item['batch'] == 1 and item['buy_signal'] in ['STRONG_BUY', 'BUY']]
        if batch1_ready:
            report.append(f"• 第1批防御性资产可买入: {', '.join([item['symbol'] for item in batch1_ready])}")
        
        batch2_wait = [item for item in watchlist_data if item['batch'] == 2]
        if batch2_wait:
            report.append(f"• 第2批科技股继续等待回调: {', '.join([item['symbol'] for item in batch2_wait])}")
        
        report.append(f"• 重点关注BRK-B是否跌至$475以下")
        report.append(f"• 如SPY跌破$580，考虑加快买入节奏")
        
        report.append("\n" + "=" * 120)
        report.append("📋 声明: 本报告仅供参考，投资需谨慎，请结合实际情况做决策")
        report.append("=" * 120)
        
        return '\n'.join(report)

def main():
    """主函数"""
    monitor = EnhancedPortfolioMonitor()
    
    # 生成综合报告
    report = monitor.generate_comprehensive_report()
    print(report)
    
    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'enhanced_portfolio_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✅ 增强版投资组合监控报告生成完成")

if __name__ == "__main__":
    main() 