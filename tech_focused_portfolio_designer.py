#!/usr/bin/env python3
"""
科技偏重投资组合设计器
基于独立分析设计能超过20%收益的科技重仓组合，并给出具体买卖点
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

class TechFocusedPortfolioDesigner:
    """科技偏重投资组合设计器"""
    
    def __init__(self):
        """初始化设计器"""
        # 科技股票池 - 经过筛选的优质科技股
        self.tech_universe = {
            # 超大盘科技龙头
            'NVDA': {'category': 'AI/芯片', 'risk_level': 'HIGH', 'expected_growth': 'VERY_HIGH'},
            'MSFT': {'category': '云计算/AI', 'risk_level': 'MEDIUM', 'expected_growth': 'HIGH'},
            'GOOGL': {'category': '搜索/AI', 'risk_level': 'MEDIUM', 'expected_growth': 'HIGH'},
            'AAPL': {'category': '消费电子', 'risk_level': 'LOW', 'expected_growth': 'MEDIUM'},
            'META': {'category': '社交/VR', 'risk_level': 'MEDIUM', 'expected_growth': 'HIGH'},
            'AMZN': {'category': '电商/云', 'risk_level': 'MEDIUM', 'expected_growth': 'HIGH'},
            'TSLA': {'category': '电动车/AI', 'risk_level': 'VERY_HIGH', 'expected_growth': 'HIGH'},
            
            # 中盘成长科技股
            'AMD': {'category': 'AI/芯片', 'risk_level': 'HIGH', 'expected_growth': 'HIGH'},
            'CRM': {'category': '企业软件', 'risk_level': 'MEDIUM', 'expected_growth': 'HIGH'},
            'ADBE': {'category': '创意软件', 'risk_level': 'MEDIUM', 'expected_growth': 'MEDIUM'},
            'NOW': {'category': '企业软件', 'risk_level': 'MEDIUM', 'expected_growth': 'HIGH'},
            'PLTR': {'category': '数据分析', 'risk_level': 'HIGH', 'expected_growth': 'HIGH'},
            
            # 防御性科技股
            'ORCL': {'category': '数据库', 'risk_level': 'LOW', 'expected_growth': 'MEDIUM'},
            'CSCO': {'category': '网络设备', 'risk_level': 'LOW', 'expected_growth': 'LOW'},
            'IBM': {'category': '云/AI', 'risk_level': 'LOW', 'expected_growth': 'LOW'},
            
            # 科技ETF
            'QQQ': {'category': '科技ETF', 'risk_level': 'MEDIUM', 'expected_growth': 'MEDIUM'},
            'XLK': {'category': '科技板块ETF', 'risk_level': 'MEDIUM', 'expected_growth': 'MEDIUM'}
        }
        
        # 非科技防御性资产 (用于平衡)
        self.defensive_universe = {
            'BRK-B': {'category': '价值投资', 'risk_level': 'LOW', 'expected_growth': 'MEDIUM'},
            'JPM': {'category': '金融', 'risk_level': 'MEDIUM', 'expected_growth': 'MEDIUM'},
            'JNJ': {'category': '医疗', 'risk_level': 'LOW', 'expected_growth': 'LOW'},
            'PG': {'category': '消费品', 'risk_level': 'LOW', 'expected_growth': 'LOW'},
            'SPY': {'category': '市场ETF', 'risk_level': 'LOW', 'expected_growth': 'MEDIUM'}
        }
        
        logger.info("🚀 科技偏重投资组合设计器初始化完成")
    
    def analyze_stock_performance(self, symbol, period="5y"):
        """分析个股历史表现和技术指标"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            
            if hist.empty:
                return None
            
            # 基础统计
            total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            years = len(hist) / 252
            annual_return = (1 + total_return) ** (1/years) - 1
            
            daily_returns = hist['Close'].pct_change().dropna()
            annual_volatility = daily_returns.std() * np.sqrt(252)
            
            # 最大回撤
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 技术指标
            current_price = hist['Close'].iloc[-1]
            ma20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma50 = hist['Close'].rolling(50).mean().iloc[-1]
            ma200 = hist['Close'].rolling(200).mean().iloc[-1]
            
            # RSI计算
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 支撑阻力位
            recent_high = hist['High'].rolling(252).max().iloc[-1]  # 52周高点
            recent_low = hist['Low'].rolling(252).min().iloc[-1]    # 52周低点
            
            # 基本面数据
            pe_ratio = info.get('trailingPE', None)
            peg_ratio = info.get('pegRatio', None)
            market_cap = info.get('marketCap', 0)
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': (annual_return - 0.04) / annual_volatility,
                'current_price': current_price,
                'ma20': ma20,
                'ma50': ma50,
                'ma200': ma200,
                'rsi': current_rsi,
                'high_52w': recent_high,
                'low_52w': recent_low,
                'pe_ratio': pe_ratio,
                'peg_ratio': peg_ratio,
                'market_cap': market_cap,
                'price_vs_ma20': (current_price / ma20 - 1) * 100,
                'price_vs_ma50': (current_price / ma50 - 1) * 100,
                'price_vs_ma200': (current_price / ma200 - 1) * 100,
                'distance_from_high': (current_price / recent_high - 1) * 100,
                'distance_from_low': (current_price / recent_low - 1) * 100
            }
            
        except Exception as e:
            logger.warning(f"分析{symbol}时出错: {e}")
            return None
    
    def calculate_optimal_weights(self, stock_data, tech_weight_target=0.65):
        """基于风险收益优化计算最佳权重"""
        
        # 按夏普比率和预期收益排序
        scored_stocks = []
        for symbol, data in stock_data.items():
            if data is None:
                continue
                
            # 综合评分
            return_score = min(data['annual_return'] / 0.30, 1.0)  # 归一化到30%
            sharpe_score = min(data['sharpe_ratio'] / 2.0, 1.0)   # 归一化到2.0
            
            # 技术面评分
            tech_score = 0
            if data['rsi'] < 70:  # 非超买
                tech_score += 0.3
            if data['price_vs_ma20'] > -10:  # 距20日均线不太远
                tech_score += 0.3
            if data['distance_from_high'] > -30:  # 距高点不太远
                tech_score += 0.4
            
            # 估值评分
            value_score = 0.5  # 默认中性
            if data['pe_ratio'] and data['pe_ratio'] < 30:
                value_score = 0.7
            elif data['pe_ratio'] and data['pe_ratio'] < 50:
                value_score = 0.6
            
            combined_score = (return_score * 0.4 + sharpe_score * 0.3 + 
                            tech_score * 0.2 + value_score * 0.1)
            
            scored_stocks.append({
                'symbol': symbol,
                'score': combined_score,
                'annual_return': data['annual_return'],
                'volatility': data['annual_volatility'],
                'sharpe_ratio': data['sharpe_ratio'],
                'is_tech': symbol in self.tech_universe
            })
        
        # 排序并选择
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        # 构建组合
        portfolio = []
        tech_weight = 0
        total_weight = 0
        
        # 优先选择科技股到目标权重
        for stock in scored_stocks:
            if stock['is_tech'] and tech_weight < tech_weight_target:
                weight = min(0.15, tech_weight_target - tech_weight)  # 单只股票最大15%
                portfolio.append({
                    'symbol': stock['symbol'],
                    'weight': weight,
                    'category': 'tech'
                })
                tech_weight += weight
                total_weight += weight
        
        # 添加防御性资产
        for stock in scored_stocks:
            if not stock['is_tech'] and total_weight < 0.95:
                weight = min(0.10, 0.95 - total_weight)  # 防御性资产权重较小
                portfolio.append({
                    'symbol': stock['symbol'],
                    'weight': weight,
                    'category': 'defensive'
                })
                total_weight += weight
                if total_weight >= 0.95:
                    break
        
        return portfolio
    
    def generate_trading_signals(self, symbol, data):
        """生成具体的买卖信号"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'current_action': 'HOLD',
            'strength': 'MEDIUM'
        }
        
        price = data['current_price']
        rsi = data['rsi']
        
        # 买入信号
        buy_score = 0
        
        # RSI超卖
        if rsi < 30:
            signals['buy_signals'].append(f"RSI超卖({rsi:.1f})")
            buy_score += 2
        elif rsi < 40:
            signals['buy_signals'].append(f"RSI偏低({rsi:.1f})")
            buy_score += 1
            
        # 价格相对均线
        if data['price_vs_ma20'] < -5:
            signals['buy_signals'].append(f"跌破20日均线{data['price_vs_ma20']:.1f}%")
            buy_score += 1
        if data['price_vs_ma50'] < -10:
            signals['buy_signals'].append(f"跌破50日均线{data['price_vs_ma50']:.1f}%")
            buy_score += 1
            
        # 距离高点回调
        if data['distance_from_high'] < -20:
            signals['buy_signals'].append(f"距高点回调{-data['distance_from_high']:.1f}%")
            buy_score += 1
            
        # 卖出信号
        sell_score = 0
        
        # RSI超买
        if rsi > 80:
            signals['sell_signals'].append(f"RSI超买({rsi:.1f})")
            sell_score += 2
        elif rsi > 70:
            signals['sell_signals'].append(f"RSI偏高({rsi:.1f})")
            sell_score += 1
            
        # 接近高点
        if data['distance_from_high'] > -5:
            signals['sell_signals'].append(f"接近52周高点({data['distance_from_high']:.1f}%)")
            sell_score += 1
            
        # 判断主要动作
        if buy_score >= 3:
            signals['current_action'] = 'STRONG_BUY'
            signals['strength'] = 'HIGH'
        elif buy_score >= 2:
            signals['current_action'] = 'BUY'
            signals['strength'] = 'MEDIUM'
        elif sell_score >= 3:
            signals['current_action'] = 'STRONG_SELL'
            signals['strength'] = 'HIGH'
        elif sell_score >= 2:
            signals['current_action'] = 'SELL'
            signals['strength'] = 'MEDIUM'
        
        return signals
    
    def calculate_entry_exit_points(self, symbol, data):
        """计算具体的进出场点位"""
        current_price = data['current_price']
        
        # 支撑位计算
        support_levels = []
        support_levels.append(data['ma20'] * 0.98)  # 20日均线下方2%
        support_levels.append(data['ma50'] * 0.97)  # 50日均线下方3%
        support_levels.append(data['low_52w'] * 1.05)  # 52周低点上方5%
        
        # 阻力位计算
        resistance_levels = []
        resistance_levels.append(data['ma20'] * 1.02)  # 20日均线上方2%
        resistance_levels.append(data['ma50'] * 1.03)  # 50日均线上方3%
        resistance_levels.append(data['high_52w'] * 0.98)  # 52周高点下方2%
        
        # 最佳买入价格区间
        buy_zone_low = min(support_levels)
        buy_zone_high = current_price * 0.95  # 当前价格下方5%
        
        # 最佳卖出价格区间
        sell_zone_low = current_price * 1.20  # 20%利润目标
        sell_zone_high = max(resistance_levels)
        
        # 止损价格
        stop_loss = current_price * 0.85  # 15%止损
        
        return {
            'current_price': current_price,
            'buy_zone': (buy_zone_low, buy_zone_high),
            'sell_zone': (sell_zone_low, sell_zone_high),
            'stop_loss': stop_loss,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'target_profit': 0.20  # 20%利润目标
        }
    
    def design_portfolio(self, target_amount=30000, tech_weight=0.65):
        """设计完整的投资组合"""
        logger.info("🔍 开始分析股票池...")
        
        # 分析所有股票
        all_stocks = {**self.tech_universe, **self.defensive_universe}
        stock_data = {}
        
        for symbol in all_stocks.keys():
            logger.info(f"分析 {symbol}...")
            data = self.analyze_stock_performance(symbol)
            if data:
                stock_data[symbol] = data
        
        # 计算最佳权重
        portfolio_weights = self.calculate_optimal_weights(stock_data, tech_weight)
        
        # 生成投资组合详情
        portfolio_details = []
        total_expected_return = 0
        total_risk = 0
        
        for position in portfolio_weights:
            symbol = position['symbol']
            weight = position['weight']
            
            if symbol in stock_data:
                data = stock_data[symbol]
                signals = self.generate_trading_signals(symbol, data)
                entry_exit = self.calculate_entry_exit_points(symbol, data)
                
                investment_amount = target_amount * weight
                shares = int(investment_amount / data['current_price'])
                actual_investment = shares * data['current_price']
                
                portfolio_details.append({
                    'symbol': symbol,
                    'category': self.tech_universe.get(symbol, self.defensive_universe.get(symbol, {})).get('category', 'Unknown'),
                    'weight': weight,
                    'investment_amount': actual_investment,
                    'shares': shares,
                    'current_price': data['current_price'],
                    'annual_return': data['annual_return'],
                    'volatility': data['annual_volatility'],
                    'sharpe_ratio': data['sharpe_ratio'],
                    'pe_ratio': data['pe_ratio'],
                    'signals': signals,
                    'entry_exit': entry_exit
                })
                
                total_expected_return += data['annual_return'] * weight
                total_risk += (data['annual_volatility'] ** 2) * (weight ** 2)
        
        portfolio_summary = {
            'target_amount': target_amount,
            'tech_weight_target': tech_weight,
            'expected_annual_return': total_expected_return,
            'estimated_volatility': np.sqrt(total_risk),
            'expected_sharpe_ratio': (total_expected_return - 0.04) / np.sqrt(total_risk),
            'positions': portfolio_details
        }
        
        return portfolio_summary
    
    def generate_detailed_report(self, portfolio):
        """生成详细的投资报告"""
        report = []
        report.append("=" * 120)
        report.append("🚀 科技偏重投资组合设计报告")
        report.append("💰 目标：超过20%年化收益的科技重仓组合")
        report.append(f"📅 设计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 120)
        
        # 组合概览
        report.append(f"\n📊 投资组合概览:")
        report.append("-" * 100)
        report.append(f"• 投资总额: ${portfolio['target_amount']:,.0f}")
        report.append(f"• 科技股权重目标: {portfolio['tech_weight_target']:.1%}")
        report.append(f"• 预期年化收益率: {portfolio['expected_annual_return']:.1%}")
        report.append(f"• 预期波动率: {portfolio['estimated_volatility']:.1%}")
        report.append(f"• 预期夏普比率: {portfolio['expected_sharpe_ratio']:.2f}")
        
        # 判断是否达到目标
        if portfolio['expected_annual_return'] >= 0.20:
            report.append(f"✅ 预期收益达到20%+目标！")
        else:
            report.append(f"⚠️  预期收益{portfolio['expected_annual_return']:.1%}，未达到20%目标")
        
        # 持仓详情
        report.append(f"\n📋 持仓配置详情:")
        report.append("-" * 100)
        report.append(f"{'股票':<6} {'类别':<12} {'权重':<8} {'金额':<12} {'股数':<6} {'当前价':<10} {'年化收益':<10} {'PE':<8} {'信号':<12}")
        report.append("-" * 100)
        
        tech_weight_actual = 0
        for pos in portfolio['positions']:
            if pos['symbol'] in self.tech_universe:
                tech_weight_actual += pos['weight']
                
            pe_str = f"{pos['pe_ratio']:.1f}" if pos['pe_ratio'] else "N/A"
            
            report.append(f"{pos['symbol']:<6} {pos['category']:<12} {pos['weight']:<7.1%} "
                         f"${pos['investment_amount']:<11,.0f} {pos['shares']:<6} "
                         f"${pos['current_price']:<9.2f} {pos['annual_return']:<9.1%} "
                         f"{pe_str:<8} {pos['signals']['current_action']:<12}")
        
        report.append(f"\n实际科技股权重: {tech_weight_actual:.1%}")
        
        # 买卖点位详情
        report.append(f"\n💡 具体买卖点位指导:")
        report.append("=" * 120)
        
        for pos in portfolio['positions']:
            symbol = pos['symbol']
            entry_exit = pos['entry_exit']
            signals = pos['signals']
            
            report.append(f"\n🎯 {symbol} - {pos['category']}")
            report.append("-" * 80)
            report.append(f"• 当前价格: ${entry_exit['current_price']:.2f}")
            report.append(f"• 建议买入区间: ${entry_exit['buy_zone'][0]:.2f} - ${entry_exit['buy_zone'][1]:.2f}")
            report.append(f"• 目标卖出区间: ${entry_exit['sell_zone'][0]:.2f} - ${entry_exit['sell_zone'][1]:.2f}")
            report.append(f"• 止损价格: ${entry_exit['stop_loss']:.2f}")
            report.append(f"• 当前信号: {signals['current_action']} ({signals['strength']})")
            
            if signals['buy_signals']:
                report.append(f"• 买入理由: {', '.join(signals['buy_signals'])}")
            if signals['sell_signals']:
                report.append(f"• 卖出警示: {', '.join(signals['sell_signals'])}")
            
            # 支撑阻力位
            report.append(f"• 支撑位: {[f'${p:.2f}' for p in entry_exit['support_levels']]}")
            report.append(f"• 阻力位: {[f'${p:.2f}' for p in entry_exit['resistance_levels']]}")
        
        # 执行策略
        report.append(f"\n📈 执行策略建议:")
        report.append("-" * 100)
        report.append(f"1. 分批建仓策略:")
        report.append(f"   • 第1批(40%): 当前价格区间立即买入防御性强的股票")
        report.append(f"   • 第2批(35%): 等待技术回调到买入区间下限")
        report.append(f"   • 第3批(25%): 市场恐慌时抄底机会")
        
        report.append(f"\n2. 风险控制:")
        report.append(f"   • 单只股票止损线: -15%")
        report.append(f"   • 组合整体止损线: -20%")
        report.append(f"   • 分批获利了结: 达到+20%开始减仓")
        
        report.append(f"\n3. 定期调仓:")
        report.append(f"   • 每季度重新评估权重")
        report.append(f"   • 根据基本面变化调整持仓")
        report.append(f"   • 新兴科技趋势的机会捕捉")
        
        # 风险提示
        report.append(f"\n⚠️ 风险提示:")
        report.append("-" * 100)
        report.append(f"• 科技股集中度高，需承受较大波动")
        report.append(f"• 宏观利率变化对科技股影响较大")
        report.append(f"• 个股基本面变化需要密切跟踪")
        report.append(f"• 市场情绪变化可能带来系统性风险")
        
        report.append("\n" + "=" * 120)
        report.append("📋 免责声明: 本报告基于历史数据分析，投资有风险，请根据个人情况独立决策")
        report.append("=" * 120)
        
        return '\n'.join(report)

def main():
    """主函数"""
    designer = TechFocusedPortfolioDesigner()
    
    # 设计投资组合
    portfolio = designer.design_portfolio(target_amount=30000, tech_weight=0.65)
    
    # 生成报告
    report = designer.generate_detailed_report(portfolio)
    print(report)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'tech_focused_portfolio_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2, default=str)
    
    with open(f'tech_portfolio_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✅ 科技偏重投资组合设计完成")

if __name__ == "__main__":
    main()