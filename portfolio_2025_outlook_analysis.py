import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Portfolio2025OutlookAnalyzer:
    """2025年投资组合前瞻分析器"""
    
    def __init__(self):
        # 当前组合配置
        self.current_portfolio = {
            # AI/科技成长股 50%
            'NVDA': 0.08,
            'GOOG': 0.12,
            'AMD': 0.05,
            'META': 0.12,
            'AMZN': 0.08,
            'PLTR': 0.05,
            
            # 价值成长 25%
            'JPM': 0.08,
            'BRK-B': 0.08,
            'ORCL': 0.05,
            'IBM': 0.04,
            
            # 防御股 25%
            'MRK': 0.08,
            'JNJ': 0.07,
            'VZ': 0.05,
            'CVX': 0.05
        }
        
        # 2025年增长主题分析
        self.growth_themes_2025 = {
            'ai_stocks': {
                'symbols': ['NVDA', 'AMD', 'PLTR', 'META'],
                'expected_growth_decline': 0.35,  # 用户预期AI增长减少35%
                'reasoning': 'AI泡沫挤压，增长回归理性'
            },
            'cloud_and_enterprise': {
                'symbols': ['GOOG', 'AMZN', 'ORCL', 'IBM'],
                'expected_multiplier': 1.1,  # 企业数字化持续
                'reasoning': '企业云迁移和数字化转型持续'
            },
            'financial_services': {
                'symbols': ['JPM', 'BRK-B'],
                'expected_multiplier': 1.2,  # 利率环境稳定受益
                'reasoning': '高利率环境稳定，银行净息差改善'
            },
            'defense_energy': {
                'symbols': ['CVX', 'MRK', 'JNJ', 'VZ'],
                'expected_multiplier': 0.9,  # 防御股表现平平
                'reasoning': '经济增长环境下防御需求降低'
            }
        }
        
        # 2025年新增长点候选
        self.new_growth_candidates = {
            # 生物技术/医疗创新
            'biotech': {
                'symbols': ['ABBV', 'GILD', 'BIIB'],
                'expected_return': 0.25,
                'reasoning': 'GLP-1药物、癌症免疫疗法突破'
            },
            # 清洁能源/储能
            'clean_energy': {
                'symbols': ['TSLA', 'ENPH', 'FSLR'],
                'expected_return': 0.30,
                'reasoning': '政策支持+技术突破+成本下降'
            },
            # 印度/新兴市场
            'emerging_markets': {
                'symbols': ['INDA', 'EEM', 'ASML'],
                'expected_return': 0.20,
                'reasoning': '制造业转移+人口红利+基建投资'
            },
            # 网络安全
            'cybersecurity': {
                'symbols': ['CRWD', 'ZS', 'PANW'],
                'expected_return': 0.28,
                'reasoning': 'AI威胁增加+企业安全需求激增'
            },
            # 消费升级/奢侈品
            'luxury_consumer': {
                'symbols': ['LVMH', 'NKE', 'SBUX'],
                'expected_return': 0.18,
                'reasoning': '中产阶级扩大+体验消费升级'
            }
        }
        
    def get_current_market_data(self):
        """获取当前市场数据"""
        print("📊 获取当前市场数据...")
        
        all_symbols = list(self.current_portfolio.keys())
        for theme in self.new_growth_candidates.values():
            all_symbols.extend(theme['symbols'])
        all_symbols = list(set(all_symbols))  # 去重
        
        market_data = {}
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='1y')
                
                if len(hist) > 50:
                    current_price = hist['Close'].iloc[-1]
                    ytd_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                    
                    market_data[symbol] = {
                        'current_price': current_price,
                        'ytd_return': ytd_return,
                        'volatility': volatility,
                        'pe_ratio': info.get('trailingPE', 'N/A'),
                        'market_cap': info.get('marketCap', 'N/A')
                    }
                    print(f"✓ {symbol}: ${current_price:.2f}, YTD: {ytd_return:.1%}")
                else:
                    print(f"⚠ {symbol}: 数据不足")
            except Exception as e:
                print(f"✗ {symbol}: {e}")
                
        return market_data
    
    def analyze_ai_slowdown_impact(self, market_data):
        """分析AI增长放缓的影响"""
        print("\n🤖 AI增长放缓影响分析:")
        print("-" * 60)
        
        ai_stocks = self.growth_themes_2025['ai_stocks']['symbols']
        current_ai_weight = sum(self.current_portfolio.get(symbol, 0) for symbol in ai_stocks)
        
        print(f"当前AI相关权重: {current_ai_weight:.1%}")
        
        # 计算AI股票2024年表现
        ai_performance = {}
        for symbol in ai_stocks:
            if symbol in market_data:
                ytd = market_data[symbol]['ytd_return']
                ai_performance[symbol] = ytd
                print(f"  {symbol}: 2024年YTD {ytd:+.1%}")
        
        # 估算AI减速35%的影响
        expected_ai_returns_2025 = {}
        total_impact = 0
        
        print(f"\n2025年AI股票预期调整:")
        for symbol in ai_stocks:
            if symbol in ai_performance:
                # 假设2024年收益的65%作为2025年基准
                base_2025 = ai_performance[symbol] * 0.65  # 减少35%
                weight = self.current_portfolio.get(symbol, 0)
                impact = base_2025 * weight
                total_impact += impact
                
                expected_ai_returns_2025[symbol] = base_2025
                print(f"  {symbol}: {base_2025:+.1%} (权重{weight:.1%}, 贡献{impact:+.1%})")
        
        print(f"\nAI减速对组合整体影响: {total_impact:+.1%}")
        return expected_ai_returns_2025, total_impact
    
    def identify_2025_growth_opportunities(self, market_data):
        """识别2025年增长机会"""
        print("\n🚀 2025年增长机会分析:")
        print("-" * 60)
        
        growth_analysis = {}
        
        for theme_name, theme_data in self.new_growth_candidates.items():
            print(f"\n【{theme_name.upper()}】- {theme_data['reasoning']}")
            
            theme_symbols = theme_data['symbols']
            expected_return = theme_data['expected_return']
            
            theme_analysis = {
                'expected_return': expected_return,
                'symbols': [],
                'avg_pe': [],
                'avg_volatility': []
            }
            
            for symbol in theme_symbols:
                if symbol in market_data:
                    data = market_data[symbol]
                    ytd = data['ytd_return']
                    pe = data['pe_ratio']
                    vol = data['volatility']
                    
                    theme_analysis['symbols'].append({
                        'symbol': symbol,
                        'ytd_return': ytd,
                        'pe_ratio': pe,
                        'volatility': vol
                    })
                    
                    if isinstance(pe, (int, float)) and pe > 0:
                        theme_analysis['avg_pe'].append(pe)
                    theme_analysis['avg_volatility'].append(vol)
                    
                    print(f"  {symbol}: YTD {ytd:+.1%}, PE {pe}, 波动率 {vol:.1%}")
            
            # 计算主题平均指标
            if theme_analysis['avg_pe']:
                avg_pe = np.mean(theme_analysis['avg_pe'])
                print(f"  平均PE: {avg_pe:.1f}")
            
            if theme_analysis['avg_volatility']:
                avg_vol = np.mean(theme_analysis['avg_volatility'])
                print(f"  平均波动率: {avg_vol:.1%}")
            
            print(f"  预期2025年收益: {expected_return:.1%}")
            
            growth_analysis[theme_name] = theme_analysis
            
        return growth_analysis
    
    def design_optimized_2025_portfolio(self, ai_impact, growth_opportunities):
        """设计2025年优化组合"""
        print("\n🎯 2025年优化组合设计:")
        print("-" * 60)
        
        # 基于分析调整权重
        optimized_portfolio = self.current_portfolio.copy()
        
        print("调整策略:")
        print("1. 降低AI权重，增加传统科技")
        print("2. 增加生物技术和清洁能源敞口")
        print("3. 适度增加网络安全主题")
        print("4. 保持金融股配置")
        
        # 调整建议
        adjustments = {
            # 减少AI权重
            'NVDA': -0.03,  # 8% -> 5%
            'PLTR': -0.02,  # 5% -> 3%
            'AMD': -0.02,   # 5% -> 3%
            
            # 增加生物技术
            'ABBV': +0.04,  # 新增4%
            'GILD': +0.03,  # 新增3%
            
            # 增加清洁能源
            'TSLA': +0.03,  # 新增3% (考虑FSD进展)
            
            # 增加网络安全
            'CRWD': +0.03,  # 新增3%
            
            # 调整其他
            'GOOG': +0.01,  # 12% -> 13% (搜索+云稳定)
            'JPM': +0.01,   # 8% -> 9% (银行受益高利率)
        }
        
        print(f"\n权重调整建议:")
        for symbol, change in adjustments.items():
            current = optimized_portfolio.get(symbol, 0)
            new_weight = current + change
            if new_weight > 0:
                optimized_portfolio[symbol] = new_weight
                print(f"  {symbol}: {current:.1%} -> {new_weight:.1%} ({change:+.1%})")
            elif current > 0:
                print(f"  {symbol}: {current:.1%} (保持不变)")
        
        # 验证权重总和
        total_weight = sum(optimized_portfolio.values())
        print(f"\n总权重: {total_weight:.1%}")
        
        return optimized_portfolio
    
    def calculate_2025_expected_returns(self, optimized_portfolio, market_data):
        """计算2025年预期收益"""
        print("\n📊 2025年预期收益计算:")
        print("-" * 60)
        
        # 各类股票2025年预期收益率假设
        expected_returns_2025 = {
            # AI股票 (减速35%)
            'NVDA': 0.25,   # 从40%+ 降至25%
            'AMD': 0.20,    # 从30%+ 降至20%
            'PLTR': 0.30,   # 仍有数据增长空间
            'META': 0.22,   # VR/AR长期布局
            
            # 传统科技 (稳定增长)
            'GOOG': 0.18,   # 搜索稳定+AI应用
            'AMZN': 0.20,   # 云计算+物流优化
            'ORCL': 0.25,   # 数据库+云转型
            'IBM': 0.15,    # AI转型缓慢
            
            # 金融 (受益高利率)
            'JPM': 0.20,    # 净息差稳定
            'BRK-B': 0.12,  # 稳健价值投资
            
            # 防御股 (平稳)
            'MRK': 0.08,    # 制药增长有限
            'JNJ': 0.06,    # 面临法律风险
            'VZ': 0.05,     # 5G投资收益递减
            'CVX': 0.10,    # 油价稳定假设
            
            # 新增长点
            'ABBV': 0.25,   # GLP-1等新药
            'GILD': 0.22,   # HIV+肿瘤治疗
            'TSLA': 0.35,   # FSD突破+储能
            'CRWD': 0.30,   # 网络安全需求激增
        }
        
        # 计算加权收益
        portfolio_expected_return = 0
        category_returns = {}
        
        print("个股预期收益贡献:")
        for symbol, weight in optimized_portfolio.items():
            if symbol in expected_returns_2025:
                expected_return = expected_returns_2025[symbol]
                contribution = expected_return * weight
                portfolio_expected_return += contribution
                
                print(f"  {symbol:6s}: {expected_return:5.1%} × {weight:5.1%} = {contribution:+5.2%}")
        
        print(f"\n组合预期年化收益率: {portfolio_expected_return:.1%}")
        
        # 情景分析
        scenarios = {
            'bull_case': {
                'multiplier': 1.3,
                'description': 'AI应用爆发+新主题超预期'
            },
            'base_case': {
                'multiplier': 1.0,
                'description': '基准情况'
            },
            'bear_case': {
                'multiplier': 0.7,
                'description': 'AI泡沫破裂+经济放缓'
            }
        }
        
        print(f"\n情景分析:")
        for scenario, params in scenarios.items():
            scenario_return = portfolio_expected_return * params['multiplier']
            print(f"  {params['description']:20s}: {scenario_return:5.1%}")
        
        return portfolio_expected_return, scenarios
    
    def comprehensive_2025_analysis(self):
        """2025年综合分析"""
        print("🔍 2025年投资组合前瞻分析")
        print("=" * 80)
        
        # 1. 获取市场数据
        market_data = self.get_current_market_data()
        
        # 2. 分析AI减速影响
        ai_returns, ai_impact = self.analyze_ai_slowdown_impact(market_data)
        
        # 3. 识别新增长机会
        growth_opportunities = self.identify_2025_growth_opportunities(market_data)
        
        # 4. 设计优化组合
        optimized_portfolio = self.design_optimized_2025_portfolio(ai_impact, growth_opportunities)
        
        # 5. 计算预期收益
        expected_return, scenarios = self.calculate_2025_expected_returns(optimized_portfolio, market_data)
        
        return {
            'market_data': market_data,
            'ai_impact': ai_impact,
            'growth_opportunities': growth_opportunities,
            'optimized_portfolio': optimized_portfolio,
            'expected_return': expected_return,
            'scenarios': scenarios
        }

if __name__ == "__main__":
    analyzer = Portfolio2025OutlookAnalyzer()
    results = analyzer.comprehensive_2025_analysis() 