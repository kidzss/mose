import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProfessionalPortfolioValidator2025:
    """2025年专业投资组合验证器"""
    
    def __init__(self):
        # 修正后的组合配置
        self.optimized_portfolio = {
            # 核心科技股 45%
            'GOOG': 0.12,   # 搜索稳定+AI应用
            'META': 0.11,   # VR/AR+广告
            'AMZN': 0.07,   # 云+电商
            'NVDA': 0.05,   # AI龙头但减权
            'ORCL': 0.05,   # 数据库+云
            'AMD': 0.03,    # AI芯片
            'PLTR': 0.02,   # 大数据，大幅减权
            
            # 新增长主题 25%
            'ABBV': 0.04,   # 生物技术龙头
            'TSLA': 0.04,   # 电动车+FSD
            'CRWD': 0.03,   # 网络安全
            'GILD': 0.03,   # 生物技术
            'JPM': 0.09,    # 金融受益高利率
            'IBM': 0.02,    # AI转型
            
            # 价值防御 30%
            'BRK-B': 0.08,  # 巴菲特价值
            'MRK': 0.06,    # 制药稳定
            'JNJ': 0.06,    # 医疗健康
            'VZ': 0.05,     # 电信股息
            'CVX': 0.05     # 能源
        }
        
        # 2025年宏观环境假设
        self.macro_environment_2025 = {
            'federal_funds_rate': 0.045,  # 4.5%联邦基金利率
            'inflation_rate': 0.025,      # 2.5%通胀率
            'gdp_growth': 0.022,          # 2.2%GDP增长
            'unemployment': 0.042,        # 4.2%失业率
            'dollar_strength': 0.05,      # 美元升值5%
            'oil_price': 75,              # 油价75美元/桶
            'vix_level': 18               # VIX恐慌指数18
        }
        
        # 地缘政治风险因子
        self.geopolitical_risks = {
            'china_us_tension': {
                'probability': 0.7,
                'impact_on_tech': -0.15,    # 科技股负面影响15%
                'impact_on_defense': 0.08,  # 防御股正面影响8%
                'tariff_escalation': 0.6    # 关税升级概率60%
            },
            'europe_instability': {
                'probability': 0.4,
                'impact_on_us_stocks': -0.05,
                'safe_haven_boost': 0.03
            },
            'middle_east_tension': {
                'probability': 0.6,
                'impact_on_energy': 0.12,
                'impact_on_market': -0.08
            }
        }
        
        # 行业特定风险
        self.sector_risks_2025 = {
            'tech_regulation': {
                'probability': 0.8,
                'impact_on_mega_tech': -0.12,
                'affected_stocks': ['GOOG', 'META', 'AMZN']
            },
            'ai_bubble_burst': {
                'probability': 0.4,
                'impact_on_ai_stocks': -0.35,
                'affected_stocks': ['NVDA', 'AMD', 'PLTR']
            },
            'healthcare_reform': {
                'probability': 0.6,
                'impact_on_pharma': -0.08,
                'affected_stocks': ['ABBV', 'GILD', 'MRK', 'JNJ']
            },
            'financial_regulation': {
                'probability': 0.3,
                'impact_on_banks': -0.10,
                'affected_stocks': ['JPM']
            }
        }
        
    def get_comprehensive_market_data(self, period='2y'):
        """获取全面的市场数据"""
        print("📊 获取全面市场数据...")
        
        symbols = list(self.optimized_portfolio.keys())
        # 添加市场指数和宏观指标
        market_indices = ['^GSPC', '^IXIC', '^DJI', '^VIX', 'GLD', 'TLT', 'DXY']
        all_symbols = symbols + market_indices
        
        market_data = {}
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
                
                if len(hist) > 100:
                    # 基本数据
                    current_price = hist['Close'].iloc[-1]
                    returns = hist['Close'].pct_change().dropna()
                    
                    # 技术指标
                    volatility = returns.std() * np.sqrt(252)
                    sharpe_ratio = (returns.mean() * 252 - 0.04) / volatility
                    max_drawdown = self.calculate_max_drawdown(hist['Close'])
                    
                    # 估值指标
                    pe_ratio = info.get('trailingPE', 'N/A')
                    pb_ratio = info.get('priceToBook', 'N/A')
                    dividend_yield = info.get('dividendYield', 0)
                    
                    market_data[symbol] = {
                        'price': current_price,
                        'returns': returns,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'pe_ratio': pe_ratio,
                        'pb_ratio': pb_ratio,
                        'dividend_yield': dividend_yield,
                        'market_cap': info.get('marketCap', 'N/A'),
                        'beta': info.get('beta', 'N/A')
                    }
                    
                    print(f"✓ {symbol}: 夏普{sharpe_ratio:.2f}, 波动率{volatility:.1%}")
                else:
                    print(f"⚠ {symbol}: 数据不足")
            except Exception as e:
                print(f"✗ {symbol}: {e}")
                
        return market_data
    
    def calculate_max_drawdown(self, price_series):
        """计算最大回撤"""
        cumulative = (1 + price_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def analyze_correlation_structure(self, market_data):
        """分析相关性结构"""
        print("\n📈 相关性结构分析:")
        print("-" * 60)
        
        # 构建收益率矩阵
        returns_data = {}
        for symbol, weight in self.optimized_portfolio.items():
            if symbol in market_data:
                returns_data[symbol] = market_data[symbol]['returns']
        
        returns_df = pd.DataFrame(returns_data).dropna()
        correlation_matrix = returns_df.corr()
        
        # 分析高相关性风险
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    stock1 = correlation_matrix.columns[i]
                    stock2 = correlation_matrix.columns[j]
                    weight1 = self.optimized_portfolio[stock1]
                    weight2 = self.optimized_portfolio[stock2]
                    combined_weight = weight1 + weight2
                    
                    high_corr_pairs.append({
                        'pair': f"{stock1}-{stock2}",
                        'correlation': corr,
                        'combined_weight': combined_weight,
                        'risk_score': abs(corr) * combined_weight
                    })
        
        print("高相关性风险对:")
        for pair in sorted(high_corr_pairs, key=lambda x: x['risk_score'], reverse=True):
            print(f"  {pair['pair']:12s}: 相关性{pair['correlation']:+.2f}, 权重{pair['combined_weight']:.1%}, 风险评分{pair['risk_score']:.3f}")
        
        # 计算组合分散度
        portfolio_weights = np.array([self.optimized_portfolio[s] for s in correlation_matrix.columns])
        portfolio_variance = np.dot(portfolio_weights, np.dot(correlation_matrix.values, portfolio_weights))
        average_correlation = (correlation_matrix.values.sum() - len(correlation_matrix)) / (len(correlation_matrix) * (len(correlation_matrix) - 1))
        
        print(f"\n组合分散度分析:")
        print(f"  平均相关系数: {average_correlation:.3f}")
        print(f"  组合方差: {portfolio_variance:.3f}")
        print(f"  分散度评分: {1 - average_correlation:.3f} (越高越好)")
        
        return correlation_matrix, high_corr_pairs
    
    def stress_test_scenarios(self, market_data):
        """压力测试场景"""
        print("\n🔥 压力测试场景分析:")
        print("-" * 60)
        
        stress_scenarios = {
            'china_tariff_escalation': {
                'description': '中美关税升级',
                'probability': 0.6,
                'impacts': {
                    'GOOG': -0.15, 'META': -0.12, 'AMZN': -0.20,
                    'NVDA': -0.25, 'AMD': -0.30, 'TSLA': -0.35,
                    'JPM': -0.05, 'CVX': 0.05, 'BRK-B': -0.08
                }
            },
            'ai_regulation_crackdown': {
                'description': 'AI监管收紧',
                'probability': 0.4,
                'impacts': {
                    'NVDA': -0.30, 'AMD': -0.25, 'GOOG': -0.20,
                    'META': -0.18, 'PLTR': -0.40, 'CRWD': 0.10
                }
            },
            'healthcare_price_controls': {
                'description': '医疗价格管制',
                'probability': 0.5,
                'impacts': {
                    'ABBV': -0.20, 'GILD': -0.15, 'MRK': -0.12,
                    'JNJ': -0.10
                }
            },
            'recession_scenario': {
                'description': '经济衰退',
                'probability': 0.3,
                'impacts': {
                    'GOOG': -0.25, 'META': -0.30, 'AMZN': -0.35,
                    'NVDA': -0.40, 'TSLA': -0.45, 'JPM': -0.20,
                    'ABBV': -0.10, 'VZ': -0.05, 'CVX': -0.15
                }
            },
            'market_crash': {
                'description': '市场崩盘(-30%)',
                'probability': 0.1,
                'impacts': {symbol: -0.30 for symbol in self.optimized_portfolio.keys()}
            }
        }
        
        scenario_results = {}
        for scenario_name, scenario in stress_scenarios.items():
            portfolio_impact = 0
            affected_value = 0
            
            print(f"\n【{scenario['description']}】(概率{scenario['probability']:.1%}):")
            
            for symbol, impact in scenario['impacts'].items():
                if symbol in self.optimized_portfolio:
                    weight = self.optimized_portfolio[symbol]
                    contribution = impact * weight
                    portfolio_impact += contribution
                    affected_value += weight
                    
                    print(f"  {symbol:6s}: {impact:+.1%} × {weight:.1%} = {contribution:+.2%}")
            
            expected_loss = portfolio_impact * scenario['probability']
            scenario_results[scenario_name] = {
                'total_impact': portfolio_impact,
                'expected_loss': expected_loss,
                'affected_weight': affected_value,
                'probability': scenario['probability']
            }
            
            print(f"  总影响: {portfolio_impact:+.1%}")
            print(f"  期望损失: {expected_loss:+.2%}")
        
        # 计算综合风险评分
        total_expected_loss = sum(s['expected_loss'] for s in scenario_results.values())
        worst_case_loss = min(s['total_impact'] for s in scenario_results.values())
        
        print(f"\n压力测试汇总:")
        print(f"  总期望损失: {total_expected_loss:+.2%}")
        print(f"  最坏情况损失: {worst_case_loss:+.1%}")
        print(f"  风险评分: {abs(total_expected_loss * 10):.1f}/10")
        
        return scenario_results
    
    def calculate_risk_adjusted_returns(self, market_data):
        """计算风险调整后收益"""
        print("\n💰 风险调整后收益计算:")
        print("-" * 60)
        
        # 基于2025年宏观环境的收益预期
        macro_adjusted_returns = {
            # 科技股 - 受监管和关税影响
            'GOOG': 0.15,   # 搜索稳定但面临监管
            'META': 0.12,   # 广告增长但监管压力
            'AMZN': 0.14,   # 云增长但电商放缓
            'NVDA': 0.20,   # AI需求但增长放缓
            'ORCL': 0.18,   # 企业软件稳定
            'AMD': 0.10,    # 竞争激烈
            'PLTR': 0.15,   # 数据分析需求
            
            # 新增长主题
            'ABBV': 0.22,   # 生物技术创新
            'TSLA': 0.25,   # 电动车+FSD
            'CRWD': 0.28,   # 网络安全需求
            'GILD': 0.20,   # 生物技术
            'JPM': 0.18,    # 高利率受益
            'IBM': 0.12,    # 转型缓慢
            
            # 价值防御
            'BRK-B': 0.10,  # 稳健价值
            'MRK': 0.08,    # 制药稳定
            'JNJ': 0.06,    # 法律风险
            'VZ': 0.05,     # 成熟行业
            'CVX': 0.12     # 能源稳定
        }
        
        # 计算组合预期收益
        portfolio_expected_return = 0
        portfolio_risk = 0
        
        print("个股风险调整收益:")
        for symbol, weight in self.optimized_portfolio.items():
            if symbol in macro_adjusted_returns and symbol in market_data:
                expected_return = macro_adjusted_returns[symbol]
                volatility = market_data[symbol]['volatility']
                
                # 风险调整
                risk_free_rate = self.macro_environment_2025['federal_funds_rate']
                risk_premium = expected_return - risk_free_rate
                risk_adjusted_return = risk_free_rate + (risk_premium * 0.8)  # 保守调整
                
                contribution = risk_adjusted_return * weight
                portfolio_expected_return += contribution
                portfolio_risk += (volatility * weight) ** 2
                
                print(f"  {symbol:6s}: 预期{expected_return:.1%} → 调整后{risk_adjusted_return:.1%} × {weight:.1%} = {contribution:+.2%}")
        
        portfolio_volatility = np.sqrt(portfolio_risk)
        portfolio_sharpe = (portfolio_expected_return - self.macro_environment_2025['federal_funds_rate']) / portfolio_volatility
        
        print(f"\n组合风险调整指标:")
        print(f"  预期收益: {portfolio_expected_return:.1%}")
        print(f"  预期波动率: {portfolio_volatility:.1%}")
        print(f"  夏普比率: {portfolio_sharpe:.2f}")
        
        return portfolio_expected_return, portfolio_volatility, portfolio_sharpe
    
    def sector_rotation_analysis(self, market_data):
        """行业轮动分析"""
        print("\n🔄 2025年行业轮动分析:")
        print("-" * 60)
        
        # 定义行业分类
        sectors = {
            'Technology': ['GOOG', 'META', 'AMZN', 'NVDA', 'ORCL', 'AMD', 'PLTR', 'IBM'],
            'Healthcare': ['ABBV', 'GILD', 'MRK', 'JNJ'],
            'Financial': ['JPM', 'BRK-B'],
            'Cyclical': ['TSLA'],
            'Defense': ['CRWD'],
            'Utilities': ['VZ'],
            'Energy': ['CVX']
        }
        
        # 2025年行业前景评估
        sector_outlook = {
            'Technology': {'trend': 'Mixed', 'score': 0.6, 'reasoning': 'AI放缓但数字化持续'},
            'Healthcare': {'trend': 'Positive', 'score': 0.75, 'reasoning': '人口老龄化+新药突破'},
            'Financial': {'trend': 'Positive', 'score': 0.7, 'reasoning': '高利率环境受益'},
            'Cyclical': {'trend': 'Volatile', 'score': 0.65, 'reasoning': '技术突破vs竞争'},
            'Defense': {'trend': 'Strong', 'score': 0.8, 'reasoning': '网络安全需求激增'},
            'Utilities': {'trend': 'Stable', 'score': 0.5, 'reasoning': '成熟行业增长有限'},
            'Energy': {'trend': 'Stable', 'score': 0.6, 'reasoning': '油价稳定但转型压力'}
        }
        
        print("行业配置分析:")
        total_sector_score = 0
        for sector, stocks in sectors.items():
            sector_weight = sum(self.optimized_portfolio.get(stock, 0) for stock in stocks)
            outlook = sector_outlook[sector]
            weighted_score = sector_weight * outlook['score']
            total_sector_score += weighted_score
            
            print(f"  {sector:12s}: {sector_weight:5.1%} 权重 × {outlook['score']:.2f} 评分 = {weighted_score:.3f}")
            print(f"                   {outlook['trend']} - {outlook['reasoning']}")
        
        print(f"\n组合行业轮动评分: {total_sector_score:.2f}/1.00")
        
        return total_sector_score
    
    def comprehensive_validation(self):
        """综合验证"""
        print("🔍 2025年投资组合专业验证")
        print("=" * 80)
        
        # 1. 获取市场数据
        market_data = self.get_comprehensive_market_data()
        
        # 2. 相关性分析
        correlation_matrix, high_corr_pairs = self.analyze_correlation_structure(market_data)
        
        # 3. 压力测试
        stress_results = self.stress_test_scenarios(market_data)
        
        # 4. 风险调整收益
        expected_return, volatility, sharpe_ratio = self.calculate_risk_adjusted_returns(market_data)
        
        # 5. 行业轮动分析
        sector_score = self.sector_rotation_analysis(market_data)
        
        # 6. 综合评分
        print(f"\n🎯 综合评估结果:")
        print("-" * 60)
        
        # 风险评分 (越低越好)
        risk_score = len(high_corr_pairs) * 0.1 + abs(sum(s['expected_loss'] for s in stress_results.values())) * 20
        
        # 收益评分
        return_score = min(expected_return / 0.25, 1.0)  # 以25%为满分
        
        # 分散度评分
        diversification_score = min(sector_score, 0.8) / 0.8
        
        # 夏普比率评分
        sharpe_score = min(sharpe_ratio / 1.5, 1.0)  # 以1.5为满分
        
        # 综合评分
        overall_score = (return_score * 0.3 + diversification_score * 0.25 + 
                        sharpe_score * 0.25 + (1 - risk_score/10) * 0.2)
        
        print(f"预期年化收益: {expected_return:.1%}")
        print(f"预期波动率: {volatility:.1%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"风险评分: {risk_score:.1f}/10 (越低越好)")
        print(f"综合评分: {overall_score:.2f}/1.00")
        
        # 投资建议
        if overall_score >= 0.8:
            recommendation = "强烈推荐 ⭐⭐⭐⭐⭐"
        elif overall_score >= 0.7:
            recommendation = "推荐 ⭐⭐⭐⭐"
        elif overall_score >= 0.6:
            recommendation = "谨慎推荐 ⭐⭐⭐"
        else:
            recommendation = "需要调整 ⭐⭐"
        
        print(f"\n投资建议: {recommendation}")
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'risk_score': risk_score,
            'overall_score': overall_score,
            'recommendation': recommendation
        }

if __name__ == "__main__":
    validator = ProfessionalPortfolioValidator2025()
    results = validator.comprehensive_validation() 