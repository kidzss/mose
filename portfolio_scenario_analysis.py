import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioScenarioAnalyzer:
    """投资组合情景分析器"""
    
    def __init__(self):
        # 我的推荐组合配置
        self.portfolio = {
            # 成长股 50%
            'NVDA': 0.08,   # 8% - AI芯片龙头
            'GOOG': 0.12,   # 12% - 搜索+AI
            'AMD': 0.05,    # 5% - 芯片+AI
            'META': 0.12,   # 12% - 社交+元宇宙
            'AMZN': 0.08,   # 8% - 云计算+电商
            'PLTR': 0.05,   # 5% - 大数据分析
            
            # 价值成长 25%
            'JPM': 0.08,    # 8% - 银行龙头
            'BRK-B': 0.08,  # 8% - 巴菲特价值投资
            'ORCL': 0.05,   # 5% - 数据库云转型
            'IBM': 0.04,    # 4% - AI转型
            
            # 防御股 25%
            'MRK': 0.08,    # 8% - 制药巨头
            'JNJ': 0.07,    # 7% - 医疗健康
            'VZ': 0.05,     # 5% - 电信高股息
            'CVX': 0.05     # 5% - 能源防通胀
        }
        
        # 验证权重总和
        total_weight = sum(self.portfolio.values())
        print(f"组合权重总和: {total_weight:.1%}")
        
    def get_historical_data(self, period='3y'):
        """获取历史数据"""
        symbols = list(self.portfolio.keys())
        data = {}
        
        print("📊 获取历史数据...")
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if len(hist) > 100:  # 至少需要100个交易日
                    data[symbol] = hist['Close']
                    print(f"✓ {symbol}: {len(hist)} 个交易日")
                else:
                    print(f"⚠ {symbol}: 数据不足 ({len(hist)} 个交易日)")
            except Exception as e:
                print(f"✗ {symbol}: 获取失败 - {e}")
                
        df = pd.DataFrame(data).dropna()
        print(f"📊 最终数据集: {len(df)} 个交易日, {len(df.columns)} 个股票")
        return df
    
    def calculate_returns(self, data):
        """计算收益率"""
        # 日收益率
        daily_returns = data.pct_change().dropna()
        
        # 年化收益率
        annual_returns = {}
        for symbol in data.columns:
            total_return = (data[symbol].iloc[-1] / data[symbol].iloc[0]) - 1
            years = len(data) / 252  # 252个交易日/年
            annual_return = (1 + total_return) ** (1/years) - 1
            annual_returns[symbol] = annual_return
            
        return daily_returns, annual_returns
    
    def calculate_portfolio_metrics(self, daily_returns, annual_returns):
        """计算组合指标"""
        # 组合日收益率
        portfolio_daily_returns = (daily_returns * pd.Series(self.portfolio)).sum(axis=1)
        
        # 组合年化收益率
        portfolio_annual_return = sum(annual_returns[symbol] * weight 
                                    for symbol, weight in self.portfolio.items())
        
        # 组合波动率(年化)
        portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
        
        # 夏普比率(假设无风险利率4%)
        risk_free_rate = 0.04
        sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_volatility
        
        # 最大回撤
        portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
        portfolio_drawdown = (portfolio_cumulative / portfolio_cumulative.cummax() - 1)
        max_drawdown = portfolio_drawdown.min()
        
        return {
            'annual_return': portfolio_annual_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_returns': portfolio_daily_returns
        }
    
    def scenario_analysis(self, daily_returns):
        """市场情景分析"""
        scenarios = {
            'bull_market': {'condition': 'SPY涨幅前25%', 'multiplier': 1.3},
            'bear_market': {'condition': 'SPY跌幅前25%', 'multiplier': 0.6},
            'normal_market': {'condition': '正常市场', 'multiplier': 1.0},
            'high_volatility': {'condition': '高波动期', 'multiplier': 0.8},
            'recession': {'condition': '经济衰退', 'multiplier': 0.4}
        }
        
        # 计算基础收益率
        base_return = 0
        for symbol, weight in self.portfolio.items():
            if symbol in daily_returns.columns:
                symbol_annual_return = daily_returns[symbol].mean() * 252
                base_return += symbol_annual_return * weight
        
        # 计算各情景下的收益
        scenario_results = {}
        
        for scenario, params in scenarios.items():
            scenario_return = base_return * params['multiplier']
            scenario_results[scenario] = {
                'annual_return': float(scenario_return),
                'description': params['condition']
            }
            
        return scenario_results
    
    def detailed_stock_analysis(self, data, annual_returns):
        """详细个股分析"""
        print("\n📈 个股历史表现分析:")
        print("-" * 80)
        
        categories = {
            '成长股': ['NVDA', 'GOOG', 'AMD', 'META', 'AMZN', 'PLTR'],
            '价值成长': ['JPM', 'BRK-B', 'ORCL', 'IBM'],
            '防御股': ['MRK', 'JNJ', 'VZ', 'CVX']
        }
        
        category_returns = {}
        
        for category, symbols in categories.items():
            print(f"\n{category}:")
            category_total_return = 0
            category_total_weight = 0
            
            for symbol in symbols:
                if symbol in annual_returns and symbol in self.portfolio:
                    ret = annual_returns[symbol]
                    weight = self.portfolio[symbol]
                    volatility = data[symbol].pct_change().std() * np.sqrt(252)
                    
                    print(f"  {symbol:6s}: {ret:6.1%} 年化 | {weight:4.1%} 权重 | {volatility:5.1%} 波动率")
                    
                    category_total_return += ret * weight
                    category_total_weight += weight
            
            if category_total_weight > 0:
                category_avg_return = category_total_return / category_total_weight
                category_returns[category] = {
                    'weighted_return': category_total_return,
                    'avg_return': category_avg_return,
                    'weight': category_total_weight
                }
                print(f"  {category} 加权收益: {category_total_return:6.1%} | 平均收益: {category_avg_return:6.1%}")
        
        return category_returns
    
    def monte_carlo_simulation(self, daily_returns, num_simulations=1000):
        """蒙特卡洛模拟"""
        print(f"\n🎲 蒙特卡洛模拟 ({num_simulations}次)...")
        
        # 计算组合的日收益率统计
        portfolio_daily_returns = []
        for symbol, weight in self.portfolio.items():
            if symbol in daily_returns.columns:
                portfolio_daily_returns.append(daily_returns[symbol] * weight)
        
        if not portfolio_daily_returns:
            return None
            
        portfolio_returns = pd.concat(portfolio_daily_returns, axis=1).sum(axis=1)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # 模拟未来1年的收益
        simulated_annual_returns = []
        
        for _ in range(num_simulations):
            # 生成252个交易日的随机收益
            random_returns = np.random.normal(mean_return, std_return, 252)
            annual_return = (1 + random_returns).prod() - 1
            simulated_annual_returns.append(annual_return)
        
        simulated_returns = np.array(simulated_annual_returns)
        
        return {
            'mean': simulated_returns.mean(),
            'median': np.median(simulated_returns),
            'percentile_10': np.percentile(simulated_returns, 10),
            'percentile_25': np.percentile(simulated_returns, 25),
            'percentile_75': np.percentile(simulated_returns, 75),
            'percentile_90': np.percentile(simulated_returns, 90),
            'std': simulated_returns.std(),
            'prob_positive': (simulated_returns > 0).mean(),
            'prob_above_20': (simulated_returns > 0.20).mean(),
            'prob_above_25': (simulated_returns > 0.25).mean()
        }
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("🔍 开始组合全面分析...")
        print("=" * 80)
        
        # 1. 获取历史数据
        data = self.get_historical_data()
        if data.empty:
            print("❌ 无法获取足够的历史数据")
            return
            
        # 2. 计算收益率
        daily_returns, annual_returns = self.calculate_returns(data)
        
        # 3. 计算组合指标
        portfolio_metrics = self.calculate_portfolio_metrics(daily_returns, annual_returns)
        
        print(f"\n📊 组合历史表现 (过去3年):")
        print("-" * 50)
        print(f"年化收益率: {portfolio_metrics['annual_return']:6.1%}")
        print(f"年化波动率: {portfolio_metrics['volatility']:6.1%}")
        print(f"夏普比率:   {portfolio_metrics['sharpe_ratio']:6.2f}")
        print(f"最大回撤:   {portfolio_metrics['max_drawdown']:6.1%}")
        
        # 4. 详细个股分析
        category_returns = self.detailed_stock_analysis(data, annual_returns)
        
        # 5. 情景分析
        scenarios = self.scenario_analysis(daily_returns)
        print(f"\n🎭 市场情景分析:")
        print("-" * 50)
        for scenario, result in scenarios.items():
            desc = result['description']
            ret = result['annual_return']
            print(f"{desc:12s}: {ret:6.1%}")
        
        # 6. 蒙特卡洛模拟
        mc_results = self.monte_carlo_simulation(daily_returns)
        if mc_results:
            print(f"\n🎲 蒙特卡洛模拟结果:")
            print("-" * 50)
            print(f"预期收益:     {mc_results['mean']:6.1%}")
            print(f"中位数收益:   {mc_results['median']:6.1%}")
            print(f"10%分位数:    {mc_results['percentile_10']:6.1%}")
            print(f"90%分位数:    {mc_results['percentile_90']:6.1%}")
            print(f"盈利概率:     {mc_results['prob_positive']:6.1%}")
            print(f"超20%概率:    {mc_results['prob_above_20']:6.1%}")
            print(f"超25%概率:    {mc_results['prob_above_25']:6.1%}")
        
        return {
            'historical_metrics': portfolio_metrics,
            'category_analysis': category_returns,
            'scenarios': scenarios,
            'monte_carlo': mc_results
        }

if __name__ == "__main__":
    analyzer = PortfolioScenarioAnalyzer()
    results = analyzer.comprehensive_analysis() 