#!/usr/bin/env python3
"""
综合投资组合情景分析系统 - 专业验证方案
严格的数据驱动验证，多角度分析投资组合的有效性
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensivePortfolioScenarioAnalyzer:
    """综合投资组合情景分析器"""
    
    def __init__(self):
        # 读取当前持仓配置
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 当前持仓（基于真实配置）
        self.current_portfolio = {
            'NVDA': 0.183,  # 18.3%
            'GOOG': 0.183,  # 18.3%
            'AMD': 0.140,   # 14.0%
            'PFE': 0.070,   # 7.0%
            'MRK': 0.023,   # 2.3%
            'BRK-B': 0.035, # 3.5%
            # 现金及货币基金
            'CASH': 0.366   # 36.6%
        }
        
        # 推荐配置（目标组合）
        self.recommended_portfolio = {
            # 成长股 50%
            'NVDA': 0.08,   # 8% - 从18.3%减至合理权重
            'GOOG': 0.12,   # 12% - 保持较高权重
            'AMD': 0.05,    # 5% - 大幅减仓
            'META': 0.12,   # 12% - 新增
            'AMZN': 0.08,   # 8% - 新增
            'PLTR': 0.05,   # 5% - 新增
            
            # 价值成长 25%
            'JPM': 0.08,    # 8% - 银行龙头
            'BRK-B': 0.08,  # 8% - 提升权重
            'ORCL': 0.05,   # 5% - 新增
            'IBM': 0.04,    # 4% - 新增
            
            # 防御股 25%
            'MRK': 0.08,    # 8% - 提升权重
            'JNJ': 0.07,    # 7% - 新增
            'VZ': 0.05,     # 5% - 新增
            'CVX': 0.05,    # 5% - 新增
            
            # 现金储备
            'CASH': 0.05    # 5% - 降低现金比例
        }
        
        print(f"🔍 当前配置权重总和: {sum(v for k, v in self.current_portfolio.items() if k != 'CASH'):.1%}")
        print(f"🎯 推荐配置权重总和: {sum(v for k, v in self.recommended_portfolio.items() if k != 'CASH'):.1%}")
    
    def fetch_historical_data(self, symbols, period="3y"):
        """获取历史数据"""
        print("📊 获取历史数据...")
        data = {}
        failed_symbols = []
        
        for symbol in symbols:
            if symbol == 'CASH':
                continue
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close']
                    print(f"✅ {symbol}: {len(hist)} 个交易日")
                else:
                    failed_symbols.append(symbol)
                    print(f"❌ {symbol}: 无数据")
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"❌ {symbol}: {str(e)}")
        
        if failed_symbols:
            print(f"⚠️ 失败的股票: {failed_symbols}")
        
        return pd.DataFrame(data)
    
    def calculate_portfolio_metrics(self, returns, portfolio_weights, name):
        """计算投资组合指标"""
        # 计算组合收益
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        
        # 基础指标
        annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR (95%置信度)
        var_95 = np.percentile(portfolio_returns, 5)
        
        return {
            'name': name,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative
        }
    
    def stress_test_scenarios(self, returns, portfolio_weights):
        """压力测试情景分析"""
        scenarios = {
            '2008金融危机': {'start': '2008-09-01', 'end': '2009-03-01'},
            '2020新冠暴跌': {'start': '2020-02-01', 'end': '2020-04-01'},
            '2022加息周期': {'start': '2022-01-01', 'end': '2022-10-01'},
        }
        
        results = {}
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        
        for scenario_name, period in scenarios.items():
            try:
                mask = (portfolio_returns.index >= period['start']) & (portfolio_returns.index <= period['end'])
                scenario_returns = portfolio_returns[mask]
                
                if len(scenario_returns) > 0:
                    total_return = (1 + scenario_returns).prod() - 1
                    volatility = scenario_returns.std() * np.sqrt(252)
                    worst_day = scenario_returns.min()
                    
                    results[scenario_name] = {
                        'total_return': total_return,
                        'volatility': volatility,
                        'worst_day': worst_day,
                        'days': len(scenario_returns)
                    }
            except:
                continue
        
        return results
    
    def monte_carlo_simulation(self, returns, portfolio_weights, num_simulations=1000, time_horizon=252):
        """蒙特卡洛模拟"""
        print("🎲 执行蒙特卡洛模拟...")
        
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # 生成随机路径
        random_returns = np.random.normal(mean_return, std_return, (time_horizon, num_simulations))
        cumulative_returns = np.cumprod(1 + random_returns, axis=0) - 1
        final_returns = cumulative_returns[-1]
        
        # 统计结果
        results = {
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'std_return': np.std(final_returns),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_95': np.percentile(final_returns, 95),
            'prob_positive': np.sum(final_returns > 0) / num_simulations,
            'prob_target_25': np.sum(final_returns > 0.25) / num_simulations,
            'all_returns': final_returns
        }
        
        return results
    
    def sector_correlation_analysis(self, returns):
        """行业相关性分析"""
        sectors = {
            'Technology': ['NVDA', 'GOOG', 'AMD', 'META', 'AMZN', 'PLTR', 'ORCL', 'IBM'],
            'Financial': ['JPM', 'BRK-B'],
            'Healthcare': ['MRK', 'JNJ', 'PFE'],
            'Telecom': ['VZ'],
            'Energy': ['CVX']
        }
        
        sector_returns = {}
        for sector, stocks in sectors.items():
            available_stocks = [s for s in stocks if s in returns.columns]
            if available_stocks:
                sector_returns[sector] = returns[available_stocks].mean(axis=1)
        
        correlation_matrix = pd.DataFrame(sector_returns).corr()
        return correlation_matrix, sector_returns
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("🔬 开始综合投资组合验证分析")
        print("=" * 80)
        
        # 1. 获取数据
        all_symbols = list(set(list(self.current_portfolio.keys()) + list(self.recommended_portfolio.keys())))
        all_symbols = [s for s in all_symbols if s != 'CASH']
        
        data = self.fetch_historical_data(all_symbols)
        if data.empty:
            print("❌ 无法获取足够的历史数据")
            return
        
        # 计算收益率
        returns = data.pct_change().dropna()
        print(f"📈 分析期间: {returns.index[0].strftime('%Y-%m-%d')} 至 {returns.index[-1].strftime('%Y-%m-%d')}")
        print(f"🗓️ 交易日数: {len(returns)}")
        
        # 2. 构建可比较的权重
        current_weights = {}
        recommended_weights = {}
        
        # 只对有数据的股票计算权重
        available_symbols = list(returns.columns)
        
        # 当前配置权重标准化
        current_stock_weight = sum(v for k, v in self.current_portfolio.items() if k in available_symbols)
        for symbol in available_symbols:
            current_weights[symbol] = self.current_portfolio.get(symbol, 0) / current_stock_weight if current_stock_weight > 0 else 0
        
        # 推荐配置权重标准化  
        recommended_stock_weight = sum(v for k, v in self.recommended_portfolio.items() if k in available_symbols)
        for symbol in available_symbols:
            recommended_weights[symbol] = self.recommended_portfolio.get(symbol, 0) / recommended_stock_weight if recommended_stock_weight > 0 else 0
        
        current_weights_series = pd.Series(current_weights, index=available_symbols)
        recommended_weights_series = pd.Series(recommended_weights, index=available_symbols)
        
        print(f"\n📊 当前配置 (股票部分权重):")
        for symbol, weight in current_weights_series[current_weights_series > 0].items():
            print(f"  {symbol}: {weight:.1%}")
        
        print(f"\n🎯 推荐配置 (股票部分权重):")
        for symbol, weight in recommended_weights_series[recommended_weights_series > 0].items():
            print(f"  {symbol}: {weight:.1%}")
        
        # 3. 计算组合指标
        current_metrics = self.calculate_portfolio_metrics(returns, current_weights_series, "当前配置")
        recommended_metrics = self.calculate_portfolio_metrics(returns, recommended_weights_series, "推荐配置")
        
        # 4. 输出对比分析
        print("\n" + "=" * 80)
        print("📊 历史回测对比分析 (过去3年真实数据)")
        print("=" * 80)
        
        metrics_comparison = [
            ("年化收益率", current_metrics['annual_return'], recommended_metrics['annual_return'], "%"),
            ("年化波动率", current_metrics['annual_volatility'], recommended_metrics['annual_volatility'], "%"),
            ("夏普比率", current_metrics['sharpe_ratio'], recommended_metrics['sharpe_ratio'], ""),
            ("最大回撤", current_metrics['max_drawdown'], recommended_metrics['max_drawdown'], "%"),
            ("95% VaR", current_metrics['var_95'], recommended_metrics['var_95'], "%")
        ]
        
        for metric_name, current_val, recommended_val, unit in metrics_comparison:
            if unit == "%":
                current_str = f"{current_val:.1%}"
                recommended_str = f"{recommended_val:.1%}"
                diff = recommended_val - current_val
                diff_str = f"{diff:+.1%}"
            else:
                current_str = f"{current_val:.2f}"
                recommended_str = f"{recommended_val:.2f}"
                diff = recommended_val - current_val
                diff_str = f"{diff:+.2f}"
            
            improvement = "🟢" if diff > 0 else "🔴" if diff < 0 else "🟡"
            print(f"{metric_name:12} | 当前: {current_str:8} | 推荐: {recommended_str:8} | 差异: {diff_str:8} {improvement}")
        
        # 5. 压力测试
        print("\n" + "=" * 80)
        print("⚠️ 压力测试分析")
        print("=" * 80)
        
        current_stress = self.stress_test_scenarios(returns, current_weights_series)
        recommended_stress = self.stress_test_scenarios(returns, recommended_weights_series)
        
        for scenario in current_stress.keys():
            if scenario in recommended_stress:
                current_ret = current_stress[scenario]['total_return']
                recommended_ret = recommended_stress[scenario]['total_return']
                print(f"{scenario:12} | 当前: {current_ret:8.1%} | 推荐: {recommended_ret:8.1%} | 差异: {recommended_ret-current_ret:+7.1%}")
        
        # 6. 蒙特卡洛模拟
        print("\n" + "=" * 80)
        print("🎲 蒙特卡洛模拟 (1000次，1年期)")
        print("=" * 80)
        
        current_monte = self.monte_carlo_simulation(returns, current_weights_series)
        recommended_monte = self.monte_carlo_simulation(returns, recommended_weights_series)
        
        monte_metrics = [
            ("预期收益", current_monte['mean_return'], recommended_monte['mean_return']),
            ("中位数收益", current_monte['median_return'], recommended_monte['median_return']),
            ("盈利概率", current_monte['prob_positive'], recommended_monte['prob_positive']),
            ("超25%概率", current_monte['prob_target_25'], recommended_monte['prob_target_25'])
        ]
        
        for metric_name, current_val, recommended_val in monte_metrics:
            if "概率" in metric_name:
                current_str = f"{current_val:.1%}"
                recommended_str = f"{recommended_val:.1%}"
                diff = recommended_val - current_val
                diff_str = f"{diff:+.1%}"
            else:
                current_str = f"{current_val:.1%}"
                recommended_str = f"{recommended_val:.1%}"
                diff = recommended_val - current_val
                diff_str = f"{diff:+.1%}"
            
            improvement = "🟢" if diff > 0 else "🔴" if diff < 0 else "🟡"
            print(f"{metric_name:12} | 当前: {current_str:8} | 推荐: {recommended_str:8} | 差异: {diff_str:8} {improvement}")
        
        # 7. 行业相关性分析
        print("\n" + "=" * 80)
        print("🔗 行业相关性分析")
        print("=" * 80)
        
        correlation_matrix, sector_returns = self.sector_correlation_analysis(returns)
        print("\n行业间相关系数:")
        print(correlation_matrix.round(2))
        
        # 8. 个股表现分析
        print("\n" + "=" * 80)
        print("📈 个股历史表现 (过去3年)")
        print("=" * 80)
        
        individual_performance = {}
        for symbol in returns.columns:
            symbol_returns = returns[symbol]
            annual_ret = (1 + symbol_returns.mean()) ** 252 - 1
            annual_vol = symbol_returns.std() * np.sqrt(252)
            individual_performance[symbol] = {
                'annual_return': annual_ret,
                'annual_volatility': annual_vol,
                'sharpe': annual_ret / annual_vol if annual_vol > 0 else 0
            }
        
        # 按收益率排序
        sorted_performance = sorted(individual_performance.items(), 
                                  key=lambda x: x[1]['annual_return'], reverse=True)
        
        print(f"{'股票':6} | {'年化收益':8} | {'年化波动':8} | {'夏普比率':8} | {'当前权重':8} | {'推荐权重':8}")
        print("-" * 70)
        for symbol, perf in sorted_performance:
            current_w = current_weights_series.get(symbol, 0)
            recommended_w = recommended_weights_series.get(symbol, 0)
            print(f"{symbol:6} | {perf['annual_return']:8.1%} | {perf['annual_volatility']:8.1%} | " + 
                  f"{perf['sharpe']:8.2f} | {current_w:8.1%} | {recommended_w:8.1%}")
        
        # 9. 综合评估结论
        print("\n" + "=" * 80)
        print("📋 综合评估结论")
        print("=" * 80)
        
        # 计算改进指标
        improvements = {
            'return': recommended_metrics['annual_return'] - current_metrics['annual_return'],
            'volatility': current_metrics['annual_volatility'] - recommended_metrics['annual_volatility'],  # 波动率降低是好事
            'sharpe': recommended_metrics['sharpe_ratio'] - current_metrics['sharpe_ratio'],
            'drawdown': current_metrics['max_drawdown'] - recommended_metrics['max_drawdown']  # 回撤降低是好事
        }
        
        positive_count = sum(1 for v in improvements.values() if v > 0)
        total_metrics = len(improvements)
        
        print(f"✅ 关键指标改进情况: {positive_count}/{total_metrics}")
        print(f"📈 预期年化收益提升: {improvements['return']:+.1%}")
        print(f"📉 年化波动率变化: {-improvements['volatility']:+.1%}")
        print(f"⚖️ 夏普比率提升: {improvements['sharpe']:+.2f}")
        print(f"🛡️ 最大回撤改善: {improvements['drawdown']:+.1%}")
        
        # 最终结论
        if positive_count >= 3:
            conclusion = "🎯 推荐配置显著优于当前配置"
            recommendation = "建议按计划执行调整"
        elif positive_count >= 2:
            conclusion = "✅ 推荐配置整体优于当前配置"
            recommendation = "建议谨慎执行调整"
        else:
            conclusion = "⚠️ 推荐配置改进有限"
            recommendation = "建议重新审视配置方案"
        
        print(f"\n🏆 最终结论: {conclusion}")
        print(f"📝 操作建议: {recommendation}")
        
        # 保存分析结果
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {k: float(v) for k, v in current_metrics.items() if isinstance(v, (int, float))},
            'recommended_metrics': {k: float(v) for k, v in recommended_metrics.items() if isinstance(v, (int, float))},
            'improvements': improvements,
            'conclusion': conclusion,
            'recommendation': recommendation,
            'monte_carlo_current': {k: float(v) for k, v in current_monte.items() if isinstance(v, (int, float))},
            'monte_carlo_recommended': {k: float(v) for k, v in recommended_monte.items() if isinstance(v, (int, float))},
        }
        
        with open('portfolio_verification_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 分析结果已保存至: portfolio_verification_results.json")
        
        return analysis_results

def main():
    """主函数"""
    analyzer = ComprehensivePortfolioScenarioAnalyzer()
    results = analyzer.comprehensive_analysis()
    
    if results:
        print("\n" + "=" * 80)
        print("🎉 分析完成！基于历史真实数据的专业验证已完成。")

if __name__ == "__main__":
    main() 