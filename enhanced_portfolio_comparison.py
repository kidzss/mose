#!/usr/bin/env python3
"""
增强投资组合对比分析系统
包含TSLA和MSFT的多方案严格数据验证
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

class EnhancedPortfolioComparison:
    """增强投资组合对比分析器"""
    
    def __init__(self):
        # 当前持仓（基于真实配置）
        self.current_portfolio = {
            'NVDA': 0.183,  # 18.3%
            'GOOG': 0.183,  # 18.3%
            'AMD': 0.140,   # 14.0%
            'PFE': 0.070,   # 7.0%
            'MRK': 0.023,   # 2.3%
            'BRK-B': 0.035, # 3.5%
            'CASH': 0.366   # 36.6%
        }
        
        # 方案1：原推荐配置
        self.portfolio_v1_original = {
            # 成长股 50%
            'NVDA': 0.08,   'GOOG': 0.12,   'AMD': 0.05,
            'META': 0.12,   'AMZN': 0.08,   'PLTR': 0.05,
            # 价值成长 25%
            'JPM': 0.08,    'BRK-B': 0.08,  'ORCL': 0.05,   'IBM': 0.04,
            # 防御股 25%
            'MRK': 0.08,    'JNJ': 0.07,    'VZ': 0.05,     'CVX': 0.05,
            'CASH': 0.05
        }
        
        # 方案2：加入TSLA和MSFT（用户建议）
        self.portfolio_v2_tsla_msft = {
            # 成长股 50% (加入TSLA和MSFT)
            'NVDA': 0.08,   'GOOG': 0.10,   'AMD': 0.04,
            'META': 0.10,   'AMZN': 0.06,   'PLTR': 0.04,
            'TSLA': 0.04,   'MSFT': 0.04,   # 新增
            # 价值成长 25%
            'JPM': 0.08,    'BRK-B': 0.08,  'ORCL': 0.05,   'IBM': 0.04,
            # 防御股 25%
            'MRK': 0.08,    'JNJ': 0.07,    'VZ': 0.05,     'CVX': 0.05,
            'CASH': 0.05
        }
        
        # 方案3：TSLA和MSFT重权重配置
        self.portfolio_v3_heavy_tsla_msft = {
            # 成长股 50% (TSLA和MSFT高权重)
            'NVDA': 0.08,   'GOOG': 0.08,   'AMD': 0.04,
            'META': 0.08,   'AMZN': 0.06,   'PLTR': 0.04,
            'TSLA': 0.08,   'MSFT': 0.08,   # 高权重
            # 价值成长 25%
            'JPM': 0.08,    'BRK-B': 0.08,  'ORCL': 0.05,   'IBM': 0.04,
            # 防御股 25%
            'MRK': 0.08,    'JNJ': 0.07,    'VZ': 0.05,     'CVX': 0.05,
            'CASH': 0.05
        }
        
        # 方案4：科技巨头均衡配置
        self.portfolio_v4_balanced_tech = {
            # 成长股 50% (科技巨头均衡)
            'NVDA': 0.08,   'GOOG': 0.08,   'AMD': 0.06,
            'META': 0.08,   'AMZN': 0.08,   'TSLA': 0.06,   'MSFT': 0.06,
            # 价值成长 25%
            'JPM': 0.08,    'BRK-B': 0.08,  'ORCL': 0.05,   'IBM': 0.04,
            # 防御股 25%
            'MRK': 0.08,    'JNJ': 0.07,    'VZ': 0.05,     'CVX': 0.05,
            'CASH': 0.05
        }
        
        # 方案5：保守型（降低科技股权重）
        self.portfolio_v5_conservative = {
            # 成长股 40% (降低科技股权重)
            'NVDA': 0.06,   'GOOG': 0.06,   'AMD': 0.04,
            'META': 0.06,   'AMZN': 0.06,   'TSLA': 0.06,   'MSFT': 0.06,
            # 价值成长 30%
            'JPM': 0.10,    'BRK-B': 0.10,  'ORCL': 0.06,   'IBM': 0.04,
            # 防御股 30%
            'MRK': 0.10,    'JNJ': 0.08,    'VZ': 0.06,     'CVX': 0.06,
            'CASH': 0.05
        }
        
        self.portfolios = {
            '当前持仓': self.current_portfolio,
            'V1-原推荐': self.portfolio_v1_original,
            'V2-加入TSLA/MSFT': self.portfolio_v2_tsla_msft,
            'V3-TSLA/MSFT高权重': self.portfolio_v3_heavy_tsla_msft,
            'V4-科技巨头均衡': self.portfolio_v4_balanced_tech,
            'V5-保守配置': self.portfolio_v5_conservative
        }
        
        print("🔍 投资组合方案设计完成")
        for name, portfolio in self.portfolios.items():
            stock_weight = sum(v for k, v in portfolio.items() if k != 'CASH')
            print(f"{name}: 股票权重 {stock_weight:.1%}, 现金 {portfolio.get('CASH', 0):.1%}")
    
    def fetch_historical_data(self, period="3y"):
        """获取历史数据"""
        print("\n📊 获取历史数据...")
        
        # 获取所有需要的股票代码
        all_symbols = set()
        for portfolio in self.portfolios.values():
            all_symbols.update([s for s in portfolio.keys() if s != 'CASH'])
        
        data = {}
        failed_symbols = []
        
        for symbol in all_symbols:
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
    
    def calculate_portfolio_performance(self, returns, weights, name):
        """计算投资组合表现"""
        # 只计算有数据的股票
        available_symbols = [s for s in weights.keys() if s in returns.columns and s != 'CASH']
        
        # 重新标准化权重（排除现金）
        total_stock_weight = sum(weights[s] for s in available_symbols)
        if total_stock_weight == 0:
            return None
        
        normalized_weights = {s: weights[s] / total_stock_weight for s in available_symbols}
        weights_series = pd.Series(normalized_weights, index=available_symbols)
        
        # 计算组合收益
        portfolio_returns = (returns[available_symbols] * weights_series).sum(axis=1)
        
        # 计算指标
        annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR和其他指标
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # 胜率（正收益日比例）
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        # 收益分布
        positive_returns = portfolio_returns[portfolio_returns > 0]
        negative_returns = portfolio_returns[portfolio_returns < 0]
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'name': name,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative,
            'weights': normalized_weights
        }
    
    def monte_carlo_simulation(self, returns, weights, num_simulations=1000, time_horizon=252):
        """蒙特卡洛模拟"""
        available_symbols = [s for s in weights.keys() if s in returns.columns and s != 'CASH']
        total_stock_weight = sum(weights[s] for s in available_symbols)
        
        if total_stock_weight == 0:
            return None
        
        normalized_weights = {s: weights[s] / total_stock_weight for s in available_symbols}
        weights_series = pd.Series(normalized_weights, index=available_symbols)
        
        portfolio_returns = (returns[available_symbols] * weights_series).sum(axis=1)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # 生成随机路径
        random_returns = np.random.normal(mean_return, std_return, (time_horizon, num_simulations))
        cumulative_returns = np.cumprod(1 + random_returns, axis=0) - 1
        final_returns = cumulative_returns[-1]
        
        return {
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'std_return': np.std(final_returns),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_95': np.percentile(final_returns, 95),
            'prob_positive': np.sum(final_returns > 0) / num_simulations,
            'prob_target_20': np.sum(final_returns > 0.20) / num_simulations,
            'prob_target_25': np.sum(final_returns > 0.25) / num_simulations,
            'prob_target_30': np.sum(final_returns > 0.30) / num_simulations,
        }
    
    def analyze_individual_stocks(self, returns):
        """分析个股表现"""
        print("\n📈 个股历史表现分析 (过去3年)")
        print("=" * 80)
        
        stock_performance = {}
        for symbol in returns.columns:
            stock_returns = returns[symbol]
            annual_ret = (1 + stock_returns.mean()) ** 252 - 1
            annual_vol = stock_returns.std() * np.sqrt(252)
            sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
            
            # 最大回撤
            cumulative = (1 + stock_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            stock_performance[symbol] = {
                'annual_return': annual_ret,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            }
        
        # 按夏普比率排序
        sorted_stocks = sorted(stock_performance.items(), 
                             key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        print(f"{'股票':6} | {'年化收益':8} | {'年化波动':8} | {'夏普比率':8} | {'最大回撤':8} | {'评级':6}")
        print("-" * 70)
        
        for symbol, perf in sorted_stocks:
            rating = "🟢优秀" if perf['sharpe_ratio'] > 1.5 else "🟡良好" if perf['sharpe_ratio'] > 1.0 else "🟠一般" if perf['sharpe_ratio'] > 0.5 else "🔴较差"
            print(f"{symbol:6} | {perf['annual_return']:8.1%} | {perf['annual_volatility']:8.1%} | " + 
                  f"{perf['sharpe_ratio']:8.2f} | {perf['max_drawdown']:8.1%} | {rating:6}")
        
        return stock_performance
    
    def comprehensive_comparison(self):
        """综合对比分析"""
        print("🔬 开始增强投资组合对比分析")
        print("=" * 80)
        
        # 获取数据
        data = self.fetch_historical_data()
        if data.empty:
            print("❌ 无法获取足够的历史数据")
            return
        
        returns = data.pct_change().dropna()
        print(f"\n📈 分析期间: {returns.index[0].strftime('%Y-%m-%d')} 至 {returns.index[-1].strftime('%Y-%m-%d')}")
        print(f"🗓️ 交易日数: {len(returns)}")
        
        # 分析个股表现
        stock_performance = self.analyze_individual_stocks(returns)
        
        # 计算各组合表现
        print(f"\n📊 投资组合对比分析")
        print("=" * 100)
        
        portfolio_results = {}
        for name, weights in self.portfolios.items():
            result = self.calculate_portfolio_performance(returns, weights, name)
            if result:
                portfolio_results[name] = result
        
        # 输出对比表格
        print(f"{'组合名称':15} | {'年化收益':8} | {'年化波动':8} | {'夏普比率':8} | {'最大回撤':8} | {'胜率':6} | {'盈亏比':6}")
        print("-" * 95)
        
        # 按夏普比率排序
        sorted_results = sorted(portfolio_results.items(), 
                              key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        for name, result in sorted_results:
            print(f"{name:15} | {result['annual_return']:8.1%} | {result['annual_volatility']:8.1%} | " + 
                  f"{result['sharpe_ratio']:8.2f} | {result['max_drawdown']:8.1%} | " +
                  f"{result['win_rate']:6.1%} | {result['profit_loss_ratio']:6.2f}")
        
        # 蒙特卡洛模拟对比
        print(f"\n🎲 蒙特卡洛模拟对比 (1000次，1年期)")
        print("=" * 80)
        
        monte_carlo_results = {}
        for name, weights in self.portfolios.items():
            mc_result = self.monte_carlo_simulation(returns, weights)
            if mc_result:
                monte_carlo_results[name] = mc_result
        
        print(f"{'组合名称':15} | {'预期收益':8} | {'盈利概率':8} | {'超20%概率':9} | {'超25%概率':9} | {'超30%概率':9}")
        print("-" * 85)
        
        for name, mc_result in monte_carlo_results.items():
            print(f"{name:15} | {mc_result['mean_return']:8.1%} | {mc_result['prob_positive']:8.1%} | " + 
                  f"{mc_result['prob_target_20']:9.1%} | {mc_result['prob_target_25']:9.1%} | " +
                  f"{mc_result['prob_target_30']:9.1%}")
        
        # 详细分析最佳方案
        print(f"\n🏆 最佳方案详细分析")
        print("=" * 80)
        
        best_portfolio = sorted_results[0]
        best_name = best_portfolio[0]
        best_result = best_portfolio[1]
        
        print(f"🥇 最佳方案: {best_name}")
        print(f"📈 年化收益: {best_result['annual_return']:.1%}")
        print(f"📊 年化波动: {best_result['annual_volatility']:.1%}")
        print(f"⚖️ 夏普比率: {best_result['sharpe_ratio']:.2f}")
        print(f"📉 最大回撤: {best_result['max_drawdown']:.1%}")
        print(f"🎯 胜率: {best_result['win_rate']:.1%}")
        print(f"💰 盈亏比: {best_result['profit_loss_ratio']:.2f}")
        
        print(f"\n📋 最佳方案权重配置:")
        sorted_weights = sorted(best_result['weights'].items(), 
                              key=lambda x: x[1], reverse=True)
        for symbol, weight in sorted_weights:
            stock_perf = stock_performance.get(symbol, {})
            stock_sharpe = stock_perf.get('sharpe_ratio', 0)
            print(f"  {symbol:6}: {weight:6.1%} (个股夏普: {stock_sharpe:.2f})")
        
        # 风险分析
        print(f"\n⚠️ 风险分析")
        print("-" * 50)
        
        current_result = portfolio_results.get('当前持仓')
        if current_result and best_name != '当前持仓':
            risk_improvement = current_result['annual_volatility'] - best_result['annual_volatility']
            return_diff = best_result['annual_return'] - current_result['annual_return']
            sharpe_improvement = best_result['sharpe_ratio'] - current_result['sharpe_ratio']
            
            print(f"相比当前持仓:")
            print(f"  收益变化: {return_diff:+.1%}")
            print(f"  风险降低: {risk_improvement:+.1%}")
            print(f"  夏普提升: {sharpe_improvement:+.2f}")
        
        # 保存结果
        analysis_summary = {
            'timestamp': datetime.now().isoformat(),
            'best_portfolio': best_name,
            'portfolio_results': {name: {k: float(v) for k, v in result.items() 
                                       if isinstance(v, (int, float))} 
                                for name, result in portfolio_results.items()},
            'monte_carlo_results': monte_carlo_results,
            'stock_performance': stock_performance
        }
        
        with open('../enhanced_portfolio_comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细分析结果已保存至: enhanced_portfolio_comparison_results.json")
        
        return analysis_summary

def main():
    """主函数"""
    analyzer = EnhancedPortfolioComparison()
    results = analyzer.comprehensive_comparison()
    
    if results:
        print("\n" + "=" * 80)
        print("🎉 增强投资组合对比分析完成！")
        print("📊 已生成包含TSLA和MSFT的多方案严格数据验证。")

if __name__ == "__main__":
    main() 