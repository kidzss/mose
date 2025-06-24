#!/usr/bin/env python3
"""
投资组合收益预测对比系统
客观分析当前持仓 vs 推荐持仓的预期表现
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioPerformancePredictor:
    """投资组合收益预测器"""
    
    def __init__(self):
        """初始化预测器"""
        # 从配置文件读取当前持仓
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 当前持仓配置
        self.current_portfolio = {
            'NVDA': 0.274,  # 27.4%
            'GOOG': 0.246,  # 24.6%
            'AMD': 0.239,   # 23.9%
            'PFE': 0.090,   # 9.0%
            'TSLA': 0.061,  # 6.1%
            'MRK': 0.052,   # 5.2%
            'JPM': 0.039    # 3.9%
        }
        
        # 推荐持仓配置 (基于之前的设计)
        self.recommended_portfolio = {
            'META': 0.133,  # 13.3%
            'NVDA': 0.133,  # 13.3%
            'GOOG': 0.133,  # 13.3%
            'ABT': 0.067,   # 6.7%
            'JNJ': 0.067,   # 6.7%
            'PFE': 0.067,   # 6.7%
            'JPM': 0.050,   # 5.0%
            'WFC': 0.050,   # 5.0%
            'BAC': 0.050,   # 5.0%
            'WMT': 0.050,   # 5.0%
            'KO': 0.050,    # 5.0%
            'COST': 0.050,  # 5.0%
            'CAT': 0.025,   # 2.5%
            'BA': 0.025,    # 2.5%
            'XOM': 0.015,   # 1.5%
            'CVX': 0.015,   # 1.5%
            'SPY': 0.010,   # 1.0%
            'QQQ': 0.010    # 1.0%
        }
        
        # 预测参数
        self.prediction_periods = [252, 504, 756]  # 1年, 2年, 3年
        self.monte_carlo_simulations = 1000
        
        logger.info("📊 投资组合收益预测器初始化完成")
    
    def get_historical_data(self, symbols, period="3y"):
        """获取历史数据"""
        historical_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty and len(hist) > 252:  # 至少1年数据
                    # 计算日收益率
                    daily_returns = hist['Close'].pct_change().dropna()
                    
                    historical_data[symbol] = {
                        'prices': hist['Close'],
                        'returns': daily_returns,
                        'mean_return': daily_returns.mean(),
                        'std_return': daily_returns.std(),
                        'annual_return': daily_returns.mean() * 252,
                        'annual_volatility': daily_returns.std() * np.sqrt(252),
                        'sharpe_ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
                    }
                    
                    logger.info(f"📈 {symbol}: 年化收益 {historical_data[symbol]['annual_return']:.1%}, "
                              f"波动率 {historical_data[symbol]['annual_volatility']:.1%}")
                
            except Exception as e:
                logger.warning(f"获取{symbol}历史数据失败: {e}")
        
        return historical_data
    
    def calculate_portfolio_metrics(self, portfolio_weights, historical_data):
        """计算投资组合指标"""
        # 确保所有股票都有数据
        valid_symbols = [symbol for symbol in portfolio_weights.keys() if symbol in historical_data]
        
        if not valid_symbols:
            return None
        
        # 重新标准化权重
        total_weight = sum([portfolio_weights[symbol] for symbol in valid_symbols])
        normalized_weights = {symbol: portfolio_weights[symbol] / total_weight for symbol in valid_symbols}
        
        # 计算组合收益率
        portfolio_returns = []
        
        # 获取共同的日期范围
        common_dates = None
        for symbol in valid_symbols:
            if common_dates is None:
                common_dates = historical_data[symbol]['returns'].index
            else:
                common_dates = common_dates.intersection(historical_data[symbol]['returns'].index)
        
        if len(common_dates) < 252:  # 至少需要1年数据
            return None
        
        # 计算每日组合收益率
        for date in common_dates:
            daily_portfolio_return = 0
            for symbol in valid_symbols:
                if date in historical_data[symbol]['returns'].index:
                    daily_portfolio_return += normalized_weights[symbol] * historical_data[symbol]['returns'][date]
            portfolio_returns.append(daily_portfolio_return)
        
        portfolio_returns = pd.Series(portfolio_returns, index=common_dates)
        
        # 计算组合指标
        portfolio_metrics = {
            'weights': normalized_weights,
            'daily_returns': portfolio_returns,
            'mean_daily_return': portfolio_returns.mean(),
            'daily_volatility': portfolio_returns.std(),
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),
            'var_99': np.percentile(portfolio_returns, 1)
        }
        
        return portfolio_metrics
    
    def calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def monte_carlo_simulation(self, portfolio_metrics, days=252):
        """蒙特卡洛模拟"""
        if portfolio_metrics is None:
            return None
        
        mean_return = portfolio_metrics['mean_daily_return']
        volatility = portfolio_metrics['daily_volatility']
        
        # 运行模拟
        simulation_results = []
        
        for _ in range(self.monte_carlo_simulations):
            # 生成随机收益率序列
            random_returns = np.random.normal(mean_return, volatility, days)
            
            # 计算累积收益
            cumulative_return = (1 + pd.Series(random_returns)).prod() - 1
            simulation_results.append(cumulative_return)
        
        simulation_results = np.array(simulation_results)
        
        return {
            'mean_return': np.mean(simulation_results),
            'median_return': np.median(simulation_results),
            'std_return': np.std(simulation_results),
            'percentile_5': np.percentile(simulation_results, 5),
            'percentile_25': np.percentile(simulation_results, 25),
            'percentile_75': np.percentile(simulation_results, 75),
            'percentile_95': np.percentile(simulation_results, 95),
            'probability_positive': np.mean(simulation_results > 0),
            'probability_target_20': np.mean(simulation_results > 0.20),
            'all_results': simulation_results
        }
    
    def compare_portfolios(self, current_metrics, recommended_metrics):
        """对比两个投资组合"""
        comparison = {}
        
        if current_metrics and recommended_metrics:
            comparison = {
                'annual_return': {
                    'current': current_metrics['annual_return'],
                    'recommended': recommended_metrics['annual_return'],
                    'difference': recommended_metrics['annual_return'] - current_metrics['annual_return']
                },
                'annual_volatility': {
                    'current': current_metrics['annual_volatility'],
                    'recommended': recommended_metrics['annual_volatility'],
                    'difference': recommended_metrics['annual_volatility'] - current_metrics['annual_volatility']
                },
                'sharpe_ratio': {
                    'current': current_metrics['sharpe_ratio'],
                    'recommended': recommended_metrics['sharpe_ratio'],
                    'difference': recommended_metrics['sharpe_ratio'] - current_metrics['sharpe_ratio']
                },
                'max_drawdown': {
                    'current': current_metrics['max_drawdown'],
                    'recommended': recommended_metrics['max_drawdown'],
                    'difference': recommended_metrics['max_drawdown'] - current_metrics['max_drawdown']
                }
            }
        
        return comparison
    
    def generate_prediction_report(self, current_metrics, recommended_metrics, 
                                 current_simulations, recommended_simulations, comparison):
        """生成预测报告"""
        report = []
        report.append("=" * 120)
        report.append("📊 投资组合收益预测对比报告")
        report.append(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📈 基于过去3年历史数据 + 蒙特卡洛模拟({self.monte_carlo_simulations}次)")
        report.append("=" * 120)
        
        # 当前持仓分析
        if current_metrics:
            report.append(f"\n📈 当前持仓历史表现分析:")
            report.append("-" * 100)
            report.append(f"• 年化收益率: {current_metrics['annual_return']:+.1%}")
            report.append(f"• 年化波动率: {current_metrics['annual_volatility']:.1%}")
            report.append(f"• 夏普比率: {current_metrics['sharpe_ratio']:.2f}")
            report.append(f"• 最大回撤: {current_metrics['max_drawdown']:+.1%}")
            
            report.append(f"\n当前持仓权重分布:")
            for symbol, weight in sorted(current_metrics['weights'].items(), key=lambda x: x[1], reverse=True):
                report.append(f"  • {symbol}: {weight:.1%}")
        
        # 推荐持仓分析
        if recommended_metrics:
            report.append(f"\n🎯 推荐持仓历史表现分析:")
            report.append("-" * 100)
            report.append(f"• 年化收益率: {recommended_metrics['annual_return']:+.1%}")
            report.append(f"• 年化波动率: {recommended_metrics['annual_volatility']:.1%}")
            report.append(f"• 夏普比率: {recommended_metrics['sharpe_ratio']:.2f}")
            report.append(f"• 最大回撤: {recommended_metrics['max_drawdown']:+.1%}")
            
            report.append(f"\n推荐持仓权重分布 (前10大):")
            sorted_weights = sorted(recommended_metrics['weights'].items(), key=lambda x: x[1], reverse=True)
            for symbol, weight in sorted_weights[:10]:
                report.append(f"  • {symbol}: {weight:.1%}")
        
        # 对比分析
        if comparison:
            report.append(f"\n⚖️ 历史表现对比:")
            report.append("-" * 100)
            
            annual_return_diff = comparison['annual_return']['difference']
            volatility_diff = comparison['annual_volatility']['difference']
            sharpe_diff = comparison['sharpe_ratio']['difference']
            drawdown_diff = comparison['max_drawdown']['difference']
            
            report.append(f"• 年化收益差异: {annual_return_diff:+.1%} "
                         f"({'推荐方案更优' if annual_return_diff > 0 else '当前方案更优' if annual_return_diff < 0 else '基本相当'})")
            
            report.append(f"• 波动率差异: {volatility_diff:+.1%} "
                         f"({'推荐方案风险更高' if volatility_diff > 0 else '推荐方案风险更低' if volatility_diff < 0 else '风险相当'})")
            
            report.append(f"• 夏普比率差异: {sharpe_diff:+.2f} "
                         f"({'推荐方案效率更高' if sharpe_diff > 0 else '当前方案效率更高' if sharpe_diff < 0 else '效率相当'})")
            
            report.append(f"• 最大回撤差异: {drawdown_diff:+.1%} "
                         f"({'推荐方案回撤更大' if drawdown_diff < 0 else '推荐方案回撤更小' if drawdown_diff > 0 else '回撤相当'})")
        
        # 蒙特卡洛预测结果
        report.append(f"\n🎲 蒙特卡洛模拟预测 (未来1年):")
        report.append("-" * 100)
        
        if current_simulations:
            report.append(f"\n【当前持仓预测】")
            report.append(f"• 预期收益率: {current_simulations['mean_return']:.1%}")
            report.append(f"• 中位数收益率: {current_simulations['median_return']:.1%}")
            report.append(f"• 收益率标准差: {current_simulations['std_return']:.1%}")
            report.append(f"• 95%置信区间: [{current_simulations['percentile_5']:.1%}, {current_simulations['percentile_95']:.1%}]")
            report.append(f"• 盈利概率: {current_simulations['probability_positive']:.1%}")
            report.append(f"• 达到20%收益概率: {current_simulations['probability_target_20']:.1%} ⭐")
        
        if recommended_simulations:
            report.append(f"\n【推荐持仓预测】")
            report.append(f"• 预期收益率: {recommended_simulations['mean_return']:.1%}")
            report.append(f"• 中位数收益率: {recommended_simulations['median_return']:.1%}")
            report.append(f"• 收益率标准差: {recommended_simulations['std_return']:.1%}")
            report.append(f"• 95%置信区间: [{recommended_simulations['percentile_5']:.1%}, {recommended_simulations['percentile_95']:.1%}]")
            report.append(f"• 盈利概率: {recommended_simulations['probability_positive']:.1%}")
            report.append(f"• 达到20%收益概率: {recommended_simulations['probability_target_20']:.1%} ⭐")
        
        # 客观结论
        report.append(f"\n🔍 客观分析结论:")
        report.append("-" * 100)
        
        if current_simulations and recommended_simulations:
            current_20_prob = current_simulations['probability_target_20']
            recommended_20_prob = recommended_simulations['probability_target_20']
            
            report.append(f"• 20%收益目标实现概率对比:")
            report.append(f"  - 当前持仓: {current_20_prob:.1%}")
            report.append(f"  - 推荐持仓: {recommended_20_prob:.1%}")
            report.append(f"  - 概率提升: {recommended_20_prob - current_20_prob:+.1%}")
            
            if recommended_20_prob > current_20_prob:
                if recommended_20_prob >= 0.5:
                    report.append(f"✅ 推荐方案有较高概率({recommended_20_prob:.1%})达到20%年化收益目标")
                else:
                    report.append(f"🔶 推荐方案虽优于当前方案，但达到20%目标概率仍较低({recommended_20_prob:.1%})")
            else:
                report.append(f"❌ 推荐方案达到20%目标的概率低于当前方案")
            
            # 风险收益权衡
            current_expected = current_simulations['mean_return']
            recommended_expected = recommended_simulations['mean_return']
            current_risk = current_simulations['std_return']
            recommended_risk = recommended_simulations['std_return']
            
            report.append(f"\n• 风险收益权衡:")
            report.append(f"  - 预期收益提升: {recommended_expected - current_expected:+.1%}")
            report.append(f"  - 风险变化: {recommended_risk - current_risk:+.1%}")
            
            if recommended_expected > current_expected and recommended_risk < current_risk:
                report.append(f"🎯 推荐方案实现了更高收益和更低风险的帕累托改进")
            elif recommended_expected > current_expected:
                report.append(f"📈 推荐方案提升收益，但风险也相应增加")
            else:
                report.append(f"⚠️ 推荐方案在预期收益方面未显示明显优势")
        
        # 模型局限性说明
        report.append(f"\n⚠️ 预测模型局限性:")
        report.append("-" * 100)
        report.append(f"• 基于历史数据，未来表现可能与历史不符")
        report.append(f"• 假设收益率服从正态分布，实际市场存在厚尾风险")
        report.append(f"• 未考虑宏观经济、政策变化等外部因素")
        report.append(f"• 交易成本、税收等实际因素未纳入模型")
        report.append(f"• 相关性假设为静态，实际相关性会动态变化")
        
        report.append(f"\n💡 投资建议:")
        report.append("-" * 100)
        if current_simulations and recommended_simulations:
            if recommended_simulations['probability_target_20'] > 0.4:
                report.append(f"• 基于模型预测，推荐方案有合理概率实现20%目标")
                report.append(f"• 建议采用推荐配置，但需要定期审查和调整")
            else:
                report.append(f"• 20%年化收益目标较为激进，需要承担相应风险")
                report.append(f"• 建议适当降低收益预期或增加风险承受能力")
        
        report.append(f"• 无论采用哪种方案，都应该:")
        report.append(f"  - 定期再平衡投资组合")
        report.append(f"  - 根据市场变化调整策略")
        report.append(f"  - 保持充足的现金储备")
        report.append(f"  - 严格执行止损纪律")
        
        report.append("\n" + "=" * 120)
        
        return '\n'.join(report)
    
    def create_visualization(self, current_simulations, recommended_simulations):
        """创建可视化图表"""
        if not current_simulations or not recommended_simulations:
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('投资组合收益预测对比', fontsize=16, fontweight='bold')
        
        # 1. 收益率分布对比
        axes[0, 0].hist(current_simulations['all_results'], bins=50, alpha=0.7, label='当前持仓', density=True)
        axes[0, 0].hist(recommended_simulations['all_results'], bins=50, alpha=0.7, label='推荐持仓', density=True)
        axes[0, 0].axvline(0.2, color='red', linestyle='--', label='20%目标线')
        axes[0, 0].set_xlabel('年化收益率')
        axes[0, 0].set_ylabel('概率密度')
        axes[0, 0].set_title('收益率分布对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 概率对比
        metrics = ['盈利概率', '达到20%概率']
        current_probs = [current_simulations['probability_positive'], current_simulations['probability_target_20']]
        recommended_probs = [recommended_simulations['probability_positive'], recommended_simulations['probability_target_20']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, current_probs, width, label='当前持仓', alpha=0.8)
        axes[0, 1].bar(x + width/2, recommended_probs, width, label='推荐持仓', alpha=0.8)
        axes[0, 1].set_ylabel('概率')
        axes[0, 1].set_title('关键概率指标对比')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 风险收益散点图
        current_return = current_simulations['mean_return']
        current_risk = current_simulations['std_return']
        recommended_return = recommended_simulations['mean_return']
        recommended_risk = recommended_simulations['std_return']
        
        axes[1, 0].scatter(current_risk, current_return, s=100, label='当前持仓', alpha=0.8)
        axes[1, 0].scatter(recommended_risk, recommended_return, s=100, label='推荐持仓', alpha=0.8)
        axes[1, 0].axhline(0.2, color='red', linestyle='--', alpha=0.7, label='20%目标')
        axes[1, 0].set_xlabel('风险 (标准差)')
        axes[1, 0].set_ylabel('预期收益率')
        axes[1, 0].set_title('风险收益对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 置信区间对比
        current_ci = [current_simulations['percentile_5'], current_simulations['percentile_95']]
        recommended_ci = [recommended_simulations['percentile_5'], recommended_simulations['percentile_95']]
        
        axes[1, 1].errorbar(['当前持仓'], [current_simulations['mean_return']], 
                           yerr=[[current_simulations['mean_return'] - current_ci[0]], 
                                [current_ci[1] - current_simulations['mean_return']]], 
                           fmt='o', capsize=5, capthick=2, label='当前持仓')
        
        axes[1, 1].errorbar(['推荐持仓'], [recommended_simulations['mean_return']], 
                           yerr=[[recommended_simulations['mean_return'] - recommended_ci[0]], 
                                [recommended_ci[1] - recommended_simulations['mean_return']]], 
                           fmt='o', capsize=5, capthick=2, label='推荐持仓')
        
        axes[1, 1].axhline(0.2, color='red', linestyle='--', alpha=0.7, label='20%目标')
        axes[1, 1].set_ylabel('收益率')
        axes[1, 1].set_title('95%置信区间对比')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 确保temp_pic目录存在
        import os
        os.makedirs('temp_pic', exist_ok=True)
        
        # 保存图表到temp_pic目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'temp_pic/portfolio_prediction_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"📊 可视化图表已保存: temp_pic/portfolio_prediction_comparison_{timestamp}.png")

def main():
    """主函数"""
    predictor = PortfolioPerformancePredictor()
    
    # 获取所有股票的历史数据
    all_symbols = list(set(list(predictor.current_portfolio.keys()) + list(predictor.recommended_portfolio.keys())))
    historical_data = predictor.get_historical_data(all_symbols)
    
    if len(historical_data) >= 5:
        # 计算当前持仓指标
        current_metrics = predictor.calculate_portfolio_metrics(predictor.current_portfolio, historical_data)
        
        # 计算推荐持仓指标
        recommended_metrics = predictor.calculate_portfolio_metrics(predictor.recommended_portfolio, historical_data)
        
        # 对比分析
        comparison = predictor.compare_portfolios(current_metrics, recommended_metrics)
        
        # 蒙特卡洛模拟
        current_simulations = predictor.monte_carlo_simulation(current_metrics, 252) if current_metrics else None
        recommended_simulations = predictor.monte_carlo_simulation(recommended_metrics, 252) if recommended_metrics else None
        
        # 生成报告
        report = predictor.generate_prediction_report(
            current_metrics, recommended_metrics, 
            current_simulations, recommended_simulations, comparison)
        
        print(report)
        
        # 创建可视化
        if current_simulations and recommended_simulations:
            predictor.create_visualization(current_simulations, recommended_simulations)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细数据
        output_data = {
            'timestamp': timestamp,
            'current_portfolio': predictor.current_portfolio,
            'recommended_portfolio': predictor.recommended_portfolio,
            'current_metrics': current_metrics,
            'recommended_metrics': recommended_metrics,
            'comparison': comparison,
            'current_simulations': current_simulations,
            'recommended_simulations': recommended_simulations
        }
        
        with open(f'portfolio_prediction_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        with open(f'portfolio_prediction_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("✅ 投资组合收益预测分析完成，报告已保存")
        
    else:
        logger.error("❌ 无法获取足够的历史数据进行分析")

if __name__ == "__main__":
    main() 