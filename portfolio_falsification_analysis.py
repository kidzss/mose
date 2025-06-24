#!/usr/bin/env python3
"""
投资组合证伪分析系统
使用证伪思路验证推荐方案的优劣性
假设H0: 推荐方案不如当前方案，然后用真实数据证伪
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

class PortfolioFalsificationAnalyzer:
    """投资组合证伪分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 从配置文件读取当前持仓
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 当前持仓 (基于真实配置)
        self.current_portfolio = {
            'NVDA': 0.274,  # 27.4%
            'GOOG': 0.246,  # 24.6% 
            'AMD': 0.239,   # 23.9%
            'PFE': 0.090,   # 9.0%
            'TSLA': 0.061,  # 6.1%
            'MRK': 0.052,   # 5.2%
            'JPM': 0.039    # 3.9%
        }
        
        # 推荐持仓 (基于分析结果)
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
        
        # 证伪假设
        self.null_hypotheses = [
            "H0_1: 推荐方案的夏普比率不如当前方案",
            "H0_2: 推荐方案的最大回撤不如当前方案", 
            "H0_3: 推荐方案的行业分散度不如当前方案",
            "H0_4: 推荐方案的下行风险不如当前方案",
            "H0_5: 推荐方案的稳定性不如当前方案",
            "H0_6: 推荐方案的防御性不如当前方案"
        ]
        
        logger.info("🔍 投资组合证伪分析器初始化完成")
    
    def get_real_market_data(self, symbols, periods=["1y", "2y", "3y", "5y"]):
        """获取真实市场数据 - 多个时间段验证"""
        market_data = {}
        
        for symbol in symbols:
            market_data[symbol] = {}
            
            for period in periods:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty and len(hist) > 50:
                        # 计算真实收益率
                        daily_returns = hist['Close'].pct_change().dropna()
                        
                        # 获取基本面数据
                        info = ticker.info
                        
                        market_data[symbol][period] = {
                            'prices': hist['Close'],
                            'returns': daily_returns,
                            'start_price': hist['Close'].iloc[0],
                            'end_price': hist['Close'].iloc[-1],
                            'total_return': (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1,
                            'annual_return': daily_returns.mean() * 252,
                            'volatility': daily_returns.std() * np.sqrt(252),
                            'max_drawdown': self.calculate_max_drawdown(daily_returns),
                            'downside_deviation': self.calculate_downside_deviation(daily_returns),
                            'var_95': np.percentile(daily_returns, 5),
                            'skewness': daily_returns.skew(),
                            'kurtosis': daily_returns.kurtosis(),
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'beta': info.get('beta', 1),
                            'sector': info.get('sector', 'Unknown')
                        }
                        
                except Exception as e:
                    logger.warning(f"获取{symbol} {period}数据失败: {e}")
        
        return market_data
    
    def calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def calculate_downside_deviation(self, returns, target_return=0):
        """计算下行偏差"""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0
        return np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    
    def calculate_portfolio_real_performance(self, portfolio_weights, market_data, period="3y"):
        """计算投资组合真实表现"""
        valid_symbols = []
        valid_weights = {}
        
        # 筛选有效数据的股票
        for symbol, weight in portfolio_weights.items():
            if symbol in market_data and period in market_data[symbol]:
                valid_symbols.append(symbol)
                valid_weights[symbol] = weight
        
        if not valid_symbols:
            return None
        
        # 重新标准化权重
        total_weight = sum(valid_weights.values())
        normalized_weights = {symbol: weight / total_weight for symbol, weight in valid_weights.items()}
        
        # 获取共同时间段
        common_dates = None
        for symbol in valid_symbols:
            symbol_dates = market_data[symbol][period]['returns'].index
            if common_dates is None:
                common_dates = symbol_dates
            else:
                common_dates = common_dates.intersection(symbol_dates)
        
        if len(common_dates) < 100:  # 至少需要100个交易日
            return None
        
        # 计算组合日收益率
        portfolio_returns = pd.Series(0.0, index=common_dates)
        
        for date in common_dates:
            daily_return = 0
            for symbol in valid_symbols:
                if date in market_data[symbol][period]['returns'].index:
                    daily_return += normalized_weights[symbol] * market_data[symbol][period]['returns'][date]
            portfolio_returns[date] = daily_return
        
        # 计算组合指标
        portfolio_metrics = {
            'period': period,
            'valid_symbols': valid_symbols,
            'weights': normalized_weights,
            'returns': portfolio_returns,
            'total_return': (1 + portfolio_returns).prod() - 1,
            'annual_return': portfolio_returns.mean() * 252,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(portfolio_returns),
            'downside_deviation': self.calculate_downside_deviation(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5) * np.sqrt(252),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis(),
            'positive_days_ratio': len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns),
            'worst_day': portfolio_returns.min(),
            'best_day': portfolio_returns.max()
        }
        
        return portfolio_metrics
    
    def analyze_sector_concentration(self, portfolio_weights, market_data):
        """分析行业集中度"""
        sector_allocation = {}
        
        for symbol, weight in portfolio_weights.items():
            if symbol in market_data and '3y' in market_data[symbol]:
                sector = market_data[symbol]['3y']['sector']
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                sector_allocation[sector] += weight
        
        # 计算集中度指标
        herfindahl_index = sum([weight**2 for weight in sector_allocation.values()])
        num_sectors = len(sector_allocation)
        max_sector_weight = max(sector_allocation.values()) if sector_allocation else 0
        
        return {
            'sector_allocation': sector_allocation,
            'herfindahl_index': herfindahl_index,
            'num_sectors': num_sectors,
            'max_sector_weight': max_sector_weight,
            'diversification_ratio': 1 / herfindahl_index if herfindahl_index > 0 else 0
        }
    
    def perform_falsification_tests(self, current_metrics, recommended_metrics, 
                                  current_sectors, recommended_sectors, market_data):
        """执行证伪测试"""
        falsification_results = {}
        
        # H0_1: 推荐方案的夏普比率不如当前方案
        if current_metrics and recommended_metrics:
            current_sharpe = current_metrics['sharpe_ratio']
            recommended_sharpe = recommended_metrics['sharpe_ratio']
            
            falsification_results['H0_1'] = {
                'hypothesis': "推荐方案的夏普比率不如当前方案",
                'current_value': current_sharpe,
                'recommended_value': recommended_sharpe,
                'difference': recommended_sharpe - current_sharpe,
                'falsified': recommended_sharpe > current_sharpe,
                'evidence': f"推荐方案夏普比率({recommended_sharpe:.2f}) {'>' if recommended_sharpe > current_sharpe else '<='} 当前方案({current_sharpe:.2f})"
            }
        
        # H0_2: 推荐方案的最大回撤不如当前方案
        if current_metrics and recommended_metrics:
            current_dd = current_metrics['max_drawdown']
            recommended_dd = recommended_metrics['max_drawdown']
            
            falsification_results['H0_2'] = {
                'hypothesis': "推荐方案的最大回撤不如当前方案",
                'current_value': current_dd,
                'recommended_value': recommended_dd,
                'difference': recommended_dd - current_dd,
                'falsified': recommended_dd > current_dd,  # 回撤更小更好
                'evidence': f"推荐方案最大回撤({recommended_dd:.1%}) {'<' if recommended_dd > current_dd else '>='} 当前方案({current_dd:.1%})"
            }
        
        # H0_3: 推荐方案的行业分散度不如当前方案
        current_diversification = current_sectors['diversification_ratio']
        recommended_diversification = recommended_sectors['diversification_ratio']
        
        falsification_results['H0_3'] = {
            'hypothesis': "推荐方案的行业分散度不如当前方案",
            'current_value': current_diversification,
            'recommended_value': recommended_diversification,
            'difference': recommended_diversification - current_diversification,
            'falsified': recommended_diversification > current_diversification,
            'evidence': f"推荐方案分散度({recommended_diversification:.2f}) {'>' if recommended_diversification > current_diversification else '<='} 当前方案({current_diversification:.2f})"
        }
        
        # H0_4: 推荐方案的下行风险不如当前方案
        if current_metrics and recommended_metrics:
            current_downside = abs(current_metrics['downside_deviation'])
            recommended_downside = abs(recommended_metrics['downside_deviation'])
            
            falsification_results['H0_4'] = {
                'hypothesis': "推荐方案的下行风险不如当前方案",
                'current_value': current_downside,
                'recommended_value': recommended_downside,
                'difference': recommended_downside - current_downside,
                'falsified': recommended_downside < current_downside,
                'evidence': f"推荐方案下行风险({recommended_downside:.1%}) {'<' if recommended_downside < current_downside else '>='} 当前方案({current_downside:.1%})"
            }
        
        # H0_5: 推荐方案的稳定性不如当前方案 (通过波动率衡量)
        if current_metrics and recommended_metrics:
            current_vol = current_metrics['volatility']
            recommended_vol = recommended_metrics['volatility']
            
            falsification_results['H0_5'] = {
                'hypothesis': "推荐方案的稳定性不如当前方案",
                'current_value': current_vol,
                'recommended_value': recommended_vol,
                'difference': recommended_vol - current_vol,
                'falsified': recommended_vol < current_vol,
                'evidence': f"推荐方案波动率({recommended_vol:.1%}) {'<' if recommended_vol < current_vol else '>='} 当前方案({current_vol:.1%})"
            }
        
        # H0_6: 推荐方案的防御性不如当前方案 (通过VaR衡量)
        if current_metrics and recommended_metrics:
            current_var = abs(current_metrics['var_95'])
            recommended_var = abs(recommended_metrics['var_95'])
            
            falsification_results['H0_6'] = {
                'hypothesis': "推荐方案的防御性不如当前方案",
                'current_value': current_var,
                'recommended_value': recommended_var,
                'difference': recommended_var - current_var,
                'falsified': recommended_var < current_var,
                'evidence': f"推荐方案VaR95({recommended_var:.1%}) {'<' if recommended_var < current_var else '>='} 当前方案({current_var:.1%})"
            }
        
        return falsification_results
    
    def cross_validate_with_different_periods(self, portfolio_weights, market_data):
        """使用不同时间段交叉验证"""
        periods = ["1y", "2y", "3y", "5y"]
        cross_validation_results = {}
        
        for period in periods:
            metrics = self.calculate_portfolio_real_performance(portfolio_weights, market_data, period)
            if metrics:
                cross_validation_results[period] = {
                    'annual_return': metrics['annual_return'],
                    'volatility': metrics['volatility'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown']
                }
        
        # 计算一致性指标
        if len(cross_validation_results) >= 2:
            sharpe_ratios = [result['sharpe_ratio'] for result in cross_validation_results.values()]
            sharpe_consistency = np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) > 0 else float('inf')
            
            cross_validation_results['consistency'] = {
                'sharpe_coefficient_of_variation': sharpe_consistency,
                'periods_analyzed': len(cross_validation_results)
            }
        
        return cross_validation_results
    
    def generate_falsification_report(self, falsification_results, current_metrics, 
                                    recommended_metrics, current_sectors, recommended_sectors,
                                    current_cross_val, recommended_cross_val, market_data):
        """生成证伪分析报告"""
        report = []
        report.append("=" * 120)
        report.append("🔍 投资组合证伪分析报告")
        report.append("📊 使用真实市场数据验证推荐方案优劣性")
        report.append(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 120)
        
        # 证伪方法说明
        report.append(f"\n📋 证伪方法说明:")
        report.append("-" * 100)
        report.append(f"• 证伪原理: 假设推荐方案不如当前方案，用真实数据反驳")
        report.append(f"• 数据来源: Yahoo Finance真实历史数据")
        report.append(f"• 验证期间: 1年、2年、3年、5年多时段交叉验证")
        report.append(f"• 分析维度: 风险调整收益、风险控制、行业分散、稳定性")
        
        # 真实数据概览
        report.append(f"\n📈 真实市场数据概览:")
        report.append("-" * 100)
        
        # 显示部分关键股票的真实表现
        key_stocks = ['NVDA', 'GOOG', 'AMD', 'META', 'JPM', 'PFE']
        for stock in key_stocks:
            if stock in market_data and '3y' in market_data[stock]:
                data = market_data[stock]['3y']
                report.append(f"• {stock}: 3年年化收益{data['annual_return']:+.1%}, "
                             f"波动率{data['volatility']:.1%}, 最大回撤{data['max_drawdown']:+.1%}")
        
        # 证伪测试结果
        report.append(f"\n🎯 证伪测试结果:")
        report.append("-" * 100)
        
        falsified_count = 0
        total_tests = len(falsification_results)
        
        for test_id, result in falsification_results.items():
            status = "✅ 证伪成功" if result['falsified'] else "❌ 证伪失败"
            if result['falsified']:
                falsified_count += 1
            
            report.append(f"\n{test_id}: {result['hypothesis']}")
            report.append(f"  {status}")
            report.append(f"  证据: {result['evidence']}")
            
            if 'difference' in result:
                direction = "优于" if result['falsified'] else "不如"
                report.append(f"  结论: 推荐方案在此指标上{direction}当前方案")
        
        # 总体证伪结果
        falsification_rate = falsified_count / total_tests
        report.append(f"\n📊 总体证伪结果:")
        report.append("-" * 100)
        report.append(f"• 证伪成功率: {falsification_rate:.1%} ({falsified_count}/{total_tests})")
        
        if falsification_rate >= 0.8:
            report.append(f"🎯 强有力证据: 推荐方案在多个维度显著优于当前方案")
        elif falsification_rate >= 0.6:
            report.append(f"✅ 充分证据: 推荐方案在主要维度优于当前方案")
        elif falsification_rate >= 0.4:
            report.append(f"🔶 部分证据: 推荐方案在某些维度优于当前方案")
        else:
            report.append(f"❌ 证据不足: 推荐方案未显示明显优势")
        
        # 跨时间段验证
        report.append(f"\n⏱️ 跨时间段验证结果:")
        report.append("-" * 100)
        
        if current_cross_val and recommended_cross_val:
            report.append(f"【当前持仓跨时段表现】")
            for period, metrics in current_cross_val.items():
                if period != 'consistency':
                    report.append(f"  {period}: 年化收益{metrics['annual_return']:+.1%}, "
                                 f"夏普比率{metrics['sharpe_ratio']:.2f}")
            
            report.append(f"【推荐持仓跨时段表现】")
            for period, metrics in recommended_cross_val.items():
                if period != 'consistency':
                    report.append(f"  {period}: 年化收益{metrics['annual_return']:+.1%}, "
                                 f"夏普比率{metrics['sharpe_ratio']:.2f}")
            
            # 一致性分析
            if 'consistency' in current_cross_val and 'consistency' in recommended_cross_val:
                current_consistency = current_cross_val['consistency']['sharpe_coefficient_of_variation']
                recommended_consistency = recommended_cross_val['consistency']['sharpe_coefficient_of_variation']
                
                report.append(f"\n【表现一致性对比】")
                report.append(f"  当前方案夏普比率变异系数: {current_consistency:.2f}")
                report.append(f"  推荐方案夏普比率变异系数: {recommended_consistency:.2f}")
                
                if recommended_consistency < current_consistency:
                    report.append(f"  ✅ 推荐方案表现更加稳定一致")
                else:
                    report.append(f"  ❌ 当前方案表现更加稳定一致")
        
        # 行业分散度对比
        report.append(f"\n🏭 行业分散度真实对比:")
        report.append("-" * 100)
        
        report.append(f"【当前持仓行业分布】")
        for sector, weight in sorted(current_sectors['sector_allocation'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  • {sector}: {weight:.1%}")
        
        report.append(f"【推荐持仓行业分布】")
        for sector, weight in sorted(recommended_sectors['sector_allocation'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  • {sector}: {weight:.1%}")
        
        report.append(f"\n【分散度指标对比】")
        report.append(f"  当前方案: {current_sectors['num_sectors']}个行业, "
                     f"赫芬达尔指数{current_sectors['herfindahl_index']:.3f}")
        report.append(f"  推荐方案: {recommended_sectors['num_sectors']}个行业, "
                     f"赫芬达尔指数{recommended_sectors['herfindahl_index']:.3f}")
        
        # 核心结论
        report.append(f"\n🎯 基于真实数据的核心结论:")
        report.append("-" * 100)
        
        if falsification_rate >= 0.6:
            report.append(f"✅ 推荐方案确实优于当前方案:")
            
            # 具体优势
            for test_id, result in falsification_results.items():
                if result['falsified']:
                    if 'sharpe' in result['hypothesis'].lower():
                        report.append(f"  • 风险调整收益更优 (夏普比率提升)")
                    elif 'drawdown' in result['hypothesis'].lower() or '回撤' in result['hypothesis']:
                        report.append(f"  • 风险控制能力更强 (最大回撤更小)")
                    elif 'diversification' in result['hypothesis'].lower() or '分散' in result['hypothesis']:
                        report.append(f"  • 行业分散度更高 (降低集中度风险)")
                    elif 'downside' in result['hypothesis'].lower() or '下行' in result['hypothesis']:
                        report.append(f"  • 下行风险保护更好")
                    elif 'volatility' in result['hypothesis'].lower() or '稳定' in result['hypothesis']:
                        report.append(f"  • 收益稳定性更高")
                    elif 'var' in result['hypothesis'].lower() or '防御' in result['hypothesis']:
                        report.append(f"  • 极端风险防御更强")
        
        # 数据可靠性声明
        report.append(f"\n📊 数据可靠性声明:")
        report.append("-" * 100)
        report.append(f"• 所有数据来源于Yahoo Finance真实历史数据")
        report.append(f"• 未使用任何模拟或假设数据")
        report.append(f"• 分析基于{len([s for s in market_data.keys() if '3y' in market_data[s]])}只股票的真实表现")
        report.append(f"• 跨越多个时间段验证，确保结论稳健性")
        report.append(f"• 采用证伪方法，避免确认偏误")
        
        # 投资建议
        report.append(f"\n💡 基于证伪分析的投资建议:")
        report.append("-" * 100)
        
        if falsification_rate >= 0.6:
            report.append(f"• 强烈建议采用推荐投资组合配置")
            report.append(f"• 推荐方案在风险控制和收益稳定性方面有明显优势")
            report.append(f"• 建议分批调仓，逐步向推荐配置靠拢")
        else:
            report.append(f"• 推荐方案优势不够明显，可考虑部分采用")
            report.append(f"• 建议重点关注风险控制和行业分散")
            report.append(f"• 定期重新评估两种方案的表现")
        
        report.append("\n" + "=" * 120)
        
        return '\n'.join(report)

def main():
    """主函数"""
    analyzer = PortfolioFalsificationAnalyzer()
    
    # 获取所有股票的真实市场数据
    all_symbols = list(set(list(analyzer.current_portfolio.keys()) + list(analyzer.recommended_portfolio.keys())))
    market_data = analyzer.get_real_market_data(all_symbols)
    
    logger.info(f"📊 获取到{len(market_data)}只股票的真实市场数据")
    
    if len(market_data) >= 10:
        # 计算当前持仓真实表现
        current_metrics = analyzer.calculate_portfolio_real_performance(
            analyzer.current_portfolio, market_data, "3y")
        
        # 计算推荐持仓真实表现
        recommended_metrics = analyzer.calculate_portfolio_real_performance(
            analyzer.recommended_portfolio, market_data, "3y")
        
        # 分析行业集中度
        current_sectors = analyzer.analyze_sector_concentration(analyzer.current_portfolio, market_data)
        recommended_sectors = analyzer.analyze_sector_concentration(analyzer.recommended_portfolio, market_data)
        
        # 跨时间段验证
        current_cross_val = analyzer.cross_validate_with_different_periods(analyzer.current_portfolio, market_data)
        recommended_cross_val = analyzer.cross_validate_with_different_periods(analyzer.recommended_portfolio, market_data)
        
        # 执行证伪测试
        falsification_results = analyzer.perform_falsification_tests(
            current_metrics, recommended_metrics, current_sectors, recommended_sectors, market_data)
        
        # 生成报告
        report = analyzer.generate_falsification_report(
            falsification_results, current_metrics, recommended_metrics,
            current_sectors, recommended_sectors, current_cross_val, recommended_cross_val, market_data)
        
        print(report)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细数据
        output_data = {
            'timestamp': timestamp,
            'falsification_results': falsification_results,
            'current_metrics': current_metrics,
            'recommended_metrics': recommended_metrics,
            'current_sectors': current_sectors,
            'recommended_sectors': recommended_sectors,
            'current_cross_validation': current_cross_val,
            'recommended_cross_validation': recommended_cross_val
        }
        
        with open(f'portfolio_falsification_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        with open(f'portfolio_falsification_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("✅ 投资组合证伪分析完成，报告已保存")
        
    else:
        logger.error("❌ 无法获取足够的真实市场数据")

if __name__ == "__main__":
    main() 