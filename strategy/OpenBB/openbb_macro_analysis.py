import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from openbb import obb
from ...strategy.strategy_base import Strategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MacroAnalysis:
    """
    使用OpenBB进行宏观经济分析的工具类
    """
    
    def __init__(self, output_dir: str = None):
        """
        初始化宏观分析工具
        
        Args:
            output_dir: 输出目录，用于保存图表和报告
        """
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def analyze_economic_indicators(self) -> Dict[str, Any]:
        """
        分析关键经济指标
        
        Returns:
            Dict: 包含经济指标分析结果的字典
        """
        results = {
            'timestamp': datetime.now(),
            'indicators': {},
            'summary': {},
        }
        
        try:
            # 获取GDP增长率
            gdp_data = obb.economy.gdp.real().to_df()
            if not gdp_data.empty:
                gdp_growth = gdp_data['value'].pct_change().iloc[-1] * 100 if len(gdp_data) > 1 else None
                results['indicators']['gdp_growth'] = gdp_growth
                
                # GDP增长趋势
                if len(gdp_data) > 4:
                    gdp_trend = gdp_data['value'].pct_change().iloc[-4:].mean() * 100
                    results['indicators']['gdp_trend'] = gdp_trend
            
            # 获取通胀率(CPI)
            cpi_data = obb.economy.cpi().to_df()
            if not cpi_data.empty:
                cpi_latest = cpi_data['value'].iloc[-1]
                cpi_yoy = ((cpi_data['value'].iloc[-1] / cpi_data['value'].iloc[-13]) - 1) * 100 if len(cpi_data) >= 13 else None
                results['indicators']['cpi_latest'] = cpi_latest
                results['indicators']['cpi_yoy'] = cpi_yoy
            
            # 获取失业率
            unemployment_data = obb.economy.unemployment().to_df()
            if not unemployment_data.empty:
                unemployment_latest = unemployment_data['value'].iloc[-1]
                unemployment_trend = unemployment_data['value'].diff().iloc[-3:].mean() if len(unemployment_data) >= 3 else None
                results['indicators']['unemployment_latest'] = unemployment_latest
                results['indicators']['unemployment_trend'] = unemployment_trend
            
            # 获取联邦基金利率
            fed_rate_data = obb.economy.fred_series(series_id="FEDFUNDS").to_df()
            if not fed_rate_data.empty:
                fed_rate_latest = fed_rate_data['value'].iloc[-1]
                fed_rate_trend = fed_rate_data['value'].diff().iloc[-3:].mean() if len(fed_rate_data) >= 3 else None
                results['indicators']['fed_rate_latest'] = fed_rate_latest
                results['indicators']['fed_rate_trend'] = fed_rate_trend
            
            # 获取消费者信心指数
            try:
                consumer_sentiment_data = obb.economy.fred_series(series_id="UMCSENT").to_df()
                if not consumer_sentiment_data.empty:
                    consumer_sentiment_latest = consumer_sentiment_data['value'].iloc[-1]
                    consumer_sentiment_trend = consumer_sentiment_data['value'].diff().iloc[-3:].mean() if len(consumer_sentiment_data) >= 3 else None
                    results['indicators']['consumer_sentiment_latest'] = consumer_sentiment_latest
                    results['indicators']['consumer_sentiment_trend'] = consumer_sentiment_trend
            except:
                logger.warning("获取消费者信心指数时出错")
            
            # 基于数据创建经济状况摘要
            summary = self._create_economic_summary(results['indicators'])
            results['summary'] = summary
            
            return results
            
        except Exception as e:
            logger.error(f"分析经济指标时出错: {str(e)}")
            results['error'] = str(e)
            return results
    
    def _create_economic_summary(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """
        根据经济指标创建经济状况摘要
        
        Args:
            indicators: 经济指标字典
            
        Returns:
            Dict: 经济状况摘要
        """
        summary = {
            'economic_health': None,  # 经济健康状况: strong, moderate, weak
            'inflation_status': None,  # 通胀状况: high, moderate, low
            'employment_status': None,  # 就业状况: strong, moderate, weak
            'monetary_policy': None,  # 货币政策: tightening, neutral, easing
            'consumer_health': None,  # 消费者健康状况: strong, moderate, weak
            'outlook': None,  # 经济展望: positive, neutral, negative
        }
        
        # 经济健康状况
        gdp_growth = indicators.get('gdp_growth')
        if gdp_growth is not None:
            if gdp_growth > 2.5:
                summary['economic_health'] = 'strong'
            elif gdp_growth > 0:
                summary['economic_health'] = 'moderate'
            else:
                summary['economic_health'] = 'weak'
        
        # 通胀状况
        cpi_yoy = indicators.get('cpi_yoy')
        if cpi_yoy is not None:
            if cpi_yoy > 4:
                summary['inflation_status'] = 'high'
            elif cpi_yoy > 2:
                summary['inflation_status'] = 'moderate'
            else:
                summary['inflation_status'] = 'low'
        
        # 就业状况
        unemployment_latest = indicators.get('unemployment_latest')
        if unemployment_latest is not None:
            if unemployment_latest < 4:
                summary['employment_status'] = 'strong'
            elif unemployment_latest < 6:
                summary['employment_status'] = 'moderate'
            else:
                summary['employment_status'] = 'weak'
        
        # 货币政策
        fed_rate_trend = indicators.get('fed_rate_trend')
        if fed_rate_trend is not None:
            if fed_rate_trend > 0.1:
                summary['monetary_policy'] = 'tightening'
            elif fed_rate_trend < -0.1:
                summary['monetary_policy'] = 'easing'
            else:
                summary['monetary_policy'] = 'neutral'
        
        # 消费者健康状况
        consumer_sentiment_latest = indicators.get('consumer_sentiment_latest')
        if consumer_sentiment_latest is not None:
            if consumer_sentiment_latest > 90:
                summary['consumer_health'] = 'strong'
            elif consumer_sentiment_latest > 75:
                summary['consumer_health'] = 'moderate'
            else:
                summary['consumer_health'] = 'weak'
        
        # 经济展望
        economic_factors = [
            summary.get('economic_health'),
            summary.get('employment_status'),
            summary.get('consumer_health')
        ]
        
        positive_count = economic_factors.count('strong')
        negative_count = economic_factors.count('weak')
        
        if positive_count > negative_count:
            summary['outlook'] = 'positive'
        elif negative_count > positive_count:
            summary['outlook'] = 'negative'
        else:
            summary['outlook'] = 'neutral'
        
        return summary
    
    def plot_economic_indicators(self, save_to_file: bool = True) -> Optional[plt.Figure]:
        """
        绘制关键经济指标
        
        Args:
            save_to_file: 是否保存图表到文件
            
        Returns:
            matplotlib Figure对象
        """
        try:
            # 创建一个包含多个子图的图表
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('宏观经济指标分析', fontsize=16)
            
            # 获取数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)  # 3年数据
            
            # 1. GDP增长率
            try:
                gdp_data = obb.economy.gdp.real().to_df()
                if not gdp_data.empty:
                    gdp_data['growth'] = gdp_data['value'].pct_change() * 100
                    gdp_data.plot(y='growth', ax=axes[0, 0], title='GDP年增长率 (%)')
                    axes[0, 0].set_ylabel('增长率 (%)')
                    axes[0, 0].grid(True)
            except Exception as e:
                logger.warning(f"绘制GDP增长率时出错: {str(e)}")
            
            # 2. 通胀率(CPI)
            try:
                cpi_data = obb.economy.cpi().to_df()
                if not cpi_data.empty:
                    # 计算同比增长
                    cpi_data['yoy_change'] = cpi_data['value'].pct_change(12) * 100
                    cpi_data.plot(y='yoy_change', ax=axes[0, 1], title='通货膨胀率(CPI同比, %)')
                    axes[0, 1].set_ylabel('CPI同比变化 (%)')
                    axes[0, 1].grid(True)
            except Exception as e:
                logger.warning(f"绘制通胀率时出错: {str(e)}")
            
            # 3. 失业率
            try:
                unemployment_data = obb.economy.unemployment().to_df()
                if not unemployment_data.empty:
                    unemployment_data.plot(y='value', ax=axes[1, 0], title='失业率 (%)')
                    axes[1, 0].set_ylabel('失业率 (%)')
                    axes[1, 0].grid(True)
            except Exception as e:
                logger.warning(f"绘制失业率时出错: {str(e)}")
            
            # 4. 联邦基金利率
            try:
                fed_rate_data = obb.economy.fred_series(series_id="FEDFUNDS").to_df()
                if not fed_rate_data.empty:
                    fed_rate_data.plot(y='value', ax=axes[1, 1], title='联邦基金利率 (%)')
                    axes[1, 1].set_ylabel('利率 (%)')
                    axes[1, 1].grid(True)
            except Exception as e:
                logger.warning(f"绘制联邦基金利率时出错: {str(e)}")
            
            # 5. 消费者信心指数
            try:
                consumer_sentiment_data = obb.economy.fred_series(series_id="UMCSENT").to_df()
                if not consumer_sentiment_data.empty:
                    consumer_sentiment_data.plot(y='value', ax=axes[2, 0], title='消费者信心指数')
                    axes[2, 0].set_ylabel('指数值')
                    axes[2, 0].grid(True)
            except Exception as e:
                logger.warning(f"绘制消费者信心指数时出错: {str(e)}")
            
            # 6. 收益率曲线
            try:
                yield_curve = obb.fixedincome.government.yield_curve().to_df()
                if not yield_curve.empty:
                    # 获取最新一期的收益率曲线
                    latest_date = yield_curve.index[-1]
                    yield_data = yield_curve.loc[latest_date]
                    
                    # 绘制收益率曲线
                    axes[2, 1].plot(yield_data.index, yield_data.values)
                    axes[2, 1].set_title(f'美国国债收益率曲线 ({latest_date.strftime("%Y-%m-%d")})')
                    axes[2, 1].set_xlabel('期限')
                    axes[2, 1].set_ylabel('收益率 (%)')
                    axes[2, 1].grid(True)
            except Exception as e:
                logger.warning(f"绘制收益率曲线时出错: {str(e)}")
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # 保存图表
            if save_to_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = os.path.join(self.output_dir, f'macro_indicators_{timestamp}.png')
                plt.savefig(file_path)
                logger.info(f"宏观经济指标图表已保存到: {file_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制经济指标时出错: {str(e)}")
            return None
    
    def analyze_market_impact(self) -> Dict[str, Any]:
        """
        分析宏观经济对市场的影响
        
        Returns:
            Dict: 对不同资产类别的影响分析
        """
        results = {
            'timestamp': datetime.now(),
            'impact': {},
        }
        
        try:
            # 获取经济指标分析
            economic_analysis = self.analyze_economic_indicators()
            summary = economic_analysis.get('summary', {})
            
            # 预测对不同资产类别的影响
            market_impact = {
                'equity': {
                    'impact': None,  # positive, neutral, negative
                    'reasoning': []
                },
                'fixed_income': {
                    'impact': None,
                    'reasoning': []
                },
                'commodities': {
                    'impact': None,
                    'reasoning': []
                },
                'recommended_sectors': [],
                'cautious_sectors': []
            }
            
            # 分析对股票市场的影响
            equity_score = 0
            
            # 经济健康状况对股票的影响
            if summary.get('economic_health') == 'strong':
                equity_score += 1
                market_impact['equity']['reasoning'].append("经济增长强劲，有利于企业盈利")
            elif summary.get('economic_health') == 'weak':
                equity_score -= 1
                market_impact['equity']['reasoning'].append("经济增长疲软，可能影响企业盈利")
            
            # 就业状况对股票的影响
            if summary.get('employment_status') == 'strong':
                equity_score += 1
                market_impact['equity']['reasoning'].append("就业市场强劲，支持消费和经济增长")
            elif summary.get('employment_status') == 'weak':
                equity_score -= 1
                market_impact['equity']['reasoning'].append("就业市场疲软，可能限制消费和经济增长")
            
            # 通胀状况对股票的影响
            if summary.get('inflation_status') == 'high':
                equity_score -= 1
                market_impact['equity']['reasoning'].append("高通胀可能导致利率上升，对股票估值产生压力")
            elif summary.get('inflation_status') == 'low':
                equity_score += 0.5
                market_impact['equity']['reasoning'].append("低通胀有利于维持宽松货币政策，支持股票估值")
            
            # 货币政策对股票的影响
            if summary.get('monetary_policy') == 'easing':
                equity_score += 1
                market_impact['equity']['reasoning'].append("宽松货币政策提供流动性，支持股票市场")
            elif summary.get('monetary_policy') == 'tightening':
                equity_score -= 1
                market_impact['equity']['reasoning'].append("紧缩货币政策收紧流动性，对股票市场构成压力")
            
            # 消费者健康状况对股票的影响
            if summary.get('consumer_health') == 'strong':
                equity_score += 1
                market_impact['equity']['reasoning'].append("消费者信心强劲，支持消费和企业盈利")
            elif summary.get('consumer_health') == 'weak':
                equity_score -= 1
                market_impact['equity']['reasoning'].append("消费者信心疲软，可能限制消费和企业盈利")
            
            # 确定对股票市场的总体影响
            if equity_score > 1:
                market_impact['equity']['impact'] = 'positive'
            elif equity_score < -1:
                market_impact['equity']['impact'] = 'negative'
            else:
                market_impact['equity']['impact'] = 'neutral'
            
            # 分析对固定收益市场的影响
            fixed_income_score = 0
            
            # 通胀状况对债券的影响
            if summary.get('inflation_status') == 'high':
                fixed_income_score -= 1
                market_impact['fixed_income']['reasoning'].append("高通胀环境下债券实际收益率下降")
            elif summary.get('inflation_status') == 'low':
                fixed_income_score += 1
                market_impact['fixed_income']['reasoning'].append("低通胀环境对债券有利")
            
            # 货币政策对债券的影响
            if summary.get('monetary_policy') == 'tightening':
                fixed_income_score -= 1
                market_impact['fixed_income']['reasoning'].append("货币紧缩导致利率上升，债券价格下跌")
            elif summary.get('monetary_policy') == 'easing':
                fixed_income_score += 1
                market_impact['fixed_income']['reasoning'].append("货币宽松导致利率下降，债券价格上涨")
            
            # 经济状况对债券的影响
            if summary.get('economic_health') == 'strong':
                fixed_income_score -= 0.5
                market_impact['fixed_income']['reasoning'].append("强劲经济可能导致通胀预期上升，对债券不利")
            elif summary.get('economic_health') == 'weak':
                fixed_income_score += 0.5
                market_impact['fixed_income']['reasoning'].append("经济疲软时债券作为避险资产可能受益")
            
            # 确定对固定收益市场的总体影响
            if fixed_income_score > 1:
                market_impact['fixed_income']['impact'] = 'positive'
            elif fixed_income_score < -1:
                market_impact['fixed_income']['impact'] = 'negative'
            else:
                market_impact['fixed_income']['impact'] = 'neutral'
            
            # 推荐的行业/需谨慎的行业
            if market_impact['equity']['impact'] == 'positive':
                if summary.get('economic_health') == 'strong':
                    market_impact['recommended_sectors'].extend(['technology', 'consumer_cyclical', 'industrials', 'financial_services'])
                    market_impact['cautious_sectors'].extend(['utilities', 'consumer_defensive'])
                else:
                    market_impact['recommended_sectors'].extend(['healthcare', 'technology', 'communication_services'])
                    market_impact['cautious_sectors'].extend(['energy', 'real_estate'])
            elif market_impact['equity']['impact'] == 'negative':
                market_impact['recommended_sectors'].extend(['utilities', 'consumer_defensive', 'healthcare'])
                market_impact['cautious_sectors'].extend(['technology', 'consumer_cyclical', 'industrials'])
            else:
                market_impact['recommended_sectors'].extend(['healthcare', 'consumer_defensive', 'technology'])
                market_impact['cautious_sectors'].extend(['energy', 'real_estate'])
            
            results['impact'] = market_impact
            results['summary'] = summary
            
            return results
            
        except Exception as e:
            logger.error(f"分析市场影响时出错: {str(e)}")
            results['error'] = str(e)
            return results

def run_macro_analysis():
    """运行宏观经济分析并打印结果"""
    try:
        logger.info("开始宏观经济分析...")
        
        # 创建分析实例
        analyzer = MacroAnalysis()
        
        # 分析经济指标
        economic_results = analyzer.analyze_economic_indicators()
        
        # 分析市场影响
        market_impact = analyzer.analyze_market_impact()
        
        # 绘制经济指标图表
        analyzer.plot_economic_indicators()
        
        # 打印分析结果
        print("\n===== 宏观经济分析结果 =====")
        
        # 打印经济指标
        print("\n--- 经济指标 ---")
        for name, value in economic_results.get('indicators', {}).items():
            if value is not None:
                print(f"{name}: {value:.2f}")
        
        # 打印经济状况摘要
        print("\n--- 经济状况摘要 ---")
        for name, status in economic_results.get('summary', {}).items():
            if status:
                print(f"{name}: {status}")
        
        # 打印市场影响
        print("\n--- 对市场的影响 ---")
        for asset_class, impact in market_impact.get('impact', {}).items():
            if isinstance(impact, dict) and 'impact' in impact:
                print(f"\n{asset_class.upper()}:")
                print(f"  影响: {impact['impact']}")
                print("  分析:")
                for reason in impact.get('reasoning', []):
                    print(f"  - {reason}")
        
        # 打印行业建议
        print("\n--- 行业配置建议 ---")
        print("推荐关注的行业:")
        for sector in market_impact.get('impact', {}).get('recommended_sectors', []):
            print(f"  - {sector}")
        
        print("\n谨慎配置的行业:")
        for sector in market_impact.get('impact', {}).get('cautious_sectors', []):
            print(f"  - {sector}")
        
        logger.info("宏观经济分析完成")
        
    except Exception as e:
        logger.error(f"运行宏观分析时出错: {str(e)}")
        raise

if __name__ == "__main__":
    run_macro_analysis() 