"""
投资组合宏观集成模块
将宏观因子分析集成到现有的投资组合分析系统中
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
from analysis.macro_factor_analyzer import MacroFactorAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioMacroIntegration:
    """投资组合宏观集成器"""
    
    def __init__(self, portfolio_config_path: str = "portfolio_config.json"):
        """
        初始化投资组合宏观集成器
        
        Args:
            portfolio_config_path: 投资组合配置文件路径
        """
        self.portfolio_config_path = portfolio_config_path
        self.macro_analyzer = MacroFactorAnalyzer()
        self.portfolio_config = self._load_portfolio_config()
        
        # 宏观调整配置
        self.macro_adjustment_config = {
            'max_adjustment_pct': 0.15,  # 最大调整幅度15%
            'technology_sensitivity': {
                'interest_rate': -0.8,    # 利率上升对科技股负面影响
                'vix': 1.2               # VIX上升对科技股影响更大
            },
            'financial_sensitivity': {
                'interest_rate': 0.6,     # 利率上升对金融股正面影响
                'dollar_strength': -0.4   # 美元走强对金融股负面影响
            },
            'energy_sensitivity': {
                'dollar_strength': -0.7,  # 美元走强对能源股负面影响
                'inflation': 0.5         # 通胀对能源股正面影响
            }
        }
        
    def _load_portfolio_config(self) -> Dict:
        """加载投资组合配置"""
        try:
            with open(self.portfolio_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载投资组合配置失败: {e}")
            return {}
    
    def analyze_macro_impact_on_portfolio(self) -> Dict:
        """分析宏观因子对当前投资组合的影响"""
        try:
            # 获取宏观数据和分析
            logger.info("获取宏观数据...")
            macro_data = self.macro_analyzer.fetch_macro_data()
            
            logger.info("计算宏观得分...")
            macro_score = self.macro_analyzer.calculate_macro_score()
            
            # 获取行业影响
            sector_impact = self.macro_analyzer.get_sector_impact(macro_score)
            
            # 分析对当前持仓的影响
            portfolio_impact = self._analyze_portfolio_impact(macro_score, sector_impact)
            
            # 生成调整建议
            adjustment_recommendations = self._generate_adjustment_recommendations(
                macro_score, sector_impact, portfolio_impact
            )
            
            return {
                'macro_analysis': macro_score,
                'sector_impact': sector_impact,
                'portfolio_impact': portfolio_impact,
                'adjustment_recommendations': adjustment_recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"分析宏观影响失败: {e}")
            return {}
    
    def _analyze_portfolio_impact(self, macro_score: Dict, sector_impact: Dict) -> Dict:
        """分析对当前投资组合的具体影响"""
        try:
            positions = self.portfolio_config.get('positions', {})
            impact_analysis = {}
            
            total_portfolio_value = self.portfolio_config.get('portfolio', {}).get('total_value', 1)
            
            for symbol, position_info in positions.items():
                sector = position_info.get('sector', 'Other')
                weight = position_info.get('weight', 0) / 100  # 转换为小数
                investment_amount = position_info.get('investment_amount', 0)
                
                # 获取该行业的宏观影响得分
                sector_score = sector_impact.get(sector, 0.5)
                
                # 计算宏观调整因子
                macro_adjustment = self._calculate_macro_adjustment(
                    symbol, sector, macro_score.get('components', {})
                )
                
                # 计算影响评估
                impact_score = (sector_score + macro_adjustment) / 2
                
                # 分类影响程度
                if impact_score >= 0.7:
                    impact_level = 'positive'
                    impact_desc = '宏观环境对该持仓有利'
                elif impact_score >= 0.5:
                    impact_level = 'neutral'
                    impact_desc = '宏观环境对该持仓中性'
                elif impact_score >= 0.3:
                    impact_level = 'negative'
                    impact_desc = '宏观环境对该持仓不利'
                else:
                    impact_level = 'very_negative'
                    impact_desc = '宏观环境对该持仓非常不利'
                
                impact_analysis[symbol] = {
                    'sector': sector,
                    'current_weight': weight,
                    'investment_amount': investment_amount,
                    'sector_score': sector_score,
                    'macro_adjustment': macro_adjustment,
                    'impact_score': impact_score,
                    'impact_level': impact_level,
                    'impact_description': impact_desc,
                    'risk_level': self._assess_position_risk(symbol, impact_score, weight)
                }
            
            # 计算组合整体宏观敏感度
            portfolio_macro_score = self._calculate_portfolio_macro_score(impact_analysis)
            
            return {
                'individual_impacts': impact_analysis,
                'portfolio_macro_score': portfolio_macro_score,
                'high_risk_positions': [
                    symbol for symbol, info in impact_analysis.items() 
                    if info['risk_level'] == 'high'
                ],
                'vulnerable_sectors': self._identify_vulnerable_sectors(impact_analysis)
            }
            
        except Exception as e:
            logger.error(f"分析投资组合影响失败: {e}")
            return {}
    
    def _calculate_macro_adjustment(self, symbol: str, sector: str, macro_components: Dict) -> float:
        """计算个股的宏观调整因子"""
        try:
            adjustment = 0.5  # 基准值
            
            # 利率环境影响
            if 'interest_rate' in macro_components:
                rate_score = macro_components['interest_rate'].get('rate_trend_score', 0.5)
                if sector == 'Technology':
                    # 科技股对利率敏感
                    adjustment += (rate_score - 0.5) * self.macro_adjustment_config['technology_sensitivity']['interest_rate']
                elif sector == 'Financial':
                    # 金融股受益于利率上升
                    adjustment += (1 - rate_score - 0.5) * self.macro_adjustment_config['financial_sensitivity']['interest_rate']
            
            # 市场情绪影响
            if 'market_sentiment' in macro_components:
                vix_score = macro_components['market_sentiment'].get('vix_score', 0.5)
                if sector == 'Technology':
                    # 科技股对市场情绪敏感
                    adjustment += (vix_score - 0.5) * self.macro_adjustment_config['technology_sensitivity']['vix']
            
            # 美元强度影响
            if 'dollar_strength' in macro_components:
                dollar_score = macro_components['dollar_strength'].get('dollar_score', 0.5)
                if sector == 'Energy':
                    # 能源股受美元强度影响
                    adjustment += (dollar_score - 0.5) * self.macro_adjustment_config['energy_sensitivity']['dollar_strength']
                elif sector == 'Financial':
                    adjustment += (dollar_score - 0.5) * self.macro_adjustment_config['financial_sensitivity']['dollar_strength']
            
            # 确保调整因子在合理范围内
            return max(0.1, min(0.9, adjustment))
            
        except Exception as e:
            logger.error(f"计算宏观调整因子失败: {e}")
            return 0.5
    
    def _assess_position_risk(self, symbol: str, impact_score: float, weight: float) -> str:
        """评估单个持仓的风险等级"""
        # 结合宏观影响和权重评估风险
        if impact_score < 0.3 and weight > 0.15:  # 宏观不利且权重较高
            return 'high'
        elif impact_score < 0.4 or weight > 0.20:  # 任一因素风险较高
            return 'medium'
        else:
            return 'low'
    
    def _calculate_portfolio_macro_score(self, impact_analysis: Dict) -> Dict:
        """计算投资组合整体宏观得分"""
        try:
            weighted_score = 0
            total_weight = 0
            
            for symbol, info in impact_analysis.items():
                weight = info['current_weight']
                score = info['impact_score']
                weighted_score += weight * score
                total_weight += weight
            
            portfolio_score = weighted_score / total_weight if total_weight > 0 else 0.5
            
            # 分析脆弱性
            vulnerability_factors = []
            if portfolio_score < 0.4:
                vulnerability_factors.append('整体宏观环境不利')
            
            # 检查集中度风险
            tech_weight = sum(
                info['current_weight'] for info in impact_analysis.values() 
                if info['sector'] == 'Technology'
            )
            if tech_weight > 0.6:
                vulnerability_factors.append('科技股集中度过高')
            
            return {
                'overall_score': portfolio_score,
                'risk_level': 'high' if portfolio_score < 0.4 else 'medium' if portfolio_score < 0.6 else 'low',
                'vulnerability_factors': vulnerability_factors,
                'macro_resilience': 'strong' if portfolio_score > 0.7 else 'moderate' if portfolio_score > 0.5 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"计算投资组合宏观得分失败: {e}")
            return {'overall_score': 0.5, 'risk_level': 'medium'}
    
    def _identify_vulnerable_sectors(self, impact_analysis: Dict) -> List[str]:
        """识别脆弱行业"""
        sector_scores = {}
        
        for info in impact_analysis.values():
            sector = info['sector']
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append(info['impact_score'])
        
        vulnerable_sectors = []
        for sector, scores in sector_scores.items():
            avg_score = np.mean(scores)
            if avg_score < 0.4:
                vulnerable_sectors.append(sector)
        
        return vulnerable_sectors
    
    def _generate_adjustment_recommendations(self, macro_score: Dict, 
                                           sector_impact: Dict, 
                                           portfolio_impact: Dict) -> Dict:
        """生成调整建议"""
        try:
            recommendations = {
                'immediate_actions': [],
                'medium_term_actions': [],
                'monitoring_points': [],
                'risk_mitigation': []
            }
            
            overall_macro_score = macro_score.get('macro_score', 0.5)
            portfolio_score = portfolio_impact.get('portfolio_macro_score', {}).get('overall_score', 0.5)
            
            # 基于宏观环境生成建议
            if overall_macro_score < 0.4:
                recommendations['immediate_actions'].append(
                    "宏观环境恶化，建议降低整体仓位至70-80%"
                )
                recommendations['risk_mitigation'].append(
                    "增加现金和债券等防御性资产配置"
                )
            elif overall_macro_score > 0.7:
                recommendations['medium_term_actions'].append(
                    "宏观环境有利，可适当增加权益配置"
                )
            
            # 使用通胀分析器的动态结果生成行业建议
            try:
                from analysis.inflation_sector_analyzer import InflationSectorAnalyzer
                inflation_analyzer = InflationSectorAnalyzer()
                inflation_report = inflation_analyzer.generate_inflation_sector_report()
                
                # 基于通胀分析生成行业建议
                if 'sector_analysis' in inflation_report:
                    for sector_name, sector_data in inflation_report['sector_analysis'].items():
                        suggestion = sector_data.get('investment_suggestion', '')
                        if '减少' in suggestion or '降低' in suggestion:
                            recommendations['immediate_actions'].append(
                                f"基于通胀分析：{suggestion}"
                            )
                        elif '增加' in suggestion or '加大' in suggestion:
                            recommendations['medium_term_actions'].append(
                                f"基于通胀分析：{suggestion}"
                            )
                
                # 添加通胀相关的投资建议
                if 'investment_recommendations' in inflation_report:
                    for rec in inflation_report['investment_recommendations'][:3]:  # 取前3条
                        if '立即' in rec or '紧急' in rec:
                            recommendations['immediate_actions'].append(rec)
                        else:
                            recommendations['medium_term_actions'].append(rec)
                            
            except Exception as e:
                logger.warning(f"获取通胀分析建议失败，使用基础建议: {e}")
                # 降级处理：基于传统sector_impact生成建议
                for sector, score in sector_impact.items():
                    if score < 0.4:
                        recommendations['immediate_actions'].append(
                            f"重点关注{sector}行业，宏观环境对该行业影响较大"
                        )
                    elif score > 0.7:
                        recommendations['medium_term_actions'].append(
                            f"可考虑增加{sector}行业配置，宏观环境对该行业有利"
                        )
            
            # 基于个股影响生成建议
            high_risk_positions = portfolio_impact.get('high_risk_positions', [])
            if high_risk_positions:
                recommendations['immediate_actions'].append(
                    f"重点关注高风险持仓: {', '.join(high_risk_positions)}"
                )
                recommendations['risk_mitigation'].append(
                    "考虑降低高风险持仓权重或设置更严格的止损"
                )
            
            # 监控要点（保留，这些是合理的固定监控项）
            recommendations['monitoring_points'].extend([
                "关注美联储政策变化对利率环境的影响",
                "监控VIX指数变化，警惕市场情绪恶化",
                "跟踪美元指数走势对不同行业的影响"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成调整建议失败: {e}")
            return {}
    
    def generate_macro_report(self) -> Dict:
        """生成宏观分析报告"""
        try:
            logger.info("开始生成宏观分析报告...")
            
            # 执行完整分析
            analysis_result = self.analyze_macro_impact_on_portfolio()
            
            if not analysis_result:
                return {'error': '分析失败'}
            
            # 生成报告摘要
            macro_analysis = analysis_result.get('macro_analysis', {})
            portfolio_impact = analysis_result.get('portfolio_impact', {})
            recommendations = analysis_result.get('adjustment_recommendations', {})
            
            report = {
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'executive_summary': {
                    'macro_score': macro_analysis.get('macro_score', 0),
                    'macro_recommendation': macro_analysis.get('recommendation', ''),
                    'portfolio_risk_level': portfolio_impact.get('portfolio_macro_score', {}).get('risk_level', 'medium'),
                    'key_concerns': portfolio_impact.get('high_risk_positions', []),
                    'immediate_actions_count': len(recommendations.get('immediate_actions', []))
                },
                'detailed_analysis': analysis_result,
                'action_plan': {
                    'priority_1': recommendations.get('immediate_actions', []),
                    'priority_2': recommendations.get('medium_term_actions', []),
                    'monitoring': recommendations.get('monitoring_points', []),
                    'risk_management': recommendations.get('risk_mitigation', [])
                }
            }
            
            logger.info("宏观分析报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"生成宏观报告失败: {e}")
            return {'error': f'报告生成失败: {e}'}


if __name__ == "__main__":
    # 测试宏观集成系统
    integration = PortfolioMacroIntegration()
    
    print("正在生成投资组合宏观分析报告...")
    report = integration.generate_macro_report()
    
    if 'error' not in report:
        print("\n=== 投资组合宏观分析报告 ===")
        print(f"报告日期: {report['report_date']}")
        print(f"宏观得分: {report['executive_summary']['macro_score']:.2f}")
        print(f"宏观建议: {report['executive_summary']['macro_recommendation']}")
        print(f"投资组合风险等级: {report['executive_summary']['portfolio_risk_level']}")
        
        if report['executive_summary']['key_concerns']:
            print(f"重点关注: {', '.join(report['executive_summary']['key_concerns'])}")
        
        print(f"\n立即行动项 ({len(report['action_plan']['priority_1'])} 项):")
        for action in report['action_plan']['priority_1']:
            print(f"  • {action}")
    else:
        print(f"报告生成失败: {report['error']}") 