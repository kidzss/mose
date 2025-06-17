"""
增强股票筛选器 - 集成专家建议的智能选股系统
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yfinance as yf
import pandas as pd

from monitor.enhanced_stock_analyzer import EnhancedStockAnalyzer
from monitor.financial_analyzer import FinancialAnalyzer

logger = logging.getLogger(__name__)

class EnhancedStockScreener:
    """增强股票筛选器 - 基于专家建议的智能选股"""
    
    def __init__(self):
        """初始化增强筛选器"""
        self.analyzer = EnhancedStockAnalyzer()
        self.financial_analyzer = FinancialAnalyzer()
        
        # 专家建议权重配置
        self.scoring_weights = {
            # 传统财务指标 (60%)
            'valuation': 0.15,      # 估值 15%
            'profitability': 0.15,  # 盈利能力 15%
            'financial_health': 0.15, # 财务健康 15%
            'analyst_sentiment': 0.15, # 分析师情绪 15%
            
            # 专家建议指标 (40%)
            'growth_quality': 0.15,    # 成长性质量 15% (EPS+营收增长)
            'fcf_strength': 0.10,      # 自由现金流强度 10%
            'industry_position': 0.10, # 行业地位 10%
            'risk_control': 0.05,      # 风险控制 5% (预警系统)
        }
        
        # 风险过滤标准
        self.risk_filters = {
            'max_pe_multiple': 2.0,    # PE不超过行业基准2倍
            'min_fcf_ratio': 0.1,      # FCF转换率不低于10%
            'min_growth_score': 0.3,   # 成长性评分不低于0.3
            'max_warning_level': 'medium'  # 预警等级不超过medium
        }
        
        logger.info("🚀 增强股票筛选器初始化完成")
    
    def screen_stocks(self, symbols: List[str], min_score: float = 0.6) -> List[Dict]:
        """
        筛选股票
        
        Args:
            symbols: 股票代码列表
            min_score: 最低评分要求
            
        Returns:
            List[Dict]: 筛选结果，按评分排序
        """
        results = []
        
        logger.info(f"开始筛选 {len(symbols)} 只股票，最低评分要求: {min_score}")
        
        for symbol in symbols:
            try:
                # 获取增强分析结果
                analysis = self.analyzer.analyze_stock_enhanced(symbol)
                
                # 计算增强评分
                enhanced_score = self._calculate_enhanced_score(analysis)
                
                # 风险过滤
                risk_passed, risk_reasons = self._check_risk_filters(analysis)
                
                # 获取当前价格用于买卖点计算
                current_price = self._get_current_price(symbol)
                
                # 计算买卖点
                price_targets = self._calculate_price_targets(analysis, current_price)
                
                result = {
                    'symbol': symbol,
                    'enhanced_score': enhanced_score,
                    'traditional_score': analysis.get('overall_score', 0),
                    'growth_score': analysis.get('growth_score', 0),
                    'industry_score': analysis.get('industry_score', 0),
                    'current_price': current_price,
                    'price_targets': price_targets,
                    'risk_passed': risk_passed,
                    'risk_reasons': risk_reasons,
                    'analysis_time': datetime.now().isoformat(),
                    'recommendations': analysis.get('recommendations', []),
                    'warnings': analysis.get('warnings', [])
                }
                
                # 只保留通过风险过滤且评分达标的股票
                if risk_passed and enhanced_score >= min_score:
                    results.append(result)
                    logger.info(f"✅ {symbol} 通过筛选 - 增强评分: {enhanced_score:.3f}")
                else:
                    logger.info(f"❌ {symbol} 未通过筛选 - 评分: {enhanced_score:.3f}, 风险: {not risk_passed}")
                    
            except Exception as e:
                logger.error(f"筛选 {symbol} 失败: {e}")
                continue
        
        # 按增强评分排序
        results.sort(key=lambda x: x['enhanced_score'], reverse=True)
        
        logger.info(f"筛选完成，{len(results)} 只股票通过筛选")
        return results
    
    def _calculate_enhanced_score(self, analysis: Dict) -> float:
        """计算增强评分 - 集成专家建议"""
        try:
            enhanced_features = analysis.get('enhanced_features', {})
            financial_analysis = enhanced_features.get('financial_analysis', {})
            
            if not financial_analysis:
                return 0.0
            
            dimensions = financial_analysis.get('dimensions', {})
            
            # 传统财务指标评分
            valuation_score = dimensions.get('valuation', {}).get('score', 0)
            profitability_score = dimensions.get('profitability', {}).get('score', 0)
            health_score = dimensions.get('financial_health', {}).get('score', 0)
            sentiment_score = dimensions.get('analyst_sentiment', {}).get('score', 0)
            
            # 专家建议指标评分
            growth_data = dimensions.get('growth', {})
            industry_data = dimensions.get('industry_comparison', {})
            
            # 成长性质量评分 (EPS + 营收增长的综合)
            growth_quality_score = self._calculate_growth_quality_score(growth_data)
            
            # 自由现金流强度评分
            fcf_strength_score = self._calculate_fcf_strength_score(growth_data)
            
            # 行业地位评分
            industry_position_score = industry_data.get('industry_adjusted_score', 0)
            
            # 风险控制评分 (基于预警系统)
            risk_control_score = self._calculate_risk_control_score(financial_analysis)
            
            # 计算加权总分
            enhanced_score = (
                valuation_score * self.scoring_weights['valuation'] +
                profitability_score * self.scoring_weights['profitability'] +
                health_score * self.scoring_weights['financial_health'] +
                sentiment_score * self.scoring_weights['analyst_sentiment'] +
                growth_quality_score * self.scoring_weights['growth_quality'] +
                fcf_strength_score * self.scoring_weights['fcf_strength'] +
                industry_position_score * self.scoring_weights['industry_position'] +
                risk_control_score * self.scoring_weights['risk_control']
            )
            
            return min(enhanced_score, 1.0)  # 确保不超过1.0
            
        except Exception as e:
            logger.error(f"计算增强评分失败: {e}")
            return 0.0
    
    def _calculate_growth_quality_score(self, growth_data: Dict) -> float:
        """计算成长性质量评分"""
        try:
            details = growth_data.get('details', {})
            eps_growth = details.get('eps_growth', {}).get('value', 0)
            revenue_growth = details.get('revenue_growth', {}).get('value', 0)
            
            # EPS和营收增长的综合评分
            eps_score = min(eps_growth * 2, 1.0) if eps_growth > 0 else 0  # EPS增长50%得满分
            revenue_score = min(revenue_growth, 1.0) if revenue_growth > 0 else 0  # 营收增长100%得满分
            
            # 综合评分 (EPS权重60%，营收权重40%)
            quality_score = eps_score * 0.6 + revenue_score * 0.4
            
            return quality_score
            
        except Exception as e:
            logger.error(f"计算成长性质量评分失败: {e}")
            return 0.0
    
    def _calculate_fcf_strength_score(self, growth_data: Dict) -> float:
        """计算自由现金流强度评分"""
        try:
            details = growth_data.get('details', {})
            fcf_data = details.get('free_cash_flow', {})
            fcf_ratio = fcf_data.get('value', 0)
            
            # FCF转换率评分：80%以上得满分，20%以下得0分
            if fcf_ratio >= 0.8:
                return 1.0
            elif fcf_ratio >= 0.6:
                return 0.8
            elif fcf_ratio >= 0.4:
                return 0.6
            elif fcf_ratio >= 0.2:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"计算FCF强度评分失败: {e}")
            return 0.0
    
    def _calculate_risk_control_score(self, financial_analysis: Dict) -> float:
        """计算风险控制评分"""
        try:
            warning_alerts = financial_analysis.get('warning_alerts', {})
            alert_level = warning_alerts.get('alert_level', 'low')
            
            # 根据预警等级给分
            if alert_level == 'low':
                return 1.0
            elif alert_level == 'medium':
                return 0.6
            elif alert_level == 'high':
                return 0.2
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"计算风险控制评分失败: {e}")
            return 0.5
    
    def _check_risk_filters(self, analysis: Dict) -> Tuple[bool, List[str]]:
        """检查风险过滤条件"""
        risk_reasons = []
        
        try:
            enhanced_features = analysis.get('enhanced_features', {})
            financial_analysis = enhanced_features.get('financial_analysis', {})
            
            if not financial_analysis:
                return False, ["财务数据不可用"]
            
            dimensions = financial_analysis.get('dimensions', {})
            
            # 1. PE倍数检查
            # 这里需要实现PE相对行业基准的检查
            
            # 2. FCF转换率检查
            growth_data = dimensions.get('growth', {})
            fcf_ratio = growth_data.get('details', {}).get('free_cash_flow', {}).get('value', 0)
            if fcf_ratio < self.risk_filters['min_fcf_ratio']:
                risk_reasons.append(f"FCF转换率过低: {fcf_ratio:.1%}")
            
            # 3. 成长性评分检查
            growth_score = analysis.get('growth_score', 0)
            if growth_score < self.risk_filters['min_growth_score']:
                risk_reasons.append(f"成长性评分过低: {growth_score:.3f}")
            
            # 4. 预警等级检查
            warning_alerts = financial_analysis.get('warning_alerts', {})
            alert_level = warning_alerts.get('alert_level', 'low')
            if alert_level == 'high':
                risk_reasons.append(f"高风险预警: {alert_level}")
            
            return len(risk_reasons) == 0, risk_reasons
            
        except Exception as e:
            logger.error(f"风险过滤检查失败: {e}")
            return False, [f"风险检查失败: {str(e)}"]
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前股价"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice', info.get('regularMarketPrice', 0))
        except Exception as e:
            logger.error(f"获取 {symbol} 价格失败: {e}")
            return 0.0
    
    def _calculate_price_targets(self, analysis: Dict, current_price: float) -> Dict:
        """计算买卖点价格目标"""
        try:
            if current_price <= 0:
                return {'buy_price': 0, 'sell_price': 0, 'stop_loss': 0}
            
            enhanced_score = self._calculate_enhanced_score(analysis)
            growth_score = analysis.get('growth_score', 0)
            industry_score = analysis.get('industry_score', 0)
            
            # 基于评分计算价格目标
            if enhanced_score >= 0.8:
                # 高分股票：激进策略
                buy_discount = 0.02  # 当前价格98%买入
                sell_premium = 0.25  # 目标涨幅25%
                stop_loss_ratio = 0.12  # 止损12%
            elif enhanced_score >= 0.65:
                # 中高分股票：平衡策略
                buy_discount = 0.05  # 当前价格95%买入
                sell_premium = 0.20  # 目标涨幅20%
                stop_loss_ratio = 0.10  # 止损10%
            else:
                # 中等分股票：保守策略
                buy_discount = 0.08  # 当前价格92%买入
                sell_premium = 0.15  # 目标涨幅15%
                stop_loss_ratio = 0.08  # 止损8%
            
            # 根据成长性调整目标
            if growth_score > 0.8:
                sell_premium += 0.05  # 高成长股提高目标5%
            
            # 根据行业地位调整
            if industry_score > 0.7:
                buy_discount -= 0.01  # 行业领先股减少折扣1%
            
            return {
                'buy_price': round(current_price * (1 - buy_discount), 2),
                'sell_price': round(current_price * (1 + sell_premium), 2),
                'stop_loss': round(current_price * (1 - stop_loss_ratio), 2),
                'current_price': current_price,
                'buy_discount': f"{buy_discount:.1%}",
                'sell_premium': f"{sell_premium:.1%}",
                'stop_loss_ratio': f"{stop_loss_ratio:.1%}"
            }
            
        except Exception as e:
            logger.error(f"计算价格目标失败: {e}")
            return {'buy_price': 0, 'sell_price': 0, 'stop_loss': 0}
    
    def generate_screening_report(self, results: List[Dict]) -> str:
        """生成筛选报告"""
        if not results:
            return "📊 筛选结果：未找到符合条件的股票"
        
        report = f"""
📊 增强选股报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

🎯 筛选结果：{len(results)} 只股票通过筛选

"""
        
        for i, result in enumerate(results[:10], 1):  # 显示前10只
            symbol = result['symbol']
            enhanced_score = result['enhanced_score']
            growth_score = result['growth_score']
            industry_score = result['industry_score']
            price_targets = result['price_targets']
            
            report += f"""
{i}. {symbol} - 增强评分: {enhanced_score:.3f}
   💰 当前价格: ${price_targets.get('current_price', 0):.2f}
   📈 建议买入: ${price_targets.get('buy_price', 0):.2f} ({price_targets.get('buy_discount', 'N/A')})
   🎯 目标卖出: ${price_targets.get('sell_price', 0):.2f} ({price_targets.get('sell_premium', 'N/A')})
   🛡️ 止损价格: ${price_targets.get('stop_loss', 0):.2f} ({price_targets.get('stop_loss_ratio', 'N/A')})
   📊 成长性: {growth_score:.3f} | 行业地位: {industry_score:.3f}
   
"""
        
        report += f"""
{'='*60}
📋 评分权重说明：
• 传统指标 (60%): 估值15% + 盈利15% + 财务健康15% + 分析师情绪15%
• 专家建议 (40%): 成长性质量15% + FCF强度10% + 行业地位10% + 风险控制5%

⚠️ 投资提示：建议价格仅供参考，请结合市场环境和个人风险承受能力决策
"""
        
        return report 