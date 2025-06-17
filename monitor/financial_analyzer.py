"""
财务分析器模块
负责分析yfinance_cache中的财务数据，提供基本面分析和估值评估
"""

import json
import os
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """财务数据分析器"""
    
    def __init__(self, cache_dir: str = "data/yfinance_cache"):
        self.cache_dir = cache_dir
        
        # 财务指标权重配置
        self.weights = {
            'valuation': 0.25,      # 估值指标
            'profitability': 0.25,  # 盈利能力
            'growth': 0.20,         # 成长性
            'financial_health': 0.20, # 财务健康度
            'analyst_sentiment': 0.10  # 分析师情绪
        }
        
        # 行业基准值（可以根据具体行业调整）
        self.benchmarks = {
            'pe_ratio': {'excellent': 15, 'good': 20, 'fair': 25, 'poor': 30},
            'peg_ratio': {'excellent': 1.0, 'good': 1.5, 'fair': 2.0, 'poor': 2.5},
            'debt_to_equity': {'excellent': 0.3, 'good': 0.6, 'fair': 1.0, 'poor': 1.5},
            'roe': {'excellent': 0.15, 'good': 0.12, 'fair': 0.08, 'poor': 0.05},
            'profit_margin': {'excellent': 0.15, 'good': 0.10, 'fair': 0.05, 'poor': 0.02},
            'revenue_growth': {'excellent': 0.15, 'good': 0.10, 'fair': 0.05, 'poor': 0.0}
        }
    
    def load_financial_data(self, symbol: str) -> Optional[Dict]:
        """加载股票的财务数据"""
        try:
            file_path = os.path.join(self.cache_dir, f"{symbol}_info.json")
            if not os.path.exists(file_path):
                logger.warning(f"财务数据文件不存在: {file_path}")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"加载 {symbol} 财务数据失败: {e}")
            return None
    
    def analyze_valuation(self, data: Dict) -> Dict:
        """分析估值指标"""
        valuation_score = 0
        details = {}
        
        # PE比率分析
        pe_ratio = data.get('trailingPE')
        if pe_ratio:
            if pe_ratio <= self.benchmarks['pe_ratio']['excellent']:
                pe_score = 100
                pe_level = "优秀"
            elif pe_ratio <= self.benchmarks['pe_ratio']['good']:
                pe_score = 80
                pe_level = "良好"
            elif pe_ratio <= self.benchmarks['pe_ratio']['fair']:
                pe_score = 60
                pe_level = "一般"
            else:
                pe_score = 40
                pe_level = "偏高"
            
            details['pe_ratio'] = {
                'value': pe_ratio,
                'score': pe_score,
                'level': pe_level,
                'comment': f"PE比率 {pe_ratio:.2f}，估值{pe_level}"
            }
            valuation_score += pe_score * 0.4
        
        # PEG比率分析
        peg_ratio = data.get('trailingPegRatio')
        if peg_ratio:
            if peg_ratio <= self.benchmarks['peg_ratio']['excellent']:
                peg_score = 100
                peg_level = "优秀"
            elif peg_ratio <= self.benchmarks['peg_ratio']['good']:
                peg_score = 80
                peg_level = "良好"
            elif peg_ratio <= self.benchmarks['peg_ratio']['fair']:
                peg_score = 60
                peg_level = "一般"
            else:
                peg_score = 40
                peg_level = "偏高"
            
            details['peg_ratio'] = {
                'value': peg_ratio,
                'score': peg_score,
                'level': peg_level,
                'comment': f"PEG比率 {peg_ratio:.2f}，成长性估值{peg_level}"
            }
            valuation_score += peg_score * 0.3
        
        # 市净率分析
        pb_ratio = data.get('priceToBook')
        if pb_ratio:
            if pb_ratio <= 1.5:
                pb_score = 100
                pb_level = "优秀"
            elif pb_ratio <= 2.5:
                pb_score = 80
                pb_level = "良好"
            elif pb_ratio <= 4.0:
                pb_score = 60
                pb_level = "一般"
            else:
                pb_score = 40
                pb_level = "偏高"
            
            details['pb_ratio'] = {
                'value': pb_ratio,
                'score': pb_score,
                'level': pb_level,
                'comment': f"市净率 {pb_ratio:.2f}，账面价值{pb_level}"
            }
            valuation_score += pb_score * 0.3
        
        return {
            'score': valuation_score / 100 if valuation_score > 0 else 0.5,
            'details': details,
            'summary': self._get_level_summary(valuation_score / 100 if valuation_score > 0 else 0.5)
        }
    
    def analyze_profitability(self, data: Dict) -> Dict:
        """分析盈利能力"""
        profitability_score = 0
        details = {}
        
        # 净利润率
        profit_margin = data.get('profitMargins')
        if profit_margin:
            if profit_margin >= self.benchmarks['profit_margin']['excellent']:
                pm_score = 100
                pm_level = "优秀"
            elif profit_margin >= self.benchmarks['profit_margin']['good']:
                pm_score = 80
                pm_level = "良好"
            elif profit_margin >= self.benchmarks['profit_margin']['fair']:
                pm_score = 60
                pm_level = "一般"
            else:
                pm_score = 40
                pm_level = "较低"
            
            details['profit_margin'] = {
                'value': profit_margin,
                'score': pm_score,
                'level': pm_level,
                'comment': f"净利润率 {profit_margin:.2%}，盈利能力{pm_level}"
            }
            profitability_score += pm_score * 0.4
        
        # ROE (净资产收益率)
        roe = data.get('returnOnEquity')
        if roe:
            if roe >= self.benchmarks['roe']['excellent']:
                roe_score = 100
                roe_level = "优秀"
            elif roe >= self.benchmarks['roe']['good']:
                roe_score = 80
                roe_level = "良好"
            elif roe >= self.benchmarks['roe']['fair']:
                roe_score = 60
                roe_level = "一般"
            else:
                roe_score = 40
                roe_level = "较低"
            
            details['roe'] = {
                'value': roe,
                'score': roe_score,
                'level': roe_level,
                'comment': f"净资产收益率 {roe:.2%}，股东回报{roe_level}"
            }
            profitability_score += roe_score * 0.3
        
        # 毛利率
        gross_margin = data.get('grossMargins')
        if gross_margin:
            if gross_margin >= 0.5:
                gm_score = 100
                gm_level = "优秀"
            elif gross_margin >= 0.3:
                gm_score = 80
                gm_level = "良好"
            elif gross_margin >= 0.2:
                gm_score = 60
                gm_level = "一般"
            else:
                gm_score = 40
                gm_level = "较低"
            
            details['gross_margin'] = {
                'value': gross_margin,
                'score': gm_score,
                'level': gm_level,
                'comment': f"毛利率 {gross_margin:.2%}，产品竞争力{gm_level}"
            }
            profitability_score += gm_score * 0.3
        
        return {
            'score': profitability_score / 100 if profitability_score > 0 else 0.5,
            'details': details,
            'summary': self._get_level_summary(profitability_score / 100 if profitability_score > 0 else 0.5)
        }
    
    def analyze_growth(self, data: Dict) -> Dict:
        """分析成长性"""
        growth_score = 0
        details = {}
        
        # 收入增长率
        revenue_growth = data.get('revenueGrowth')
        if revenue_growth:
            if revenue_growth >= self.benchmarks['revenue_growth']['excellent']:
                rg_score = 100
                rg_level = "优秀"
            elif revenue_growth >= self.benchmarks['revenue_growth']['good']:
                rg_score = 80
                rg_level = "良好"
            elif revenue_growth >= self.benchmarks['revenue_growth']['fair']:
                rg_score = 60
                rg_level = "一般"
            elif revenue_growth >= 0:
                rg_score = 40
                rg_level = "较低"
            else:
                rg_score = 20
                rg_level = "负增长"
            
            details['revenue_growth'] = {
                'value': revenue_growth,
                'score': rg_score,
                'level': rg_level,
                'comment': f"收入增长率 {revenue_growth:.2%}，业务扩张{rg_level}"
            }
            growth_score += rg_score * 0.5
        
        # 盈利增长率
        earnings_growth = data.get('earningsGrowth')
        if earnings_growth:
            if earnings_growth >= 0.15:
                eg_score = 100
                eg_level = "优秀"
            elif earnings_growth >= 0.10:
                eg_score = 80
                eg_level = "良好"
            elif earnings_growth >= 0.05:
                eg_score = 60
                eg_level = "一般"
            elif earnings_growth >= 0:
                eg_score = 40
                eg_level = "较低"
            else:
                eg_score = 20
                eg_level = "负增长"
            
            details['earnings_growth'] = {
                'value': earnings_growth,
                'score': eg_score,
                'level': eg_level,
                'comment': f"盈利增长率 {earnings_growth:.2%}，利润增长{eg_level}"
            }
            growth_score += eg_score * 0.5
        
        return {
            'score': growth_score / 100 if growth_score > 0 else 0.5,
            'details': details,
            'summary': self._get_level_summary(growth_score / 100 if growth_score > 0 else 0.5)
        }
    
    def analyze_financial_health(self, data: Dict) -> Dict:
        """分析财务健康度"""
        health_score = 0
        details = {}
        
        # 债务股权比
        debt_to_equity = data.get('debtToEquity')
        if debt_to_equity:
            debt_ratio = debt_to_equity / 100  # 转换为小数
            if debt_ratio <= self.benchmarks['debt_to_equity']['excellent']:
                de_score = 100
                de_level = "优秀"
            elif debt_ratio <= self.benchmarks['debt_to_equity']['good']:
                de_score = 80
                de_level = "良好"
            elif debt_ratio <= self.benchmarks['debt_to_equity']['fair']:
                de_score = 60
                de_level = "一般"
            else:
                de_score = 40
                de_level = "偏高"
            
            details['debt_to_equity'] = {
                'value': debt_ratio,
                'score': de_score,
                'level': de_level,
                'comment': f"债务股权比 {debt_ratio:.2f}，财务杠杆{de_level}"
            }
            health_score += de_score * 0.4
        
        # 流动比率
        current_ratio = data.get('currentRatio')
        if current_ratio:
            if current_ratio >= 2.0:
                cr_score = 100
                cr_level = "优秀"
            elif current_ratio >= 1.5:
                cr_score = 80
                cr_level = "良好"
            elif current_ratio >= 1.0:
                cr_score = 60
                cr_level = "一般"
            else:
                cr_score = 40
                cr_level = "较低"
            
            details['current_ratio'] = {
                'value': current_ratio,
                'score': cr_score,
                'level': cr_level,
                'comment': f"流动比率 {current_ratio:.2f}，短期偿债能力{cr_level}"
            }
            health_score += cr_score * 0.3
        
        # 自由现金流
        free_cashflow = data.get('freeCashflow')
        if free_cashflow:
            if free_cashflow > 1000000000:  # 10亿以上
                fcf_score = 100
                fcf_level = "优秀"
            elif free_cashflow > 0:
                fcf_score = 80
                fcf_level = "良好"
            elif free_cashflow > -500000000:  # -5亿以上
                fcf_score = 60
                fcf_level = "一般"
            else:
                fcf_score = 40
                fcf_level = "较差"
            
            details['free_cashflow'] = {
                'value': free_cashflow,
                'score': fcf_score,
                'level': fcf_level,
                'comment': f"自由现金流 ${free_cashflow/1000000:.0f}M，现金创造能力{fcf_level}"
            }
            health_score += fcf_score * 0.3
        
        return {
            'score': health_score / 100 if health_score > 0 else 0.5,
            'details': details,
            'summary': self._get_level_summary(health_score / 100 if health_score > 0 else 0.5)
        }
    
    def analyze_analyst_sentiment(self, data: Dict) -> Dict:
        """分析师情绪分析"""
        sentiment_score = 0
        details = {}
        
        # 分析师评级
        recommendation = data.get('recommendationKey', '').lower()
        rec_mean = data.get('recommendationMean', 3.0)
        
        if recommendation == 'strong_buy' or rec_mean <= 1.5:
            rec_score = 100
            rec_level = "强烈买入"
        elif recommendation == 'buy' or rec_mean <= 2.0:
            rec_score = 80
            rec_level = "买入"
        elif recommendation == 'hold' or rec_mean <= 3.0:
            rec_score = 60
            rec_level = "持有"
        elif recommendation == 'sell' or rec_mean <= 4.0:
            rec_score = 40
            rec_level = "卖出"
        else:
            rec_score = 20
            rec_level = "强烈卖出"
        
        details['recommendation'] = {
            'value': recommendation,
            'score': rec_score,
            'level': rec_level,
            'comment': f"分析师评级: {rec_level} (均值: {rec_mean:.2f})"
        }
        sentiment_score += rec_score * 0.6
        
        # 目标价位分析
        current_price = data.get('currentPrice')
        target_mean = data.get('targetMeanPrice')
        
        if current_price and target_mean:
            upside_potential = (target_mean - current_price) / current_price
            if upside_potential >= 0.2:
                target_score = 100
                target_level = "高上涨潜力"
            elif upside_potential >= 0.1:
                target_score = 80
                target_level = "中等上涨潜力"
            elif upside_potential >= 0:
                target_score = 60
                target_level = "小幅上涨潜力"
            else:
                target_score = 40
                target_level = "下跌风险"
            
            details['target_price'] = {
                'current_price': current_price,
                'target_price': target_mean,
                'upside_potential': upside_potential,
                'score': target_score,
                'level': target_level,
                'comment': f"目标价 ${target_mean:.2f}，上涨空间 {upside_potential:.1%}"
            }
            sentiment_score += target_score * 0.4
        
        return {
            'score': sentiment_score / 100 if sentiment_score > 0 else 0.5,
            'details': details,
            'summary': self._get_level_summary(sentiment_score / 100 if sentiment_score > 0 else 0.5)
        }
    
    def _get_level_summary(self, score: float) -> str:
        """根据分数获取等级总结"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "较差"
    
    def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """综合分析股票"""
        try:
            # 加载财务数据
            data = self.load_financial_data(symbol)
            if not data:
                return None
            
            # 各维度分析
            valuation = self.analyze_valuation(data)
            profitability = self.analyze_profitability(data)
            growth = self.analyze_growth(data)
            health = self.analyze_financial_health(data)
            sentiment = self.analyze_analyst_sentiment(data)
            
            # 计算综合得分
            total_score = (
                valuation['score'] * self.weights['valuation'] +
                profitability['score'] * self.weights['profitability'] +
                growth['score'] * self.weights['growth'] +
                health['score'] * self.weights['financial_health'] +
                sentiment['score'] * self.weights['analyst_sentiment']
            )
            
            # 基本信息
            basic_info = {
                'company_name': data.get('longName', symbol),
                'sector': data.get('sector', '未知'),
                'industry': data.get('industry', '未知'),
                'market_cap': data.get('marketCap', 0),
                'current_price': data.get('currentPrice', 0),
                'currency': data.get('currency', 'USD')
            }
            
            # 生成投资建议
            investment_advice = self._generate_investment_advice(total_score, valuation, profitability, growth, health, sentiment)
            
            return {
                'symbol': symbol,
                'basic_info': basic_info,
                'analysis_date': datetime.now().isoformat(),
                'total_score': total_score,
                'overall_rating': self._get_level_summary(total_score),
                'dimensions': {
                    'valuation': valuation,
                    'profitability': profitability,
                    'growth': growth,
                    'financial_health': health,
                    'analyst_sentiment': sentiment
                },
                'investment_advice': investment_advice
            }
            
        except Exception as e:
            logger.error(f"分析股票 {symbol} 失败: {e}")
            return None
    
    def _generate_investment_advice(self, total_score: float, valuation: Dict, profitability: Dict, 
                                  growth: Dict, health: Dict, sentiment: Dict) -> Dict:
        """生成投资建议"""
        advice = {
            'recommendation': '',
            'confidence': 0,
            'key_strengths': [],
            'key_concerns': [],
            'action_items': []
        }
        
        # 确定总体建议
        if total_score >= 0.8:
            advice['recommendation'] = '强烈买入'
            advice['confidence'] = 95
        elif total_score >= 0.65:
            advice['recommendation'] = '买入'
            advice['confidence'] = 80
        elif total_score >= 0.5:
            advice['recommendation'] = '持有'
            advice['confidence'] = 65
        elif total_score >= 0.35:
            advice['recommendation'] = '考虑卖出'
            advice['confidence'] = 70
        else:
            advice['recommendation'] = '卖出'
            advice['confidence'] = 80
        
        # 分析优势
        if valuation['score'] >= 0.7:
            advice['key_strengths'].append('估值合理，具有投资价值')
        if profitability['score'] >= 0.7:
            advice['key_strengths'].append('盈利能力强，经营效率高')
        if growth['score'] >= 0.7:
            advice['key_strengths'].append('业务增长强劲，发展前景良好')
        if health['score'] >= 0.7:
            advice['key_strengths'].append('财务状况健康，风险较低')
        if sentiment['score'] >= 0.7:
            advice['key_strengths'].append('分析师看好，市场情绪积极')
        
        # 分析担忧
        if valuation['score'] <= 0.4:
            advice['key_concerns'].append('估值偏高，存在泡沫风险')
        if profitability['score'] <= 0.4:
            advice['key_concerns'].append('盈利能力较弱，经营压力大')
        if growth['score'] <= 0.4:
            advice['key_concerns'].append('增长乏力，业务发展受限')
        if health['score'] <= 0.4:
            advice['key_concerns'].append('财务状况不佳，债务风险较高')
        if sentiment['score'] <= 0.4:
            advice['key_concerns'].append('分析师不看好，市场情绪消极')
        
        # 行动建议
        if advice['recommendation'] in ['强烈买入', '买入']:
            advice['action_items'].append('可以考虑逐步建仓')
            advice['action_items'].append('关注财报发布日期')
            advice['action_items'].append('设置止损位以控制风险')
        elif advice['recommendation'] == '持有':
            advice['action_items'].append('继续持有，密切关注业绩变化')
            advice['action_items'].append('关注行业发展趋势')
        else:
            advice['action_items'].append('考虑减仓或清仓')
            advice['action_items'].append('寻找更好的投资机会')
        
        return advice