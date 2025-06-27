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
            'pb_ratio': {'excellent': 1.5, 'good': 3.0, 'fair': 5.0, 'poor': 7.0},
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
        """分析成长性指标"""
        growth_score = 0
        details = {}
        
        # EPS增长率分析
        eps_growth = data.get('earningsQuarterlyGrowth')  # 季度EPS增长
        eps_growth_yoy = data.get('earningsGrowth')       # 年度EPS增长
        
        if eps_growth_yoy:
            if eps_growth_yoy >= self.benchmarks['revenue_growth']['excellent']:
                eps_score = 100
                eps_level = "优秀"
            elif eps_growth_yoy >= self.benchmarks['revenue_growth']['good']:
                eps_score = 80
                eps_level = "良好"
            elif eps_growth_yoy >= self.benchmarks['revenue_growth']['fair']:
                eps_score = 60
                eps_level = "一般"
            elif eps_growth_yoy >= 0:
                eps_score = 40
                eps_level = "较低"
            else:
                eps_score = 20
                eps_level = "负增长"
            
            details['eps_growth'] = {
                'value': eps_growth_yoy,
                'score': eps_score,
                'level': eps_level,
                'comment': f"EPS增长率 {eps_growth_yoy:.2%}，成长性{eps_level}"
            }
            growth_score += eps_score * 0.4
        
        # 营收增长率分析
        revenue_growth = data.get('revenueGrowth')
        if revenue_growth:
            if revenue_growth >= self.benchmarks['revenue_growth']['excellent']:
                rev_score = 100
                rev_level = "优秀"
            elif revenue_growth >= self.benchmarks['revenue_growth']['good']:
                rev_score = 80
                rev_level = "良好"
            elif revenue_growth >= self.benchmarks['revenue_growth']['fair']:
                rev_score = 60
                rev_level = "一般"
            elif revenue_growth >= 0:
                rev_score = 40
                rev_level = "较低"
            else:
                rev_score = 20
                rev_level = "负增长"
            
            details['revenue_growth'] = {
                'value': revenue_growth,
                'score': rev_score,
                'level': rev_level,
                'comment': f"营收增长率 {revenue_growth:.2%}，业务增长{rev_level}"
            }
            growth_score += rev_score * 0.3
        
        # 自由现金流分析 (新增)
        free_cash_flow = data.get('freeCashflow')
        operating_cash_flow = data.get('operatingCashflow')
        
        if free_cash_flow and operating_cash_flow:
            fcf_ratio = free_cash_flow / operating_cash_flow if operating_cash_flow != 0 else 0
            
            if fcf_ratio >= 0.8:
                fcf_score = 100
                fcf_level = "优秀"
            elif fcf_ratio >= 0.6:
                fcf_score = 80
                fcf_level = "良好"
            elif fcf_ratio >= 0.4:
                fcf_score = 60
                fcf_level = "一般"
            elif fcf_ratio >= 0.2:
                fcf_score = 40
                fcf_level = "较低"
            else:
                fcf_score = 20
                fcf_level = "很低"
            
            details['free_cash_flow'] = {
                'value': fcf_ratio,
                'score': fcf_score,
                'level': fcf_level,
                'comment': f"自由现金流转换率 {fcf_ratio:.2%}，现金产生能力{fcf_level}"
            }
            growth_score += fcf_score * 0.3

        return {
            'score': growth_score / 100 if growth_score > 0 else 0.3,
            'details': details,
            'summary': self._get_level_summary(growth_score / 100 if growth_score > 0 else 0.3)
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
                'comment': f"自由现金流 ${free_cashflow/1000000:.0f}M，现金创造能力{fcf_level}" if free_cashflow != 0 else f"自由现金流 $0M，现金创造能力{fcf_level}"
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
        
        if current_price and target_mean and current_price > 0:
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
            
            # 新增：行业对比分析
            industry_comparison = self.analyze_industry_comparison(symbol, data)
            
            # 调整综合得分权重，加入行业对比因子
            adjusted_weights = {
                'valuation': 0.20,      # 估值指标权重略降
                'profitability': 0.20,  # 盈利能力权重略降
                'growth': 0.25,         # 成长性权重增加
                'financial_health': 0.15, # 财务健康度权重略降
                'analyst_sentiment': 0.10,  # 分析师情绪
                'industry_comparison': 0.10  # 新增行业对比权重
            }
            
            # 计算综合得分
            total_score = (
                valuation['score'] * adjusted_weights['valuation'] +
                profitability['score'] * adjusted_weights['profitability'] +
                growth['score'] * adjusted_weights['growth'] +
                health['score'] * adjusted_weights['financial_health'] +
                sentiment['score'] * adjusted_weights['analyst_sentiment'] +
                industry_comparison['industry_adjusted_score'] * adjusted_weights['industry_comparison']
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
            
            # 新增：生成预警信息（第二个专家建议）
            warning_alerts = self._generate_warning_alerts(symbol, data, total_score)
            
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
                    'analyst_sentiment': sentiment,
                    'industry_comparison': industry_comparison  # 新增行业对比维度
                },
                'investment_advice': investment_advice,
                'warning_alerts': warning_alerts  # 新增预警信息
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
    
    def analyze_industry_comparison(self, symbol: str, data: Dict) -> Dict:
        """行业对比分析 - 第三个专家建议的实现"""
        try:
            sector = data.get('sector', 'Unknown')
            industry = data.get('industry', 'Unknown')
            
            # 行业基准值（根据不同行业调整）
            industry_benchmarks = self._get_industry_benchmarks(sector)
            
            comparison_result = {
                'sector': sector,
                'industry': industry,
                'industry_adjusted_score': 0,
                'relative_metrics': {},
                'industry_ranking': {}
            }
            
            # 获取股票的关键指标
            current_pe = data.get('trailingPE', 0)
            current_roe = data.get('returnOnEquity', 0)
            current_pb = data.get('priceToBook', 0)
            current_debt_ratio = data.get('debtToEquity', 0)
            
            # 与行业基准对比
            industry_score = 0
            
            # PE行业对比
            if current_pe > 0:
                industry_pe_benchmark = industry_benchmarks['pe_ratio']['good']
                if current_pe <= industry_pe_benchmark * 0.8:
                    pe_industry_score = 100
                    pe_relative = "行业内低估值"
                elif current_pe <= industry_pe_benchmark:
                    pe_industry_score = 80
                    pe_relative = "行业内合理估值"
                elif current_pe <= industry_pe_benchmark * 1.3:
                    pe_industry_score = 60
                    pe_relative = "行业内略高估值"
                else:
                    pe_industry_score = 40
                    pe_relative = "行业内高估值"
                
                comparison_result['relative_metrics']['pe_comparison'] = {
                    'value': current_pe,
                    'industry_benchmark': industry_pe_benchmark,
                    'relative_position': pe_relative,
                    'score': pe_industry_score
                }
                industry_score += pe_industry_score * 0.3
            
            # ROE行业对比
            if current_roe > 0:
                industry_roe_benchmark = industry_benchmarks['roe']['good']
                if current_roe >= industry_roe_benchmark * 1.3:
                    roe_industry_score = 100
                    roe_relative = "行业内优秀盈利"
                elif current_roe >= industry_roe_benchmark:
                    roe_industry_score = 80
                    roe_relative = "行业内良好盈利"
                elif current_roe >= industry_roe_benchmark * 0.7:
                    roe_industry_score = 60
                    roe_relative = "行业内平均盈利"
                else:
                    roe_industry_score = 40
                    roe_relative = "行业内较低盈利"
                
                comparison_result['relative_metrics']['roe_comparison'] = {
                    'value': current_roe,
                    'industry_benchmark': industry_roe_benchmark,
                    'relative_position': roe_relative,
                    'score': roe_industry_score
                }
                industry_score += roe_industry_score * 0.3
            
            # PB行业对比
            try:
                current_pb = data.get('priceToBook', data.get('pb_ratio', 0))
                if current_pb and current_pb > 0:
                    # 确保行业基准存在pb_ratio字段
                    if 'pb_ratio' not in industry_benchmarks:
                        industry_benchmarks['pb_ratio'] = {'excellent': 1.5, 'good': 3.0, 'fair': 5.0, 'poor': 7.0}
                    industry_pb_benchmark = industry_benchmarks['pb_ratio']['good']
                    if current_pb <= industry_pb_benchmark * 0.7:
                        pb_industry_score = 100
                        pb_relative = "行业内低账面价值"
                    elif current_pb <= industry_pb_benchmark:
                        pb_industry_score = 80
                        pb_relative = "行业内合理账面价值"
                    elif current_pb <= industry_pb_benchmark * 1.5:
                        pb_industry_score = 60
                        pb_relative = "行业内略高账面价值"
                    else:
                        pb_industry_score = 40
                        pb_relative = "行业内高账面价值"
                    
                    comparison_result['relative_metrics']['pb_comparison'] = {
                        'value': current_pb,
                        'industry_benchmark': industry_pb_benchmark,
                        'relative_position': pb_relative,
                        'score': pb_industry_score
                    }
                    industry_score += pb_industry_score * 0.2
                else:
                    # PB数据缺失，给予中性评分
                    industry_score += 60 * 0.2
            except Exception as pb_error:
                logger.warning(f"PB行业对比计算失败 {symbol}: {pb_error}")
                industry_score += 60 * 0.2  # 给予中性评分
            
            # 债务水平行业对比
            if current_debt_ratio > 0:
                industry_debt_benchmark = industry_benchmarks['debt_to_equity']['good']
                if current_debt_ratio <= industry_debt_benchmark * 0.5:
                    debt_industry_score = 100
                    debt_relative = "行业内低债务"
                elif current_debt_ratio <= industry_debt_benchmark:
                    debt_industry_score = 80
                    debt_relative = "行业内适度债务"
                elif current_debt_ratio <= industry_debt_benchmark * 1.5:
                    debt_industry_score = 60
                    debt_relative = "行业内较高债务"
                else:
                    debt_industry_score = 40
                    debt_relative = "行业内高债务"
                
                comparison_result['relative_metrics']['debt_comparison'] = {
                    'value': current_debt_ratio,
                    'industry_benchmark': industry_debt_benchmark,
                    'relative_position': debt_relative,
                    'score': debt_industry_score
                }
                industry_score += debt_industry_score * 0.2
            
            comparison_result['industry_adjusted_score'] = industry_score / 100 if industry_score > 0 else 0.5
            comparison_result['summary'] = self._get_industry_summary(industry_score / 100 if industry_score > 0 else 0.5)
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"行业对比分析失败 {symbol}: {e}")
            return {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'industry_adjusted_score': 0.5,
                'relative_metrics': {},
                'summary': '无法进行行业对比'
            }
    
    def _get_industry_benchmarks(self, sector: str) -> Dict:
        """根据行业获取基准值"""
        # 不同行业的基准值
        industry_specific_benchmarks = {
            'Technology': {
                'pe_ratio': {'excellent': 25, 'good': 40, 'fair': 60, 'poor': 80},
                'roe': {'excellent': 0.25, 'good': 0.20, 'fair': 0.15, 'poor': 0.10},
                'pb_ratio': {'excellent': 5.0, 'good': 10.0, 'fair': 20.0, 'poor': 30.0},
                'debt_to_equity': {'excellent': 0.1, 'good': 0.2, 'fair': 0.4, 'poor': 0.8}
            },
            'Communication Services': {
                'pe_ratio': {'excellent': 15, 'good': 25, 'fair': 35, 'poor': 45},
                'roe': {'excellent': 0.30, 'good': 0.25, 'fair': 0.20, 'poor': 0.15},
                'pb_ratio': {'excellent': 3.0, 'good': 6.0, 'fair': 10.0, 'poor': 15.0},
                'debt_to_equity': {'excellent': 0.05, 'good': 0.15, 'fair': 0.30, 'poor': 0.50}
            },
            'Financial Services': {
                'pe_ratio': {'excellent': 10, 'good': 15, 'fair': 20, 'poor': 25},
                'roe': {'excellent': 0.15, 'good': 0.12, 'fair': 0.08, 'poor': 0.05},
                'pb_ratio': {'excellent': 0.8, 'good': 1.2, 'fair': 1.8, 'poor': 2.5},
                'debt_to_equity': {'excellent': 2.0, 'good': 4.0, 'fair': 6.0, 'poor': 8.0}
            },
            'Healthcare': {
                'pe_ratio': {'excellent': 15, 'good': 25, 'fair': 35, 'poor': 45},
                'roe': {'excellent': 0.18, 'good': 0.15, 'fair': 0.10, 'poor': 0.06},
                'pb_ratio': {'excellent': 1.5, 'good': 3.0, 'fair': 5.0, 'poor': 7.0},
                'debt_to_equity': {'excellent': 0.3, 'good': 0.6, 'fair': 1.0, 'poor': 1.5}
            },
            'Energy': {
                'pe_ratio': {'excellent': 8, 'good': 12, 'fair': 18, 'poor': 25},
                'roe': {'excellent': 0.15, 'good': 0.10, 'fair': 0.06, 'poor': 0.03},
                'pb_ratio': {'excellent': 0.8, 'good': 1.5, 'fair': 2.5, 'poor': 4.0},
                'debt_to_equity': {'excellent': 0.4, 'good': 0.8, 'fair': 1.5, 'poor': 2.5}
            },
            'Consumer Cyclical': {
                'pe_ratio': {'excellent': 12, 'good': 18, 'fair': 25, 'poor': 35},
                'roe': {'excellent': 0.18, 'good': 0.12, 'fair': 0.08, 'poor': 0.04},
                'pb_ratio': {'excellent': 1.2, 'good': 2.5, 'fair': 4.0, 'poor': 6.0},
                'debt_to_equity': {'excellent': 0.3, 'good': 0.6, 'fair': 1.2, 'poor': 2.0}
            },
            'Consumer Defensive': {
                'pe_ratio': {'excellent': 15, 'good': 20, 'fair': 28, 'poor': 35},
                'roe': {'excellent': 0.20, 'good': 0.15, 'fair': 0.10, 'poor': 0.06},
                'pb_ratio': {'excellent': 2.0, 'good': 3.5, 'fair': 5.0, 'poor': 7.0},
                'debt_to_equity': {'excellent': 0.4, 'good': 0.8, 'fair': 1.5, 'poor': 2.5}
            }
        }
        
        # 如果找不到特定行业，使用默认基准
        return industry_specific_benchmarks.get(sector, self.benchmarks)
    
    def _get_industry_summary(self, score: float) -> str:
        """生成行业对比总结"""
        if score >= 0.75:
            return "行业内表现优秀"
        elif score >= 0.60:
            return "行业内表现良好"
        elif score >= 0.45:
            return "行业内表现平均"
        elif score >= 0.30:
            return "行业内表现较差"
        else:
            return "行业内表现落后"
    
    def _generate_warning_alerts(self, symbol: str, data: Dict, total_score: float) -> Dict:
        """生成预警提示（第二个专家建议的实现）"""
        alerts = {
            'valuation_alerts': [],
            'fundamental_alerts': [],
            'risk_alerts': [],
            'alert_level': 'low'  # low, medium, high
        }
        
        try:
            # 估值预警
            pe_ratio = data.get('trailingPE', 0)
            if pe_ratio > 0:
                # 获取历史PE均值（简化处理，使用行业基准作为历史均值）
                sector = data.get('sector', 'Unknown')
                industry_benchmarks = self._get_industry_benchmarks(sector)
                historical_pe_avg = industry_benchmarks['pe_ratio']['good']
                
                if pe_ratio > historical_pe_avg * 1.5:
                    alerts['valuation_alerts'].append({
                        'type': 'PE_OVERVALUATION',
                        'message': f'PE比率 {pe_ratio:.2f} 超过历史平均的150%，估值过高预警',
                        'severity': 'high',
                        'current_value': pe_ratio,
                        'benchmark': historical_pe_avg * 1.5
                    })
                    alerts['alert_level'] = 'high'
            
            # 基本面恶化预警
            roe = data.get('returnOnEquity', 0)
            profit_margin = data.get('profitMargins', 0)
            
            if roe < 0.05 and profit_margin < 0.02:  # ROE低于5%且利润率低于2%
                alerts['fundamental_alerts'].append({
                    'type': 'FUNDAMENTAL_DETERIORATION',
                    'message': f'基本面恶化：ROE {roe:.2%}，利润率 {profit_margin:.2%}，基本面恶化提示',
                    'severity': 'high',
                    'roe': roe,
                    'profit_margin': profit_margin
                })
                if alerts['alert_level'] != 'high':
                    alerts['alert_level'] = 'medium'
            
            # 波动风险预警
            beta = data.get('beta', 1.0)
            debt_to_equity = data.get('debtToEquity', 0) / 100  # 转换为比率
            
            # 模拟VIX检查（实际应该从外部数据获取）
            simulated_vix = 20  # 默认VIX值，实际使用时应该获取真实VIX
            
            if simulated_vix > 30 and beta > 1.5:
                alerts['risk_alerts'].append({
                    'type': 'VOLATILITY_RISK',
                    'message': f'波动风险预警：VIX {simulated_vix}，Beta {beta:.2f}，市场波动风险较高',
                    'severity': 'medium',
                    'vix': simulated_vix,
                    'beta': beta
                })
                if alerts['alert_level'] == 'low':
                    alerts['alert_level'] = 'medium'
            
            # 债务风险预警
            if debt_to_equity > 2.0:  # 债务权益比超过200%
                alerts['risk_alerts'].append({
                    'type': 'DEBT_RISK',
                    'message': f'债务风险预警：债务权益比 {debt_to_equity:.2f}，财务杠杆过高',
                    'severity': 'medium',
                    'debt_to_equity': debt_to_equity
                })
                if alerts['alert_level'] == 'low':
                    alerts['alert_level'] = 'medium'
            
            # 综合评分下降预警
            if total_score < 0.4:
                alerts['fundamental_alerts'].append({
                    'type': 'SCORE_DECLINE',
                    'message': f'综合评分 {total_score:.2f} 偏低，整体投资价值下降',
                    'severity': 'medium',
                    'score': total_score
                })
                if alerts['alert_level'] == 'low':
                    alerts['alert_level'] = 'medium'
            
            return alerts
            
        except Exception as e:
            logger.error(f"生成预警信息失败 {symbol}: {e}")
            return alerts