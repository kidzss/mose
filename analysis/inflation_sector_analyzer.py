#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通胀-行业影响分析器
专业级通胀预期对不同行业影响的分析系统

基于现代宏观经济理论，分析通胀环境对各行业的具体影响
为个人投资者提供行业轮动策略建议

Author: AI Investment Expert
Version: 1.0 - 专业级通胀行业分析
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InflationRegime(Enum):
    """通胀环境分类"""
    DEFLATION = "通缩环境"           # <0%
    LOW_INFLATION = "低通胀环境"      # 0-2%
    MODERATE_INFLATION = "温和通胀"   # 2-4%
    HIGH_INFLATION = "高通胀环境"     # 4-6%
    HYPERINFLATION = "超高通胀"      # >6%

@dataclass
class SectorInflationImpact:
    """行业通胀影响数据类"""
    sector_name: str
    inflation_beta: float          # 通胀敏感度
    pricing_power: float           # 定价能力 (0-1)
    cost_sensitivity: float        # 成本敏感度 (0-1)
    demand_elasticity: float       # 需求弹性 (-2 to 2)
    real_asset_exposure: float     # 实物资产敞口 (0-1)
    overall_impact_score: float    # 综合影响评分 (0-1)
    investment_suggestion: str     # 投资建议
    key_drivers: List[str]         # 关键驱动因素

class InflationSectorAnalyzer:
    """通胀-行业影响分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 通胀指标数据源 - 修正数据源符号
        self.inflation_indicators = {
            'TIPS_10Y': '^FVX',         # 5年期TIPS（通胀保护债券）
            'GOLD': 'GC=F',             # 黄金期货（通胀对冲）
            'COMMODITIES': '^GSCI',     # 商品指数
            'OIL': 'CL=F',              # 原油期货
            'COPPER': 'HG=F',           # 铜期货
            'DOLLAR_INDEX': 'DX-Y.NYB', # 美元指数
            'TREASURY_10Y': '^TNX',     # 10年期国债收益率
            'VIX': '^VIX',              # 波动率指数
            # 移除不可用的BREAKEVEN数据源，改用替代指标
            'INFLATION_ETF': 'VTIP',    # 短期TIPS ETF作为通胀预期指标
            'REAL_ESTATE': 'XLRE',      # 房地产ETF（通胀对冲资产）
        }
        
        # 行业ETF映射
        self.sector_etfs = {
            'Technology': 'XLK',        # 科技
            'Healthcare': 'XLV',        # 医疗保健
            'Financial': 'XLF',         # 金融
            'Consumer_Discretionary': 'XLY',  # 消费者可选
            'Consumer_Staples': 'XLP',        # 消费者必需品
            'Energy': 'XLE',            # 能源
            'Materials': 'XLB',         # 材料
            'Industrials': 'XLI',       # 工业
            'Utilities': 'XLU',         # 公用事业
            'Real_Estate': 'XLRE',      # 房地产
            'Communication': 'XLC',     # 通讯服务
        }
        
        # 行业通胀特征配置（基于历史经验和理论）
        self.sector_inflation_profiles = {
            'Technology': {
                'inflation_beta': -0.3,     # 科技股通常受通胀负面影响
                'pricing_power': 0.7,       # 较强定价能力
                'cost_sensitivity': 0.4,    # 中等成本敏感度
                'demand_elasticity': -0.8,  # 需求相对稳定
                'real_asset_exposure': 0.2, # 低实物资产敞口
                'key_factors': ['利率敏感性', '成长股特征', '估值压缩风险', '创新驱动']
            },
            'Energy': {
                'inflation_beta': 0.8,      # 能源股受益于通胀
                'pricing_power': 0.9,       # 很强的定价能力
                'cost_sensitivity': 0.3,    # 低成本敏感度
                'demand_elasticity': -0.3,  # 需求相对刚性
                'real_asset_exposure': 0.9, # 高实物资产敞口
                'key_factors': ['原油价格', '供需关系', '地缘政治', '能源转型']
            },
            'Materials': {
                'inflation_beta': 0.6,      # 材料股通常受益于通胀
                'pricing_power': 0.6,       # 一般定价能力
                'cost_sensitivity': 0.6,    # 较高成本敏感度
                'demand_elasticity': -0.5,  # 需求有一定弹性
                'real_asset_exposure': 0.7, # 较高实物资产敞口
                'key_factors': ['商品价格', '基建需求', '制造业景气度', '库存周期']
            },
            'Financial': {
                'inflation_beta': 0.2,      # 金融股对通胀有复杂影响
                'pricing_power': 0.5,       # 中等定价能力
                'cost_sensitivity': 0.4,    # 中等成本敏感度
                'demand_elasticity': -0.4,  # 需求相对稳定
                'real_asset_exposure': 0.3, # 中等实物资产敞口
                'key_factors': ['利率环境', '信贷需求', '净息差', '违约风险']
            },
            'Consumer_Staples': {
                'inflation_beta': -0.1,     # 必需消费品相对防御性
                'pricing_power': 0.8,       # 较强定价能力
                'cost_sensitivity': 0.7,    # 较高成本敏感度
                'demand_elasticity': -0.2,  # 需求刚性
                'real_asset_exposure': 0.4, # 中等实物资产敞口
                'key_factors': ['成本转嫁能力', '品牌溢价', '消费习惯', '供应链']
            },
            'Consumer_Discretionary': {
                'inflation_beta': -0.4,     # 可选消费品受通胀负面影响
                'pricing_power': 0.4,       # 较弱定价能力
                'cost_sensitivity': 0.8,    # 高成本敏感度
                'demand_elasticity': -1.2,  # 需求弹性较大
                'real_asset_exposure': 0.3, # 中等实物资产敞口
                'key_factors': ['消费者信心', '可支配收入', '价格敏感性', '替代效应']
            },
            'Healthcare': {
                'inflation_beta': -0.2,     # 医疗保健相对防御性
                'pricing_power': 0.6,       # 较强定价能力
                'cost_sensitivity': 0.5,    # 中等成本敏感度
                'demand_elasticity': -0.3,  # 需求相对刚性
                'real_asset_exposure': 0.3, # 中等实物资产敞口
                'key_factors': ['人口老龄化', '医保政策', '药价控制', '创新需求']
            },
            'Utilities': {
                'inflation_beta': 0.1,      # 公用事业相对稳定
                'pricing_power': 0.7,       # 较强定价能力（监管保护）
                'cost_sensitivity': 0.6,    # 较高成本敏感度
                'demand_elasticity': -0.1,  # 需求非常刚性
                'real_asset_exposure': 0.8, # 高实物资产敞口
                'key_factors': ['监管政策', '资本支出', '利率敏感性', 'ESG要求']
            },
            'Real_Estate': {
                'inflation_beta': 0.5,      # 房地产是传统通胀对冲
                'pricing_power': 0.8,       # 较强定价能力
                'cost_sensitivity': 0.5,    # 中等成本敏感度
                'demand_elasticity': -0.8,  # 需求有一定弹性
                'real_asset_exposure': 0.9, # 极高实物资产敞口
                'key_factors': ['房价趋势', '租金收入', '利率环境', '供需关系']
            },
            'Industrials': {
                'inflation_beta': 0.3,      # 工业股受通胀中等影响
                'pricing_power': 0.5,       # 中等定价能力
                'cost_sensitivity': 0.7,    # 较高成本敏感度
                'demand_elasticity': -0.6,  # 需求有一定弹性
                'real_asset_exposure': 0.6, # 较高实物资产敞口
                'key_factors': ['制造业PMI', '基础设施投资', '全球贸易', '成本转嫁']
            },
            'Communication': {
                'inflation_beta': -0.2,     # 通讯服务受通胀轻微负面影响
                'pricing_power': 0.6,       # 较强定价能力
                'cost_sensitivity': 0.4,    # 中等成本敏感度
                'demand_elasticity': -0.4,  # 需求相对稳定
                'real_asset_exposure': 0.4, # 中等实物资产敞口
                'key_factors': ['用户增长', 'ARPU', '5G投资', '内容成本']
            }
        }
        
        self.cache = {}
        self.cache_expiry = timedelta(hours=1)
    
    def get_inflation_indicators(self, lookback_days: int = 252) -> Dict[str, pd.DataFrame]:
        """获取通胀相关指标数据"""
        cache_key = f"inflation_data_{lookback_days}"
        
        if (cache_key in self.cache and 
            datetime.now() - self.cache[cache_key]['timestamp'] < self.cache_expiry):
            return self.cache[cache_key]['data']
        
        inflation_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for indicator, symbol in self.inflation_indicators.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    inflation_data[indicator] = data
                    logger.info(f"✅ 获取{indicator}({symbol})数据成功: {len(data)}条记录")
                else:
                    logger.warning(f"⚠️ 获取{indicator}({symbol})数据为空")
            except Exception as e:
                logger.warning(f"⚠️ 获取{indicator}({symbol})数据失败: {e}")
                # 不终止整个分析过程，继续获取其他数据
                continue
        
        # 确保至少有一些核心数据
        if len(inflation_data) == 0:
            logger.error("❌ 所有通胀数据源都获取失败，使用模拟数据")
            # 创建模拟数据以防止系统崩溃
            mock_data = pd.DataFrame({
                'Open': [100] * 30,
                'High': [102] * 30,
                'Low': [98] * 30,
                'Close': [101] * 30,
                'Volume': [1000] * 30
            }, index=pd.date_range(start=start_date, periods=30))
            inflation_data['MOCK_DATA'] = mock_data
        
        # 缓存数据
        self.cache[cache_key] = {
            'data': inflation_data,
            'timestamp': datetime.now()
        }
        
        logger.info(f"✅ 通胀数据获取完成，成功获取 {len(inflation_data)} 个数据源")
        return inflation_data
    
    def detect_inflation_regime(self, inflation_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """检测当前通胀环境"""
        try:
            analysis = {
                'regime': InflationRegime.LOW_INFLATION,
                'confidence': 0.5,
                'indicators': {},
                'trend': 'stable',
                'risk_level': 'medium'
            }
            
            # 分析盈亏平衡通胀率
            if 'TREASURY_10Y' in inflation_data and not inflation_data['TREASURY_10Y'].empty:
                treasury_10y = inflation_data['TREASURY_10Y']['Close']
                current_10y = treasury_10y.iloc[-1]
                analysis['indicators']['treasury_10y'] = current_10y
                
                # 通胀预期趋势
                if len(treasury_10y) >= 20:
                    trend_change = (treasury_10y.iloc[-1] - treasury_10y.iloc[-20]) / treasury_10y.iloc[-20]
                    analysis['indicators']['10y_trend'] = trend_change
                    
                    if trend_change > 0.1:
                        analysis['trend'] = 'rising'
                    elif trend_change < -0.1:
                        analysis['trend'] = 'falling'
            
            # 分析商品价格
            if 'COMMODITIES' in inflation_data and not inflation_data['COMMODITIES'].empty:
                commodities = inflation_data['COMMODITIES']['Close']
                if len(commodities) >= 60:
                    commodity_momentum = (commodities.iloc[-1] - commodities.iloc[-60]) / commodities.iloc[-60]
                    analysis['indicators']['commodity_pressure'] = commodity_momentum
            
            # 分析原油价格
            if 'OIL' in inflation_data and not inflation_data['OIL'].empty:
                oil = inflation_data['OIL']['Close']
                current_oil = oil.iloc[-1]
                analysis['indicators']['oil_price'] = current_oil
                
                if len(oil) >= 20:
                    oil_change = (oil.iloc[-1] - oil.iloc[-20]) / oil.iloc[-20]
                    analysis['indicators']['oil_momentum'] = oil_change
            
            # 美元强度影响
            if 'DOLLAR_INDEX' in inflation_data and not inflation_data['DOLLAR_INDEX'].empty:
                dxy = inflation_data['DOLLAR_INDEX']['Close']
                if len(dxy) >= 20:
                    dollar_momentum = (dxy.iloc[-1] - dxy.iloc[-20]) / dxy.iloc[-20]
                    analysis['indicators']['dollar_strength'] = dollar_momentum
            
            # 综合判断通胀环境
            inflation_signals = 0
            total_signals = 0
            
            # 信号1: 10年期国债收益率
            if 'treasury_10y' in analysis['indicators']:
                treasury_10y = analysis['indicators']['treasury_10y']
                total_signals += 1
                if treasury_10y > 3.0:
                    inflation_signals += 0.8
                elif treasury_10y > 2.5:
                    inflation_signals += 0.6
                elif treasury_10y > 2.0:
                    inflation_signals += 0.4
                else:
                    inflation_signals += 0.2
            
            # 信号2: 商品价格
            if 'commodity_pressure' in analysis['indicators']:
                commodity_momentum = analysis['indicators']['commodity_pressure']
                total_signals += 1
                if commodity_momentum > 0.2:
                    inflation_signals += 0.7
                elif commodity_momentum > 0.1:
                    inflation_signals += 0.5
                elif commodity_momentum > 0:
                    inflation_signals += 0.3
            
            # 信号3: 原油价格
            if 'oil_momentum' in analysis['indicators']:
                oil_momentum = analysis['indicators']['oil_momentum']
                total_signals += 1
                if oil_momentum > 0.3:
                    inflation_signals += 0.8
                elif oil_momentum > 0.15:
                    inflation_signals += 0.6
                elif oil_momentum > 0:
                    inflation_signals += 0.4
            
            # 信号4: 黄金价格（通胀对冲）
            if 'GOLD' in inflation_data and not inflation_data['GOLD'].empty:
                gold = inflation_data['GOLD']['Close']
                if len(gold) >= 30:
                    gold_momentum = (gold.iloc[-1] - gold.iloc[-30]) / gold.iloc[-30]
                    analysis['indicators']['gold_momentum'] = gold_momentum
                    total_signals += 1
                    if gold_momentum > 0.1:
                        inflation_signals += 0.5
                    elif gold_momentum > 0.05:
                        inflation_signals += 0.3
            
            # 信号5: 美元指数（负相关）
            if 'DOLLAR_INDEX' in inflation_data and not inflation_data['DOLLAR_INDEX'].empty:
                dollar = inflation_data['DOLLAR_INDEX']['Close']
                if len(dollar) >= 30:
                    dollar_momentum = (dollar.iloc[-1] - dollar.iloc[-30]) / dollar.iloc[-30]
                    analysis['indicators']['dollar_momentum'] = dollar_momentum
                    total_signals += 1
                    if dollar_momentum < -0.05:  # 美元走弱通常推高通胀
                        inflation_signals += 0.4
                    elif dollar_momentum < -0.02:
                        inflation_signals += 0.2
            
            # 计算综合信心度
            if total_signals > 0:
                confidence = inflation_signals / total_signals
                analysis['confidence'] = min(confidence, 1.0)
            
            # 确定通胀环境
            if confidence > 0.7:
                if inflation_signals > 2.5:
                    analysis['regime'] = InflationRegime.HIGH_INFLATION
                elif inflation_signals > 1.5:
                    analysis['regime'] = InflationRegime.MODERATE_INFLATION
                else:
                    analysis['regime'] = InflationRegime.LOW_INFLATION
            elif confidence > 0.4:
                analysis['regime'] = InflationRegime.MODERATE_INFLATION
            else:
                analysis['regime'] = InflationRegime.LOW_INFLATION
            
            # 设置风险等级
            if analysis['regime'] in [InflationRegime.HIGH_INFLATION, InflationRegime.HYPERINFLATION]:
                analysis['risk_level'] = 'high'
            elif analysis['regime'] == InflationRegime.MODERATE_INFLATION:
                analysis['risk_level'] = 'medium'
            else:
                analysis['risk_level'] = 'low'
                
            return analysis
            
        except Exception as e:
            logger.error(f"检测通胀环境时发生错误: {e}")
            return {
                'regime': InflationRegime.LOW_INFLATION,
                'confidence': 0.5,
                'indicators': {},
                'trend': 'stable',
                'risk_level': 'medium'
            }
    
    def analyze_sector_inflation_impact(self, inflation_regime: Dict, 
                                      current_inflation_level: float = 0.02) -> Dict[str, SectorInflationImpact]:
        """分析各行业受通胀影响"""
        try:
            sector_impacts = {}
            regime = inflation_regime['regime']
            confidence = inflation_regime['confidence']
            
            for sector, profile in self.sector_inflation_profiles.items():
                # 计算通胀调整后的影响评分
                base_beta = profile['inflation_beta']
                pricing_power = profile['pricing_power']
                cost_sensitivity = profile['cost_sensitivity']
                demand_elasticity = profile['demand_elasticity']
                real_asset_exposure = profile['real_asset_exposure']
                
                # 根据通胀环境调整影响
                if regime == InflationRegime.HIGH_INFLATION:
                    # 高通胀环境下的调整
                    inflation_adjustment = base_beta * 1.5 * confidence
                    cost_impact = -cost_sensitivity * 0.8
                    pricing_benefit = pricing_power * 0.6
                    
                elif regime == InflationRegime.MODERATE_INFLATION:
                    # 温和通胀环境下的调整
                    inflation_adjustment = base_beta * 1.0 * confidence
                    cost_impact = -cost_sensitivity * 0.5
                    pricing_benefit = pricing_power * 0.4
                    
                else:  # LOW_INFLATION or DEFLATION
                    # 低通胀环境下的调整
                    inflation_adjustment = base_beta * 0.5 * confidence
                    cost_impact = -cost_sensitivity * 0.2
                    pricing_benefit = pricing_power * 0.2
                
                # 计算综合影响评分
                overall_impact = (
                    0.4 * inflation_adjustment +     # 直接通胀敏感性
                    0.3 * (pricing_benefit + cost_impact) +  # 成本-定价动态
                    0.2 * real_asset_exposure * base_beta +  # 实物资产敞口
                    0.1 * (-demand_elasticity * current_inflation_level)  # 需求弹性影响
                )
                
                # 标准化到0-1区间
                normalized_score = max(0, min(1, (overall_impact + 1) / 2))
                
                # 生成投资建议
                if normalized_score > 0.7:
                    suggestion = f"在{regime.value}下，{sector}行业受益显著，建议增配"
                elif normalized_score > 0.5:
                    suggestion = f"在{regime.value}下，{sector}行业表现中性，维持配置"
                else:
                    suggestion = f"在{regime.value}下，{sector}行业面临挑战，建议减配"
                
                sector_impacts[sector] = SectorInflationImpact(
                    sector_name=sector,
                    inflation_beta=base_beta,
                    pricing_power=pricing_power,
                    cost_sensitivity=cost_sensitivity,
                    demand_elasticity=demand_elasticity,
                    real_asset_exposure=real_asset_exposure,
                    overall_impact_score=normalized_score,
                    investment_suggestion=suggestion,
                    key_drivers=profile['key_factors']
                )
            
            return sector_impacts
            
        except Exception as e:
            logger.error(f"分析行业通胀影响失败: {e}")
            return {}
    
    def generate_inflation_sector_report(self) -> Dict:
        """生成完整的通胀-行业影响分析报告"""
        logger.info("🔍 开始生成通胀-行业影响分析报告...")
        
        try:
            # 获取通胀数据
            inflation_data = self.get_inflation_indicators()
            if not inflation_data:
                logger.error("❌ 无法获取通胀数据")
                return self._generate_fallback_report()
            
            # 检测通胀环境
            inflation_regime = self.detect_inflation_regime(inflation_data)
            logger.info(f"📊 检测到通胀环境: {inflation_regime['regime'].value}")
            
            # 分析行业影响
            sector_impacts = self.analyze_sector_inflation_impact(inflation_regime)
            
            # 生成投资建议
            recommendations = self._generate_investment_recommendations(inflation_regime, sector_impacts)
            
            # 生成风险警告
            risk_warnings = self._generate_risk_warnings(inflation_regime, sector_impacts)
            
            # 生成执行摘要
            executive_summary = self._generate_executive_summary(inflation_regime, sector_impacts)
            
            # 构建完整报告
            report = {
                'timestamp': datetime.now().isoformat(),
                'inflation_regime': {
                    'type': inflation_regime['regime'].value,
                    'confidence': inflation_regime['confidence'],
                    'trend': inflation_regime['trend'],
                    'risk_level': inflation_regime['risk_level'],
                    'indicators': inflation_regime['indicators']
                },
                'sector_analysis': {},
                'investment_recommendations': recommendations,
                'risk_warnings': risk_warnings,
                'executive_summary': executive_summary,
                'methodology_note': "基于多因子通胀环境检测模型和行业特征分析"
            }
            
            # 整理行业分析数据
            for sector, impact in sector_impacts.items():
                report['sector_analysis'][sector] = {
                    'sector_name': impact.sector_name,
                    'overall_score': impact.overall_impact_score,
                    'inflation_beta': impact.inflation_beta,
                    'pricing_power': impact.pricing_power,
                    'investment_suggestion': impact.investment_suggestion,
                    'key_drivers': impact.key_drivers
                }
            
            logger.info(f"✅ 通胀-行业影响分析报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"❌ 生成通胀分析报告时发生错误: {e}")
            return self._generate_fallback_report()
    
    def _generate_fallback_report(self) -> Dict:
        """生成降级报告（当主要分析失败时）"""
        return {
            'timestamp': datetime.now().isoformat(),
            'inflation_regime': {
                'type': '温和通胀环境',
                'confidence': 0.5,
                'trend': 'stable',
                'risk_level': 'medium',
                'indicators': {}
            },
            'sector_analysis': {
                'Technology': {
                    'sector_name': '科技',
                    'overall_score': 0.5,
                    'inflation_beta': -0.3,
                    'pricing_power': 0.7,
                    'investment_suggestion': '谨慎关注',
                    'key_drivers': ['利率敏感性', '估值影响']
                }
            },
            'investment_recommendations': [
                "由于数据获取问题，建议维持当前投资组合配置",
                "关注实物资产配置以对冲通胀风险",
                "建议咨询投资顾问获取更详细分析"
            ],
            'risk_warnings': [
                "当前分析基于有限数据，请谨慎使用",
                "建议结合其他信息源进行投资决策"
            ],
            'executive_summary': '由于数据源问题，本次分析采用降级模式。建议保持投资组合多样化。',
            'methodology_note': "降级分析模式 - 建议后续重新运行完整分析"
        }
    
    def _generate_investment_recommendations(self, inflation_regime: Dict, 
                                           sector_impacts: Dict) -> List[str]:
        """生成投资建议"""
        recommendations = []
        
        # 按影响评分排序
        sorted_sectors = sorted(
            sector_impacts.items(), 
            key=lambda x: x[1].overall_impact_score, 
            reverse=True
        )
        
        regime = inflation_regime['regime']
        confidence = inflation_regime['confidence']
        
        # 顶级推荐（前3名）
        if len(sorted_sectors) >= 3:
            top_sectors = [sector for sector, _ in sorted_sectors[:3]]
            recommendations.append(
                f"📈 在{regime.value}下，优先配置：{', '.join(top_sectors)}"
            )
        
        # 避开的行业（后3名）
        if len(sorted_sectors) >= 3:
            bottom_sectors = [sector for sector, _ in sorted_sectors[-3:]]
            recommendations.append(
                f"📉 建议减少配置：{', '.join(bottom_sectors)}"
            )
        
        # 基于通胀环境的特殊建议
        if regime == InflationRegime.HIGH_INFLATION:
            recommendations.extend([
                "💰 考虑增加实物资产敞口（能源、材料、房地产）",
                "🛡️ 关注具备定价能力的必需消费品公司",
                "⚠️ 避免高估值成长股和利率敏感股"
            ])
        elif regime == InflationRegime.MODERATE_INFLATION:
            recommendations.extend([
                "⚖️ 平衡配置周期性和防御性行业",
                "📊 关注能够有效转嫁成本的公司",
                "🔄 考虑行业轮动策略"
            ])
        elif regime == InflationRegime.LOW_INFLATION:
            recommendations.extend([
                "🚀 可以适度配置成长股",
                "💡 关注科技和消费者可选行业",
                "🏦 金融股在低通胀环境下可能承压"
            ])
        
        return recommendations
    
    def _generate_risk_warnings(self, inflation_regime: Dict, 
                               sector_impacts: Dict) -> List[str]:
        """生成风险警告"""
        warnings = []
        
        regime = inflation_regime['regime']
        confidence = inflation_regime['confidence']
        trend = inflation_regime['trend']
        
        # 基于通胀环境的风险
        if regime == InflationRegime.HIGH_INFLATION and confidence > 0.7:
            warnings.extend([
                "🔥 高通胀环境可能导致央行激进加息",
                "📉 估值收缩风险加大，特别是成长股",
                "💸 消费者可支配收入下降，影响可选消费"
            ])
        
        if trend == 'rising':
            warnings.append("📈 通胀预期上升，需密切关注央行政策动向")
        
        # 行业特定风险
        high_risk_sectors = [
            sector for sector, impact in sector_impacts.items()
            if impact.overall_impact_score < 0.3
        ]
        
        if high_risk_sectors:
            warnings.append(
                f"⚠️ 以下行业在当前通胀环境下风险较高：{', '.join(high_risk_sectors)}"
            )
        
        # 美元强度风险
        if inflation_regime.get('indicators', {}).get('dollar_strength', 0) > 0.1:
            warnings.append("💵 美元强势可能抑制出口导向型行业表现")
        
        return warnings
    
    def _generate_executive_summary(self, inflation_regime: Dict, 
                                   sector_impacts: Dict) -> str:
        """生成执行摘要"""
        regime = inflation_regime['regime'].value
        confidence = inflation_regime['confidence']
        
        # 找出最受益和最受损的行业
        sorted_sectors = sorted(
            sector_impacts.items(), 
            key=lambda x: x[1].overall_impact_score, 
            reverse=True
        )
        
        best_sector = sorted_sectors[0][0] if sorted_sectors else "未知"
        worst_sector = sorted_sectors[-1][0] if sorted_sectors else "未知"
        
        summary = f"""
当前通胀环境分析：{regime}（信心度：{confidence:.1%}）

🏆 最受益行业：{best_sector}
📉 最受挑战行业：{worst_sector}

投资策略要点：
• 通胀环境下应重点关注定价能力强、实物资产敞口高的行业
• 避免成本敏感度高、需求弹性大的行业
• 建议采用行业轮动策略，根据通胀预期动态调整配置

风险提示：通胀预期具有不确定性，建议分散投资并密切关注宏观经济指标变化。
        """.strip()
        
        return summary

# 工厂函数
def create_inflation_sector_analyzer() -> InflationSectorAnalyzer:
    """创建通胀-行业分析器实例"""
    return InflationSectorAnalyzer()

if __name__ == "__main__":
    # 示例用法
    analyzer = create_inflation_sector_analyzer()
    report = analyzer.generate_inflation_sector_report()
    
    if report:
        print("=" * 60)
        print("通胀-行业影响分析报告")
        print("=" * 60)
        print(report['executive_summary'])
        print("\n投资建议：")
        for rec in report['investment_recommendations']:
            print(f"  • {rec}")
    else:
        print("报告生成失败") 