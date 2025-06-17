"""
宏观因子分析器
用于集成宏观经济指标到投资决策系统中
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacroFactorAnalyzer:
    """宏观因子分析器"""
    
    def __init__(self):
        """初始化宏观因子分析器"""
        self.macro_symbols = {
            'VIX': '^VIX',          # 恐慌指数
            'DXY': 'DX-Y.NYB',      # 美元指数
            'TNX': '^TNX',          # 10年期国债收益率
            'FVX': '^FVX',          # 5年期国债收益率
            'IRX': '^IRX',          # 3个月国债收益率
            'GOLD': 'GC=F',         # 黄金期货
            'OIL': 'CL=F',          # 原油期货
            'SPY': 'SPY',           # 标普500ETF
            'QQQ': 'QQQ',           # 纳斯达克100ETF
        }
        
        # 宏观因子权重
        self.factor_weights = {
            'interest_rate_environment': 0.25,  # 利率环境
            'market_sentiment': 0.30,           # 市场情绪
            'dollar_strength': 0.20,            # 美元强度
            'inflation_expectation': 0.15,      # 通胀预期
            'economic_growth': 0.10             # 经济增长
        }
        
        self.cache = {}
        self.last_update = None
        
    def fetch_macro_data(self, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """获取宏观经济数据"""
        try:
            macro_data = {}
            
            for name, symbol in self.macro_symbols.items():
                try:
                    data = yf.download(symbol, period=period, interval="1d")
                    if not data.empty:
                        macro_data[name] = data
                        logger.info(f"成功获取 {name} 数据")
                    else:
                        logger.warning(f"无法获取 {name} 数据")
                except Exception as e:
                    logger.error(f"获取 {name} 数据失败: {e}")
                    continue
                    
            self.cache['macro_data'] = macro_data
            self.last_update = datetime.now()
            
            return macro_data
            
        except Exception as e:
            logger.error(f"获取宏观数据失败: {e}")
            return {}
    
    def analyze_interest_rate_environment(self, macro_data: Dict) -> Dict:
        """分析利率环境"""
        try:
            analysis = {}
            
            # 收益率曲线分析
            if ('TNX' in macro_data and 'FVX' in macro_data and 'IRX' in macro_data and 
                not macro_data['TNX'].empty and not macro_data['FVX'].empty and not macro_data['IRX'].empty):
                tnx = macro_data['TNX']['Close'].iloc[-1].item()  # 10年期
                fvx = macro_data['FVX']['Close'].iloc[-1].item()  # 5年期
                irx = macro_data['IRX']['Close'].iloc[-1].item()  # 3个月
                
                # 收益率曲线斜率
                yield_curve_slope = tnx - irx
                analysis['yield_curve_slope'] = yield_curve_slope
                
                # 收益率曲线形态
                if yield_curve_slope > 2:
                    analysis['curve_shape'] = 'steep'
                    analysis['curve_score'] = 0.8  # 有利于股票
                elif yield_curve_slope > 0:
                    analysis['curve_shape'] = 'normal'
                    analysis['curve_score'] = 0.6
                else:
                    analysis['curve_shape'] = 'inverted'
                    analysis['curve_score'] = 0.2  # 衰退信号
                
                # 利率趋势
                if len(macro_data['TNX']) >= 20:
                    tnx_change = (macro_data['TNX']['Close'].iloc[-1].item() / 
                                macro_data['TNX']['Close'].iloc[-20].item() - 1)
                else:
                    tnx_change = 0
                analysis['rate_trend'] = tnx_change
                
                if tnx_change < -0.05:
                    analysis['rate_trend_score'] = 0.8  # 利率下降有利
                elif tnx_change > 0.05:
                    analysis['rate_trend_score'] = 0.3  # 利率上升不利
                else:
                    analysis['rate_trend_score'] = 0.5
                    
            return analysis
            
        except Exception as e:
            logger.error(f"分析利率环境失败: {e}")
            return {}
    
    def analyze_market_sentiment(self, macro_data: Dict) -> Dict:
        """分析市场情绪"""
        try:
            analysis = {}
            
            # VIX恐慌指数分析
            if 'VIX' in macro_data and not macro_data['VIX'].empty:
                vix_current = macro_data['VIX']['Close'].iloc[-1].item()
                vix_ma20 = macro_data['VIX']['Close'].rolling(20).mean().iloc[-1].item()
                
                analysis['vix_level'] = vix_current
                analysis['vix_vs_ma20'] = vix_current / vix_ma20 - 1
                
                # VIX情绪评分
                if vix_current < 15:
                    analysis['vix_sentiment'] = 'complacent'
                    analysis['vix_score'] = 0.3  # 过度乐观，谨慎
                elif vix_current < 25:
                    analysis['vix_sentiment'] = 'normal'
                    analysis['vix_score'] = 0.7
                elif vix_current < 35:
                    analysis['vix_sentiment'] = 'elevated'
                    analysis['vix_score'] = 0.8  # 适度恐慌，买入机会
                else:
                    analysis['vix_sentiment'] = 'panic'
                    analysis['vix_score'] = 0.9  # 极度恐慌，绝佳买入
            
            # 市场表现分析
            if 'SPY' in macro_data and not macro_data['SPY'].empty:
                spy_data = macro_data['SPY']
                spy_returns = spy_data['Close'].pct_change()
                
                # 市场动量
                analysis['market_momentum'] = spy_returns.rolling(20).mean().iloc[-1].item()
                
                # 市场波动率
                analysis['market_volatility'] = spy_returns.rolling(20).std().iloc[-1].item()
                
                # 相对强弱分析
                if 'QQQ' in macro_data and not macro_data['QQQ'].empty:
                    qqq_returns = macro_data['QQQ']['Close'].pct_change()
                    spy_mean = spy_returns.rolling(20).mean().iloc[-1].item()
                    qqq_mean = qqq_returns.rolling(20).mean().iloc[-1].item()
                    if spy_mean != 0:
                        analysis['tech_vs_market'] = (qqq_mean / spy_mean - 1)
                    else:
                        analysis['tech_vs_market'] = 0
                    
            return analysis
            
        except Exception as e:
            logger.error(f"分析市场情绪失败: {e}")
            return {}
    
    def analyze_dollar_strength(self, macro_data: Dict) -> Dict:
        """分析美元强度"""
        try:
            analysis = {}
            
            if 'DXY' in macro_data and not macro_data['DXY'].empty:
                dxy_data = macro_data['DXY']['Close']
                dxy_current = dxy_data.iloc[-1].item()
                dxy_ma50 = dxy_data.rolling(50).mean().iloc[-1].item()
                dxy_ma200 = dxy_data.rolling(200).mean().iloc[-1].item()
                
                analysis['dxy_level'] = dxy_current
                analysis['dxy_vs_ma50'] = dxy_current / dxy_ma50 - 1
                analysis['dxy_vs_ma200'] = dxy_current / dxy_ma200 - 1
                
                # 美元强度评分
                if dxy_current > dxy_ma50 and dxy_current > dxy_ma200:
                    analysis['dollar_trend'] = 'strong'
                    analysis['dollar_score'] = 0.4  # 强美元对股票不利
                elif dxy_current < dxy_ma50 and dxy_current < dxy_ma200:
                    analysis['dollar_trend'] = 'weak'
                    analysis['dollar_score'] = 0.8  # 弱美元有利股票
                else:
                    analysis['dollar_trend'] = 'neutral'
                    analysis['dollar_score'] = 0.6
                    
            return analysis
            
        except Exception as e:
            logger.error(f"分析美元强度失败: {e}")
            return {}
    
    def calculate_macro_score(self) -> Dict:
        """计算综合宏观得分"""
        try:
            if not self.cache.get('macro_data'):
                logger.warning("没有宏观数据，先获取数据")
                return {}
                
            macro_data = self.cache['macro_data']
            
            # 各项分析
            interest_analysis = self.analyze_interest_rate_environment(macro_data)
            sentiment_analysis = self.analyze_market_sentiment(macro_data)
            dollar_analysis = self.analyze_dollar_strength(macro_data)
            
            # 计算综合得分
            total_score = 0
            weight_sum = 0
            
            if interest_analysis.get('curve_score'):
                total_score += (interest_analysis['curve_score'] * 
                              self.factor_weights['interest_rate_environment'])
                weight_sum += self.factor_weights['interest_rate_environment']
                
            if sentiment_analysis.get('vix_score'):
                total_score += (sentiment_analysis['vix_score'] * 
                              self.factor_weights['market_sentiment'])
                weight_sum += self.factor_weights['market_sentiment']
                
            if dollar_analysis.get('dollar_score'):
                total_score += (dollar_analysis['dollar_score'] * 
                              self.factor_weights['dollar_strength'])
                weight_sum += self.factor_weights['dollar_strength']
            
            # 归一化得分
            final_score = total_score / weight_sum if weight_sum > 0 else 0.5
            
            return {
                'macro_score': final_score,
                'components': {
                    'interest_rate': interest_analysis,
                    'market_sentiment': sentiment_analysis,
                    'dollar_strength': dollar_analysis
                },
                'recommendation': self._get_recommendation(final_score),
                'last_update': self.last_update
            }
            
        except Exception as e:
            logger.error(f"计算宏观得分失败: {e}")
            return {}
    
    def _get_recommendation(self, score: float) -> str:
        """根据得分生成建议"""
        if score >= 0.7:
            return "宏观环境有利，建议积极投资"
        elif score >= 0.5:
            return "宏观环境中性，保持谨慎乐观"
        elif score >= 0.3:
            return "宏观环境偏弱，建议降低仓位"
        else:
            return "宏观环境恶化，建议保守策略"
    
    def get_sector_impact(self, macro_score: Dict) -> Dict:
        """获取宏观因子对不同行业的影响"""
        try:
            sector_impact = {}
            
            components = macro_score.get('components', {})
            
            # 科技股影响
            tech_score = 0.5
            if components.get('interest_rate'):
                # 利率上升对科技股不利
                rate_impact = 1 - components['interest_rate'].get('rate_trend_score', 0.5)
                tech_score += (rate_impact - 0.5) * 0.4
                
            if components.get('market_sentiment'):
                # 市场情绪对科技股影响较大
                sentiment_impact = components['market_sentiment'].get('vix_score', 0.5)
                tech_score += (sentiment_impact - 0.5) * 0.3
                
            sector_impact['Technology'] = max(0, min(1, tech_score))
            
            # 金融股影响
            financial_score = 0.5
            if components.get('interest_rate'):
                # 利率上升对金融股有利
                rate_impact = components['interest_rate'].get('rate_trend_score', 0.5)
                financial_score += (1 - rate_impact - 0.5) * 0.6
                
            sector_impact['Financial'] = max(0, min(1, financial_score))
            
            # 能源股影响
            energy_score = 0.5
            if components.get('dollar_strength'):
                # 美元走强对能源股不利
                dollar_impact = 1 - components['dollar_strength'].get('dollar_score', 0.5)
                energy_score += (dollar_impact - 0.5) * 0.4
                
            sector_impact['Energy'] = max(0, min(1, energy_score))
            
            # 医药股相对稳定
            sector_impact['Healthcare'] = 0.6
            
            return sector_impact
            
        except Exception as e:
            logger.error(f"计算行业影响失败: {e}")
            return {}


if __name__ == "__main__":
    # 测试宏观因子分析器
    analyzer = MacroFactorAnalyzer()
    
    print("获取宏观数据...")
    macro_data = analyzer.fetch_macro_data()
    
    print("计算宏观得分...")
    macro_score = analyzer.calculate_macro_score()
    
    print("宏观分析结果:")
    print(f"综合得分: {macro_score.get('macro_score', 0):.2f}")
    print(f"建议: {macro_score.get('recommendation', '无')}")
    
    print("\n行业影响分析:")
    sector_impact = analyzer.get_sector_impact(macro_score)
    for sector, impact in sector_impact.items():
        print(f"{sector}: {impact:.2f}") 