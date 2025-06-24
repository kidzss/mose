"""
第二阶段专业级多因子量化筛选器

基于现代投资组合理论和多因子模型的专业量化筛选系统
实现Fama-French五因子模型、机器学习增强、风险调整收益等专业功能

Author: AI Quantitative Expert
Version: 2.1 - 增加市场情绪分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 导入基础模块
from data.data_interface import DataInterface
from strategy.strategy_factory import StrategyFactory
from utils.unified_email_api import send_html, send_markdown
from utils.alpha_vantage_client import AlphaVantageClient
from utils.yfinance_client import YFinanceClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketSentiment:
    """市场情绪数据类"""
    vix_level: float
    vix_change: float
    pcr_level: float
    pcr_change: float
    market_breadth: float
    fear_greed_index: float
    sentiment_score: float

@dataclass
class FactorExposure:
    """因子暴露度数据类"""
    market_beta: float
    size_factor: float  # SMB (Small Minus Big)
    value_factor: float  # HML (High Minus Low)
    profitability_factor: float  # RMW (Robust Minus Weak)
    investment_factor: float  # CMA (Conservative Minus Aggressive)
    quality_factor: float
    momentum_factor: float
    low_volatility_factor: float
    sentiment_factor: float  # 新增：市场情绪因子

@dataclass
class RiskMetrics:
    """风险指标数据类"""
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    max_drawdown: float
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR
    beta: float
    tracking_error: float

class MarketSentimentAnalyzer:
    """市场情绪分析器"""
    
    def __init__(self):
        import yfinance as yf
        self.yf = yf
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = timedelta(minutes=15)  # 15分钟缓存
    
    def get_vix_data(self) -> Dict[str, float]:
        """获取VIX恐慌指数数据"""
        try:
            cache_key = 'vix_data'
            if (cache_key in self.cache and 
                cache_key in self.cache_time and 
                datetime.now() - self.cache_time[cache_key] < self.cache_duration):
                return self.cache[cache_key]
            
            # 获取VIX数据
            vix_ticker = self.yf.Ticker('^VIX')
            vix_data = vix_ticker.history(period='10d')
            if vix_data is None or vix_data.empty:
                return {'vix_level': 20.0, 'vix_change': 0.0}
            
            current_vix = vix_data['Close'].iloc[-1]
            prev_vix = vix_data['Close'].iloc[-2] if len(vix_data) > 1 else current_vix
            vix_change = ((current_vix - prev_vix) / prev_vix) * 100
            
            result = {
                'vix_level': float(current_vix),
                'vix_change': float(vix_change)
            }
            
            # 缓存结果
            self.cache[cache_key] = result
            self.cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.warning(f"获取VIX数据失败: {e}")
            return {'vix_level': 20.0, 'vix_change': 0.0}
    
    def get_put_call_ratio(self) -> Dict[str, float]:
        """获取Put/Call比率数据"""
        try:
            cache_key = 'pcr_data'
            if (cache_key in self.cache and 
                cache_key in self.cache_time and 
                datetime.now() - self.cache_time[cache_key] < self.cache_duration):
                return self.cache[cache_key]
            
            # 获取SPY期权数据来估算PCR
            spy_ticker = self.yf.Ticker('SPY')
            spy_data = spy_ticker.history(period='5d')
            if spy_data is None or spy_data.empty:
                return {'pcr_level': 1.0, 'pcr_change': 0.0}
            
            # 简化版PCR计算：基于价格波动性估算
            returns = spy_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # 高波动性通常对应高PCR
            pcr_level = 0.8 + (volatility * 2)  # 基础0.8，波动性影响
            pcr_level = max(0.3, min(2.0, pcr_level))  # 限制在合理范围
            
            # 计算PCR变化
            prev_volatility = returns.iloc[:-1].std() * np.sqrt(252) if len(returns) > 1 else volatility
            prev_pcr = 0.8 + (prev_volatility * 2)
            pcr_change = ((pcr_level - prev_pcr) / prev_pcr) * 100
            
            result = {
                'pcr_level': float(pcr_level),
                'pcr_change': float(pcr_change)
            }
            
            # 缓存结果
            self.cache[cache_key] = result
            self.cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.warning(f"获取PCR数据失败: {e}")
            return {'pcr_level': 1.0, 'pcr_change': 0.0}
    
    def calculate_market_breadth(self, symbols: List[str] = None) -> float:
        """计算市场宽度指标"""
        try:
            if symbols is None:
                # 使用主要指数成分股
                symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
            
            advancing = 0
            total = 0
            
            for symbol in symbols:
                try:
                    ticker = self.yf.Ticker(symbol)
                    data = ticker.history(period='2d')
                    if data is not None and len(data) >= 2:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2]
                        if current_price > prev_price:
                            advancing += 1
                        total += 1
                except:
                    continue
            
            if total == 0:
                return 0.5  # 中性值
            
            return advancing / total
            
        except Exception as e:
            logger.warning(f"计算市场宽度失败: {e}")
            return 0.5
    
    def calculate_fear_greed_index(self, vix_data: Dict[str, float], 
                                 pcr_data: Dict[str, float], 
                                 market_breadth: float) -> float:
        """计算恐惧贪婪指数"""
        try:
            vix_level = vix_data['vix_level']
            pcr_level = pcr_data['pcr_level']
            
            # VIX评分 (0-100，低VIX=高贪婪)
            if vix_level < 15:
                vix_score = 90  # 极度贪婪
            elif vix_level < 20:
                vix_score = 70  # 贪婪
            elif vix_level < 25:
                vix_score = 50  # 中性
            elif vix_level < 30:
                vix_score = 30  # 恐惧
            else:
                vix_score = 10  # 极度恐惧
            
            # PCR评分 (0-100，低PCR=高贪婪)
            if pcr_level < 0.7:
                pcr_score = 80  # 贪婪
            elif pcr_level < 1.0:
                pcr_score = 60  # 中性偏贪婪
            elif pcr_level < 1.3:
                pcr_score = 40  # 中性偏恐惧
            else:
                pcr_score = 20  # 恐惧
            
            # 市场宽度评分 (0-100)
            breadth_score = market_breadth * 100
            
            # 综合恐惧贪婪指数
            fear_greed = (vix_score * 0.4 + pcr_score * 0.3 + breadth_score * 0.3)
            
            return max(0, min(100, fear_greed))
            
        except Exception as e:
            logger.warning(f"计算恐惧贪婪指数失败: {e}")
            return 50.0
    
    def get_market_sentiment(self) -> MarketSentiment:
        """获取综合市场情绪"""
        try:
            # 获取各项指标
            vix_data = self.get_vix_data()
            pcr_data = self.get_put_call_ratio()
            market_breadth = self.calculate_market_breadth()
            
            # 计算恐惧贪婪指数
            fear_greed = self.calculate_fear_greed_index(vix_data, pcr_data, market_breadth)
            
            # 计算综合情绪得分 (-1到1，负值=悲观，正值=乐观)
            sentiment_score = (fear_greed - 50) / 50  # 转换为-1到1
            
            return MarketSentiment(
                vix_level=vix_data['vix_level'],
                vix_change=vix_data['vix_change'],
                pcr_level=pcr_data['pcr_level'],
                pcr_change=pcr_data['pcr_change'],
                market_breadth=market_breadth,
                fear_greed_index=fear_greed,
                sentiment_score=sentiment_score
            )
            
        except Exception as e:
            logger.error(f"获取市场情绪失败: {e}")
            return MarketSentiment(20.0, 0.0, 1.0, 0.0, 0.5, 50.0, 0.0)

class FamaFrenchFactors:
    """Fama-French五因子模型实现"""
    
    def calculate_market_factor(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """计算市场因子 (Beta)"""
        try:
            if len(returns) < 30 or len(market_returns) < 30:
                return 1.0
            
            # 计算Beta
            covariance = np.cov(returns.dropna(), market_returns.dropna())[0, 1]
            market_variance = np.var(market_returns.dropna())
            
            if market_variance == 0:
                return 1.0
                
            beta = covariance / market_variance
            return max(0.1, min(3.0, beta))  # 限制在合理范围内
            
        except Exception as e:
            logger.warning(f"市场因子计算失败: {e}")
            return 1.0
    
    def calculate_size_factor(self, market_cap: float, universe_market_caps: List[float]) -> float:
        """计算规模因子 SMB (Small Minus Big)"""
        try:
            if not universe_market_caps or market_cap <= 0:
                return 0.0
            
            # 计算市值分位数
            percentile = np.percentile(universe_market_caps, [30, 70])
            
            if market_cap < percentile[0]:
                return 1.0  # 小盘股
            elif market_cap > percentile[1]:
                return -1.0  # 大盘股
            else:
                return 0.0  # 中盘股
                
        except Exception as e:
            logger.warning(f"规模因子计算失败: {e}")
            return 0.0

class QualityFactors:
    """质量因子计算"""
    
    @staticmethod
    def calculate_quality_score(financial_data: Dict[str, float]) -> float:
        """计算综合质量评分 - 增强版"""
        try:
            score = 0.0
            weights = {
                'roe': 0.30,              # 净资产收益率 (最重要)
                'roa': 0.25,              # 总资产收益率
                'debt_to_equity': 0.20,   # 债务权益比
                'current_ratio': 0.15,    # 流动比率
                'gross_margin': 0.10      # 毛利率
            }
            
            # ROE评分 (净资产收益率)
            roe = financial_data.get('roe', 0)
            if roe > 20:
                score += weights['roe'] * 100    # 优秀
            elif roe > 15:
                score += weights['roe'] * 85     # 良好
            elif roe > 10:
                score += weights['roe'] * 70     # 一般
            elif roe > 5:
                score += weights['roe'] * 40     # 较差
            elif roe > 0:
                score += weights['roe'] * 20     # 很差
            # 负ROE不加分
            
            # ROA评分 (总资产收益率)
            roa = financial_data.get('roa', roe * 0.6)  # 如果没有ROA，用ROE估算
            if roa > 15:
                score += weights['roa'] * 100
            elif roa > 10:
                score += weights['roa'] * 80
            elif roa > 5:
                score += weights['roa'] * 60
            elif roa > 2:
                score += weights['roa'] * 40
            elif roa > 0:
                score += weights['roa'] * 20
            
            # 债务比率评分 (越低越好)
            debt_ratio = financial_data.get('debt_to_equity', 1.0)
            if debt_ratio < 0.2:
                score += weights['debt_to_equity'] * 100  # 极低债务
            elif debt_ratio < 0.5:
                score += weights['debt_to_equity'] * 80   # 低债务
            elif debt_ratio < 1.0:
                score += weights['debt_to_equity'] * 60   # 中等债务
            elif debt_ratio < 2.0:
                score += weights['debt_to_equity'] * 30   # 高债务
            # 超高债务不加分
            
            # 流动比率评分
            current_ratio = financial_data.get('current_ratio', 1.5)
            if current_ratio > 2.5:
                score += weights['current_ratio'] * 100  # 流动性极好
            elif current_ratio > 2.0:
                score += weights['current_ratio'] * 85   # 流动性很好
            elif current_ratio > 1.5:
                score += weights['current_ratio'] * 70   # 流动性良好
            elif current_ratio > 1.0:
                score += weights['current_ratio'] * 40   # 流动性一般
            # 流动比率<1不加分
            
            # 毛利率评分
            gross_margin = financial_data.get('gross_margin', 25)
            if gross_margin > 50:
                score += weights['gross_margin'] * 100   # 极高毛利率
            elif gross_margin > 35:
                score += weights['gross_margin'] * 80    # 高毛利率
            elif gross_margin > 25:
                score += weights['gross_margin'] * 60    # 中等毛利率
            elif gross_margin > 15:
                score += weights['gross_margin'] * 40    # 低毛利率
            elif gross_margin > 5:
                score += weights['gross_margin'] * 20    # 很低毛利率
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"质量因子计算失败: {e}")
            return 50.0

class Phase2ProfessionalScreener:
    """第二阶段专业级多因子量化筛选器"""
    
    def __init__(self):
        self.data_interface = DataInterface()
        self.strategy_factory = StrategyFactory()
        self.ff_factors = FamaFrenchFactors()
        self.quality_factors = QualityFactors()
        # Email functionality now provided by unified_email_api
        
        # 初始化流动性分析器
        try:
            from analysis.liquidity_analyzer import LiquidityAnalyzer
            self.liquidity_analyzer = LiquidityAnalyzer()
            logger.info("流动性分析模块初始化成功")
        except ImportError:
            self.liquidity_analyzer = None
            logger.warning("流动性分析模块不可用，将跳过流动性分析部分")
        
        # 初始化通胀-行业分析器
        try:
            from analysis.inflation_sector_analyzer import InflationSectorAnalyzer
            self.inflation_sector_analyzer = InflationSectorAnalyzer()
            logger.info("通胀-行业分析模块初始化成功")
        except ImportError:
            self.inflation_sector_analyzer = None
            logger.warning("通胀-行业分析模块不可用，将跳过通胀行业分析部分")
        self.alpha_vantage = AlphaVantageClient()
        self.yfinance_client = YFinanceClient()
        
        # 新增：市场情绪分析器
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
        # 财务数据缓存
        self.financial_data_cache = {}
        
        logger.info("🚀 第二阶段专业级多因子量化筛选器初始化完成 (集成Alpha Vantage + yfinance双重真实财务数据 + 市场情绪分析)")
    
    def _get_real_financial_data(self, symbol: str) -> Dict[str, float]:
        """获取真实财务数据（优先使用yfinance，备选Alpha Vantage）"""
        try:
            # 检查缓存
            if symbol in self.financial_data_cache:
                return self.financial_data_cache[symbol]
            
            # 优先使用yfinance（免费且稳定）
            stock_info = self.yfinance_client.get_stock_info(symbol)
            if stock_info:
                financial_metrics = self.yfinance_client.extract_financial_metrics(stock_info)
                # 验证数据质量
                if financial_metrics.get('roe', 0) > 0 or financial_metrics.get('market_cap', 0) > 0:
                    self.financial_data_cache[symbol] = financial_metrics
                    return financial_metrics
            
            # 备选：Alpha Vantage
            logger.info(f"yfinance数据不足，尝试Alpha Vantage获取 {symbol}")
            overview_data = self.alpha_vantage.get_company_overview(symbol)
            if overview_data:
                financial_metrics = self.alpha_vantage.extract_financial_metrics(overview_data)
                if financial_metrics.get('roe', 0) > 0 or financial_metrics.get('market_cap', 0) > 0:
                    self.financial_data_cache[symbol] = financial_metrics
                    return financial_metrics
            
            # 如果都获取失败，使用行业平均默认值
            logger.warning(f"无法获取 {symbol} 的真实财务数据，使用行业平均默认值")
            default_data = {
                'roe': 12.0, 'roa': 8.0, 'debt_to_equity': 0.5,
                'current_ratio': 1.5, 'gross_margin': 25.0,
                'pe_ratio': 15.0, 'pb_ratio': 2.0, 'market_cap': 1000000000
            }
            self.financial_data_cache[symbol] = default_data
            return default_data
                
        except Exception as e:
            logger.warning(f"获取 {symbol} 财务数据失败: {e}")
            # 返回默认值
            default_data = {
                'roe': 12.0, 'roa': 8.0, 'debt_to_equity': 0.5,
                'current_ratio': 1.5, 'gross_margin': 25.0,
                'pe_ratio': 15.0, 'pb_ratio': 2.0, 'market_cap': 1000000000
            }
            return default_data
    
    def _calculate_market_beta(self, returns: pd.Series) -> float:
        """计算市场Beta（相对于SPY）"""
        try:
            # 简化版：基于收益率波动性估算Beta
            if len(returns) < 30:
                return 1.0
            
            # 计算相对波动性
            market_vol = 0.16  # SPY年化波动率约16%
            stock_vol = returns.std() * np.sqrt(252)
            
            # Beta估算
            beta = stock_vol / market_vol
            return max(0.1, min(3.0, beta))
            
        except Exception as e:
            logger.warning(f"Beta计算失败: {e}")
            return 1.0
    
    def _calculate_size_factor(self, financial_data: Dict[str, float]) -> float:
        """基于市值计算规模因子"""
        try:
            market_cap = financial_data.get('market_cap', 1000000000)  # 默认10亿
            
            # 规模分类（美股标准）
            if market_cap > 200000000000:  # >2000亿，超大盘
                return -1.0
            elif market_cap > 10000000000:  # >100亿，大盘
                return -0.5
            elif market_cap > 2000000000:   # >20亿，中盘
                return 0.0
            elif market_cap > 300000000:    # >3亿，小盘
                return 0.5
            else:  # <3亿，微盘
                return 1.0
                
        except Exception as e:
            logger.warning(f"规模因子计算失败: {e}")
            return 0.0
    
    def _calculate_value_factor(self, financial_data: Dict[str, float]) -> float:
        """基于估值指标计算价值因子"""
        try:
            pe_ratio = financial_data.get('pe_ratio', 15.0)
            pb_ratio = financial_data.get('pb_ratio', 2.0)
            
            # PE估值评分
            pe_score = 0
            if pe_ratio > 0:
                if pe_ratio < 10:
                    pe_score = 1.0      # 低估值
                elif pe_ratio < 15:
                    pe_score = 0.5      # 合理估值
                elif pe_ratio < 25:
                    pe_score = 0.0      # 中性
                elif pe_ratio < 40:
                    pe_score = -0.5     # 高估值
                else:
                    pe_score = -1.0     # 极高估值
            
            # PB估值评分
            pb_score = 0
            if pb_ratio > 0:
                if pb_ratio < 1.0:
                    pb_score = 1.0      # 破净，可能低估
                elif pb_ratio < 2.0:
                    pb_score = 0.5      # 合理估值
                elif pb_ratio < 3.0:
                    pb_score = 0.0      # 中性
                elif pb_ratio < 5.0:
                    pb_score = -0.5     # 高估值
                else:
                    pb_score = -1.0     # 极高估值
            
            # 综合价值因子
            return (pe_score + pb_score) / 2
            
        except Exception as e:
            logger.warning(f"价值因子计算失败: {e}")
            return 0.0
    
    def _calculate_profitability_factor(self, financial_data: Dict[str, float]) -> float:
        """基于盈利能力计算盈利因子"""
        try:
            roe = financial_data.get('roe', 12.0)
            roa = financial_data.get('roa', 8.0)
            gross_margin = financial_data.get('gross_margin', 25.0)
            
            # ROE评分
            if roe > 20:
                roe_score = 1.0
            elif roe > 15:
                roe_score = 0.5
            elif roe > 10:
                roe_score = 0.0
            elif roe > 5:
                roe_score = -0.5
            else:
                roe_score = -1.0
            
            # ROA评分
            if roa > 15:
                roa_score = 1.0
            elif roa > 10:
                roa_score = 0.5
            elif roa > 5:
                roa_score = 0.0
            elif roa > 2:
                roa_score = -0.5
            else:
                roa_score = -1.0
            
            # 毛利率评分
            if gross_margin > 50:
                margin_score = 1.0
            elif gross_margin > 35:
                margin_score = 0.5
            elif gross_margin > 25:
                margin_score = 0.0
            elif gross_margin > 15:
                margin_score = -0.5
            else:
                margin_score = -1.0
            
            # 综合盈利因子
            return (roe_score * 0.5 + roa_score * 0.3 + margin_score * 0.2)
            
        except Exception as e:
            logger.warning(f"盈利因子计算失败: {e}")
            return 0.0
    
    def _calculate_investment_factor(self, financial_data: Dict[str, float]) -> float:
        """基于投资质量计算投资因子"""
        try:
            debt_to_equity = financial_data.get('debt_to_equity', 0.5)
            current_ratio = financial_data.get('current_ratio', 1.5)
            
            # 债务比率评分（保守投资偏好低债务）
            if debt_to_equity < 0.2:
                debt_score = 1.0        # 极低债务
            elif debt_to_equity < 0.5:
                debt_score = 0.5        # 低债务
            elif debt_to_equity < 1.0:
                debt_score = 0.0        # 中等债务
            elif debt_to_equity < 2.0:
                debt_score = -0.5       # 高债务
            else:
                debt_score = -1.0       # 极高债务
            
            # 流动比率评分
            if current_ratio > 2.5:
                liquidity_score = 1.0   # 流动性极好
            elif current_ratio > 2.0:
                liquidity_score = 0.5   # 流动性很好
            elif current_ratio > 1.5:
                liquidity_score = 0.0   # 流动性良好
            elif current_ratio > 1.0:
                liquidity_score = -0.5  # 流动性一般
            else:
                liquidity_score = -1.0  # 流动性差
            
            # 综合投资因子（保守 vs 激进）
            return (debt_score * 0.6 + liquidity_score * 0.4)
            
        except Exception as e:
            logger.warning(f"投资因子计算失败: {e}")
            return 0.0
    
    def calculate_factor_exposure(self, symbol: str, data: pd.DataFrame) -> FactorExposure:
        """计算因子暴露度（使用真实财务数据 + 市场情绪分析）"""
        try:
            returns = data['close'].pct_change().dropna()
            
            # 获取真实财务数据
            financial_data = self._get_real_financial_data(symbol)
            
            # 计算市场Beta
            market_beta = self._calculate_market_beta(returns)
            
            # 基于真实财务数据计算因子
            size_factor = self._calculate_size_factor(financial_data)
            value_factor = self._calculate_value_factor(financial_data)
            profitability_factor = self._calculate_profitability_factor(financial_data)
            investment_factor = self._calculate_investment_factor(financial_data)
            
            # 质量因子（基于真实财务数据）
            quality_factor = self.quality_factors.calculate_quality_score(financial_data) / 100
            
            # 动量因子（基于价格数据）
            momentum_factor = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) if len(data) >= 20 else 0
            
            # 低波动率因子
            volatility = returns.std() * np.sqrt(252)
            low_vol_factor = max(0, 1 - volatility / 0.5)
            
            # 市场情绪因子（新增）
            try:
                market_sentiment = self.sentiment_analyzer.get_market_sentiment()
                # 将情绪得分转换为0-1范围
                sentiment_factor = (market_sentiment.sentiment_score + 1) / 2
            except Exception as e:
                logger.warning(f"获取市场情绪失败: {e}")
                sentiment_factor = 0.5  # 中性值
            
            return FactorExposure(
                market_beta=market_beta,
                size_factor=size_factor,
                value_factor=value_factor,
                profitability_factor=profitability_factor,
                investment_factor=investment_factor,
                quality_factor=quality_factor,
                momentum_factor=momentum_factor,
                low_volatility_factor=low_vol_factor,
                sentiment_factor=sentiment_factor
            )
            
        except Exception as e:
            logger.error(f"因子暴露度计算失败 {symbol}: {e}")
            return FactorExposure(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def calculate_risk_metrics(self, data: pd.DataFrame) -> RiskMetrics:
        """计算风险指标"""
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 30:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 1, 0)
            
            # 夏普比率
            excess_returns = returns - 0.02/252
            sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # 索提诺比率
            downside_returns = returns[returns < 0]
            sortino = (excess_returns.mean() / downside_returns.std() * np.sqrt(252) 
                      if len(downside_returns) > 0 and downside_returns.std() > 0 else 0)
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_dd = abs(drawdown.min())
            
            # VaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            return RiskMetrics(
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                information_ratio=sharpe * 0.8,  # 简化
                max_drawdown=max_dd,
                var_95=var_95,
                cvar_95=cvar_95,
                beta=1.0,
                tracking_error=returns.std() * np.sqrt(252)
            )
            
        except Exception as e:
            logger.error(f"风险指标计算失败: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 1, 0)
    
    def calculate_multifactor_score(self, factor_exposure: FactorExposure, risk_metrics: RiskMetrics) -> float:
        """计算多因子综合评分"""
        try:
            # 因子权重 - 重点关注质量因子
            factor_weights = {
                'quality': 0.30,           # 提高质量因子权重 (最重要)
                'momentum': 0.20,          # 动量因子
                'low_volatility': 0.15,    # 低波动率因子
                'value': 0.10,             # 价值因子
                'profitability': 0.10,     # 盈利能力因子
                'risk_adjustment': 0.10,   # 风险调整
                'size': 0.03,              # 规模因子
                'investment': 0.02,         # 投资因子
                'sentiment': 0.05           # 市场情绪因子
            }
            
            # 计算各因子得分
            quality_score = factor_exposure.quality_factor * 100
            momentum_score = max(0, factor_exposure.momentum_factor * 100 + 50)
            value_score = max(0, factor_exposure.value_factor * 50 + 50)
            low_vol_score = factor_exposure.low_volatility_factor * 100
            profitability_score = max(0, factor_exposure.profitability_factor * 50 + 50)
            size_score = max(0, factor_exposure.size_factor * 50 + 50)
            investment_score = max(0, factor_exposure.investment_factor * 50 + 50)
            sentiment_score = factor_exposure.sentiment_factor * 100
            
            # 风险调整
            risk_score = 50
            if risk_metrics.sharpe_ratio > 1.0:
                risk_score = 80
            elif risk_metrics.sharpe_ratio > 0.5:
                risk_score = 60
            elif risk_metrics.sharpe_ratio < 0:
                risk_score = 20
            
            # 计算加权综合得分
            weighted_score = (
                quality_score * factor_weights['quality'] +
                momentum_score * factor_weights['momentum'] +
                value_score * factor_weights['value'] +
                low_vol_score * factor_weights['low_volatility'] +
                profitability_score * factor_weights['profitability'] +
                size_score * factor_weights['size'] +
                investment_score * factor_weights['investment'] +
                sentiment_score * factor_weights['sentiment'] +
                risk_score * factor_weights['risk_adjustment']
            )
            
            return max(0, min(100, weighted_score))
            
        except Exception as e:
            logger.error(f"多因子评分计算失败: {e}")
            return 50.0
    
    def analyze_stock_professional(self, symbol: str) -> Optional[Dict[str, Any]]:
        """专业级股票分析（包含市场情绪分析）"""
        try:
            # 获取数据
            data = self.data_interface.get_data_for_strategy(symbol, lookback_days=252)
            if data is None or len(data) < 60:
                return None
            
            # 计算因子暴露度
            factor_exposure = self.calculate_factor_exposure(symbol, data)
            
            # 计算风险指标
            risk_metrics = self.calculate_risk_metrics(data)
            
            # 计算多因子评分
            multifactor_score = self.calculate_multifactor_score(factor_exposure, risk_metrics)
            
            # 获取市场情绪数据
            try:
                market_sentiment = self.sentiment_analyzer.get_market_sentiment()
                sentiment_info = {
                    'vix_level': market_sentiment.vix_level,
                    'vix_change': market_sentiment.vix_change,
                    'pcr_level': market_sentiment.pcr_level,
                    'fear_greed_index': market_sentiment.fear_greed_index,
                    'sentiment_score': market_sentiment.sentiment_score
                }
            except Exception as e:
                logger.warning(f"获取市场情绪数据失败: {e}")
                sentiment_info = {
                    'vix_level': 20.0,
                    'vix_change': 0.0,
                    'pcr_level': 1.0,
                    'fear_greed_index': 50.0,
                    'sentiment_score': 0.0
                }
            
            # 添加流动性分析（核心功能1：增强流动性评估）
            liquidity_info = None
            if self.liquidity_analyzer:
                try:
                    liquidity_metrics = self.liquidity_analyzer.analyze_stock_liquidity(symbol)
                    liquidity_info = {
                        'liquidity_score': liquidity_metrics.liquidity_score,
                        'risk_level': liquidity_metrics.risk_level,
                        'bid_ask_spread_pct': liquidity_metrics.bid_ask_spread_pct,
                        'volume_consistency': liquidity_metrics.volume_consistency,
                        'market_cap_tier': liquidity_metrics.market_cap_tier,
                        'investment_suggestion': liquidity_metrics.investment_suggestion,
                        'risk_warning': liquidity_metrics.risk_warning
                    }
                except Exception as e:
                    logger.warning(f"流动性分析失败 {symbol}: {e}")
                    liquidity_info = None
            
            return {
                'symbol': symbol,
                'multifactor_score': multifactor_score,
                'current_price': data['close'].iloc[-1],
                'avg_volume': data['volume'].mean(),
                'quality_factor': factor_exposure.quality_factor,
                'momentum_factor': factor_exposure.momentum_factor,
                'sentiment_factor': factor_exposure.sentiment_factor,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'max_drawdown': risk_metrics.max_drawdown,
                'market_sentiment': sentiment_info,
                'liquidity_analysis': liquidity_info,  # 新增流动性分析结果
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"专业分析失败 {symbol}: {e}")
            return None
    
    def screen_stocks_professional(self, min_score: float = 60, max_results: int = 30) -> List[Dict[str, Any]]:
        """专业级股票筛选"""
        logger.info("🎯 开始专业级多因子股票筛选...")
        
        # 获取股票池 - 使用全量数据
        stock_universe = self.data_interface.get_available_symbols()  # 全量573只股票
        
        results = []
        
        for i, symbol in enumerate(stock_universe):
            try:
                # 每50只股票显示一次进度
                if i % 50 == 0:
                    logger.info(f"   进度: {i}/{len(stock_universe)} ({i/len(stock_universe)*100:.1f}%) - 已找到 {len(results)} 只优质股票")
                
                analysis = self.analyze_stock_professional(symbol)
                if analysis and analysis['multifactor_score'] >= min_score:
                    results.append(analysis)
                    # 找到优质股票时立即显示
                    logger.info(f"   ✅ 发现优质股票: {symbol} (评分: {analysis['multifactor_score']:.1f})")
                    
            except Exception as e:
                logger.warning(f"分析股票失败 {symbol}: {e}")
                continue
        
        # 按评分排序
        results.sort(key=lambda x: x['multifactor_score'], reverse=True)
        
        logger.info(f"🎯 筛选完成！发现 {len(results[:max_results])} 只优质股票")
        return results[:max_results]
    
    def _generate_screening_report_html(self, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
        """生成筛选报告HTML内容"""
        html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>专业股票筛选报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 10px; }}
                .summary {{ margin: 20px 0; }}
                .stock-table {{ border-collapse: collapse; width: 100%; }}
                .stock-table th, .stock-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stock-table th {{ background-color: #f2f2f2; }}
                .high-score {{ background-color: #e8f5e8; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 专业股票筛选报告</h1>
                <p>筛选时间: {summary.get('screening_time', '')}</p>
            </div>
            
            <div class="summary">
                <h2>📊 筛选摘要</h2>
                <ul>
                    <li>总分析股票数: {summary.get('total_stocks_analyzed', 0)}</li>
                    <li>符合条件股票: {summary.get('qualified_stocks_found', 0)}</li>
                    <li>高质量股票: {summary.get('high_quality_stocks', 0)}</li>
                    <li>最佳股票: {summary.get('best_stock', 'N/A')}</li>
                    <li>最高评分: {summary.get('best_score', 0):.1f}</li>
                </ul>
            </div>
            
            <h2>📈 推荐股票列表</h2>
            <table class="stock-table">
                <tr>
                    <th>排名</th>
                    <th>股票代码</th>
                    <th>多因子评分</th>
                    <th>质量因子</th>
                    <th>动量因子</th>
                    <th>流动性评分</th>
                    <th>夏普比率</th>
                    <th>当前价格</th>
                </tr>
        """
        
        for i, stock in enumerate(results, 1):
            row_class = "high-score" if stock['multifactor_score'] >= 70 else ""
            
            # 获取流动性评分，如果没有则显示N/A
            liquidity_score = "N/A"
            if stock.get('liquidity_analysis'):
                liquidity_score = f"{stock['liquidity_analysis']['liquidity_score']:.1f}"
            
            html += f"""
                <tr class="{row_class}">
                    <td>{i}</td>
                    <td>{stock['symbol']}</td>
                    <td>{stock['multifactor_score']:.1f}</td>
                    <td>{stock['quality_factor']:.2f}</td>
                    <td>{stock['momentum_factor']:.2f}</td>
                    <td>{liquidity_score}</td>
                    <td>{stock['sharpe_ratio']:.2f}</td>
                    <td>${stock['current_price']:.2f}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        return html
    
    def screen_and_email(self, min_score: float = 50, max_results: int = 25, 
                        send_email: bool = True, email_subject: str = None) -> List[Dict[str, Any]]:
        """
        执行筛选并发送邮件报告
        
        Args:
            min_score: 最低评分
            max_results: 最大结果数
            send_email: 是否发送邮件
            email_subject: 邮件主题
            
        Returns:
            List[Dict]: 筛选结果
        """
        try:
            # 执行筛选
            results = self.screen_stocks_professional(min_score=min_score, max_results=max_results)
            
            if send_email and results:
                # 准备摘要信息
                high_quality_stocks = [s for s in results if s['quality_factor'] > 0.6]
                summary = {
                    'total_stocks_analyzed': 573,
                    'qualified_stocks_found': len(results),
                    'high_quality_stocks': len(high_quality_stocks),
                    'medium_quality_stocks': len(results) - len(high_quality_stocks),
                    'best_stock': results[0]['symbol'] if results else None,
                    'best_score': results[0]['multifactor_score'] if results else 0,
                    'screening_time': datetime.now().isoformat(),
                    'parameters': {
                        'min_score': min_score,
                        'max_results': max_results,
                        'quality_factor_weight': 0.30
                    }
                }
                
                # 发送邮件
                subject = email_subject or f"专业股票筛选报告 - {datetime.now().strftime('%Y-%m-%d')}"
                
                # 生成HTML报告内容
                html_content = self._generate_screening_report_html(results, summary)
                success = send_html(subject=subject, html_content=html_content)
                
                if success:
                    logger.info("📧 筛选结果邮件发送成功")
                else:
                    logger.warning("📧 筛选结果邮件发送失败")
            
            return results
            
        except Exception as e:
            logger.error(f"筛选和邮件发送失败: {e}")
            return []
    
    def send_report_email(self, report_file_path: str, subject: str = None) -> bool:
        """
        发送报告文件邮件
        
        Args:
            report_file_path: 报告文件路径
            subject: 邮件主题
            
        Returns:
            bool: 发送是否成功
        """
        try:
            subject = subject or f"股票分析报告 - {datetime.now().strftime('%Y-%m-%d')}"
            if os.path.exists(report_file_path):
                with open(report_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return send_markdown(subject=subject, md_content=content)
            else:
                logger.error(f"报告文件不存在: {report_file_path}")
                return False
        except Exception as e:
            logger.error(f"报告邮件发送失败: {e}")
            return False

def test_phase2_screener():
    """测试第二阶段筛选器"""
    screener = Phase2ProfessionalScreener()
    results = screener.screen_stocks_professional(min_score=60, max_results=20)
    
    if results:
        print("\n🎯 第二阶段专业级筛选结果:")
        print("=" * 100)
        print(f"{'排名':<4} {'股票':<8} {'多因子评分':<12} {'夏普比率':<10} {'质量因子':<10} {'动量因子':<10} {'价格':<10}")
        print("=" * 100)
        
        for i, stock in enumerate(results, 1):
            print(f"{i:<4} {stock['symbol']:<8} {stock['multifactor_score']:<12.1f} "
                  f"{stock['sharpe_ratio']:<10.2f} {stock['quality_factor']:<10.2f} "
                  f"{stock['momentum_factor']:<10.2f} ${stock['current_price']:<9.2f}")
    
    return results

if __name__ == "__main__":
    test_phase2_screener()
