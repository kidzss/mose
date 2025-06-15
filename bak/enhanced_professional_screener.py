#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版专业多策略量化股票筛选器
第一阶段改进：修复逻辑错误 + 风险调整收益指标 + 行业分类轮动分析
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据接口
from data.data_interface import DataInterface

# 导入策略和指标
from strategy.strategy_factory import StrategyFactory
from strategy.indicators import TechnicalIndicators, calculate_indicators

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedProfessionalScreener")


class RiskAdjustedMetrics:
    """风险调整收益指标计算类"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        try:
            if len(returns) < 2:
                return 0
            
            excess_returns = returns.mean() * 252 - risk_free_rate  # 年化超额收益
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            if volatility == 0:
                return 0
            
            return excess_returns / volatility
        except:
            return 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率（只考虑下行风险）"""
        try:
            if len(returns) < 2:
                return 0
            
            excess_returns = returns.mean() * 252 - risk_free_rate
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf') if excess_returns > 0 else 0
            
            downside_deviation = downside_returns.std() * np.sqrt(252)
            
            if downside_deviation == 0:
                return 0
            
            return excess_returns / downside_deviation
        except:
            return 0
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, int]:
        """计算最大回撤和回撤期间"""
        try:
            if len(prices) < 2:
                return 0, 0
            
            # 计算累计最高点
            peak = prices.expanding().max()
            
            # 计算回撤
            drawdown = (prices - peak) / peak
            
            # 最大回撤
            max_dd = drawdown.min()
            
            # 回撤期间
            dd_duration = 0
            current_dd_duration = 0
            
            for dd in drawdown:
                if dd < 0:
                    current_dd_duration += 1
                    dd_duration = max(dd_duration, current_dd_duration)
                else:
                    current_dd_duration = 0
            
            return abs(max_dd), dd_duration
        except:
            return 0, 0
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """计算卡玛比率（年化收益/最大回撤）"""
        try:
            if len(returns) < 2:
                return 0
            
            annual_return = returns.mean() * 252
            prices = (1 + returns).cumprod()
            max_dd, _ = RiskAdjustedMetrics.calculate_max_drawdown(prices)
            
            if max_dd == 0:
                return float('inf') if annual_return > 0 else 0
            
            return annual_return / max_dd
        except:
            return 0


class SectorAnalyzer:
    """行业分析器"""
    
    # 简化的行业分类映射
    SECTOR_MAPPING = {
        # 科技股
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
        'AMZN': 'Technology', 'META': 'Technology', 'TSLA': 'Technology', 'NVDA': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'ORCL': 'Technology', 'CRM': 'Technology',
        'ADBE': 'Technology', 'NFLX': 'Technology', 'PYPL': 'Technology', 'UBER': 'Technology',
        
        # 金融股
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
        'MS': 'Financials', 'C': 'Financials', 'AXP': 'Financials', 'BRK.B': 'Financials',
        'V': 'Financials', 'MA': 'Financials', 'AIG': 'Financials',
        
        # 医疗保健
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
        'MRK': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
        'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
        
        # 消费品
        'PG': 'Consumer Goods', 'KO': 'Consumer Goods', 'PEP': 'Consumer Goods', 
        'WMT': 'Consumer Goods', 'HD': 'Consumer Goods', 'MCD': 'Consumer Goods',
        'NKE': 'Consumer Goods', 'SBUX': 'Consumer Goods',
        
        # 工业
        'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials', 'MMM': 'Industrials',
        'HON': 'Industrials', 'UPS': 'Industrials', 'LMT': 'Industrials',
        
        # 能源
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
        
        # 公用事业
        'ATO': 'Utilities', 'NEE': 'Utilities', 'DUK': 'Utilities',
        
        # 材料
        'LIN': 'Materials', 'APD': 'Materials',
        
        # 房地产
        'AMT': 'Real Estate', 'PLD': 'Real Estate',
        
        # 通信
        'T': 'Communication', 'VZ': 'Communication',
        
        # 零售
        'COST': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary',
        
        # 医药分销
        'CAH': 'Healthcare', 'MCK': 'Healthcare',
        
        # 银行
        'BK': 'Financials', 'STT': 'Financials',
        
        # 其他
        'CHTR': 'Communication', 'CDNS': 'Technology'
    }
    
    @staticmethod
    def get_sector(symbol: str) -> str:
        """获取股票所属行业"""
        return SectorAnalyzer.SECTOR_MAPPING.get(symbol, 'Other')
    
    @staticmethod
    def calculate_sector_momentum(sector_stocks: Dict[str, pd.DataFrame], 
                                lookback_days: int = 20) -> Dict[str, float]:
        """计算行业动量"""
        sector_momentum = {}
        
        for sector, stocks_data in sector_stocks.items():
            if not stocks_data:
                sector_momentum[sector] = 0
                continue
            
            # 计算行业平均收益率
            sector_returns = []
            for symbol, data in stocks_data.items():
                if data is not None and len(data) >= lookback_days:
                    returns = data['close'].pct_change().tail(lookback_days).mean()
                    sector_returns.append(returns)
            
            if sector_returns:
                sector_momentum[sector] = np.mean(sector_returns) * 100
            else:
                sector_momentum[sector] = 0
        
        return sector_momentum
    
    @staticmethod
    def get_sector_rotation_signal(sector_momentum: Dict[str, float]) -> Dict[str, str]:
        """获取行业轮动信号"""
        if not sector_momentum:
            return {}
        
        # 按动量排序
        sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
        
        signals = {}
        total_sectors = len(sorted_sectors)
        
        for i, (sector, momentum) in enumerate(sorted_sectors):
            if i < total_sectors * 0.3:  # 前30%
                signals[sector] = 'Strong Buy'
            elif i < total_sectors * 0.6:  # 中间30%
                signals[sector] = 'Hold'
            else:  # 后40%
                signals[sector] = 'Weak'
        
        return signals


class EnhancedProfessionalScreener:
    """增强版专业多策略量化股票筛选器"""
    
    def __init__(self, min_market_cap: float = 5e8, min_avg_volume: float = 1e5):
        """
        初始化筛选器
        
        参数:
            min_market_cap: 最小市值门槛（默认5亿美元）
            min_avg_volume: 最小平均成交量门槛（默认10万股）
        """
        self.min_market_cap = min_market_cap
        self.min_avg_volume = min_avg_volume
        
        # 初始化组件
        self.data_interface = DataInterface()
        self.strategy_factory = StrategyFactory()
        self.technical_indicators = TechnicalIndicators()
        self.risk_metrics = RiskAdjustedMetrics()
        self.sector_analyzer = SectorAnalyzer()
        
        # 创建策略实例
        self.strategies = self._initialize_strategies()
        
        # 筛选权重配置（增加风险调整权重）
        self.weights = {
            'technical': 0.25,          # 技术指标权重
            'strategy': 0.25,           # 策略信号权重
            'risk_adjusted': 0.30,      # 风险调整收益权重
            'sector_momentum': 0.15,    # 行业动量权重
            'liquidity': 0.05           # 流动性权重
        }
        
        logger.info("🚀 增强版专业多策略量化筛选器初始化完成")
    
    def _initialize_strategies(self) -> Dict[str, Any]:
        """初始化所有策略"""
        strategies = {}
        try:
            # 创建核心策略
            strategies['tdi'] = self.strategy_factory.create_strategy('TDI')
            strategies['niuniu'] = self.strategy_factory.create_strategy('NiuniuV3')
            strategies['cpgw'] = self.strategy_factory.create_strategy('CPGW')
            strategies['combined'] = self.strategy_factory.create_combined_strategy()
            
            logger.info(f"✅ 成功初始化 {len(strategies)} 个策略")
            return strategies
        except Exception as e:
            logger.error(f"❌ 策略初始化失败: {e}")
            return {}
    
    def get_stock_universe(self) -> List[str]:
        """获取股票池（带流动性筛选）"""
        try:
            # 获取所有股票
            all_stocks = self.data_interface.get_available_symbols()
            logger.info(f"📊 原始股票池: {len(all_stocks)} 只股票")
            
            # 流动性筛选
            qualified_stocks = []
            failed_count = 0
            
            for symbol in all_stocks:
                try:
                    # 获取最近60天数据进行流动性筛选
                    hist = self.data_interface.get_data_for_strategy(symbol, lookback_days=60)
                    if hist is None or len(hist) < 30:
                        failed_count += 1
                        continue
                    
                    # 平均成交量筛选
                    avg_volume = hist['volume'].mean()
                    if avg_volume < self.min_avg_volume:
                        continue
                    
                    # 价格筛选（避免仙股）
                    current_price = hist['close'].iloc[-1]
                    if current_price < 1.0:  # 低于1美元的股票
                        continue
                    
                    qualified_stocks.append(symbol)
                    
                except Exception as e:
                    failed_count += 1
                    continue
            
            logger.info(f"✅ 流动性筛选后: {len(qualified_stocks)} 只股票")
            logger.info(f"⚠️ 数据获取失败: {failed_count} 只股票")
            return qualified_stocks
            
        except Exception as e:
            logger.error(f"❌ 获取股票池失败: {e}")
            return []
    
    def calculate_risk_adjusted_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算风险调整收益评分"""
        try:
            if len(data) < 30:
                return {'risk_adjusted_total': 0}
            
            # 计算收益率
            returns = data['close'].pct_change().dropna()
            prices = data['close']
            
            if len(returns) < 20:
                return {'risk_adjusted_total': 0}
            
            # 计算各种风险调整指标
            sharpe_ratio = self.risk_metrics.calculate_sharpe_ratio(returns)
            sortino_ratio = self.risk_metrics.calculate_sortino_ratio(returns)
            max_drawdown, dd_duration = self.risk_metrics.calculate_max_drawdown(prices)
            calmar_ratio = self.risk_metrics.calculate_calmar_ratio(returns)
            
            # 计算波动率
            volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率百分比
            
            # 评分逻辑
            scores = {}
            
            # 夏普比率评分 (0-30分)
            if sharpe_ratio > 2.0:
                scores['sharpe'] = 30
            elif sharpe_ratio > 1.5:
                scores['sharpe'] = 25
            elif sharpe_ratio > 1.0:
                scores['sharpe'] = 20
            elif sharpe_ratio > 0.5:
                scores['sharpe'] = 15
            elif sharpe_ratio > 0:
                scores['sharpe'] = 10
            else:
                scores['sharpe'] = 0
            
            # 索提诺比率评分 (0-25分)
            if sortino_ratio > 3.0:
                scores['sortino'] = 25
            elif sortino_ratio > 2.0:
                scores['sortino'] = 20
            elif sortino_ratio > 1.0:
                scores['sortino'] = 15
            elif sortino_ratio > 0.5:
                scores['sortino'] = 10
            elif sortino_ratio > 0:
                scores['sortino'] = 5
            else:
                scores['sortino'] = 0
            
            # 最大回撤评分 (0-25分)
            if max_drawdown < 0.05:  # 小于5%
                scores['drawdown'] = 25
            elif max_drawdown < 0.10:  # 小于10%
                scores['drawdown'] = 20
            elif max_drawdown < 0.15:  # 小于15%
                scores['drawdown'] = 15
            elif max_drawdown < 0.20:  # 小于20%
                scores['drawdown'] = 10
            elif max_drawdown < 0.30:  # 小于30%
                scores['drawdown'] = 5
            else:
                scores['drawdown'] = 0
            
            # 波动率评分 (0-20分) - 适中波动率最好
            if 10 <= volatility <= 25:  # 适中波动率
                scores['volatility'] = 20
            elif 5 <= volatility < 10 or 25 < volatility <= 35:
                scores['volatility'] = 15
            elif volatility < 5 or 35 < volatility <= 50:
                scores['volatility'] = 10
            else:
                scores['volatility'] = 5
            
            # 综合风险调整评分
            risk_adjusted_total = (
                scores['sharpe'] * 0.35 +
                scores['sortino'] * 0.30 +
                scores['drawdown'] * 0.25 +
                scores['volatility'] * 0.10
            )
            
            scores['risk_adjusted_total'] = risk_adjusted_total
            scores['sharpe_ratio'] = sharpe_ratio
            scores['sortino_ratio'] = sortino_ratio
            scores['max_drawdown'] = max_drawdown
            scores['volatility'] = volatility
            scores['calmar_ratio'] = calmar_ratio
            
            return scores
            
        except Exception as e:
            logger.error(f"风险调整指标计算失败: {e}")
            return {'risk_adjusted_total': 0}
    
    def calculate_technical_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标综合评分"""
        try:
            # 计算所有技术指标
            indicators = calculate_indicators(data, selected_indicators=[
                'sma', 'ema', 'bb', 'rsi', 'macd', 'adx'
            ])
            
            if not indicators:
                return {'technical_total': 0}
            
            scores = {}
            
            # 1. 趋势指标评分
            trend_score = self._calculate_trend_score(data, indicators)
            scores['trend'] = trend_score
            
            # 2. 动量指标评分
            momentum_score = self._calculate_momentum_score(data, indicators)
            scores['momentum'] = momentum_score
            
            # 3. 超买超卖指标评分
            overbought_oversold_score = self._calculate_overbought_oversold_score(indicators)
            scores['overbought_oversold'] = overbought_oversold_score
            
            # 综合技术评分
            technical_score = (
                trend_score * 0.40 +
                momentum_score * 0.35 +
                overbought_oversold_score * 0.25
            )
            
            scores['technical_total'] = technical_score
            return scores
            
        except Exception as e:
            logger.error(f"技术指标计算失败: {e}")
            return {'technical_total': 0}
    
    def _calculate_trend_score(self, data: pd.DataFrame, indicators: Dict) -> float:
        """计算趋势评分"""
        try:
            close = data['close']
            score = 0
            
            # SMA趋势
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20 = indicators['sma_20'].iloc[-1]
                sma_50 = indicators['sma_50'].iloc[-1]
                current_price = close.iloc[-1]
                
                if current_price > sma_20 > sma_50:
                    score += 40  # 强势上升趋势
                elif current_price > sma_20:
                    score += 20  # 短期上升趋势
                elif current_price < sma_20 < sma_50:
                    score -= 40  # 强势下降趋势
                else:
                    score -= 20  # 短期下降趋势
            
            # ADX趋势强度
            if 'adx' in indicators:
                adx = indicators['adx'].iloc[-1]
                if adx > 25:
                    score += 30  # 强趋势
                elif adx > 20:
                    score += 15  # 中等趋势
            
            return max(-100, min(100, score))
            
        except Exception as e:
            return 0
    
    def _calculate_momentum_score(self, data: pd.DataFrame, indicators: Dict) -> float:
        """计算动量评分"""
        try:
            score = 0
            
            # MACD动量
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                macd_signal = indicators['macd_signal'].iloc[-1]
                
                if macd > macd_signal and macd > 0:
                    score += 35  # 强势上升动量
                elif macd > macd_signal:
                    score += 20  # 上升动量
                elif macd < macd_signal and macd < 0:
                    score -= 35  # 强势下降动量
                else:
                    score -= 20  # 下降动量
            
            # 价格动量
            close = data['close']
            if len(close) >= 20:
                momentum_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100
                if momentum_20 > 10:
                    score += 30
                elif momentum_20 > 5:
                    score += 15
                elif momentum_20 < -10:
                    score -= 30
                elif momentum_20 < -5:
                    score -= 15
            
            return max(-100, min(100, score))
            
        except Exception as e:
            return 0
    
    def _calculate_overbought_oversold_score(self, indicators: Dict) -> float:
        """计算超买超卖评分"""
        try:
            score = 0
            
            # RSI评分
            if 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                if 30 <= rsi <= 70:
                    score += 25  # 正常区间
                elif 20 <= rsi < 30:
                    score += 40  # 超卖，买入机会
                elif 70 < rsi <= 80:
                    score -= 25  # 超买警告
                elif rsi > 80:
                    score -= 50  # 严重超买
                elif rsi < 20:
                    score += 50  # 严重超卖，强买入机会
            
            return max(-100, min(100, score))
            
        except Exception as e:
            return 0
    
    def calculate_strategy_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算策略信号评分"""
        scores = {}
        
        for name, strategy in self.strategies.items():
            try:
                # 生成策略信号
                signals = strategy.generate_signals(data)
                
                if signals is not None and not signals.empty and 'signal' in signals.columns:
                    # 计算最近信号强度
                    recent_signals = signals['signal'].tail(10)
                    signal_strength = recent_signals.mean()
                    signal_consistency = 1 - recent_signals.std() if recent_signals.std() > 0 else 1
                    
                    # 策略评分
                    strategy_score = signal_strength * signal_consistency * 100
                    scores[name] = max(-100, min(100, strategy_score))
                else:
                    scores[name] = 0
                    
            except Exception as e:
                logger.warning(f"策略 {name} 计算失败: {e}")
                scores[name] = 0
        
        # 综合策略评分
        if scores:
            strategy_total = np.mean(list(scores.values()))
            scores['strategy_total'] = strategy_total
        else:
            scores['strategy_total'] = 0
        
        return scores
    
    def calculate_sector_score(self, symbol: str, sector_momentum: Dict[str, float]) -> float:
        """计算行业评分"""
        try:
            sector = self.sector_analyzer.get_sector(symbol)
            momentum = sector_momentum.get(sector, 0)
            
            # 行业动量评分
            if momentum > 5:
                return 50  # 强势行业
            elif momentum > 2:
                return 30  # 中等行业
            elif momentum > 0:
                return 10  # 弱势但正向行业
            elif momentum > -2:
                return -10  # 轻微负向行业
            elif momentum > -5:
                return -30  # 弱势行业
            else:
                return -50  # 严重弱势行业
                
        except Exception as e:
            return 0
    
    def calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """计算流动性评分"""
        try:
            volume = data['volume']
            avg_volume = volume.mean()
            
            # 基于平均成交量的流动性评分
            if avg_volume > 10e6:
                return 100  # 极高流动性
            elif avg_volume > 5e6:
                return 80   # 高流动性
            elif avg_volume > 2e6:
                return 60   # 中高流动性
            elif avg_volume > 1e6:
                return 40   # 中等流动性
            elif avg_volume > 5e5:
                return 30   # 中低流动性
            else:
                return 20   # 低流动性
                
        except Exception as e:
            return 50
    
    def analyze_stock(self, symbol: str, sector_momentum: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """综合分析单只股票"""
        try:
            # 获取股票数据
            data = self.data_interface.get_data_for_strategy(symbol, lookback_days=180)
            if data is None or len(data) < 60:
                return None
            
            # 各维度评分
            technical_scores = self.calculate_technical_score(data)
            strategy_scores = self.calculate_strategy_score(data)
            risk_adjusted_scores = self.calculate_risk_adjusted_score(data)
            sector_score = self.calculate_sector_score(symbol, sector_momentum)
            liquidity_score = self.calculate_liquidity_score(data)
            
            # 综合评分
            total_score = (
                technical_scores.get('technical_total', 0) * self.weights['technical'] +
                strategy_scores.get('strategy_total', 0) * self.weights['strategy'] +
                risk_adjusted_scores.get('risk_adjusted_total', 0) * self.weights['risk_adjusted'] +
                sector_score * self.weights['sector_momentum'] +
                liquidity_score * self.weights['liquidity']
            )
            
            # 构建结果
            result = {
                'symbol': symbol,
                'sector': self.sector_analyzer.get_sector(symbol),
                'total_score': total_score,
                'current_price': data['close'].iloc[-1],
                'avg_volume': data['volume'].mean(),
                
                # 技术指标评分
                'technical_total': technical_scores.get('technical_total', 0),
                'trend_score': technical_scores.get('trend', 0),
                'momentum_score': technical_scores.get('momentum', 0),
                'overbought_oversold_score': technical_scores.get('overbought_oversold', 0),
                
                # 策略评分
                'strategy_total': strategy_scores.get('strategy_total', 0),
                'tdi_score': strategy_scores.get('tdi', 0),
                'niuniu_score': strategy_scores.get('niuniu', 0),
                'cpgw_score': strategy_scores.get('cpgw', 0),
                'combined_score': strategy_scores.get('combined', 0),
                
                # 风险调整评分
                'risk_adjusted_total': risk_adjusted_scores.get('risk_adjusted_total', 0),
                'sharpe_ratio': risk_adjusted_scores.get('sharpe_ratio', 0),
                'sortino_ratio': risk_adjusted_scores.get('sortino_ratio', 0),
                'max_drawdown': risk_adjusted_scores.get('max_drawdown', 0),
                'volatility': risk_adjusted_scores.get('volatility', 0),
                'calmar_ratio': risk_adjusted_scores.get('calmar_ratio', 0),
                
                # 其他评分
                'sector_score': sector_score,
                'liquidity_score': liquidity_score,
                
                # 分析时间
                'analysis_time': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {symbol} 失败: {e}")
            return None
    
    def screen_stocks(self, min_score: float = 30, max_results: int = 50) -> List[Dict[str, Any]]:
        """执行股票筛选"""
        logger.info("🚀 开始增强版专业多策略股票筛选")
        logger.info("=" * 80)
        
        # 获取股票池
        stock_universe = self.get_stock_universe()
        if not stock_universe:
            logger.error("❌ 无法获取股票池")
            return []
        
        logger.info(f"📊 股票池规模: {len(stock_universe)} 只股票")
        logger.info(f"📈 筛选标准: 综合评分 >= {min_score}")
        logger.info(f"🎯 最大结果数: {max_results}")
        
        # 计算行业动量
        logger.info("📊 计算行业动量...")
        sector_stocks = {}
        for symbol in stock_universe:
            sector = self.sector_analyzer.get_sector(symbol)
            if sector not in sector_stocks:
                sector_stocks[sector] = {}
            
            try:
                data = self.data_interface.get_data_for_strategy(symbol, lookback_days=60)
                sector_stocks[sector][symbol] = data
            except:
                sector_stocks[sector][symbol] = None
        
        sector_momentum = self.sector_analyzer.calculate_sector_momentum(sector_stocks)
        sector_signals = self.sector_analyzer.get_sector_rotation_signal(sector_momentum)
        
        logger.info("📈 行业动量排名:")
        for sector, momentum in sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True):
            signal = sector_signals.get(sector, 'Hold')
            logger.info(f"   {sector}: {momentum:.2f}% ({signal})")
        
        # 分析所有股票
        results = []
        total_count = len(stock_universe)
        
        logger.info(f"\n⏳ 开始分析 {total_count} 只股票...")
        
        for i, symbol in enumerate(stock_universe):
            if i % 50 == 0:
                logger.info(f"   进度: {i}/{total_count} ({i/total_count*100:.1f}%)")
            
            analysis = self.analyze_stock(symbol, sector_momentum)
            if analysis and analysis['total_score'] >= min_score:
                results.append(analysis)
        
        # 按评分排序
        results.sort(key=lambda x: x['total_score'], reverse=True)
        results = results[:max_results]
        
        logger.info(f"\n🎯 筛选完成！发现 {len(results)} 只优质股票")
        
        return results


def test_enhanced_screener():
    """测试增强版筛选器"""
    print("🧪 测试增强版专业筛选器")
    print("=" * 80)
    
    # 创建筛选器
    screener = EnhancedProfessionalScreener(
        min_market_cap=5e8,    # 5亿美元市值门槛
        min_avg_volume=1e5     # 10万股成交量门槛
    )
    
    # 执行筛选
    results = screener.screen_stocks(min_score=25, max_results=20)
    
    if results:
        print(f"\n🎯 发现 {len(results)} 只优质股票:")
        print("-" * 120)
        header = f"{'排名':^4} {'股票':^8} {'行业':^12} {'总分':^6} {'技术':^6} {'策略':^6} {'风险':^6} {'夏普':^6} {'价格':^8} {'成交量':^10}"
        print(header)
        print("-" * 120)
        
        for i, stock in enumerate(results, 1):
            row = (f"{i:^4} "
                   f"{stock['symbol']:^8} "
                   f"{stock['sector'][:10]:^12} "
                   f"{stock['total_score']:^6.1f} "
                   f"{stock['technical_total']:^6.1f} "
                   f"{stock['strategy_total']:^6.1f} "
                   f"{stock['risk_adjusted_total']:^6.1f} "
                   f"{stock['sharpe_ratio']:^6.2f} "
                   f"${stock['current_price']:^7.2f} "
                   f"{stock['avg_volume']/1e6:^9.1f}M")
            print(row)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_screening_test_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📝 测试结果已保存: {filename}")
        except Exception as e:
            print(f"⚠️ 保存失败: {e}")
        
        print(f"\n✅ 测试完成！")
        return filename
    else:
        print("❌ 未发现符合条件的股票")
        return None


if __name__ == "__main__":
    test_enhanced_screener() 