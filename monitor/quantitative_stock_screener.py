#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_interface import DataInterface

# 尝试导入可选的分析组件，如果不存在则使用简化版本
try:
    from strategy.market_environment_classifier import MarketEnvironmentClassifier, MarketEnvironment
except ImportError:
    MarketEnvironmentClassifier = None
    class MarketEnvironment:
        BULL_MARKET = "牛市"
        BEAR_MARKET = "熊市"
        CONSOLIDATION = "震荡市"

try:
    from strategy.dynamic_strategy_selector import DynamicStrategySelector
except ImportError:
    DynamicStrategySelector = None

try:
    from strategy.signal_quality_evaluator import SignalQualityEvaluator
except ImportError:
    SignalQualityEvaluator = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantStockScreener")

class RiskLevel(Enum):
    """风险等级"""
    CONSERVATIVE = "保守型"      # 低风险低收益
    MODERATE = "稳健型"          # 中等风险收益
    AGGRESSIVE = "激进型"        # 高风险高收益
    SPECULATIVE = "投机型"       # 极高风险收益

@dataclass
class ScreeningCriteria:
    """筛选标准配置"""
    # 时间范围
    lookback_months: int = 6  # 回看月数
    min_trading_days: int = 120  # 最少交易天数
    
    # 技术指标标准
    min_rsi: float = 30  # RSI下限
    max_rsi: float = 70  # RSI上限
    min_volume_ratio: float = 1.2  # 成交量比例
    max_volatility: float = 0.4  # 最大波动率
    
    # 基本面标准
    min_market_cap: float = 1e9  # 最小市值（10亿美元）
    min_avg_volume: float = 1e6   # 最小平均成交量
    
    # 收益风险标准
    min_sharpe_ratio: float = 1.0  # 最小夏普比率
    max_max_drawdown: float = 0.2  # 最大回撤限制
    min_win_rate: float = 0.5      # 最小胜率
    
    # 信号质量标准
    min_signal_quality: float = 0.65  # 最小信号质量
    min_strategy_score: float = 0.6   # 最小策略分数

@dataclass 
class StockScore:
    """股票评分结果"""
    symbol: str
    total_score: float
    risk_level: RiskLevel
    
    # 分项评分
    technical_score: float    # 技术分析评分
    fundamental_score: float  # 基本面评分
    momentum_score: float     # 动量评分
    value_score: float        # 价值评分
    quality_score: float      # 质量评分
    
    # 风险指标
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    var_95: float  # 95%置信度风险价值
    
    # 收益指标
    expected_return: float
    win_rate: float
    profit_loss_ratio: float
    
    # 市场环境适应性
    market_env: MarketEnvironment
    strategy_recommendation: str
    signal_quality: float
    
    # 买入建议
    buy_price: float
    stop_loss: float
    target_price: float
    position_size: float  # 建议仓位大小
    
    # 详细分析
    strengths: List[str]
    risks: List[str]
    market_timing: str  # 市场时机评估

class QuantitativeStockScreener:
    """专业量化股票筛选器
    
    基于多因子模型、风险调整收益和机器学习的股票筛选系统
    """
    
    def __init__(self, criteria: ScreeningCriteria = None):
        """
        初始化量化筛选器
        
        Args:
            criteria: 筛选标准配置
        """
        self.criteria = criteria or ScreeningCriteria()
        
        # 初始化数据接口
        self.data_interface = DataInterface()
        
        # 初始化分析组件（如果可用）
        self.market_classifier = MarketEnvironmentClassifier() if MarketEnvironmentClassifier else None
        self.strategy_selector = DynamicStrategySelector() if DynamicStrategySelector else None
        self.signal_evaluator = SignalQualityEvaluator() if SignalQualityEvaluator else None
        
        # 因子权重配置
        self.factor_weights = {
            'technical': 0.25,      # 技术分析权重
            'fundamental': 0.20,    # 基本面权重
            'momentum': 0.20,       # 动量权重
            'value': 0.15,          # 价值权重
            'quality': 0.20         # 质量权重
        }
        
        # 风险偏好配置
        self.risk_profiles = {
            RiskLevel.CONSERVATIVE: {
                'max_volatility': 0.15,
                'min_sharpe': 1.5,
                'max_drawdown': 0.10,
                'position_limit': 0.05
            },
            RiskLevel.MODERATE: {
                'max_volatility': 0.25,
                'min_sharpe': 1.0,
                'max_drawdown': 0.15,
                'position_limit': 0.08
            },
            RiskLevel.AGGRESSIVE: {
                'max_volatility': 0.35,
                'min_sharpe': 0.8,
                'max_drawdown': 0.20,
                'position_limit': 0.12
            },
            RiskLevel.SPECULATIVE: {
                'max_volatility': 0.50,
                'min_sharpe': 0.5,
                'max_drawdown': 0.30,
                'position_limit': 0.15
            }
        }
        
        logger.info("✅ 量化股票筛选器初始化完成")
    
    def screen_stocks(self, symbols: List[str] = None, 
                     risk_preference: RiskLevel = RiskLevel.MODERATE) -> List[StockScore]:
        """
        执行股票筛选
        
        Args:
            symbols: 股票列表，如果为None则使用默认股票池
            risk_preference: 风险偏好
            
        Returns:
            按评分排序的股票列表
        """
        logger.info("🚀 开始执行量化股票筛选...")
        
        # 获取股票池
        if symbols is None:
            symbols = self._get_default_stock_universe()
        
        results = []
        total_stocks = len(symbols)
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"📊 分析股票 {symbol} ({i}/{total_stocks})")
                
                # 获取数据
                data = self._get_stock_data(symbol)
                if data is None or len(data) < self.criteria.min_trading_days:
                    logger.warning(f"⚠️ {symbol} 数据不足，跳过")
                    continue
                
                # 执行多因子分析
                score = self._analyze_stock(symbol, data, risk_preference)
                if score:
                    results.append(score)
                    
            except Exception as e:
                logger.error(f"❌ 分析 {symbol} 时出错: {e}")
                continue
        
        # 按总分排序
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        logger.info(f"✅ 筛选完成，发现 {len(results)} 只合格股票")
        return results
    
    def _get_default_stock_universe(self) -> List[str]:
        """获取默认股票池"""
        try:
            # 获取数据库中所有可用股票
            available_symbols = self.data_interface.get_available_symbols()
            
            # 过滤掉一些不适合的股票（比如ETF、债券等）
            filtered_symbols = []
            for symbol in available_symbols:
                if (not symbol.startswith('^') and  # 排除指数
                    not '.' in symbol and           # 排除外国股票
                    len(symbol) <= 5):              # 排除过长的代码
                    filtered_symbols.append(symbol)
            
            return filtered_symbols[:100]  # 限制为前100只股票进行测试
            
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            # 使用默认股票池
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                   'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL']
    
    def _get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.criteria.lookback_months * 30)
            
            data = self.data_interface.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None or data.empty:
                return None
                
            # 添加技术指标
            data = self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        try:
            # 基本技术指标
            data['rsi'] = self._calculate_rsi(data['close'])
            data['macd'] = self._calculate_macd(data['close'])
            data['bb_upper'], data['bb_lower'] = self._calculate_bollinger_bands(data['close'])
            
            # 移动平均线
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['sma_200'] = data['close'].rolling(200).mean()
            
            # 成交量指标
            data['volume_sma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            
            # 波动率指标
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(20).std() * np.sqrt(252)
            
            # ATR
            data['atr'] = self._calculate_atr(data)
            
            return data
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return data
    
    def _analyze_stock(self, symbol: str, data: pd.DataFrame, 
                      risk_preference: RiskLevel) -> Optional[StockScore]:
        """执行单只股票的多因子分析"""
        try:
            # 1. 技术分析评分
            technical_score = self._calculate_technical_score(data)
            
            # 2. 基本面评分 (简化版本)
            fundamental_score = self._calculate_fundamental_score(symbol, data)
            
            # 3. 动量评分
            momentum_score = self._calculate_momentum_score(data)
            
            # 4. 价值评分
            value_score = self._calculate_value_score(data)
            
            # 5. 质量评分
            quality_score = self._calculate_quality_score(data)
            
            # 6. 风险指标计算
            risk_metrics = self._calculate_risk_metrics(data)
            
            # 7. 市场环境分析
            if self.market_classifier:
                market_env_result = self.market_classifier.classify_environment(data)
            else:
                market_env_result = {
                    'environment': MarketEnvironment.CONSOLIDATION,
                    'confidence': 0.7
                }
            
            # 8. 策略推荐
            if self.strategy_selector:
                strategy_result = self.strategy_selector.get_best_strategy(data)
            else:
                strategy_result = {
                    'primary_strategy': '多因子量化策略',
                    'confidence': 0.8
                }
            
            # 9. 信号质量评估
            signal_quality = self._evaluate_signal_quality(data, market_env_result)
            
            # 计算综合评分
            total_score = (
                technical_score * self.factor_weights['technical'] +
                fundamental_score * self.factor_weights['fundamental'] +
                momentum_score * self.factor_weights['momentum'] +
                value_score * self.factor_weights['value'] +
                quality_score * self.factor_weights['quality']
            )
            
            # 风险调整
            risk_profile = self.risk_profiles[risk_preference]
            if (risk_metrics['volatility'] > risk_profile['max_volatility'] or
                risk_metrics['sharpe_ratio'] < risk_profile['min_sharpe'] or
                risk_metrics['max_drawdown'] > risk_profile['max_drawdown']):
                total_score *= 0.7  # 降低评分
            
            # 计算买入建议
            buy_signals = self._generate_buy_signals(data, market_env_result, risk_metrics)
            
            # 生成评分对象
            score = StockScore(
                symbol=symbol,
                total_score=total_score,
                risk_level=self._classify_risk_level(risk_metrics),
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                momentum_score=momentum_score,
                value_score=value_score,
                quality_score=quality_score,
                sharpe_ratio=risk_metrics['sharpe_ratio'],
                max_drawdown=risk_metrics['max_drawdown'],
                volatility=risk_metrics['volatility'],
                var_95=risk_metrics['var_95'],
                expected_return=risk_metrics['expected_return'],
                win_rate=risk_metrics['win_rate'],
                profit_loss_ratio=risk_metrics['profit_loss_ratio'],
                market_env=market_env_result['environment'],
                strategy_recommendation=strategy_result['primary_strategy'],
                signal_quality=signal_quality,
                buy_price=buy_signals['buy_price'],
                stop_loss=buy_signals['stop_loss'],
                target_price=buy_signals['target_price'],
                position_size=buy_signals['position_size'],
                strengths=self._identify_strengths(data, risk_metrics),
                risks=self._identify_risks(data, risk_metrics),
                market_timing=self._assess_market_timing(data, market_env_result)
            )
            
            return score
            
        except Exception as e:
            logger.error(f"分析股票 {symbol} 失败: {e}")
            return None
    
    def _calculate_technical_score(self, data: pd.DataFrame) -> float:
        """计算技术分析评分"""
        score = 0.0
        factors = 0
        
        try:
            latest = data.iloc[-1]
            
            # RSI评分 (30-70为优)
            rsi = latest['rsi']
            if 30 <= rsi <= 70:
                score += 1.0
                factors += 1
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                score += 0.7
                factors += 1
            elif rsi < 20 or rsi > 80:
                score += 0.3
                factors += 1
            
            # 移动平均线评分
            price = latest['close']
            if price > latest['sma_20'] > latest['sma_50'] > latest['sma_200']:
                score += 1.0  # 完美上升趋势
            elif price > latest['sma_20'] > latest['sma_50']:
                score += 0.8  # 短期上升趋势
            elif price > latest['sma_20']:
                score += 0.6  # 短期强势
            else:
                score += 0.2  # 弱势
            factors += 1
            
            # 成交量评分
            volume_ratio = latest['volume_ratio']
            if volume_ratio > 1.5:
                score += 1.0  # 放量
            elif volume_ratio > 1.2:
                score += 0.8
            elif volume_ratio > 0.8:
                score += 0.6
            else:
                score += 0.3  # 缩量
            factors += 1
            
            # 布林带位置评分
            bb_position = (price - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            if 0.2 <= bb_position <= 0.8:
                score += 1.0  # 在布林带中轨附近
            elif bb_position < 0.2:
                score += 0.7  # 接近下轨，可能超卖
            elif bb_position > 0.8:
                score += 0.5  # 接近上轨，可能超买
            factors += 1
            
            return score / factors if factors > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算技术评分失败: {e}")
            return 0.0
    
    def _calculate_fundamental_score(self, symbol: str, data: pd.DataFrame) -> float:
        """计算基本面评分（简化版本）"""
        try:
            # 由于缺乏基本面数据，使用技术指标来近似
            score = 0.0
            
            # 价格趋势稳定性
            returns = data['returns'].dropna()
            price_stability = 1 - returns.std()
            score += min(price_stability * 2, 1.0) * 0.3
            
            # 成交量稳定性
            volume_cv = data['volume'].std() / data['volume'].mean()
            volume_stability = max(0, 1 - volume_cv)
            score += volume_stability * 0.3
            
            # 长期趋势
            long_term_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1)
            if long_term_return > 0:
                score += min(long_term_return, 0.5) * 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"计算基本面评分失败: {e}")
            return 0.5  # 默认中性评分
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """计算动量评分"""
        try:
            score = 0.0
            
            # 价格动量
            price_momentum = self._calculate_price_momentum(data)
            score += price_momentum * 0.4
            
            # 相对强弱指数动量
            rsi_trend = self._calculate_rsi_trend(data)
            score += rsi_trend * 0.3
            
            # 成交量动量
            volume_momentum = self._calculate_volume_momentum(data)
            score += volume_momentum * 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"计算动量评分失败: {e}")
            return 0.0
    
    def _calculate_value_score(self, data: pd.DataFrame) -> float:
        """计算价值评分"""
        try:
            score = 0.0
            
            # 基于技术指标的相对价值
            latest_price = data['close'].iloc[-1]
            
            # 相对于移动平均线的价值
            ma_200 = data['sma_200'].iloc[-1]
            if latest_price < ma_200 * 0.9:  # 相对于长期均线有折扣
                score += 0.4
            elif latest_price < ma_200:
                score += 0.2
            
            # 相对于波动区间的价值
            high_52w = data['close'].rolling(252).max().iloc[-1]
            low_52w = data['close'].rolling(252).min().iloc[-1]
            position_in_range = (latest_price - low_52w) / (high_52w - low_52w)
            
            if position_in_range < 0.3:  # 接近52周低点
                score += 0.6
            elif position_in_range < 0.5:
                score += 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"计算价值评分失败: {e}")
            return 0.0
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """计算质量评分"""
        try:
            score = 0.0
            
            # 价格稳定性
            returns = data['returns'].dropna()
            volatility = returns.std()
            if volatility < 0.02:  # 日波动率小于2%
                score += 0.3
            elif volatility < 0.03:
                score += 0.2
            
            # 成交量一致性
            volume_cv = data['volume'].std() / data['volume'].mean()
            if volume_cv < 1.0:  # 成交量变异系数小于1
                score += 0.3
            elif volume_cv < 1.5:
                score += 0.2
            
            # 趋势一致性
            trend_consistency = self._calculate_trend_consistency(data)
            score += trend_consistency * 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"计算质量评分失败: {e}")
            return 0.0
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算风险指标"""
        try:
            returns = data['returns'].dropna()
            
            # 夏普比率
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # 波动率
            volatility = returns.std() * np.sqrt(252)
            
            # VaR (95%)
            var_95 = abs(returns.quantile(0.05))
            
            # 预期收益
            expected_return = returns.mean() * 252
            
            # 胜率
            win_rate = (returns > 0).mean()
            
            # 盈亏比
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            profit_loss_ratio = (positive_returns.mean() / abs(negative_returns.mean()) 
                               if len(negative_returns) > 0 else float('inf'))
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'var_95': var_95,
                'expected_return': expected_return,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio
            }
            
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'volatility': 1.0,
                'var_95': 0.1,
                'expected_return': 0.0,
                'win_rate': 0.5,
                'profit_loss_ratio': 1.0
            }
    
    # 辅助计算方法
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """计算MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        return exp1 - exp2
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """计算布林带"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window).mean()
    
    # 集成辅助方法
    def _calculate_price_momentum(self, data: pd.DataFrame) -> float:
        """计算价格动量评分"""
        try:
            if len(data) < 60:
                return 0.0
                
            # 短期动量 (5天)
            short_momentum = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1)
            
            # 中期动量 (20天)
            medium_momentum = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1)
            
            # 长期动量 (60天)
            long_momentum = (data['close'].iloc[-1] / data['close'].iloc[-60] - 1)
            
            # 综合动量评分
            momentum_score = 0.0
            
            # 短期动量评分
            if short_momentum > 0.05:  # 5天涨幅超过5%
                momentum_score += 0.4
            elif short_momentum > 0.02:
                momentum_score += 0.3
            elif short_momentum > 0:
                momentum_score += 0.2
            
            # 中期动量评分
            if medium_momentum > 0.15:  # 20天涨幅超过15%
                momentum_score += 0.3
            elif medium_momentum > 0.05:
                momentum_score += 0.2
            elif medium_momentum > 0:
                momentum_score += 0.1
            
            # 长期动量评分
            if long_momentum > 0.25:  # 60天涨幅超过25%
                momentum_score += 0.3
            elif long_momentum > 0.10:
                momentum_score += 0.2
            elif long_momentum > 0:
                momentum_score += 0.1
            
            return min(momentum_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_rsi_trend(self, data: pd.DataFrame) -> float:
        """计算RSI趋势评分"""
        try:
            rsi = data['rsi'].dropna()
            if len(rsi) < 10:
                return 0.0
            
            # RSI趋势方向
            recent_rsi = rsi.tail(5).mean()
            previous_rsi = rsi.iloc[-10:-5].mean()
            
            score = 0.0
            
            # RSI上升趋势
            if recent_rsi > previous_rsi:
                score += 0.5
            
            # RSI水平评分
            if 40 <= recent_rsi <= 60:  # 中性区域
                score += 0.5
            elif 30 <= recent_rsi < 40:  # 超卖区域回升
                score += 0.7
            elif recent_rsi < 30:  # 深度超卖
                score += 0.3
            elif 60 < recent_rsi <= 70:  # 强势但未超买
                score += 0.4
            elif recent_rsi > 70:  # 超买区域
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> float:
        """计算成交量动量评分"""
        try:
            # 近期平均成交量
            recent_volume = data['volume'].tail(5).mean()
            # 历史平均成交量
            historical_volume = data['volume'].tail(20).mean()
            
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
            
            score = 0.0
            
            # 成交量放大评分
            if volume_ratio > 2.0:  # 放量2倍以上
                score += 1.0
            elif volume_ratio > 1.5:  # 放量1.5倍
                score += 0.8
            elif volume_ratio > 1.2:  # 温和放量
                score += 0.6
            elif volume_ratio > 0.8:  # 正常成交量
                score += 0.4
            else:  # 缩量
                score += 0.2
            
            return score
            
        except Exception:
            return 0.0
    
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """计算趋势一致性评分"""
        try:
            if len(data) < 200:
                return 0.5
                
            # 计算短期、中期、长期均线的趋势方向
            short_ma = data['sma_20'].tail(10)
            medium_ma = data['sma_50'].tail(10)
            long_ma = data['sma_200'].tail(10)
            
            # 趋势方向 (1为上升，-1为下降，0为横盘)
            short_trend = 1 if short_ma.iloc[-1] > short_ma.iloc[0] else -1
            medium_trend = 1 if medium_ma.iloc[-1] > medium_ma.iloc[0] else -1
            long_trend = 1 if long_ma.iloc[-1] > long_ma.iloc[0] else -1
            
            # 趋势一致性评分
            trend_consistency = 0.0
            
            # 三个时间框架趋势一致
            if short_trend == medium_trend == long_trend:
                trend_consistency = 1.0
            # 两个时间框架一致
            elif (short_trend == medium_trend or 
                  short_trend == long_trend or 
                  medium_trend == long_trend):
                trend_consistency = 0.6
            # 完全不一致
            else:
                trend_consistency = 0.2
            
            return trend_consistency
            
        except Exception:
            return 0.0
    
    def _evaluate_signal_quality(self, data: pd.DataFrame, market_env_result: Dict) -> float:
        """评估信号质量"""
        try:
            signal_quality = 0.0
            
            # 基于市场环境调整信号质量
            if market_env_result['environment'] == MarketEnvironment.BULL_MARKET:
                signal_quality += 0.3
            elif market_env_result['environment'] == MarketEnvironment.CONSOLIDATION:
                signal_quality += 0.2
            elif market_env_result['environment'] == MarketEnvironment.BEAR_MARKET:
                signal_quality += 0.1
            
            # 基于技术指标一致性
            latest_data = data.iloc[-1]
            
            # 价格与移动平均线关系
            price = latest_data['close']
            if (price > latest_data['sma_20'] > 
                latest_data['sma_50'] > latest_data['sma_200']):
                signal_quality += 0.3
            elif price > latest_data['sma_20']:
                signal_quality += 0.2
            
            # RSI信号质量
            rsi = latest_data['rsi']
            if 30 <= rsi <= 70:
                signal_quality += 0.2
            elif rsi < 30:  # 超卖可能反弹
                signal_quality += 0.15
            
            # 成交量确认
            if latest_data['volume_ratio'] > 1.2:
                signal_quality += 0.2
            
            return min(signal_quality, 1.0)
            
        except Exception:
            return 0.5
    
    def _generate_buy_signals(self, data: pd.DataFrame, market_env_result: Dict, risk_metrics: Dict = None) -> Dict[str, float]:
        """生成买入信号"""
        try:
            latest_data = data.iloc[-1]
            current_price = latest_data['close']
            
            # 安全获取ATR值
            atr = latest_data.get('atr', np.nan)
            if pd.isna(atr) or atr <= 0:
                # 如果ATR无效，使用价格的2%作为替代
                atr = current_price * 0.02
            
            # 计算买入价格 (当前价格附近，略微优化)
            buy_price = current_price * 0.995  # 稍微低于当前价格
            
            # 计算止损价格 (基于ATR)
            stop_loss_distance = atr * 2
            stop_loss = current_price - stop_loss_distance
            
            # 确保止损价格合理 (不超过8%损失)
            max_stop_loss = current_price * 0.92
            stop_loss = max(stop_loss, max_stop_loss)
            
            # 计算目标价格 (风险收益比1:2.5)
            risk = current_price - stop_loss
            target_price = current_price + (risk * 2.5)
            
            # 根据市场环境和风险指标调整仓位大小
            base_position = 0.05  # 基础仓位5%
            
            # 市场环境调整（支持字符串和枚举）
            env = market_env_result.get('environment', 'CONSOLIDATION')
            if hasattr(env, 'value'):
                env_str = env.value
            else:
                env_str = str(env)
            
            if 'BULL' in env_str or '牛市' in env_str:
                market_multiplier = 1.5  # 牛市增加仓位
            elif 'CONSOLIDATION' in env_str or '震荡' in env_str:
                market_multiplier = 1.0  # 震荡市正常仓位
            else:  # 熊市
                market_multiplier = 0.6  # 熊市减少仓位
            
            # 根据止损距离调整仓位（风险越大，仓位越小）
            risk_distance = (current_price - stop_loss) / current_price
            if risk_distance > 0.06:  # 止损距离超过6%
                risk_multiplier = 0.7
            elif risk_distance > 0.04:  # 止损距离超过4%
                risk_multiplier = 0.85
            else:  # 止损距离较小
                risk_multiplier = 1.0
            
            position_size = base_position * market_multiplier * risk_multiplier
            
            # 根据夏普比率进一步调整仓位
            if risk_metrics:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 1.0)
                volatility = risk_metrics.get('volatility', 0.2)
                
                # 夏普比率越高，可以适当增加仓位
                if sharpe_ratio > 2.0:
                    position_size *= 1.3
                elif sharpe_ratio > 1.5:
                    position_size *= 1.15
                elif sharpe_ratio < 0.5:
                    position_size *= 0.7
                
                # 波动率越高，减少仓位
                if volatility > 0.35:
                    position_size *= 0.7
                elif volatility > 0.25:
                    position_size *= 0.85
            
            # 确保价格都是正数，仓位在合理范围内
            return {
                'buy_price': round(max(buy_price, 0.01), 2),
                'stop_loss': round(max(stop_loss, 0.01), 2),
                'target_price': round(max(target_price, current_price * 1.01), 2),  # 目标价至少比当前价高1%
                'position_size': min(max(position_size, 0.02), 0.12)  # 仓位在2%-12%之间
            }
            
        except Exception as e:
            logger.error(f"生成买入信号失败: {e}")
            # 返回基于当前价格的合理默认值
            try:
                current_price = data.iloc[-1]['close']
                return {
                    'buy_price': current_price * 0.995,
                    'stop_loss': current_price * 0.92,  # 8%止损
                    'target_price': current_price * 1.15,  # 15%目标
                    'position_size': 0.05
                }
            except:
                return {
                    'buy_price': 0.0,
                    'stop_loss': 0.0,
                    'target_price': 0.0,
                    'position_size': 0.05
                }
    
    def _classify_risk_level(self, risk_metrics: Dict[str, float]) -> RiskLevel:
        """分类风险等级"""
        try:
            volatility = risk_metrics.get('volatility', 0.3)
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
            max_drawdown = risk_metrics.get('max_drawdown', 0.2)
            
            # 风险评分
            risk_score = 0.0
            
            # 波动率评分 (波动率越高风险越大)
            if volatility < 0.15:
                risk_score += 1  # 低风险
            elif volatility < 0.25:
                risk_score += 2  # 中等风险
            elif volatility < 0.35:
                risk_score += 3  # 高风险
            else:
                risk_score += 4  # 极高风险
            
            # 夏普比率评分 (夏普比率越高风险调整收益越好)
            if sharpe_ratio > 2.0:
                risk_score -= 1  # 降低风险等级
            elif sharpe_ratio > 1.5:
                risk_score -= 0.5
            elif sharpe_ratio < 0.5:
                risk_score += 1  # 增加风险等级
            
            # 最大回撤评分
            if max_drawdown < 0.1:
                risk_score -= 0.5
            elif max_drawdown > 0.25:
                risk_score += 1
            
            # 根据综合评分分类
            if risk_score <= 1.5:
                return RiskLevel.CONSERVATIVE
            elif risk_score <= 2.5:
                return RiskLevel.MODERATE
            elif risk_score <= 3.5:
                return RiskLevel.AGGRESSIVE
            else:
                return RiskLevel.SPECULATIVE
                
        except Exception:
            return RiskLevel.MODERATE
    
    def _identify_strengths(self, data: pd.DataFrame, risk_metrics: Dict[str, float]) -> List[str]:
        """识别股票优势"""
        strengths = []
        
        try:
            latest_data = data.iloc[-1]
            
            # 技术面优势
            if latest_data['rsi'] < 40 and latest_data['rsi'] > 30:
                strengths.append("RSI显示超卖后企稳")
            
            if (latest_data['close'] > latest_data['sma_20'] > 
                latest_data['sma_50'] > latest_data['sma_200']):
                strengths.append("多重均线呈完美多头排列")
            
            if latest_data['volume_ratio'] > 1.5:
                strengths.append("成交量明显放大确认趋势")
            
            # 风险收益优势
            if risk_metrics.get('sharpe_ratio', 0) > 1.5:
                strengths.append("夏普比率优秀，风险调整收益突出")
            
            if risk_metrics.get('win_rate', 0.5) > 0.6:
                strengths.append("历史胜率较高")
            
            if risk_metrics.get('max_drawdown', 1.0) < 0.15:
                strengths.append("最大回撤控制良好")
            
            # 动量优势
            if len(data) >= 20:
                recent_return = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1)
                if recent_return > 0.1:
                    strengths.append("近期表现强势，动量良好")
            
            # 价值优势
            if len(data) >= 252:
                price_52w_low = data['close'].tail(252).min()
                current_price = latest_data['close']
                if current_price < price_52w_low * 1.2:
                    strengths.append("价格接近52周低点，具备价值优势")
            
            return strengths[:5]  # 最多返回5个优势
            
        except Exception:
            return ["数据分析中发现潜在机会"]
    
    def _identify_risks(self, data: pd.DataFrame, risk_metrics: Dict[str, float]) -> List[str]:
        """识别股票风险"""
        risks = []
        
        try:
            latest_data = data.iloc[-1]
            
            # 技术面风险
            if latest_data['rsi'] > 70:
                risks.append("RSI显示超买，短期回调风险")
            
            if latest_data['close'] < latest_data['sma_200']:
                risks.append("价格低于200日均线，长期趋势偏弱")
            
            if latest_data['volume_ratio'] < 0.5:
                risks.append("成交量萎缩，市场关注度不足")
            
            # 风险指标风险
            if risk_metrics.get('volatility', 0.3) > 0.35:
                risks.append("波动率较高，价格波动风险大")
            
            if risk_metrics.get('max_drawdown', 0.2) > 0.25:
                risks.append("历史最大回撤较大，抗风险能力偏弱")
            
            if risk_metrics.get('sharpe_ratio', 0) < 0.5:
                risks.append("夏普比率偏低，风险调整收益不佳")
            
            # 市场环境风险
            if len(data) >= 60:
                long_term_trend = (data['close'].iloc[-1] / data['close'].iloc[-60] - 1)
                if long_term_trend < -0.15:
                    risks.append("长期趋势向下，系统性风险")
            
            # 流动性风险
            avg_volume = data['volume'].tail(20).mean()
            if avg_volume < 1000000:  # 平均成交量小于100万
                risks.append("成交量偏小，流动性风险")
            
            return risks[:5]  # 最多返回5个风险
            
        except Exception:
            return ["存在一般市场风险"]
    
    def _assess_market_timing(self, data: pd.DataFrame, market_env_result: Dict) -> str:
        """评估市场时机"""
        try:
            latest_data = data.iloc[-1]
            
            # 技术指标时机评估
            rsi = latest_data['rsi']
            price = latest_data['close']
            ma_20 = latest_data['sma_20']
            volume_ratio = latest_data['volume_ratio']
            
            timing_score = 0
            
            # RSI时机
            if 30 <= rsi <= 50:
                timing_score += 2  # 理想买入区域
            elif 50 < rsi <= 60:
                timing_score += 1  # 可接受买入区域
            elif rsi < 30:
                timing_score += 1  # 超卖反弹机会
            
            # 价格位置
            if price > ma_20:
                timing_score += 1
            
            # 成交量确认
            if volume_ratio > 1.2:
                timing_score += 1
            
            # 市场环境调整
            if market_env_result['environment'] == MarketEnvironment.BULL_MARKET:
                timing_score += 1
            elif market_env_result['environment'] == MarketEnvironment.BEAR_MARKET:
                timing_score -= 1
            
            # 时机评估
            if timing_score >= 4:
                return "绝佳买入时机"
            elif timing_score >= 3:
                return "良好买入时机"
            elif timing_score >= 2:
                return "可考虑买入"
            elif timing_score >= 1:
                return "谨慎观察"
            else:
                return "暂不适合买入"
                
        except Exception:
            return "市场时机分析中"
    
    def generate_screening_report(self, results: List[StockScore], 
                                 top_n: int = 20) -> str:
        """生成筛选报告"""
        if not results:
            return "❌ 未发现符合条件的股票"
        
        # 取前N只股票
        top_stocks = results[:top_n]
        
        html_report = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>量化股票筛选报告</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .stock-card {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; 
                             border-radius: 8px; background: #f9f9f9; }}
                .score {{ font-size: 1.2em; font-weight: bold; }}
                .high-score {{ color: #28a745; }}
                .medium-score {{ color: #ffc107; }}
                .low-score {{ color: #dc3545; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                           gap: 10px; margin: 10px 0; }}
                .metric {{ background: white; padding: 10px; border-radius: 5px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 量化股票筛选报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>筛选结果: {len(results)} 只股票，展示前 {len(top_stocks)} 只</p>
            </div>
        """
        
        for i, stock in enumerate(top_stocks, 1):
            score_class = ("high-score" if stock.total_score >= 0.8 else 
                          "medium-score" if stock.total_score >= 0.6 else "low-score")
            
            html_report += f"""
            <div class="stock-card">
                <h2>{i}. {stock.symbol} 
                    <span class="score {score_class}">{stock.total_score:.2f}分</span>
                    <span style="color: #666; font-size: 0.8em;">({stock.risk_level.value})</span>
                </h2>
                
                <div class="metrics">
                    <div class="metric">
                        <strong>技术分析</strong><br>{stock.technical_score:.2f}
                    </div>
                    <div class="metric">
                        <strong>动量分析</strong><br>{stock.momentum_score:.2f}
                    </div>
                    <div class="metric">
                        <strong>夏普比率</strong><br>{stock.sharpe_ratio:.2f}
                    </div>
                    <div class="metric">
                        <strong>最大回撤</strong><br>{stock.max_drawdown:.1%}
                    </div>
                    <div class="metric">
                        <strong>预期收益</strong><br>{stock.expected_return:.1%}
                    </div>
                    <div class="metric">
                        <strong>胜率</strong><br>{stock.win_rate:.1%}
                    </div>
                </div>
                
                <div style="background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="margin-top: 0; color: #2c5282;">💰 交易建议</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                        <div><strong>买入价格:</strong> ${stock.buy_price:.2f}</div>
                        <div><strong>止损价格:</strong> ${stock.stop_loss:.2f}</div>
                        <div><strong>目标价格:</strong> ${stock.target_price:.2f}</div>
                        <div><strong>建议仓位:</strong> {stock.position_size:.1%}</div>
                    </div>
                    <div style="margin-top: 10px;">
                        <strong>盈亏比:</strong> {((stock.target_price - stock.buy_price) / (stock.buy_price - stock.stop_loss)):.1f}:1 |
                        <strong>上涨空间:</strong> {((stock.target_price - stock.buy_price) / stock.buy_price * 100):.1f}% |
                        <strong>下跌风险:</strong> {((stock.buy_price - stock.stop_loss) / stock.buy_price * 100):.1f}%
                    </div>
                </div>
                
                <p><strong>📊 市场环境:</strong> {stock.market_env.value}</p>
                <p><strong>🎯 推荐策略:</strong> {stock.strategy_recommendation}</p>
                <p><strong>⭐ 信号质量:</strong> {stock.signal_quality:.2f}</p>
                
                <p><strong>✅ 优势:</strong> {', '.join(stock.strengths[:3])}</p>
                <p><strong>⚠️ 风险:</strong> {', '.join(stock.risks[:3])}</p>
            </div>
            """
        
        html_report += """
        </body>
        </html>
        """
        
        return html_report

# 添加更多辅助方法
def main():
    """主函数 - 演示量化筛选器的使用"""
    try:
        # 创建筛选器
        screener = QuantitativeStockScreener()
        
        # 执行筛选
        results = screener.screen_stocks(risk_preference=RiskLevel.MODERATE)
        
        # 生成报告
        report = screener.generate_screening_report(results, top_n=10)
        
        # 保存报告
        report_file = f"量化股票筛选报告_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 筛选完成！报告已保存到: {report_file}")
        
        # 打印摘要
        if results:
            print(f"\n📊 筛选摘要:")
            print(f"总共筛选: {len(results)} 只股票")
            print(f"前5名股票:")
            for i, stock in enumerate(results[:5], 1):
                print(f"  {i}. {stock.symbol} - {stock.total_score:.2f}分 "
                     f"({stock.risk_level.value})")
        
    except Exception as e:
        logger.error(f"❌ 量化筛选执行失败: {e}")

if __name__ == "__main__":
    main()