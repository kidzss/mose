import pandas as pd
import numpy as np
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

from .indicators.indicators import TechnicalIndicators
from .indicators.volatility import atr, historical_volatility, bollinger_bandwidth
from .indicators.trend_strength import adx, aroon, directional_movement_index, supertrend
from .indicators.volume import on_balance_volume, money_flow_index
from .indicators.oscillators import stochastic_oscillator, cci
from .indicators.support_resistance import pivot_points_traditional, price_channels

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketClassifier")

class MarketEnvironment(Enum):
    """市场环境枚举类型"""
    STRONG_UPTREND = "强势上升趋势"
    WEAK_UPTREND = "弱势上升趋势"
    STRONG_DOWNTREND = "强势下降趋势"
    WEAK_DOWNTREND = "弱势下降趋势"
    RANGE_BOUND = "区间震荡"
    CHOPPY = "混沌无序"
    UNKNOWN = "未知状态"


class MarketEnvironmentClassifier:
    """
    市场环境分类器
    
    该类用于分析市场数据并将当前市场状态分类为：
    1. 趋势市场（上升/下降，强势/弱势）
    2. 区间震荡市场
    3. 混沌无序市场
    
    分类结果可用于选择最适合当前市场环境的交易策略。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化市场环境分类器
        
        参数:
            config: 配置字典，可包含分析参数
        """
        logger.info("初始化市场环境分类器")
        # 默认配置
        self.default_config = {
            # 趋势参数
            'adx_period': 14,
            'adx_trend_threshold': 25,  # ADX大于此值视为有趋势
            'adx_strong_trend_threshold': 35,  # ADX大于此值视为强趋势
            'aroon_period': 14,
            'aroon_threshold': 70,  # Aroon-Up/-Down大于此值且另一个指标小于30，视为趋势信号
            'supertrend_period': 10,
            'supertrend_multiplier': 3.0,
            
            # 波动率参数
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'bandwidth_expansion_threshold': 0.05,  # 布林带宽度扩大阈值
            'volatility_period': 21,
            'volatility_threshold': 0.3,  # 历史波动率阈值
            'atr_period': 14,
            
            # 区间震荡参数
            'range_period': 20,  # 判断区间的历史天数
            'range_threshold': 0.15,  # 波动幅度/价格不超过此值视为区间
            'range_touch_count': 2,  # 触及区间上下限的最小次数
            
            # 混沌市场参数
            'choppy_volume_threshold': 1.5,  # 成交量波动阈值
            'choppy_direction_changes': 3,  # 方向变化次数阈值
            'choppy_period': 10,  # 评估混沌的历史天数
            
            # 分析周期
            'short_term_period': 14,  # 短期趋势天数
            'medium_term_period': 30,  # 中期趋势天数
            'long_term_period': 60,  # 长期趋势天数
            
            # 多时间框架权重
            'short_term_weight': 0.5,
            'medium_term_weight': 0.3,
            'long_term_weight': 0.2,
        }
        
        # 应用用户配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
    def classify_environment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分类市场环境
        
        参数:
            data: OHLCV数据，包含open, high, low, close, volume列
            
        返回:
            包含市场环境信息的字典，包括：
            - environment: MarketEnvironment枚举类型
            - confidence: 分类置信度 (0-1)
            - details: 分类的详细原因和支持因素
            - sub_environments: 各子环境的评估结果
            - indicators: 用于分类的各项指标值
        """
        logger.info("开始对市场环境进行分类...")
        
        try:
            if data is None or len(data) < max(self.config['long_term_period'], 50):
                logger.warning(f"数据不足，需要至少{max(self.config['long_term_period'], 50)}条数据")
                return {
                    'environment': MarketEnvironment.UNKNOWN,
                    'confidence': 0,
                    'details': {'reason': '数据不足以进行市场环境分析'}
                }
            
            # 计算指标
            indicators = self._calculate_indicators(data)
            
            # 分析各环境特征
            trend_analysis = self._analyze_trend(data, indicators)
            range_analysis = self._analyze_range_bound(data, indicators)
            choppy_analysis = self._analyze_choppy(data, indicators)
            
            # 多时间框架分析
            timeframe_analysis = self._multi_timeframe_analysis(data)
            
            # 汇总环境评分
            environment_scores = {
                MarketEnvironment.STRONG_UPTREND: trend_analysis['uptrend_score'] * trend_analysis['strength_score'],
                MarketEnvironment.WEAK_UPTREND: trend_analysis['uptrend_score'] * (1 - trend_analysis['strength_score']),
                MarketEnvironment.STRONG_DOWNTREND: trend_analysis['downtrend_score'] * trend_analysis['strength_score'],
                MarketEnvironment.WEAK_DOWNTREND: trend_analysis['downtrend_score'] * (1 - trend_analysis['strength_score']),
                MarketEnvironment.RANGE_BOUND: range_analysis['range_score'],
                MarketEnvironment.CHOPPY: choppy_analysis['choppy_score']
            }
            
            # 应用多时间框架分析结果
            for env, score in environment_scores.items():
                if env in [MarketEnvironment.STRONG_UPTREND, MarketEnvironment.WEAK_UPTREND]:
                    environment_scores[env] *= timeframe_analysis['uptrend_alignment']
                elif env in [MarketEnvironment.STRONG_DOWNTREND, MarketEnvironment.WEAK_DOWNTREND]:
                    environment_scores[env] *= timeframe_analysis['downtrend_alignment']
            
            # 确定最终环境
            best_environment = max(environment_scores.items(), key=lambda x: x[1])
            environment = best_environment[0]
            confidence = best_environment[1]
            
            # 收集详细原因
            details = self._collect_classification_details(
                environment, confidence, trend_analysis, range_analysis, 
                choppy_analysis, timeframe_analysis
            )
            
            result = {
                'environment': environment,
                'confidence': confidence,
                'details': details,
                'sub_environments': {
                    'trend': trend_analysis,
                    'range_bound': range_analysis,
                    'choppy': choppy_analysis,
                    'timeframes': timeframe_analysis
                },
                'indicators': indicators
            }
            
            logger.info(f"环境分类完成: {environment.value}, 置信度: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"市场环境分析出错: {str(e)}", exc_info=True)
            return {
                'environment': MarketEnvironment.UNKNOWN,
                'confidence': 0,
                'details': {'reason': f"分析出错: {str(e)}"}
            }
    
    def get_suitable_strategies(self, environment: MarketEnvironment) -> List[str]:
        """
        根据市场环境获取适合的策略
        
        参数:
            environment: 市场环境类型
            
        返回:
            适合当前环境的策略列表
        """
        strategy_map = {
            MarketEnvironment.STRONG_UPTREND: ['trend_following_strategy', 'momentum_strategy', 'advanced_multi_factor_strategy'],
            MarketEnvironment.WEAK_UPTREND: ['momentum_strategy', 'advanced_multi_factor_strategy'],
            MarketEnvironment.STRONG_DOWNTREND: ['trend_following_strategy', 'advanced_multi_factor_strategy'],
            MarketEnvironment.WEAK_DOWNTREND: ['advanced_multi_factor_strategy'],
            MarketEnvironment.RANGE_BOUND: ['bollinger_bands_strategy', 'mean_reversion_strategy'],
            MarketEnvironment.CHOPPY: ['market_sentiment_strategy'],
            MarketEnvironment.UNKNOWN: ['market_sentiment_strategy', 'advanced_multi_factor_strategy']
        }
        
        return strategy_map.get(environment, ['advanced_multi_factor_strategy'])
        
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算用于环境分类的指标"""
        df = data.copy()
        indicators = {}
        
        # 计算ADX
        adx_result = adx(
            df['high'], df['low'], df['close'], 
            window=self.config['adx_period']
        )
        indicators['adx'] = adx_result['adx']
        indicators['plus_di'] = adx_result['plus_di']
        indicators['minus_di'] = adx_result['minus_di']
        
        # 计算Aroon指标
        aroon_result = aroon(
            df['high'], df['low'], 
            period=self.config['aroon_period']
        )
        indicators['aroon_up'] = aroon_result['aroon_up']
        indicators['aroon_down'] = aroon_result['aroon_down']
        indicators['aroon_osc'] = aroon_result['aroon_osc']
        
        # 计算Supertrend指标
        supertrend_result = supertrend(
            df['high'], df['low'], df['close'],
            period=self.config['supertrend_period'],
            multiplier=self.config['supertrend_multiplier']
        )
        indicators['supertrend'] = supertrend_result['supertrend']
        indicators['supertrend_direction'] = supertrend_result['trend']
        
        # 计算Bollinger Bands带宽
        indicators['bollinger_bandwidth'] = bollinger_bandwidth(
            df['close'],
            window=self.config['bollinger_period'],
            num_std=self.config['bollinger_std']
        )
        
        # 计算波动率
        indicators['historical_volatility'] = historical_volatility(
            df['close'], 
            window=self.config['volatility_period']
        )
        
        # 计算ATR和ATR百分比
        indicators['atr'] = atr(
            df['high'], df['low'], df['close'], 
            window=self.config['atr_period']
        )
        indicators['atr_pct'] = indicators['atr'] / df['close']
        
        # 计算支撑阻力位
        price_channel_result = price_channels(
            df['high'], df['low'], 
            period=self.config['range_period']
        )
        indicators['upper_channel'] = price_channel_result['upper']
        indicators['lower_channel'] = price_channel_result['lower']
        
        # 成交量指标
        indicators['obv'] = on_balance_volume(df['close'], df['volume'])
        
        # 移动平均线
        indicators['sma_20'] = df['close'].rolling(20).mean()
        indicators['sma_50'] = df['close'].rolling(50).mean()
        indicators['sma_200'] = df['close'].rolling(200).mean()
        
        return indicators
        
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, float]:
        """分析趋势强度和方向"""
        # 最新的指标值
        recent_idx = -1
        adx = indicators['adx'].iloc[recent_idx]
        plus_di = indicators['plus_di'].iloc[recent_idx]
        minus_di = indicators['minus_di'].iloc[recent_idx]
        aroon_up = indicators['aroon_up'].iloc[recent_idx]
        aroon_down = indicators['aroon_down'].iloc[recent_idx]
        supertrend_direction = indicators['supertrend_direction'].iloc[recent_idx]
        
        # 移动平均线关系
        sma_20 = indicators['sma_20'].iloc[recent_idx]
        sma_50 = indicators['sma_50'].iloc[recent_idx]
        sma_200 = indicators['sma_200'].iloc[recent_idx] if len(indicators['sma_200'].dropna()) > 0 else None
        
        # 价格
        close = data['close'].iloc[recent_idx]
        
        # 1. 计算趋势强度分数 (0-1)
        adx_strength = min(adx / self.config['adx_strong_trend_threshold'], 1.0)
        
        # 2. 计算上升趋势分数 (0-1)
        uptrend_signals = []
        
        # ADX方向信号
        if plus_di > minus_di:
            uptrend_signals.append(min((plus_di - minus_di) / 10, 1.0))
            
        # Aroon指标信号
        if aroon_up > self.config['aroon_threshold'] and aroon_down < 30:
            uptrend_signals.append(min(aroon_up / 100, 1.0))
            
        # Supertrend信号
        if supertrend_direction == 1:  # 1表示上升趋势
            uptrend_signals.append(1.0)
            
        # 移动平均线信号
        ma_alignment_score = 0
        if close > sma_20:
            ma_alignment_score += 0.33
        if sma_20 is not None and close > sma_50:
            ma_alignment_score += 0.33
        if sma_200 is not None and close > sma_200:
            ma_alignment_score += 0.34
        uptrend_signals.append(ma_alignment_score)
        
        # 计算最终上升趋势分数
        uptrend_score = sum(uptrend_signals) / len(uptrend_signals) if uptrend_signals else 0
        
        # 3. 计算下降趋势分数 (0-1)
        downtrend_signals = []
        
        # ADX方向信号
        if minus_di > plus_di:
            downtrend_signals.append(min((minus_di - plus_di) / 10, 1.0))
            
        # Aroon指标信号
        if aroon_down > self.config['aroon_threshold'] and aroon_up < 30:
            downtrend_signals.append(min(aroon_down / 100, 1.0))
            
        # Supertrend信号
        if supertrend_direction == -1:  # -1表示下降趋势
            downtrend_signals.append(1.0)
            
        # 移动平均线信号
        ma_alignment_score = 0
        if close < sma_20:
            ma_alignment_score += 0.33
        if sma_20 is not None and close < sma_50:
            ma_alignment_score += 0.33
        if sma_200 is not None and close < sma_200:
            ma_alignment_score += 0.34
        downtrend_signals.append(ma_alignment_score)
        
        # 计算最终下降趋势分数
        downtrend_score = sum(downtrend_signals) / len(downtrend_signals) if downtrend_signals else 0
        
        # 4. 趋势方向偏差（+1为纯上升，-1为纯下降）
        direction_bias = uptrend_score - downtrend_score
        
        # 收集趋势特征
        trend_features = []
        if adx > self.config['adx_trend_threshold']:
            trend_features.append(f"ADX ({adx:.1f}) 大于趋势阈值 {self.config['adx_trend_threshold']}")
        
        if plus_di > minus_di:
            trend_features.append(f"+DI ({plus_di:.1f}) 大于 -DI ({minus_di:.1f})")
        else:
            trend_features.append(f"-DI ({minus_di:.1f}) 大于 +DI ({plus_di:.1f})")
            
        if aroon_up > self.config['aroon_threshold']:
            trend_features.append(f"Aroon-Up ({aroon_up:.1f}) 指示上升趋势")
        if aroon_down > self.config['aroon_threshold']:
            trend_features.append(f"Aroon-Down ({aroon_down:.1f}) 指示下降趋势")
            
        trend_features.append(f"Supertrend {'看多' if supertrend_direction == 1 else '看空'}")
        
        if close > sma_50:
            trend_features.append("价格位于50日均线上方")
        else:
            trend_features.append("价格位于50日均线下方")
            
        # 返回分析结果
        return {
            'strength_score': adx_strength,
            'uptrend_score': uptrend_score,
            'downtrend_score': downtrend_score,
            'direction_bias': direction_bias,
            'trend_features': trend_features
        }
    
    def _analyze_range_bound(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """分析区间震荡特征"""
        period = self.config['range_period']
        recent_data = data.iloc[-period:]
        
        # 检查价格是否在一定范围内波动
        price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['low'].min()
        range_bound = price_range < self.config['range_threshold']
        
        # 检查是否多次触及上下边界
        upper_channel = indicators['upper_channel'].iloc[-period:]
        lower_channel = indicators['lower_channel'].iloc[-period:]
        
        # 上边界触及次数
        upper_touches = sum((recent_data['high'] >= upper_channel * 0.98).astype(int))
        # 下边界触及次数
        lower_touches = sum((recent_data['low'] <= lower_channel * 1.02).astype(int))
        
        # 波动率是否维持在低水平
        recent_volatility = indicators['historical_volatility'].iloc[-1]
        low_volatility = recent_volatility < 0.2  # 20%是低波动率的经验值
        
        # 价格在中轨附近震荡的比例
        mid_channel = (upper_channel + lower_channel) / 2
        mid_range_ratio = sum(
            (recent_data['close'] >= mid_channel * 0.95) & 
            (recent_data['close'] <= mid_channel * 1.05)
        ) / len(recent_data)
        
        # ADX是否低于趋势阈值
        adx = indicators['adx'].iloc[-1]
        weak_trend = adx < self.config['adx_trend_threshold']
        
        # 区间震荡特征
        range_features = []
        if range_bound:
            range_features.append(f"价格在过去{period}天的波动范围为{price_range:.1%}，低于阈值{self.config['range_threshold']:.1%}")
        
        if upper_touches >= self.config['range_touch_count']:
            range_features.append(f"价格多次（{upper_touches}次）触及上轨")
        if lower_touches >= self.config['range_touch_count']:
            range_features.append(f"价格多次（{lower_touches}次）触及下轨")
            
        if low_volatility:
            range_features.append(f"波动率（{recent_volatility:.1%}）维持在较低水平")
            
        if mid_range_ratio > 0.6:
            range_features.append(f"价格主要在中轨附近震荡，占比{mid_range_ratio:.1%}")
            
        if weak_trend:
            range_features.append(f"ADX（{adx:.1f}）低于趋势阈值，表明趋势弱")
        
        # 计算区间震荡总分
        touch_score = min((upper_touches + lower_touches) / (2 * self.config['range_touch_count']), 1.0)
        volatility_score = 1.0 - min(recent_volatility / 0.3, 1.0)  # 波动率越低分数越高
        adx_score = 1.0 - min(adx / self.config['adx_trend_threshold'], 1.0)  # ADX越低分数越高
        
        range_score = (touch_score + volatility_score + adx_score + mid_range_ratio) / 4
        
        return {
            'range_score': range_score,
            'price_range': price_range,
            'upper_touches': upper_touches,
            'lower_touches': lower_touches,
            'mid_range_ratio': mid_range_ratio,
            'volatility': recent_volatility,
            'range_features': range_features
        }
    
    def _analyze_choppy(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """分析混沌无序特征"""
        period = self.config['choppy_period']
        recent_data = data.iloc[-period:]
        
        # 计算方向变化次数
        price_directions = np.sign(recent_data['close'].diff())
        direction_changes = ((price_directions != price_directions.shift(1)) & 
                            (price_directions.shift(1) != 0)).sum()
        
        # 分析成交量波动
        volume_std = recent_data['volume'].std() / recent_data['volume'].mean()
        high_volume_volatility = volume_std > self.config['choppy_volume_threshold']
        
        # 价格与均线关系频繁变化
        # 确保索引匹配，先提取整个指标，然后根据recent_data的索引重新索引
        sma20 = indicators['sma_20'].reindex(recent_data.index)
        sma20_prev = sma20.shift(1)
        close = recent_data['close']
        close_prev = close.shift(1)
        
        # 现在比较相同索引的Series
        sma20_crosses = ((close > sma20) != (close_prev > sma20_prev)).sum()
        
        # 计算混沌市场程度
        direction_change_score = min(direction_changes / self.config['choppy_direction_changes'], 1.0)
        volume_volatility_score = min(volume_std / self.config['choppy_volume_threshold'], 1.0)
        sma_cross_score = min(sma20_crosses / (period/10), 1.0)  # 假设每10天最多发生一次均线交叉
        
        # 混沌市场特征
        choppy_features = []
        if direction_changes >= self.config['choppy_direction_changes']:
            choppy_features.append(f"价格方向在过去{period}天内频繁变化（{direction_changes}次）")
            
        if high_volume_volatility:
            choppy_features.append(f"成交量波动较大（标准差/均值 = {volume_std:.2f}）")
            
        if sma20_crosses >= 3:
            choppy_features.append(f"价格与20日均线频繁交叉（{sma20_crosses}次）")
        
        # 计算混沌市场总分
        choppy_score = (direction_change_score + volume_volatility_score + sma_cross_score) / 3
        
        return {
            'choppy_score': choppy_score,
            'direction_changes': direction_changes,
            'volume_volatility': volume_std,
            'sma_crosses': sma20_crosses,
            'choppy_features': choppy_features
        }
    
    def _multi_timeframe_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """多时间框架分析"""
        # 分割数据为不同时间框架
        short_term = data.iloc[-self.config['short_term_period']:]
        medium_term = data.iloc[-self.config['medium_term_period']:] if len(data) >= self.config['medium_term_period'] else short_term
        long_term = data.iloc[-self.config['long_term_period']:] if len(data) >= self.config['long_term_period'] else medium_term
        
        # 计算各时间框架的趋势方向
        # 简单使用开盘收盘对比
        short_trend = 1 if short_term['close'].iloc[-1] > short_term['close'].iloc[0] else -1
        medium_trend = 1 if medium_term['close'].iloc[-1] > medium_term['close'].iloc[0] else -1
        long_trend = 1 if long_term['close'].iloc[-1] > long_term['close'].iloc[0] else -1
        
        # 计算趋势一致性得分 (1表示完全一致，0表示完全不一致)
        uptrend_alignment = 0
        downtrend_alignment = 0
        
        # 上升趋势一致性
        if short_trend == 1:
            uptrend_alignment += self.config['short_term_weight']
        if medium_trend == 1:
            uptrend_alignment += self.config['medium_term_weight']
        if long_trend == 1:
            uptrend_alignment += self.config['long_term_weight']
            
        # 下降趋势一致性
        if short_trend == -1:
            downtrend_alignment += self.config['short_term_weight']
        if medium_trend == -1:
            downtrend_alignment += self.config['medium_term_weight']
        if long_trend == -1:
            downtrend_alignment += self.config['long_term_weight']
            
        # 收集多时间框架特征
        timeframe_features = []
        if short_trend == 1:
            timeframe_features.append(f"短期（{self.config['short_term_period']}天）为上升趋势")
        else:
            timeframe_features.append(f"短期（{self.config['short_term_period']}天）为下降趋势")
            
        if medium_trend == 1:
            timeframe_features.append(f"中期（{self.config['medium_term_period']}天）为上升趋势")
        else:
            timeframe_features.append(f"中期（{self.config['medium_term_period']}天）为下降趋势")
            
        if long_trend == 1:
            timeframe_features.append(f"长期（{self.config['long_term_period']}天）为上升趋势")
        else:
            timeframe_features.append(f"长期（{self.config['long_term_period']}天）为下降趋势")
            
        # 返回分析结果
        return {
            'short_trend': short_trend,
            'medium_trend': medium_trend,
            'long_trend': long_trend,
            'uptrend_alignment': uptrend_alignment,
            'downtrend_alignment': downtrend_alignment,
            'timeframe_features': timeframe_features
        }
    
    def _collect_classification_details(
            self, 
            environment: MarketEnvironment, 
            confidence: float,
            trend_analysis: Dict[str, Any],
            range_analysis: Dict[str, Any],
            choppy_analysis: Dict[str, Any],
            timeframe_analysis: Dict[str, Any]
        ) -> Dict[str, Any]:
        """收集分类的详细原因"""
        reasons = []
        
        if environment in [MarketEnvironment.STRONG_UPTREND, MarketEnvironment.WEAK_UPTREND]:
            reasons.extend(trend_analysis['trend_features'])
            reasons.extend([f for f in timeframe_analysis['timeframe_features'] if '上升' in f])
            reasons.append(f"趋势强度评分：{trend_analysis['strength_score']:.2f}")
            reasons.append(f"上升趋势评分：{trend_analysis['uptrend_score']:.2f}")
            
            if environment == MarketEnvironment.STRONG_UPTREND:
                reasons.append("该环境适合趋势跟踪和动量策略")
            else:
                reasons.append("该环境适合动量策略和多因子策略")
                
        elif environment in [MarketEnvironment.STRONG_DOWNTREND, MarketEnvironment.WEAK_DOWNTREND]:
            reasons.extend(trend_analysis['trend_features'])
            reasons.extend([f for f in timeframe_analysis['timeframe_features'] if '下降' in f])
            reasons.append(f"趋势强度评分：{trend_analysis['strength_score']:.2f}")
            reasons.append(f"下降趋势评分：{trend_analysis['downtrend_score']:.2f}")
            
            if environment == MarketEnvironment.STRONG_DOWNTREND:
                reasons.append("该环境适合趋势跟踪策略（做空）")
            else:
                reasons.append("该环境适合多因子策略（谨慎操作）")
                
        elif environment == MarketEnvironment.RANGE_BOUND:
            reasons.extend(range_analysis['range_features'])
            reasons.append(f"区间震荡评分：{range_analysis['range_score']:.2f}")
            reasons.append("该环境适合布林带策略和均值回归策略")
            
        elif environment == MarketEnvironment.CHOPPY:
            reasons.extend(choppy_analysis['choppy_features'])
            reasons.append(f"混沌市场评分：{choppy_analysis['choppy_score']:.2f}")
            reasons.append("该环境适合情绪市场策略或降低交易频率")
        
        return {
            'reasons': reasons,
            'suitable_strategies': self.get_suitable_strategies(environment)
        } 