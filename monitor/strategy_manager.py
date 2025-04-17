import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from strategy.strategy_factory import StrategyFactory
from monitor.technical_analysis import TechnicalAnalysis

class StrategyManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化策略管理器
        :param config: 策略配置
        """
        self.config = config or {}
        self.strategies = {}
        self.risk_params = self.config.get('risk_params', {})
        self.logger = logging.getLogger(__name__)
        
        # 初始化技术分析工具
        self.technical_analysis = TechnicalAnalysis()
        
        # 初始化策略
        self._init_strategies()
        
        # 添加市场情绪指标
        self.market_sentiment = {
            'overall': 'neutral',
            'sector_sentiment': {},
            'market_volatility': 0.0
        }
        
    def _init_strategies(self):
        """初始化所有策略"""
        try:
            # 初始化牛牛V3策略
            from strategy.niuniu_strategy_v3 import NiuniuStrategyV3
            self.strategies['NiuniuV3'] = NiuniuStrategyV3()
            
            # 初始化其他策略...
            
        except Exception as e:
            self.logger.error(f"初始化策略失败: {e}")
            
    def analyze_stock(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        分析股票
        :param df: 股票数据
        :param symbol: 股票代码
        :return: 分析结果
        """
        try:
            # 获取策略权重
            strategy_weights = self.config.get('strategy_weights', {'NiuniuV3': 1.0})
            
            # 计算技术指标
            technical_indicators = self._calculate_technical_indicators(df)
            
            # 计算波动率
            volatility = self._calculate_volatility(df)
            
            # 更新市场情绪
            self._update_market_sentiment(df, symbol)
            
            # 获取各策略信号
            signals = {}
            total_score = 0
            for strategy_name, weight in strategy_weights.items():
                if strategy_name in self.strategies:
                    strategy = self.strategies[strategy_name]
                    signal = strategy.analyze(df)
                    signals[strategy_name] = signal
                    total_score += signal.get('score', 0) * weight
                    
            # 计算综合得分
            total_weight = sum(strategy_weights.values())
            final_score = total_score / total_weight if total_weight > 0 else 0
            
            # 计算动态止损止盈
            stop_loss, take_profit = self._calculate_dynamic_levels(df, volatility)
            
            # 计算建议仓位
            position_size = self._calculate_position_size(final_score, volatility)
            
            # 生成建议
            recommendations = self._generate_recommendations(signals, final_score, volatility, technical_indicators)
            
            # 确定风险等级
            risk_level = self._determine_risk_level(final_score, volatility)
            
            return {
                'signals': signals,
                'score': final_score,
                'risk_level': risk_level,
                'stop_loss_price': stop_loss,
                'take_profit_price': take_profit,
                'recommendations': recommendations,
                'position_size': position_size,
                'volatility': volatility,
                'market_sentiment': self.market_sentiment,
                'technical_indicators': technical_indicators
            }
            
        except Exception as e:
            self.logger.error(f"分析股票 {symbol} 失败: {e}")
            return {
                'signals': {},
                'score': 0,
                'risk_level': 'high',
                'stop_loss_price': None,
                'take_profit_price': None,
                'recommendations': [f"分析失败: {str(e)}"],
                'position_size': 0,
                'volatility': 0,
                'market_sentiment': self.market_sentiment,
                'technical_indicators': {}
            }
            
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """
        计算技术指标
        :param df: 股票数据
        :return: 技术指标数据
        """
        try:
            # 计算布林带
            bollinger_bands = self.technical_analysis.calculate_bollinger_bands(df)
            
            # 计算KDJ
            kdj = self.technical_analysis.calculate_kdj(df)
            
            # 计算成交量指标
            volume_indicators = self.technical_analysis.calculate_volume_indicators(df)
            
            # 计算斐波那契回调位
            fibonacci_levels = self.technical_analysis.calculate_fibonacci_levels(df)
            
            return {
                'bollinger_bands': bollinger_bands,
                'kdj': kdj,
                'volume_indicators': volume_indicators,
                'fibonacci_levels': fibonacci_levels
            }
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            return {}
            
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        计算波动率
        :param df: 股票数据
        :return: 波动率
        """
        try:
            # 计算日收益率
            returns = df['close'].pct_change().dropna()
            
            # 计算历史波动率（20日）
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            return float(volatility)
            
        except Exception as e:
            self.logger.error(f"计算波动率失败: {e}")
            return 0.0
            
    def _update_market_sentiment(self, df: pd.DataFrame, symbol: str):
        """
        更新市场情绪
        :param df: 股票数据
        :param symbol: 股票代码
        """
        try:
            # 计算市场整体情绪
            price_trend = df['close'].pct_change(5).mean()  # 5日价格趋势
            volume_trend = df['volume'].pct_change(5).mean()  # 5日成交量趋势
            
            if price_trend > 0.02 and volume_trend > 0:
                self.market_sentiment['overall'] = 'bullish'
            elif price_trend < -0.02 and volume_trend > 0:
                self.market_sentiment['overall'] = 'bearish'
            else:
                self.market_sentiment['overall'] = 'neutral'
                
            # 更新市场波动率
            self.market_sentiment['market_volatility'] = self._calculate_volatility(df)
            
            # 更新行业情绪（这里需要行业数据，暂时留空）
            
        except Exception as e:
            self.logger.error(f"更新市场情绪失败: {e}")
            
    def _calculate_dynamic_levels(self, df: pd.DataFrame, volatility: float) -> Tuple[float, float]:
        """
        计算动态止损止盈水平
        :param df: 股票数据
        :param volatility: 波动率
        :return: (止损价, 止盈价)
        """
        try:
            current_price = df['close'].iloc[-1]
            
            # 基础止损止盈比例
            base_stop_loss = self.risk_params.get('stop_loss', 0.05)
            base_take_profit = self.risk_params.get('take_profit', 0.1)
            
            # 根据波动率调整
            volatility_factor = min(max(volatility / 0.2, 0.5), 2.0)  # 波动率调整因子
            
            # 根据市场情绪调整
            sentiment_factor = 1.0
            if self.market_sentiment['overall'] == 'bullish':
                sentiment_factor = 1.2
            elif self.market_sentiment['overall'] == 'bearish':
                sentiment_factor = 0.8
                
            # 计算最终止损止盈水平
            stop_loss_pct = base_stop_loss * volatility_factor * sentiment_factor
            take_profit_pct = base_take_profit * volatility_factor * sentiment_factor
            
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
            
            return float(stop_loss), float(take_profit)
            
        except Exception as e:
            self.logger.error(f"计算动态止损止盈失败: {e}")
            return None, None
            
    def _calculate_position_size(self, score: float, volatility: float) -> float:
        """
        计算建议仓位
        :param score: 策略得分
        :param volatility: 波动率
        :return: 建议仓位比例
        """
        try:
            # 基础仓位限制
            base_limit = self.risk_params.get('position_size_limit', 0.2)
            
            # 根据得分调整
            score_factor = min(max(score / 50, 0.5), 1.5)  # 得分调整因子
            
            # 根据波动率调整
            volatility_factor = max(1 - volatility / 0.3, 0.5)  # 波动率调整因子
            
            # 根据市场情绪调整
            sentiment_factor = 1.0
            if self.market_sentiment['overall'] == 'bullish':
                sentiment_factor = 1.2
            elif self.market_sentiment['overall'] == 'bearish':
                sentiment_factor = 0.8
                
            # 计算最终仓位
            position_size = base_limit * score_factor * volatility_factor * sentiment_factor
            
            return float(position_size)
            
        except Exception as e:
            self.logger.error(f"计算建议仓位失败: {e}")
            return 0.0
            
    def _generate_recommendations(self, signals: Dict, score: float, volatility: float, 
                                technical_indicators: Dict) -> List[str]:
        """
        生成交易建议
        :param signals: 各策略信号
        :param score: 综合得分
        :param volatility: 波动率
        :param technical_indicators: 技术指标数据
        :return: 建议列表
        """
        recommendations = []
        
        try:
            # 根据得分生成建议
            if score > 70:
                recommendations.append("强烈建议买入")
            elif score > 50:
                recommendations.append("建议买入")
            elif score > 30:
                recommendations.append("建议观望")
            else:
                recommendations.append("建议卖出")
                
            # 根据波动率添加建议
            if volatility > 0.3:
                recommendations.append("波动较大，注意风险控制")
            elif volatility < 0.1:
                recommendations.append("波动较小，适合长期持有")
                
            # 根据市场情绪添加建议
            if self.market_sentiment['overall'] == 'bullish':
                recommendations.append("市场情绪积极，可适当增加仓位")
            elif self.market_sentiment['overall'] == 'bearish':
                recommendations.append("市场情绪消极，建议谨慎操作")
                
            # 添加技术指标建议
            self._add_technical_recommendations(recommendations, technical_indicators)
                
            # 添加各策略的具体建议
            for strategy_name, signal in signals.items():
                if 'recommendations' in signal:
                    recommendations.extend(signal['recommendations'])
                    
        except Exception as e:
            self.logger.error(f"生成建议失败: {e}")
            recommendations.append("生成建议时出错")
            
        return recommendations
        
    def _add_technical_recommendations(self, recommendations: List[str], technical_indicators: Dict):
        """
        添加技术指标建议
        :param recommendations: 建议列表
        :param technical_indicators: 技术指标数据
        """
        try:
            # 布林带建议
            bb = technical_indicators.get('bollinger_bands', {})
            if bb:
                position = bb.get('position', 0)
                if position > 0.8:
                    recommendations.append("布林带显示价格接近上轨，可能面临回调")
                elif position < 0.2:
                    recommendations.append("布林带显示价格接近下轨，可能面临反弹")
                    
            # KDJ建议
            kdj = technical_indicators.get('kdj', {})
            if kdj:
                signal = kdj.get('signal', '')
                if signal == 'oversold':
                    recommendations.append("KDJ显示超卖，可能面临反弹")
                elif signal == 'overbought':
                    recommendations.append("KDJ显示超买，可能面临回调")
                    
            # 成交量建议
            volume = technical_indicators.get('volume_indicators', {})
            if volume:
                volume_ratio = volume.get('volume_ratio', 1)
                if volume_ratio > 2:
                    recommendations.append("成交量显著放大，注意价格波动")
                elif volume_ratio < 0.5:
                    recommendations.append("成交量萎缩，市场活跃度降低")
                    
            # 斐波那契建议
            fib = technical_indicators.get('fibonacci_levels', {})
            if fib:
                nearest_level = fib.get('nearest_level', (None, None))
                if nearest_level[0]:
                    recommendations.append(f"价格接近斐波那契{nearest_level[0]}回调位")
                    
        except Exception as e:
            self.logger.error(f"添加技术指标建议失败: {e}")
            
    def _determine_risk_level(self, score: float, volatility: float) -> str:
        """
        确定风险等级
        :param score: 综合得分
        :param volatility: 波动率
        :return: 风险等级
        """
        try:
            # 根据得分和波动率综合判断
            if score < 30 or volatility > 0.3:
                return 'high'
            elif score < 50 or volatility > 0.2:
                return 'warning'
            else:
                return 'normal'
                
        except Exception as e:
            self.logger.error(f"确定风险等级失败: {e}")
            return 'high'

    def update_config(self, new_config: Dict):
        """
        更新配置
        :param new_config: 新的配置字典
        """
        self.config = self._merge_config(self.config, new_config)
        self._init_strategies()
        
    def get_strategy_info(self) -> Dict:
        """
        获取策略信息
        :return: 策略信息字典
        """
        return {
            'active_strategies': list(self.strategies.keys()),
            'weights': self.config['strategy_weights'],
            'risk_params': self.config['risk_params']
        }

    def _merge_config(self, default: Dict, custom: Dict) -> Dict:
        """
        合并配置
        """
        result = default.copy()
        for key, value in custom.items():
            if isinstance(value, dict) and key in result:
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result 