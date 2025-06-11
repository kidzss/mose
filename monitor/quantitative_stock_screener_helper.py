#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

# 简化的MarketEnvironment类
class MarketEnvironment:
    BULL_MARKET = "牛市"
    BEAR_MARKET = "熊市" 
    CONSOLIDATION = "震荡市"

class RiskLevel:
    CONSERVATIVE = "保守型"
    MODERATE = "稳健型"
    AGGRESSIVE = "激进型"
    SPECULATIVE = "投机型"

class QuantitativeStockScreenerHelper:
    """量化股票筛选器辅助方法类"""
    
    @staticmethod
    def calculate_price_momentum(data: pd.DataFrame) -> float:
        """计算价格动量评分"""
        try:
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
    
    @staticmethod
    def calculate_rsi_trend(data: pd.DataFrame) -> float:
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
    
    @staticmethod
    def calculate_volume_momentum(data: pd.DataFrame) -> float:
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
    
    @staticmethod
    def calculate_trend_consistency(data: pd.DataFrame) -> float:
        """计算趋势一致性评分"""
        try:
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
    
    @staticmethod
    def evaluate_signal_quality(data: pd.DataFrame, market_env_result: Dict) -> float:
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
    
    @staticmethod
    def generate_buy_signals(data: pd.DataFrame, market_env_result: Dict) -> Dict[str, float]:
        """生成买入信号"""
        try:
            latest_data = data.iloc[-1]
            current_price = latest_data['close']
            atr = latest_data['atr']
            
            # 计算买入价格 (当前价格附近)
            buy_price = current_price
            
            # 计算止损价格 (基于ATR)
            stop_loss = current_price - (atr * 2)
            
            # 计算目标价格 (风险收益比1:2)
            risk = current_price - stop_loss
            target_price = current_price + (risk * 2)
            
            # 根据市场环境调整仓位大小
            base_position = 0.05  # 基础仓位5%
            
            if market_env_result['environment'] == MarketEnvironment.BULL_MARKET:
                position_size = base_position * 1.5  # 牛市增加仓位
            elif market_env_result['environment'] == MarketEnvironment.CONSOLIDATION:
                position_size = base_position  # 震荡市正常仓位
            else:  # 熊市
                position_size = base_position * 0.5  # 熊市减少仓位
            
            return {
                'buy_price': buy_price,
                'stop_loss': max(stop_loss, current_price * 0.9),  # 止损不超过10%
                'target_price': target_price,
                'position_size': min(position_size, 0.1)  # 单只股票仓位不超过10%
            }
            
        except Exception:
            return {
                'buy_price': 0.0,
                'stop_loss': 0.0,
                'target_price': 0.0,
                'position_size': 0.05
            }
    
    @staticmethod
    def classify_risk_level(risk_metrics: Dict[str, float]) -> RiskLevel:
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
    
    @staticmethod
    def identify_strengths(data: pd.DataFrame, risk_metrics: Dict[str, float]) -> List[str]:
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
    
    @staticmethod
    def identify_risks(data: pd.DataFrame, risk_metrics: Dict[str, float]) -> List[str]:
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
    
    @staticmethod
    def assess_market_timing(data: pd.DataFrame, market_env_result: Dict) -> str:
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

# 辅助方法类定义完成
# 注意：这些方法需要在主类中手动导入使用