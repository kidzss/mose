#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资决策支持系统
Investment Decision Support System

专门为避免抄底抄到半山腰、避免卖到半路而设计
提供明确的买卖时机判断和建议记录功能
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np

class DecisionSupportSystem:
    """投资决策支持系统"""
    
    def __init__(self):
        self.decisions_file = "investment_decisions.json"
        self.decisions_history = self._load_decisions_history()
        
        # 决策规则配置
        self.rules = {
            # 避免抄底的规则
            'anti_bottom_fishing': {
                'trend_confirmation_days': 5,  # 趋势确认天数
                'volume_threshold': 1.2,       # 成交量放大阈值
                'ma_alignment_required': True,  # 需要均线多头排列
                'rsi_min_threshold': 35,       # RSI最低阈值（避免抄在刀刃上）
            },
            
            # 避免卖到半路的规则
            'anti_early_exit': {
                'profit_protection_threshold': 8,   # 8%以上利润开始保护
                'trailing_stop_percentage': 5,     # 5%跟踪止损
                'rsi_exit_threshold': 75,          # RSI卖出阈值
                'volume_divergence_alert': True,   # 量价背离警告
            },
            
            # 市场环境判断
            'market_environment': {
                'vix_fear_threshold': 25,      # VIX恐慌阈值
                'market_trend_days': 10,       # 市场趋势判断天数
                'sector_correlation_check': True,  # 板块相关性检查
            }
        }
    
    def analyze_position_management(self, symbol: str, current_position_pct: float, 
                                   target_position_pct: float, current_analysis: Dict) -> Dict:
        """仓位管理分析 - 专业的加仓/减仓建议"""
        
        # 获取历史数据
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='2mo')
        
        if hist.empty:
            return {'decision': 'DATA_ERROR', 'reason': '无法获取历史数据'}
        
        current_price = hist['Close'].iloc[-1]
        
        # 技术分析
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma50 = hist['Close'].rolling(50).mean().iloc[-1]
        
        # 计算RSI - 防止除零错误
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss_safe = loss.replace(0, 1e-10)
        rs = gain / loss_safe
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # 计算关键价位
        pullback_5pct = current_price * 0.95
        pullback_8pct = current_price * 0.92
        pullback_10pct = current_price * 0.90
        
        # 分析当前技术状态
        price_vs_ma20_pct = (current_price / ma20 - 1) * 100
        is_overbought = rsi > 70
        is_oversold = rsi < 30
        is_healthy = 40 <= rsi <= 65
        
        # 仓位管理决策逻辑
        position_gap = target_position_pct - current_position_pct
        
        # 生成具体的加仓策略
        strategies = self._generate_position_strategies(
            current_price, ma20, rsi, position_gap, 
            pullback_5pct, pullback_8pct, pullback_10pct
        )
        
        # 风险评估
        risk_assessment = self._assess_position_risk(
            current_price, ma20, rsi, price_vs_ma20_pct, current_position_pct
        )
        
        # 最优加仓时机
        optimal_timing = self._calculate_optimal_timing(
            current_price, ma20, rsi, hist
        )
        
        # 综合决策
        final_decision = self._make_position_decision(
            strategies, risk_assessment, optimal_timing, position_gap
        )
        
        # 记录决策
        decision_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'decision_type': 'POSITION_MANAGEMENT',
            'current_price': float(current_price),
            'current_position': current_position_pct,
            'target_position': target_position_pct,
            'position_gap': position_gap,
            'technical_data': {
                'ma20': float(ma20),
                'ma50': float(ma50) if not pd.isna(ma50) else None,
                'rsi': float(rsi),
                'price_vs_ma20_pct': float(price_vs_ma20_pct)
            },
            'strategies': strategies,
            'risk_assessment': risk_assessment,
            'optimal_timing': optimal_timing,
            'decision': final_decision,
            'user_notes': ""
        }
        
        return decision_record
    
    def _generate_position_strategies(self, current_price: float, ma20: float, rsi: float,
                                    position_gap: float, pullback_5pct: float, 
                                    pullback_8pct: float, pullback_10pct: float) -> Dict:
        """生成具体的仓位调整策略"""
        
        strategies = {}
        
        if position_gap > 0:  # 需要加仓
            # 稳健分批策略
            strategies['conservative'] = {
                'name': '稳健分批策略',
                'recommended': True,
                'batches': [
                    {
                        'batch': 1,
                        'price_range': f"${pullback_5pct:.2f} - ${current_price-2:.2f}",
                        'position_add': min(3.0, position_gap/2),
                        'condition': '等待5%回调'
                    },
                    {
                        'batch': 2, 
                        'price_range': f"${ma20:.2f} - ${pullback_8pct:.2f}",
                        'position_add': min(4.0, position_gap - min(3.0, position_gap/2)),
                        'condition': '回调至MA20附近'
                    }
                ],
                'total_add': position_gap,
                'risk_level': 'MEDIUM'
            }
            
            # 激进追涨策略
            strategies['aggressive'] = {
                'name': '激进追涨策略',
                'recommended': False,
                'batches': [
                    {
                        'batch': 1,
                        'price_range': f"${current_price+1:.2f}以上",
                        'position_add': position_gap,
                        'condition': '突破确认'
                    }
                ],
                'total_add': position_gap,
                'risk_level': 'HIGH'
            }
            
            # 保守等待策略
            strategies['patient'] = {
                'name': '保守等待策略',
                'recommended': rsi > 65,
                'batches': [
                    {
                        'batch': 1,
                        'price_range': f"${ma20-3:.2f}以下",
                        'position_add': position_gap,
                        'condition': '深度回调'
                    }
                ],
                'total_add': position_gap,
                'risk_level': 'LOW'
            }
        
        else:  # 需要减仓
            strategies['profit_taking'] = {
                'name': '获利了结策略',
                'recommended': True,
                'batches': [
                    {
                        'batch': 1,
                        'price_range': f"当前价位${current_price:.2f}",
                        'position_reduce': abs(position_gap),
                        'condition': '立即执行'
                    }
                ],
                'total_reduce': abs(position_gap),
                'risk_level': 'LOW'
            }
        
        return strategies
    
    def _assess_position_risk(self, current_price: float, ma20: float, rsi: float,
                            price_vs_ma20_pct: float, current_position_pct: float) -> Dict:
        """评估仓位风险"""
        
        risk_factors = []
        risk_score = 0
        
        # RSI风险
        if rsi > 80:
            risk_factors.append("RSI极度超买")
            risk_score += 30
        elif rsi > 70:
            risk_factors.append("RSI超买状态")
            risk_score += 20
        elif rsi < 20:
            risk_factors.append("RSI极度超卖")
            risk_score += 15
        
        # 价格偏离风险
        if price_vs_ma20_pct > 10:
            risk_factors.append(f"价格比MA20高{price_vs_ma20_pct:.1f}%")
            risk_score += 25
        elif price_vs_ma20_pct > 5:
            risk_factors.append(f"价格比MA20高{price_vs_ma20_pct:.1f}%")
            risk_score += 15
        
        # 仓位集中度风险
        if current_position_pct > 25:
            risk_factors.append("单股仓位过高")
            risk_score += 20
        elif current_position_pct > 20:
            risk_factors.append("仓位相对集中")
            risk_score += 10
        
        # 风险等级
        if risk_score >= 50:
            risk_level = "HIGH"
            recommendation = "建议减仓或等待"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
            recommendation = "谨慎加仓"
        else:
            risk_level = "LOW"
            recommendation = "可以正常操作"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendation': recommendation
        }
    
    def _calculate_optimal_timing(self, current_price: float, ma20: float, 
                                rsi: float, hist: pd.DataFrame) -> Dict:
        """计算最优操作时机"""
        
        timing_signals = []
        
        # RSI时机
        if rsi > 70:
            timing_signals.append("等待RSI回落至60以下")
        elif rsi < 40:
            timing_signals.append("RSI超卖，适合买入")
        else:
            timing_signals.append("RSI处于健康区间")
        
        # 价格时机
        if current_price > ma20 * 1.05:
            timing_signals.append(f"等待回调至MA20(${ma20:.2f})附近")
        elif current_price < ma20 * 0.98:
            timing_signals.append("价格接近MA20支撑，适合买入")
        
        # 成交量时机
        recent_volume = hist['Volume'].tail(5).mean()
        avg_volume = hist['Volume'].tail(20).mean()
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 1.5:
            timing_signals.append("成交量放大，关注方向")
        elif volume_ratio < 0.8:
            timing_signals.append("成交量萎缩，等待放量")
        
        return {
            'signals': timing_signals,
            'best_timing': "未来1-2周内关注回调机会" if rsi > 65 else "当前时机相对合适"
        }
    
    def _make_position_decision(self, strategies: Dict, risk_assessment: Dict,
                              optimal_timing: Dict, position_gap: float) -> Dict:
        """做出最终的仓位决策"""
        
        risk_level = risk_assessment['risk_level']
        risk_score = risk_assessment['risk_score']
        
        if position_gap > 0:  # 加仓决策
            if risk_level == "HIGH":
                decision = "WAIT"
                action = "暂时不要加仓"
                confidence = 85
                reason = "技术面风险较高，建议等待回调"
                recommended_strategy = "patient"
            elif risk_level == "MEDIUM":
                decision = "CAUTIOUS_ADD"
                action = "可以小幅加仓"
                confidence = 65
                reason = "技术面偏热，建议谨慎分批加仓"
                recommended_strategy = "conservative"
            else:
                decision = "ADD"
                action = "可以按计划加仓"
                confidence = 75
                reason = "技术面健康，可以执行加仓计划"
                recommended_strategy = "conservative"
        
        else:  # 减仓决策
            decision = "REDUCE"
            action = "建议适当减仓"
            confidence = 70
            reason = "仓位过高，建议适当减仓"
            recommended_strategy = "profit_taking"
        
        return {
            'decision': decision,
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'recommended_strategy': recommended_strategy,
            'risk_level': risk_level,
            'summary': f"{action} (信心度: {confidence}%, 风险: {risk_level})"
        }

    def analyze_buy_timing(self, symbol: str, current_analysis: Dict) -> Dict:
        """分析买入时机 - 避免抄底陷阱"""
        
        # 获取历史数据进行趋势确认
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')
        
        if hist.empty:
            return {'decision': 'DATA_ERROR', 'reason': '无法获取历史数据'}
        
        current_price = hist['Close'].iloc[-1]
        
        # 1. 趋势确认分析
        trend_analysis = self._analyze_trend_confirmation(hist)
        
        # 2. 技术指标确认
        technical_confirmation = self._analyze_technical_confirmation(hist, current_analysis)
        
        # 3. 成交量确认
        volume_confirmation = self._analyze_volume_confirmation(hist)
        
        # 4. 市场环境确认
        market_confirmation = self._analyze_market_environment()
        
        # 综合决策
        decision = self._make_buy_decision(
            trend_analysis, technical_confirmation, 
            volume_confirmation, market_confirmation
        )
        
        # 记录决策过程
        decision_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'decision_type': 'BUY_TIMING',
            'current_price': float(current_price),
            'analysis': {
                'trend': trend_analysis,
                'technical': technical_confirmation,
                'volume': volume_confirmation,
                'market': market_confirmation
            },
            'decision': decision,
            'user_notes': ""  # 用户可以添加自己的想法
        }
        
        return decision_record
    
    def analyze_sell_timing(self, symbol: str, position_info: Dict, current_analysis: Dict) -> Dict:
        """分析卖出时机 - 避免卖到半路"""
        
        # 获取持仓信息
        cost_basis = position_info.get('cost_basis', 0)
        shares = position_info.get('shares', 0)
        current_price = current_analysis.get('current_price', 0)
        
        if cost_basis == 0 or current_price == 0:
            return {'decision': 'DATA_ERROR', 'reason': '持仓信息不完整'}
        
        # 计算盈亏
        profit_pct = (current_price - cost_basis) / cost_basis * 100
        
        # 获取历史数据
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')
        
        # 1. 利润保护分析
        profit_protection = self._analyze_profit_protection(profit_pct, current_analysis)
        
        # 2. 趋势延续分析
        trend_continuation = self._analyze_trend_continuation(hist)
        
        # 3. 技术指标恶化分析
        technical_deterioration = self._analyze_technical_deterioration(hist, current_analysis)
        
        # 4. 量价背离分析
        volume_divergence = self._analyze_volume_divergence(hist)
        
        # 综合卖出决策
        decision = self._make_sell_decision(
            profit_protection, trend_continuation,
            technical_deterioration, volume_divergence, profit_pct
        )
        
        # 记录决策过程
        decision_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'decision_type': 'SELL_TIMING',
            'current_price': float(current_price),
            'cost_basis': float(cost_basis),
            'profit_pct': float(profit_pct),
            'analysis': {
                'profit_protection': profit_protection,
                'trend_continuation': trend_continuation,
                'technical_deterioration': technical_deterioration,
                'volume_divergence': volume_divergence
            },
            'decision': decision,
            'user_notes': ""
        }
        
        return decision_record
    
    def _analyze_trend_confirmation(self, hist: pd.DataFrame) -> Dict:
        """趋势确认分析"""
        
        # 计算移动平均线
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['MA50'] = hist['Close'].rolling(50).mean()
        
        current_price = hist['Close'].iloc[-1]
        ma5 = hist['MA5'].iloc[-1]
        ma20 = hist['MA20'].iloc[-1]
        ma50 = hist['MA50'].iloc[-1]
        
        # 多头排列确认
        bullish_alignment = current_price > ma5 > ma20 > ma50
        
        # 均线斜率确认（避免横盘整理时的假突破）
        ma20_slope = (hist['MA20'].iloc[-1] - hist['MA20'].iloc[-6]) / hist['MA20'].iloc[-6] * 100
        ma50_slope = (hist['MA50'].iloc[-1] - hist['MA50'].iloc[-11]) / hist['MA50'].iloc[-11] * 100
        
        # 趋势强度
        if bullish_alignment and ma20_slope > 1 and ma50_slope > 0.5:
            trend_strength = "STRONG_UPTREND"
            confidence = 90
        elif bullish_alignment and ma20_slope > 0:
            trend_strength = "MODERATE_UPTREND"
            confidence = 70
        elif current_price > ma20 > ma50:
            trend_strength = "WEAK_UPTREND"
            confidence = 50
        else:
            trend_strength = "NO_UPTREND"
            confidence = 20
        
        return {
            'trend_strength': trend_strength,
            'confidence': int(confidence),
            'bullish_alignment': bool(bullish_alignment),
            'ma20_slope': float(ma20_slope),
            'ma50_slope': float(ma50_slope),
            'summary': f"趋势强度: {trend_strength}, 信心度: {confidence}%"
        }
    
    def _analyze_technical_confirmation(self, hist: pd.DataFrame, current_analysis: Dict) -> Dict:
        """技术指标确认分析"""
        
        technical_data = current_analysis.get('technical_analysis', {})
        indicators = technical_data.get('indicators', {})
        
        rsi = indicators.get('rsi', 50)
        
        # RSI确认（避免抄在刀刃上）
        if rsi > 70:
            rsi_signal = "OVERBOUGHT_RISK"
            rsi_confidence = 20
        elif rsi > 50:
            rsi_signal = "HEALTHY_RANGE"
            rsi_confidence = 80
        elif rsi > 35:
            rsi_signal = "OVERSOLD_OPPORTUNITY"
            rsi_confidence = 90
        else:
            rsi_signal = "EXTREME_OVERSOLD"
            rsi_confidence = 30  # 可能还会继续下跌
        
        # 布林带位置
        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        current_price = indicators.get('current_price', 0)
        
        if bb_upper > 0 and bb_lower > 0:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            if bb_position > 0.8:
                bb_signal = "NEAR_UPPER_BAND"
            elif bb_position > 0.5:
                bb_signal = "MIDDLE_RANGE"
            elif bb_position > 0.2:
                bb_signal = "LOWER_RANGE"
            else:
                bb_signal = "NEAR_LOWER_BAND"
        else:
            bb_signal = "DATA_UNAVAILABLE"
            bb_position = 0.5
        
        return {
            'rsi_signal': rsi_signal,
            'rsi_value': float(rsi),
            'rsi_confidence': rsi_confidence,
            'bb_signal': bb_signal,
            'bb_position': float(bb_position),
            'summary': f"RSI: {rsi:.1f} ({rsi_signal}), 布林带位置: {bb_position:.2f}"
        }
    
    def _analyze_volume_confirmation(self, hist: pd.DataFrame) -> Dict:
        """成交量确认分析"""
        
        # 计算成交量移动平均
        hist['Volume_MA20'] = hist['Volume'].rolling(20).mean()
        
        current_volume = hist['Volume'].iloc[-1]
        avg_volume = hist['Volume_MA20'].iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # 最近5天的成交量趋势
        recent_volumes = hist['Volume'].tail(5).values
        volume_trend = "INCREASING" if recent_volumes[-1] > recent_volumes[0] else "DECREASING"
        
        # 价量配合分析
        recent_prices = hist['Close'].tail(5).values
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if price_change > 0 and volume_trend == "INCREASING":
            volume_confirmation = "BULLISH_CONFIRMATION"
            confidence = 85
        elif price_change > 0 and volume_ratio > 1.2:
            volume_confirmation = "MODERATE_CONFIRMATION"
            confidence = 70
        elif price_change > 0 and volume_ratio < 0.8:
            volume_confirmation = "WEAK_CONFIRMATION"
            confidence = 40
        else:
            volume_confirmation = "NO_CONFIRMATION"
            confidence = 30
        
        return {
            'volume_confirmation': volume_confirmation,
            'confidence': confidence,
            'volume_ratio': float(volume_ratio),
            'volume_trend': volume_trend,
            'summary': f"成交量比率: {volume_ratio:.2f}, 确认度: {volume_confirmation}"
        }
    
    def _analyze_market_environment(self) -> Dict:
        """市场环境分析"""
        
        try:
            # 获取市场指数数据
            indices = {'^GSPC': 'SP500', '^IXIC': 'NASDAQ', '^VIX': 'VIX'}
            market_data = {}
            
            for symbol, name in indices.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev_week = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]
                    change_pct = (current - prev_week) / prev_week * 100
                    market_data[name] = {
                        'current': float(current),
                        'weekly_change': float(change_pct)
                    }
            
            # 市场环境判断
            if 'VIX' in market_data:
                vix_level = market_data['VIX']['current']
                if vix_level > 30:
                    market_sentiment = "HIGH_FEAR"
                    buy_recommendation = "WAIT"
                elif vix_level > 20:
                    market_sentiment = "MODERATE_FEAR"
                    buy_recommendation = "CAUTIOUS"
                else:
                    market_sentiment = "LOW_FEAR"
                    buy_recommendation = "NORMAL"
            else:
                market_sentiment = "UNKNOWN"
                buy_recommendation = "NORMAL"
            
            # 大盘趋势
            if 'SP500' in market_data and 'NASDAQ' in market_data:
                sp_change = market_data['SP500']['weekly_change']
                nasdaq_change = market_data['NASDAQ']['weekly_change']
                
                if sp_change > 2 and nasdaq_change > 2:
                    market_trend = "STRONG_BULLISH"
                elif sp_change > 0 and nasdaq_change > 0:
                    market_trend = "BULLISH"
                elif sp_change < -2 and nasdaq_change < -2:
                    market_trend = "BEARISH"
                else:
                    market_trend = "NEUTRAL"
            else:
                market_trend = "UNKNOWN"
            
            return {
                'market_sentiment': market_sentiment,
                'market_trend': market_trend,
                'buy_recommendation': buy_recommendation,
                'market_data': market_data,
                'summary': f"市场情绪: {market_sentiment}, 趋势: {market_trend}"
            }
            
        except Exception as e:
            return {
                'market_sentiment': 'ERROR',
                'market_trend': 'ERROR',
                'buy_recommendation': 'WAIT',
                'error': str(e),
                'summary': '市场数据获取失败，建议谨慎操作'
            }
    
    def _make_buy_decision(self, trend_analysis: Dict, technical_confirmation: Dict,
                          volume_confirmation: Dict, market_confirmation: Dict) -> Dict:
        """综合买入决策"""
        
        # 评分系统
        score = 0
        reasons = []
        warnings = []
        
        # 趋势分析评分
        trend_strength = trend_analysis['trend_strength']
        if trend_strength == "STRONG_UPTREND":
            score += 40
            reasons.append("强势上升趋势确认")
        elif trend_strength == "MODERATE_UPTREND":
            score += 25
            reasons.append("中等上升趋势")
        elif trend_strength == "WEAK_UPTREND":
            score += 10
            reasons.append("弱势上升趋势")
        else:
            score -= 20
            warnings.append("缺乏明确上升趋势")
        
        # 技术指标评分
        rsi_signal = technical_confirmation['rsi_signal']
        if rsi_signal == "HEALTHY_RANGE":
            score += 20
            reasons.append("RSI处于健康区间")
        elif rsi_signal == "OVERSOLD_OPPORTUNITY":
            score += 30
            reasons.append("RSI超卖提供机会")
        elif rsi_signal == "OVERBOUGHT_RISK":
            score -= 15
            warnings.append("RSI超买存在风险")
        elif rsi_signal == "EXTREME_OVERSOLD":
            score -= 10
            warnings.append("RSI极度超卖，可能继续下跌")
        
        # 成交量确认评分
        volume_confirmation_level = volume_confirmation['volume_confirmation']
        if volume_confirmation_level == "BULLISH_CONFIRMATION":
            score += 25
            reasons.append("成交量强势确认")
        elif volume_confirmation_level == "MODERATE_CONFIRMATION":
            score += 15
            reasons.append("成交量中等确认")
        elif volume_confirmation_level == "WEAK_CONFIRMATION":
            score += 5
            reasons.append("成交量弱确认")
        else:
            score -= 10
            warnings.append("成交量未确认")
        
        # 市场环境评分
        market_recommendation = market_confirmation['buy_recommendation']
        if market_recommendation == "NORMAL":
            score += 15
            reasons.append("市场环境正常")
        elif market_recommendation == "CAUTIOUS":
            score += 5
            warnings.append("市场环境需谨慎")
        else:
            score -= 15
            warnings.append("市场环境不利")
        
        # 最终决策
        if score >= 70:
            decision = "STRONG_BUY"
            action = "建议买入"
            confidence = min(95, score)
        elif score >= 50:
            decision = "BUY"
            action = "可以买入"
            confidence = min(85, score)
        elif score >= 30:
            decision = "HOLD_WAIT"
            action = "等待更好时机"
            confidence = 60
        else:
            decision = "AVOID"
            action = "避免买入"
            confidence = 70
        
        return {
            'decision': decision,
            'action': action,
            'confidence': confidence,
            'score': score,
            'reasons': reasons,
            'warnings': warnings,
            'summary': f"{action} (信心度: {confidence}%, 评分: {score})"
        }
    
    def _analyze_trend_continuation(self, hist: pd.DataFrame) -> Dict:
        """趋势延续分析"""
        
        # 计算移动平均线
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['MA50'] = hist['Close'].rolling(50).mean()
        
        current_price = hist['Close'].iloc[-1]
        ma5 = hist['MA5'].iloc[-1]
        ma20 = hist['MA20'].iloc[-1]
        ma50 = hist['MA50'].iloc[-1]
        
        # 趋势延续性分析
        if current_price > ma5 > ma20 > ma50:
            trend_status = "STRONG_UPTREND"
            continuation_probability = 85
        elif current_price > ma20 > ma50:
            trend_status = "MODERATE_UPTREND"
            continuation_probability = 70
        elif current_price > ma20:
            trend_status = "WEAK_UPTREND"
            continuation_probability = 55
        else:
            trend_status = "DOWNTREND"
            continuation_probability = 30
        
        return {
            'trend_status': trend_status,
            'continuation_probability': continuation_probability,
            'summary': f"趋势状态: {trend_status}, 延续概率: {continuation_probability}%"
        }
    
    def _analyze_technical_deterioration(self, hist: pd.DataFrame, current_analysis: Dict) -> Dict:
        """技术指标恶化分析"""
        
        technical_data = current_analysis.get('technical_analysis', {})
        indicators = technical_data.get('indicators', {})
        
        rsi = indicators.get('rsi', 50)
        
        # RSI恶化信号
        if rsi > 80:
            rsi_deterioration = "EXTREME_OVERBOUGHT"
            deterioration_score = 90
        elif rsi > 70:
            rsi_deterioration = "OVERBOUGHT"
            deterioration_score = 70
        elif rsi < 30:
            rsi_deterioration = "OVERSOLD_BOUNCE"
            deterioration_score = 60
        else:
            rsi_deterioration = "NORMAL"
            deterioration_score = 20
        
        return {
            'rsi_deterioration': rsi_deterioration,
            'deterioration_score': deterioration_score,
            'summary': f"技术恶化程度: {rsi_deterioration}, 评分: {deterioration_score}"
        }
    
    def _analyze_volume_divergence(self, hist: pd.DataFrame) -> Dict:
        """量价背离分析"""
        
        # 计算价格和成交量的相关性
        recent_prices = hist['Close'].tail(10).pct_change().dropna()
        recent_volumes = hist['Volume'].tail(10).pct_change().dropna()
        
        if len(recent_prices) > 5 and len(recent_volumes) > 5:
            correlation = recent_prices.corr(recent_volumes)
            
            if correlation < -0.3:
                divergence_status = "NEGATIVE_DIVERGENCE"
                warning_level = 80
            elif correlation < 0.1:
                divergence_status = "WEAK_CORRELATION"
                warning_level = 40
            else:
                divergence_status = "NORMAL_CORRELATION"
                warning_level = 20
        else:
            divergence_status = "INSUFFICIENT_DATA"
            warning_level = 50
        
        return {
            'divergence_status': divergence_status,
            'warning_level': warning_level,
            'summary': f"量价关系: {divergence_status}, 警告等级: {warning_level}"
        }
    
    def _analyze_profit_protection(self, profit_pct: float, current_analysis: Dict) -> Dict:
        """利润保护分析"""
        
        if profit_pct < 0:
            return {
                'protection_level': 'LOSS_CUTTING',
                'action': '考虑止损',
                'reason': f'当前亏损 {profit_pct:.1f}%'
            }
        elif profit_pct < 5:
            return {
                'protection_level': 'MINIMAL_PROFIT',
                'action': '继续持有',
                'reason': f'利润微薄 {profit_pct:.1f}%，等待更大涨幅'
            }
        elif profit_pct < 15:
            return {
                'protection_level': 'MODERATE_PROFIT',
                'action': '设置跟踪止损',
                'reason': f'利润 {profit_pct:.1f}%，开始保护'
            }
        elif profit_pct < 30:
            return {
                'protection_level': 'GOOD_PROFIT',
                'action': '考虑部分获利',
                'reason': f'利润丰厚 {profit_pct:.1f}%，可考虑减仓'
            }
        else:
            return {
                'protection_level': 'EXCELLENT_PROFIT',
                'action': '分批获利了结',
                'reason': f'利润优异 {profit_pct:.1f}%，建议分批卖出'
            }
    
    def _make_sell_decision(self, profit_protection: Dict, trend_continuation: Dict,
                           technical_deterioration: Dict, volume_divergence: Dict, profit_pct: float) -> Dict:
        """综合卖出决策"""
        
        # 简化的卖出决策逻辑
        protection_level = profit_protection['protection_level']
        
        if protection_level == 'LOSS_CUTTING' and profit_pct < -10:
            decision = "SELL_STOP_LOSS"
            action = "止损卖出"
            confidence = 90
        elif protection_level == 'EXCELLENT_PROFIT':
            decision = "SELL_PARTIAL"
            action = "分批获利"
            confidence = 80
        elif protection_level == 'GOOD_PROFIT':
            decision = "SELL_CONSIDER"
            action = "考虑减仓"
            confidence = 70
        else:
            decision = "HOLD"
            action = "继续持有"
            confidence = 60
        
        return {
            'decision': decision,
            'action': action,
            'confidence': confidence,
            'summary': f"{action} (信心度: {confidence}%)"
        }
    
    def add_user_note(self, symbol: str, note: str):
        """添加用户备注 - 增强版"""
        try:
            timestamp = datetime.now().isoformat()
            
            user_note = {
                'symbol': symbol,
                'timestamp': timestamp,
                'note': note,
                'type': 'USER_NOTE',
                'note_id': f"{symbol}_{timestamp.replace(':', '-')}",  # 唯一ID
                'status': 'active'  # 备注状态
            }
            
            if symbol not in self.decisions_history:
                self.decisions_history[symbol] = []
            
            self.decisions_history[symbol].append(user_note)
            self._save_decisions_history()
            
            # 记录到日志文件
            self._log_note(symbol, note, timestamp)
            
            return f"✅ 已成功记录 {symbol} 的备注: {note[:50]}{'...' if len(note) > 50 else ''}"
            
        except Exception as e:
            print(f"添加备注失败: {e}")
            return f"❌ 保存备注失败: {str(e)}"
    
    def _log_note(self, symbol: str, note: str, timestamp: str):
        """记录备注到日志文件"""
        try:
            log_file = "user_notes.log"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {symbol}: {note}\n")
        except Exception as e:
            print(f"记录备注日志失败: {e}")
    
    def get_user_notes(self, symbol: str, days: int = 30) -> List[Dict]:
        """获取用户备注"""
        if symbol not in self.decisions_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        user_notes = []
        for record in self.decisions_history[symbol]:
            if record.get('type') == 'USER_NOTE':
                record_date = datetime.fromisoformat(record['timestamp'])
                if record_date >= cutoff_date:
                    user_notes.append(record)
        
        return sorted(user_notes, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_user_note(self, symbol: str, note_id: str) -> str:
        """删除用户备注"""
        try:
            if symbol in self.decisions_history:
                for i, record in enumerate(self.decisions_history[symbol]):
                    if record.get('type') == 'USER_NOTE' and record.get('note_id') == note_id:
                        del self.decisions_history[symbol][i]
                        self._save_decisions_history()
                        return f"✅ 已删除备注: {record.get('note', '')[:30]}..."
            
            return "❌ 未找到要删除的备注"
            
        except Exception as e:
            return f"❌ 删除备注失败: {str(e)}"
    
    def update_user_note(self, symbol: str, note_id: str, new_note: str) -> str:
        """更新用户备注"""
        try:
            if symbol in self.decisions_history:
                for record in self.decisions_history[symbol]:
                    if record.get('type') == 'USER_NOTE' and record.get('note_id') == note_id:
                        old_note = record.get('note', '')
                        record['note'] = new_note
                        record['updated_at'] = datetime.now().isoformat()
                        self._save_decisions_history()
                        return f"✅ 已更新备注: {old_note[:30]}... → {new_note[:30]}..."
            
            return "❌ 未找到要更新的备注"
            
        except Exception as e:
            return f"❌ 更新备注失败: {str(e)}"
    
    def export_notes_for_ai(self, symbol: str = None) -> Dict:
        """导出备注数据用于AI分析"""
        try:
            export_data = {
                'export_time': datetime.now().isoformat(),
                'notes': [],
                'decisions': [],
                'summary': {}
            }
            
            symbols_to_export = [symbol] if symbol else list(self.decisions_history.keys())
            
            for sym in symbols_to_export:
                if sym in self.decisions_history:
                    symbol_data = {
                        'symbol': sym,
                        'user_notes': [],
                        'decisions': []
                    }
                    
                    for record in self.decisions_history[sym]:
                        if record.get('type') == 'USER_NOTE':
                            symbol_data['user_notes'].append({
                                'timestamp': record['timestamp'],
                                'note': record['note'],
                                'note_id': record.get('note_id', '')
                            })
                        else:
                            symbol_data['decisions'].append({
                                'timestamp': record['timestamp'],
                                'decision_type': record.get('decision_type', 'UNKNOWN'),
                                'decision': record.get('decision', {}),
                                'current_price': record.get('current_price', 0)
                            })
                    
                    export_data['notes'].append(symbol_data)
            
            # 生成摘要统计
            total_notes = sum(len(sym_data['user_notes']) for sym_data in export_data['notes'])
            total_decisions = sum(len(sym_data['decisions']) for sym_data in export_data['notes'])
            
            export_data['summary'] = {
                'total_symbols': len(symbols_to_export),
                'total_notes': total_notes,
                'total_decisions': total_decisions,
                'export_format': 'ai_analysis_ready'
            }
            
            return export_data
            
        except Exception as e:
            print(f"导出AI数据失败: {e}")
            return {'error': str(e)}
    
    def prepare_ai_analysis_prompt(self, symbol: str) -> str:
        """为AI分析准备提示词"""
        try:
            notes = self.get_user_notes(symbol, days=90)  # 获取90天的备注
            decisions = self.get_decision_history(symbol, days=90)
            
            prompt = f"""
# 投资决策分析请求

## 股票信息
- 股票代码: {symbol}
- 分析时间范围: 最近90天

## 用户备注记录 ({len(notes)} 条)
"""
            
            for note in notes[:10]:  # 最多显示10条备注
                note_time = datetime.fromisoformat(note['timestamp']).strftime('%Y-%m-%d %H:%M')
                prompt += f"- {note_time}: {note['note']}\n"
            
            prompt += f"""
## 系统决策记录 ({len(decisions)} 条)
"""
            
            for decision in decisions[:5]:  # 最多显示5条决策
                if decision.get('type') != 'USER_NOTE':
                    decision_time = datetime.fromisoformat(decision['timestamp']).strftime('%Y-%m-%d %H:%M')
                    decision_type = decision.get('decision_type', 'UNKNOWN')
                    prompt += f"- {decision_time} ({decision_type}): {decision.get('decision', {}).get('action', 'N/A')}\n"
            
            prompt += """
## 分析请求
请基于以上用户备注和系统决策记录，提供以下分析：

1. **投资思路分析**: 分析用户的主要投资思路和策略偏好
2. **决策质量评估**: 评估用户决策的合理性和改进空间
3. **风险控制建议**: 基于备注内容提供风险控制建议
4. **投资策略优化**: 提供具体的投资策略优化建议
5. **心理状态分析**: 分析用户的心理状态和投资情绪

请提供详细、专业的分析报告。
"""
            
            return prompt
            
        except Exception as e:
            return f"准备AI分析提示词失败: {str(e)}"
    
    def get_decision_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """获取决策历史"""
        if symbol not in self.decisions_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_decisions = []
        for decision in self.decisions_history[symbol]:
            decision_date = datetime.fromisoformat(decision['timestamp'])
            if decision_date >= cutoff_date:
                recent_decisions.append(decision)
        
        return sorted(recent_decisions, key=lambda x: x['timestamp'], reverse=True)
    
    def _load_decisions_history(self) -> Dict:
        """加载决策历史"""
        if os.path.exists(self.decisions_file):
            try:
                with open(self.decisions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_decisions_history(self):
        """保存决策历史"""
        try:
            with open(self.decisions_file, 'w', encoding='utf-8') as f:
                json.dump(self.decisions_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存决策历史失败: {e}")
    
    def save_decision(self, decision_record: Dict):
        """保存决策记录"""
        symbol = decision_record['symbol']
        
        if symbol not in self.decisions_history:
            self.decisions_history[symbol] = []
        
        self.decisions_history[symbol].append(decision_record)
        self._save_decisions_history()


# 便捷函数
def analyze_buy_opportunity(symbol: str, current_analysis: Dict) -> Dict:
    """分析买入机会的便捷函数"""
    dss = DecisionSupportSystem()
    return dss.analyze_buy_timing(symbol, current_analysis)

def analyze_sell_opportunity(symbol: str, position_info: Dict, current_analysis: Dict) -> Dict:
    """分析卖出机会的便捷函数"""
    dss = DecisionSupportSystem()
    return dss.analyze_sell_timing(symbol, position_info, current_analysis)


if __name__ == "__main__":
    # 测试代码
    dss = DecisionSupportSystem()
    
    # 模拟分析数据
    mock_analysis = {
        'current_price': 150.0,
        'technical_analysis': {
            'indicators': {
                'rsi': 45.0,
                'bb_upper': 155.0,
                'bb_lower': 145.0,
                'current_price': 150.0
            }
        }
    }
    
    # 测试买入时机分析
    buy_decision = dss.analyze_buy_timing("AAPL", mock_analysis)
    print("买入决策分析:")
    print(f"决策: {buy_decision['decision']['action']}")
    print(f"信心度: {buy_decision['decision']['confidence']}%")
    
    # 测试用户备注功能
    dss.add_user_note("AAPL", "我觉得这个价位不错，但想等等看") 