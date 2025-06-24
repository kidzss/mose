#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流动性分析器
为个人投资者提供简单实用的流动性风险评估

核心功能：
1. 买卖价差分析 (Bid-Ask Spread)
2. 成交量一致性评估
3. 市场深度分析 (基于bidSize/askSize)
4. 流动性风险预警
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LiquidityMetrics:
    """流动性指标数据类"""
    symbol: str
    bid_ask_spread_pct: float       # 买卖价差百分比
    avg_daily_volume: float         # 平均日成交量
    volume_consistency: float       # 成交量一致性 (0-1, 1最稳定)
    market_depth_score: float       # 市场深度评分 (0-1)
    market_cap_tier: str           # 市值等级 (large/mid/small/micro)
    liquidity_risk_level: str      # 流动性风险等级 (low/medium/high/critical)
    liquidity_score: float         # 综合流动性评分 (0-100)
    warning_message: str           # 风险警告信息
    exit_difficulty: str           # 退出难度 (easy/normal/difficult/very_difficult)
    
    # 为了兼容性添加的额外字段
    @property
    def risk_level(self) -> str:
        return self.liquidity_risk_level
    
    @property
    def risk_warning(self) -> str:
        return self.warning_message
    
    @property
    def investment_suggestion(self) -> str:
        """基于流动性风险提供投资建议"""
        if self.liquidity_risk_level == 'low':
            return "流动性充足，适合正常投资"
        elif self.liquidity_risk_level == 'medium':
            return "流动性一般，建议适度投资，注意仓位控制"
        elif self.liquidity_risk_level == 'high':
            return "流动性偏低，建议小仓位试探，避免大额投资"
        else:  # critical
            return "流动性严重不足，不建议投资"
    
    @property
    def spread_rating(self) -> str:
        """价差等级评级"""
        if self.bid_ask_spread_pct <= 0.005:
            return "优秀"
        elif self.bid_ask_spread_pct <= 0.01:
            return "良好"
        elif self.bid_ask_spread_pct <= 0.02:
            return "一般"
        elif self.bid_ask_spread_pct <= 0.05:
            return "较差"
        else:
            return "很差"
    
    @property
    def liquidity_reasons(self) -> List[str]:
        """流动性分析要点"""
        reasons = []
        
        if self.bid_ask_spread_pct <= 0.01:
            reasons.append(f"买卖价差为{self.bid_ask_spread_pct:.3f}%，交易成本较低")
        else:
            reasons.append(f"买卖价差为{self.bid_ask_spread_pct:.3f}%，交易成本偏高")
        
        if self.volume_consistency >= 0.7:
            reasons.append("成交量稳定，流动性可预期")
        elif self.volume_consistency >= 0.5:
            reasons.append("成交量波动适中")
        else:
            reasons.append("成交量波动较大，流动性不稳定")
        
        if self.market_cap_tier in ['large', 'mid']:
            reasons.append(f"属于{self.market_cap_tier.upper()}市值股票，机构参与度高")
        else:
            reasons.append(f"属于{self.market_cap_tier.upper()}市值股票，机构参与度较低")
        
        if self.market_depth_score >= 0.6:
            reasons.append("市场深度充足，大单冲击较小")
        else:
            reasons.append("市场深度较浅，大单可能造成价格冲击")
        
        return reasons

class LiquidityAnalyzer:
    """流动性分析器"""
    
    def __init__(self):
        """初始化流动性分析器"""
        self.cache = {}
        self.cache_timeout = timedelta(minutes=30)  # 30分钟缓存
        
        # 市值分级标准 (美元)
        self.market_cap_tiers = {
            'large': 10_000_000_000,      # 大盘股 >= 100亿
            'mid': 2_000_000_000,         # 中盘股 >= 20亿
            'small': 300_000_000,         # 小盘股 >= 3亿
            'micro': 50_000_000           # 微盘股 >= 5000万
        }
        
        # 风险阈值
        self.risk_thresholds = {
            'spread_pct': {
                'low': 0.005,      # 0.5%
                'medium': 0.015,   # 1.5%
                'high': 0.03,      # 3%
                'critical': 0.05   # 5%
            },
            'volume_consistency': {
                'good': 0.7,       # 70%以上一致性为好
                'fair': 0.5,       # 50%以上为可接受
                'poor': 0.3        # 30%以下为差
            },
            'min_daily_volume': {
                'large': 1_000_000,    # 大盘股最低100万股
                'mid': 500_000,        # 中盘股最低50万股
                'small': 100_000,      # 小盘股最低10万股
                'micro': 50_000        # 微盘股最低5万股
            }
        }
    
    def get_stock_info(self, symbol: str) -> Dict:
        """获取股票基础信息"""
        cache_key = f"{symbol}_info"
        current_time = datetime.now()
        
        # 检查缓存
        if (cache_key in self.cache and 
            current_time - self.cache[cache_key]['timestamp'] < self.cache_timeout):
            return self.cache[cache_key]['data']
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 获取历史数据用于成交量分析
            hist_data = ticker.history(period='30d')
            
            result = {
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'bidSize': info.get('bidSize', 0),
                'askSize': info.get('askSize', 0),
                'regularMarketPrice': info.get('regularMarketPrice', info.get('currentPrice', 0)),
                'averageVolume': info.get('averageVolume', 0),
                'averageVolume10days': info.get('averageVolume10days', 0),
                'marketCap': info.get('marketCap', 0),
                'volume_history': hist_data['Volume'].tolist() if not hist_data.empty else [],
                'symbol': symbol
            }
            
            # 缓存结果
            self.cache[cache_key] = {
                'data': result,
                'timestamp': current_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"获取股票信息失败 {symbol}: {e}")
            return {}
    
    def calculate_bid_ask_spread(self, stock_info: Dict) -> Tuple[float, str]:
        """计算买卖价差"""
        try:
            bid = stock_info.get('bid', 0)
            ask = stock_info.get('ask', 0)
            price = stock_info.get('regularMarketPrice', 0)
            
            if bid <= 0 or ask <= 0 or price <= 0:
                return 0.0, "无法获取报价数据"
            
            spread_pct = (ask - bid) / price
            
            # 评估价差等级
            thresholds = self.risk_thresholds['spread_pct']
            if spread_pct <= thresholds['low']:
                level = "优秀"
            elif spread_pct <= thresholds['medium']:
                level = "良好"
            elif spread_pct <= thresholds['high']:
                level = "一般"
            elif spread_pct <= thresholds['critical']:
                level = "较差"
            else:
                level = "很差"
            
            return spread_pct, level
            
        except Exception as e:
            logger.error(f"计算买卖价差失败: {e}")
            return 0.0, "计算失败"
    
    def analyze_stock_liquidity(self, symbol: str) -> LiquidityMetrics:
        """分析单只股票的流动性"""
        try:
            # 获取股票信息
            stock_info = self.get_stock_info(symbol)
            if not stock_info:
                return self._create_error_metrics(symbol, "无法获取股票数据")
            
            # 计算买卖价差
            spread_pct, _ = self.calculate_bid_ask_spread(stock_info)
            
            # 简化版流动性评分
            market_cap = stock_info.get('marketCap', 0)
            avg_volume = stock_info.get('averageVolume', 0)
            
            # 市值等级
            if market_cap >= 10_000_000_000:
                market_cap_tier = 'large'
            elif market_cap >= 2_000_000_000:
                market_cap_tier = 'mid'
            elif market_cap >= 300_000_000:
                market_cap_tier = 'small'
            else:
                market_cap_tier = 'micro'
            
            # 简单的流动性评分
            if spread_pct < 0.01 and market_cap_tier in ['large', 'mid']:
                liquidity_score = 85
                risk_level = 'low'
                exit_difficulty = 'easy'
                warning = "流动性良好"
            elif spread_pct < 0.02:
                liquidity_score = 70
                risk_level = 'medium'
                exit_difficulty = 'normal'
                warning = "流动性一般"
            else:
                liquidity_score = 45
                risk_level = 'high'
                exit_difficulty = 'difficult'
                warning = "⚠️ 流动性较差，注意交易成本"
            
            return LiquidityMetrics(
                symbol=symbol,
                bid_ask_spread_pct=spread_pct,
                avg_daily_volume=avg_volume,
                volume_consistency=0.7,  # 简化
                market_depth_score=0.6,  # 简化
                market_cap_tier=market_cap_tier,
                liquidity_risk_level=risk_level,
                liquidity_score=liquidity_score,
                warning_message=warning,
                exit_difficulty=exit_difficulty
            )
            
        except Exception as e:
            logger.error(f"分析股票流动性失败 {symbol}: {e}")
            return self._create_error_metrics(symbol, f"分析失败: {str(e)}")
    
    def _create_error_metrics(self, symbol: str, error_msg: str) -> LiquidityMetrics:
        """创建错误情况下的指标对象"""
        return LiquidityMetrics(
            symbol=symbol,
            bid_ask_spread_pct=0.0,
            avg_daily_volume=0,
            volume_consistency=0.0,
            market_depth_score=0.0,
            market_cap_tier='unknown',
            liquidity_risk_level='critical',
            liquidity_score=0,
            warning_message=f"❌ {error_msg}",
            exit_difficulty='very_difficult'
        )
    
    def calculate_volume_consistency(self, volume_history: List[float]) -> Tuple[float, str]:
        """计算成交量一致性"""
        try:
            if len(volume_history) < 5:
                return 0.5, "数据不足"
            
            volumes = np.array(volume_history)
            volumes = volumes[volumes > 0]  # 排除零成交量
            
            if len(volumes) < 5:
                return 0.3, "成交量数据不足"
            
            # 计算变异系数 (标准差/均值)
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            
            if mean_volume <= 0:
                return 0.2, "成交量异常"
            
            cv = std_volume / mean_volume  # 变异系数
            consistency = max(0, 1 - cv)  # 一致性评分 (变异系数越小越好)
            
            # 评估一致性等级
            thresholds = self.risk_thresholds['volume_consistency']
            if consistency >= thresholds['good']:
                level = "稳定"
            elif consistency >= thresholds['fair']:
                level = "一般"
            elif consistency >= thresholds['poor']:
                level = "波动较大"
            else:
                level = "极不稳定"
            
            return consistency, level
            
        except Exception as e:
            logger.error(f"计算成交量一致性失败: {e}")
            return 0.3, "计算失败"
    
    def calculate_market_depth(self, stock_info: Dict) -> Tuple[float, str]:
        """计算市场深度评分"""
        try:
            bid_size = stock_info.get('bidSize', 0)
            ask_size = stock_info.get('askSize', 0)
            
            # 基础深度评分
            total_size = bid_size + ask_size
            
            if total_size <= 0:
                return 0.1, "无深度数据"
            
            # 根据买卖盘数量评估深度
            if total_size >= 1000:
                depth_score = 1.0
                level = "深度充足"
            elif total_size >= 500:
                depth_score = 0.8
                level = "深度良好"
            elif total_size >= 100:
                depth_score = 0.6
                level = "深度一般"
            elif total_size >= 50:
                depth_score = 0.4
                level = "深度较浅"
            else:
                depth_score = 0.2
                level = "深度很浅"
            
            return depth_score, level
            
        except Exception as e:
            logger.error(f"计算市场深度失败: {e}")
            return 0.3, "计算失败"
    
    def classify_market_cap(self, market_cap: float) -> str:
        """分类市值等级"""
        if market_cap >= self.market_cap_tiers['large']:
            return 'large'
        elif market_cap >= self.market_cap_tiers['mid']:
            return 'mid'
        elif market_cap >= self.market_cap_tiers['small']:
            return 'small'
        elif market_cap >= self.market_cap_tiers['micro']:
            return 'micro'
        else:
            return 'nano'
    
    def assess_exit_difficulty(self, liquidity_score: float, market_cap_tier: str) -> str:
        """评估退出难度"""
        if liquidity_score >= 80 and market_cap_tier in ['large', 'mid']:
            return 'easy'
        elif liquidity_score >= 60:
            return 'normal'
        elif liquidity_score >= 40:
            return 'difficult'
        else:
            return 'very_difficult'
    
    def generate_warning_message(self, metrics: LiquidityMetrics) -> str:
        """生成风险警告信息"""
        warnings = []
        
        if metrics.bid_ask_spread_pct > 0.03:
            warnings.append("买卖价差过大，交易成本高")
        
        if metrics.volume_consistency < 0.5:
            warnings.append("成交量波动大，流动性不稳定")
        
        if metrics.market_depth_score < 0.4:
            warnings.append("市场深度浅，大单冲击成本高")
        
        if metrics.market_cap_tier in ['micro', 'nano']:
            warnings.append("小市值股票，流动性风险较高")
        
        if metrics.avg_daily_volume < self.risk_thresholds['min_daily_volume'].get(metrics.market_cap_tier, 50000):
            warnings.append("日均成交量偏低")
        
        if not warnings:
            return "流动性状况良好，无特殊风险"
        
        return "⚠️ " + "；".join(warnings)
    
    def batch_analyze_liquidity(self, symbols: List[str]) -> List[LiquidityMetrics]:
        """批量分析多只股票的流动性"""
        results = []
        for symbol in symbols:
            metrics = self.analyze_stock_liquidity(symbol)
            results.append(metrics)
            
        # 按流动性评分排序
        results.sort(key=lambda x: x.liquidity_score, reverse=True)
        return results
    
    def get_liquidity_summary(self, metrics_list: List[LiquidityMetrics]) -> Dict:
        """获取流动性分析汇总"""
        if not metrics_list:
            return {}
        
        # 统计各风险等级数量
        risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for metrics in metrics_list:
            risk_counts[metrics.liquidity_risk_level] += 1
        
        # 计算平均评分
        avg_score = sum(m.liquidity_score for m in metrics_list) / len(metrics_list)
        
        # 找出风险最高的股票
        high_risk_stocks = [m.symbol for m in metrics_list if m.liquidity_risk_level in ['high', 'critical']]
        
        return {
            'total_analyzed': len(metrics_list),
            'average_score': avg_score,
            'risk_distribution': risk_counts,
            'high_risk_stocks': high_risk_stocks,
            'recommendations': self._generate_portfolio_recommendations(metrics_list)
        }
    
    def _generate_portfolio_recommendations(self, metrics_list: List[LiquidityMetrics]) -> List[str]:
        """生成投资组合流动性建议"""
        recommendations = []
        
        # 计算高风险股票比例
        high_risk_count = sum(1 for m in metrics_list if m.liquidity_risk_level in ['high', 'critical'])
        high_risk_ratio = high_risk_count / len(metrics_list) if metrics_list else 0
        
        if high_risk_ratio > 0.3:
            recommendations.append("⚠️ 投资组合中高流动性风险股票比例过高(>30%)，建议降低仓位")
        
        if high_risk_ratio > 0.5:
            recommendations.append("🚨 超过50%的股票存在流动性风险，建议重新配置投资组合")
        
        # 检查小盘股比例
        small_cap_count = sum(1 for m in metrics_list if m.market_cap_tier in ['micro', 'nano'])
        small_cap_ratio = small_cap_count / len(metrics_list) if metrics_list else 0
        
        if small_cap_ratio > 0.2:
            recommendations.append("💡 小市值股票占比较高，注意控制单个仓位不超过5%")
        
        # 总体建议
        avg_score = sum(m.liquidity_score for m in metrics_list) / len(metrics_list) if metrics_list else 0
        if avg_score < 60:
            recommendations.append("📉 整体流动性评分偏低，建议增加大盘股配置")
        elif avg_score >= 80:
            recommendations.append("✅ 投资组合流动性状况良好")
        
        return recommendations if recommendations else ["✅ 投资组合流动性配置合理"] 