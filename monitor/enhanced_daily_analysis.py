#!/usr/bin/env python3
"""
增强版每日持股分析系统
整合宏观分析结果，提供详细通俗的分析和操作建议
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../analysis'))

from analysis.portfolio_macro_integration import PortfolioMacroIntegration
import yfinance as yf

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDailyAnalysis:
    """增强版每日持股分析器"""
    
    def __init__(self, portfolio_config_path: str = "../portfolio_config.json"):
        """初始化分析器"""
        self.portfolio_config_path = portfolio_config_path
        self.portfolio_config = self._load_portfolio_config()
        
        # 初始化宏观分析器
        self.macro_integration = PortfolioMacroIntegration(portfolio_config_path)
        
        # 股票中文名称映射
        self.stock_names = {
            'AMD': 'AMD(超威半导体)',
            'NVDA': 'NVIDIA(英伟达)',
            'GOOGL': 'Google(谷歌)',
            'TSLA': 'Tesla(特斯拉)',
            'PFE': 'Pfizer(辉瑞制药)',
            'EOG': 'EOG Resources(EOG能源)',
            'TMDX': 'TransMedics(移植医疗)',
            '9999.HK': '小米集团-W'
        }
        
        # 行业中文名称
        self.sector_names = {
            'Technology': '科技股',
            'Healthcare': '医疗股',
            'Energy': '能源股',
            'Automotive': '汽车股',
            'Financial': '金融股'
        }
    
    def _load_portfolio_config(self) -> Dict:
        """加载投资组合配置"""
        try:
            with open(self.portfolio_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return {}
    
    def get_stock_data(self, symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """计算技术指标"""
        try:
            close = data['Close']
            
            # 计算移动平均线
            ma5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else close.iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
            
            # 计算RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean() if len(delta) >= 14 else 0
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean() if len(delta) >= 14 else 0
            rs = gain / loss if loss != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs)) if rs != float('inf') else 100
            
            # 当前价格
            current_price = close.iloc[-1]
            
            # 价格变化
            if len(close) >= 2:
                price_change = (current_price - close.iloc[-2]) / close.iloc[-2] * 100
            else:
                price_change = 0
            
            return {
                'current_price': current_price,
                'ma5': ma5,
                'ma20': ma20,
                'rsi': rsi.iloc[-1] if hasattr(rsi, 'iloc') else rsi,
                'price_change': price_change,
                'volume_ratio': data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else 1.0
            }
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return {}
    
    def analyze_stock_sentiment(self, symbol: str, technical_data: Dict) -> Dict:
        """分析股票情绪"""
        sentiment = {'level': '中性', 'score': 0.5, 'signals': []}
        
        try:
            price = technical_data.get('current_price', 0)
            ma5 = technical_data.get('ma5', 0)
            ma20 = technical_data.get('ma20', 0)
            rsi = technical_data.get('rsi', 50)
            price_change = technical_data.get('price_change', 0)
            volume_ratio = technical_data.get('volume_ratio', 1)
            
            score = 0.5  # 基准分数
            
            # 均线分析
            if price > ma5 > ma20:
                sentiment['signals'].append("✅ 股价站上5日和20日均线，短期趋势向好")
                score += 0.15
            elif price < ma5 < ma20:
                sentiment['signals'].append("❌ 股价跌破5日和20日均线，短期趋势偏弱")
                score -= 0.15
            elif price > ma5:
                sentiment['signals'].append("🟡 股价站上5日均线，短期有支撑")
                score += 0.05
            
            # RSI分析
            if rsi > 70:
                sentiment['signals'].append(f"⚠️ RSI达到{rsi:.1f}，可能存在超买，注意回调风险")
                score -= 0.1
            elif rsi < 30:
                sentiment['signals'].append(f"💡 RSI降至{rsi:.1f}，可能存在超卖，关注反弹机会")
                score += 0.1
            elif 45 <= rsi <= 55:
                sentiment['signals'].append(f"🟢 RSI为{rsi:.1f}，处于健康区间")
                score += 0.05
            
            # 价格变化分析
            if price_change > 3:
                sentiment['signals'].append(f"🚀 今日大涨{price_change:.1f}%，市场情绪积极")
                score += 0.1
            elif price_change < -3:
                sentiment['signals'].append(f"📉 今日大跌{price_change:.1f}%，市场情绪悲观")
                score -= 0.1
            
            # 成交量分析
            if volume_ratio > 1.5:
                sentiment['signals'].append("📊 成交量放大，市场关注度较高")
                score += 0.05
            elif volume_ratio < 0.5:
                sentiment['signals'].append("📊 成交量萎缩，市场关注度不高")
                score -= 0.05
            
            # 确定情绪等级
            sentiment['score'] = max(0, min(1, score))
            if sentiment['score'] >= 0.7:
                sentiment['level'] = '积极'
            elif sentiment['score'] >= 0.6:
                sentiment['level'] = '偏积极'
            elif sentiment['score'] >= 0.4:
                sentiment['level'] = '中性'
            elif sentiment['score'] >= 0.3:
                sentiment['level'] = '偏悲观'
            else:
                sentiment['level'] = '悲观'
                
        except Exception as e:
            logger.error(f"分析股票情绪失败: {e}")
        
        return sentiment
    
    def generate_operation_suggestions(self, symbol: str, position_info: Dict, 
                                     technical_data: Dict, sentiment: Dict, 
                                     macro_impact: Dict) -> List[str]:
        """生成操作建议"""
        suggestions = []
        
        try:
            # 基础信息
            current_price = technical_data.get('current_price', 0)
            cost_basis = position_info.get('cost_basis', current_price)
            weight = position_info.get('weight', 0)
            
            # 收益率
            return_pct = (current_price - cost_basis) / cost_basis * 100 if cost_basis > 0 else 0
            
            # 宏观影响
            macro_score = macro_impact.get('impact_score', 0.5)
            impact_level = macro_impact.get('impact_level', 'neutral')
            
            # 技术指标
            rsi = technical_data.get('rsi', 50)
            price_change = technical_data.get('price_change', 0)
            
            # 基于收益率的建议
            if return_pct > 15:
                suggestions.append(f"🎯 **盈利丰厚**: {symbol}已盈利{return_pct:.1f}%，建议考虑部分止盈，锁定收益")
                suggestions.append(f"💡 **操作建议**: 可以先卖出30-50%的仓位，剩余部分设置追踪止损")
            elif return_pct > 8:
                suggestions.append(f"✅ **适度盈利**: {symbol}盈利{return_pct:.1f}%，可以继续持有并设置保护性止损")
                suggestions.append(f"💡 **操作建议**: 设置8-10%的追踪止损，保护已有收益")
            elif return_pct < -8:
                suggestions.append(f"⚠️ **亏损较大**: {symbol}亏损{abs(return_pct):.1f}%，需要重点关注")
                suggestions.append(f"💡 **操作建议**: 如果亏损超过15%，建议考虑止损出局")
            
            # 基于宏观环境的建议
            if impact_level == 'positive':
                suggestions.append(f"🌟 **宏观有利**: 当前宏观环境对{symbol}有利(得分{macro_score:.2f})，可以考虑逢低加仓")
            elif impact_level == 'negative':
                suggestions.append(f"🚨 **宏观不利**: 当前宏观环境对{symbol}不利(得分{macro_score:.2f})，建议降低仓位")
                suggestions.append(f"💡 **操作建议**: 建议将仓位控制在5%以下，或考虑暂时出局观望")
            elif impact_level == 'very_negative':
                suggestions.append(f"❌ **宏观恶劣**: 宏观环境对{symbol}非常不利，强烈建议减仓或清仓")
            
            # 基于技术指标的建议
            if rsi > 75:
                suggestions.append(f"📈 **技术超买**: RSI达到{rsi:.1f}，股价可能面临回调压力")
                suggestions.append(f"💡 **操作建议**: 暂时不宜追高，等待回调机会再考虑加仓")
            elif rsi < 25:
                suggestions.append(f"📉 **技术超卖**: RSI跌至{rsi:.1f}，可能存在反弹机会")
                suggestions.append(f"💡 **操作建议**: 可以考虑分批逢低买入，但要设置好止损")
            
            # 基于仓位权重的建议
            if weight > 25:
                suggestions.append(f"⚖️ **仓位过重**: {symbol}占比{weight:.1f}%过高，存在集中度风险")
                suggestions.append(f"💡 **操作建议**: 建议将单一股票仓位控制在20%以下，进行适当分散")
            elif weight > 20:
                suggestions.append(f"⚖️ **仓位较重**: {symbol}占比{weight:.1f}%偏高，注意风险控制")
            
            # 基于当日表现的建议
            if price_change > 5:
                suggestions.append(f"🚀 **今日大涨**: {symbol}今日涨幅{price_change:.1f}%，注意获利回吐风险")
                suggestions.append(f"💡 **操作建议**: 可以考虑减仓一部分，落袋为安")
            elif price_change < -5:
                suggestions.append(f"📉 **今日大跌**: {symbol}今日跌幅{price_change:.1f}%，需要分析跌因")
                if sentiment['level'] in ['积极', '偏积极']:
                    suggestions.append(f"💡 **操作建议**: 如果是技术性调整，可以考虑逢低加仓")
                else:
                    suggestions.append(f"💡 **操作建议**: 建议暂时观望，等待企稳信号")
            
            # 如果没有具体建议，给出通用建议
            if not suggestions:
                suggestions.append(f"📊 **持续关注**: {symbol}当前表现平稳，建议继续持有并密切关注")
                suggestions.append(f"💡 **操作建议**: 保持现有仓位，设置合理的止盈止损点位")
                
        except Exception as e:
            logger.error(f"生成操作建议失败: {e}")
            suggestions.append("⚠️ 暂时无法生成操作建议，请手动分析")
        
        return suggestions
    
    def generate_comprehensive_daily_report(self) -> str:
        """生成综合每日分析报告"""
        try:
            # 获取宏观分析
            logger.info("正在获取宏观分析数据...")
            macro_report = self.macro_integration.generate_macro_report()
            
            if 'error' in macro_report:
                logger.error(f"宏观分析失败: {macro_report['error']}")
                return "❌ 宏观分析获取失败，请检查网络连接"
            
            # 开始生成报告
            report_lines = []
            
            # 报告标题
            current_time = datetime.now().strftime('%Y年%m月%d日 %H:%M')
            report_lines.extend([
                "=" * 80,
                f"📊 每日持股分析报告 - {current_time}",
                "=" * 80,
                ""
            ])
            
            # 宏观环境概述
            macro_analysis = macro_report.get('detailed_analysis', {}).get('macro_analysis', {})
            macro_score = macro_analysis.get('macro_score', 0)
            macro_recommendation = macro_analysis.get('recommendation', '')
            
            report_lines.extend([
                "🌍 **宏观环境总览**",
                "-" * 40,
                f"📈 宏观得分: {macro_score:.2f}/1.00 ({int(macro_score*100)}分)",
                f"💡 宏观建议: {macro_recommendation}",
                ""
            ])
            
            # 解读宏观环境
            if macro_score >= 0.7:
                macro_desc = "宏观环境良好，有利于股市上涨，可以适当增加仓位"
                macro_emoji = "🟢"
            elif macro_score >= 0.5:
                macro_desc = "宏观环境中性，股市可能震荡整理，维持现有仓位"
                macro_emoji = "🟡"
            else:
                macro_desc = "宏观环境不利，股市面临下行压力，建议降低仓位"
                macro_emoji = "🔴"
            
            report_lines.extend([
                f"{macro_emoji} **环境解读**: {macro_desc}",
                ""
            ])
            
            # 关键宏观指标
            components = macro_analysis.get('components', {})
            if components:
                report_lines.extend([
                    "📊 **关键宏观指标**",
                    "-" * 30
                ])
                
                # 利率环境
                if 'interest_rate' in components:
                    rate_data = components['interest_rate']
                    curve_shape = rate_data.get('curve_shape', 'unknown')
                    curve_score = rate_data.get('curve_score', 0)
                    
                    if curve_shape == 'normal':
                        rate_desc = "收益率曲线正常，利率环境稳定"
                    elif curve_shape == 'inverted':
                        rate_desc = "收益率曲线倒挂，经济可能面临衰退风险"
                    else:
                        rate_desc = "收益率曲线平坦，经济增长动力不足"
                    
                    report_lines.append(f"📉 利率环境: {rate_desc} (得分: {curve_score:.1f})")
                
                # 市场情绪
                if 'market_sentiment' in components:
                    sentiment_data = components['market_sentiment']
                    vix_level = sentiment_data.get('vix_level', 0)
                    vix_sentiment = sentiment_data.get('vix_sentiment', 'unknown')
                    
                    if vix_sentiment == 'low':
                        sentiment_desc = f"VIX指数{vix_level:.1f}，市场情绪乐观，投资者风险偏好较高"
                    elif vix_sentiment == 'normal':
                        sentiment_desc = f"VIX指数{vix_level:.1f}，市场情绪正常，投资者相对理性"
                    else:
                        sentiment_desc = f"VIX指数{vix_level:.1f}，市场情绪恐慌，投资者风险规避"
                    
                    report_lines.append(f"😰 市场情绪: {sentiment_desc}")
                
                # 美元强度
                if 'dollar_strength' in components:
                    dollar_data = components['dollar_strength']
                    dollar_trend = dollar_data.get('dollar_trend', 'unknown')
                    dxy_level = dollar_data.get('dxy_level', 0)
                    
                    if dollar_trend == 'strong':
                        dollar_desc = f"美元指数{dxy_level:.1f}，美元走强，对新兴市场和大宗商品不利"
                    elif dollar_trend == 'weak':
                        dollar_desc = f"美元指数{dxy_level:.1f}，美元走弱，有利于风险资产和新兴市场"
                    else:
                        dollar_desc = f"美元指数{dxy_level:.1f}，美元震荡，对市场影响中性"
                    
                    report_lines.append(f"💵 美元强度: {dollar_desc}")
                
                report_lines.append("")
            
            # 行业影响分析
            sector_impact = macro_report.get('detailed_analysis', {}).get('sector_impact', {})
            if sector_impact:
                report_lines.extend([
                    "🏭 **行业影响分析**",
                    "-" * 30
                ])
                
                for sector, score in sector_impact.items():
                    sector_cn = self.sector_names.get(sector, sector)
                    
                    if score >= 0.6:
                        impact_desc = "宏观环境有利"
                        impact_emoji = "🟢"
                        suggestion = "可以考虑增加配置"
                    elif score >= 0.4:
                        impact_desc = "宏观环境中性"
                        impact_emoji = "🟡"
                        suggestion = "维持现有配置"
                    else:
                        impact_desc = "宏观环境不利"
                        impact_emoji = "🔴"
                        suggestion = "建议减少配置"
                    
                    report_lines.append(f"{impact_emoji} {sector_cn}: {score:.2f}分 - {impact_desc}，{suggestion}")
                
                report_lines.append("")
            
            # 个股详细分析
            portfolio_impact = macro_report.get('detailed_analysis', {}).get('portfolio_impact', {})
            individual_impacts = portfolio_impact.get('individual_impacts', {})
            
            if individual_impacts:
                report_lines.extend([
                    "📈 **个股详细分析**",
                    "=" * 50,
                    ""
                ])
                
                for symbol, impact_info in individual_impacts.items():
                    stock_name = self.stock_names.get(symbol, symbol)
                    sector = impact_info.get('sector', 'Unknown')
                    sector_cn = self.sector_names.get(sector, sector)
                    
                    report_lines.extend([
                        f"💼 **{stock_name} ({symbol})** - {sector_cn}",
                        "-" * 60
                    ])
                    
                    # 获取技术数据
                    stock_data = self.get_stock_data(symbol)
                    if stock_data is not None and not stock_data.empty:
                        technical_data = self.calculate_technical_indicators(stock_data)
                        sentiment = self.analyze_stock_sentiment(symbol, technical_data)
                        
                        # 基本信息
                        current_price = technical_data.get('current_price', 0)
                        price_change = technical_data.get('price_change', 0)
                        
                        # 持仓信息
                        position_info = self.portfolio_config.get('positions', {}).get(symbol, {})
                        cost_basis = position_info.get('cost_basis', current_price)
                        weight = position_info.get('weight', 0)
                        investment_amount = position_info.get('investment_amount', 0)
                        
                        # 收益情况
                        if cost_basis > 0:
                            return_pct = (current_price - cost_basis) / cost_basis * 100
                            profit_loss = (current_price - cost_basis) * (investment_amount / cost_basis) if cost_basis > 0 else 0
                        else:
                            return_pct = 0
                            profit_loss = 0
                        
                        # 价格变化emoji
                        if price_change > 2:
                            change_emoji = "🚀"
                        elif price_change > 0:
                            change_emoji = "📈"
                        elif price_change < -2:
                            change_emoji = "📉"
                        else:
                            change_emoji = "➡️"
                        
                        # 收益emoji
                        if return_pct > 10:
                            return_emoji = "🎉"
                        elif return_pct > 0:
                            return_emoji = "✅"
                        elif return_pct > -5:
                            return_emoji = "🟡"
                        else:
                            return_emoji = "❌"
                        
                        report_lines.extend([
                            f"💰 **价格信息**:",
                            f"   当前价格: ${current_price:.2f} {change_emoji}",
                            f"   今日涨跌: {price_change:+.2f}%",
                            f"   成本价格: ${cost_basis:.2f}",
                            f"   持仓比例: {weight:.1f}%",
                            f"   投资金额: ${investment_amount:,.0f}",
                            "",
                            f"📊 **收益情况**: {return_emoji}",
                            f"   收益率: {return_pct:+.2f}%",
                            f"   盈亏金额: ${profit_loss:+,.0f}",
                            ""
                        ])
                        
                        # 宏观影响
                        macro_score = impact_info.get('impact_score', 0.5)
                        impact_level = impact_info.get('impact_level', 'neutral')
                        impact_desc = impact_info.get('impact_description', '')
                        
                        if impact_level == 'positive':
                            macro_emoji = "🟢"
                        elif impact_level == 'neutral':
                            macro_emoji = "🟡"
                        else:
                            macro_emoji = "🔴"
                        
                        report_lines.extend([
                            f"🌍 **宏观影响**: {macro_emoji}",
                            f"   影响得分: {macro_score:.2f}/1.00",
                            f"   影响描述: {impact_desc}",
                            ""
                        ])
                        
                        # 技术分析
                        rsi = technical_data.get('rsi', 50)
                        ma5 = technical_data.get('ma5', 0)
                        ma20 = technical_data.get('ma20', 0)
                        volume_ratio = technical_data.get('volume_ratio', 1)
                        
                        report_lines.extend([
                            f"📊 **技术指标**:",
                            f"   RSI指标: {rsi:.1f} ({'超买' if rsi > 70 else '超卖' if rsi < 30 else '正常'})",
                            f"   5日均线: ${ma5:.2f} ({'站上' if current_price > ma5 else '跌破'})",
                            f"   20日均线: ${ma20:.2f} ({'站上' if current_price > ma20 else '跌破'})",
                            f"   成交量比: {volume_ratio:.1f}倍 ({'放量' if volume_ratio > 1.2 else '缩量' if volume_ratio < 0.8 else '正常'})",
                            ""
                        ])
                        
                        # 市场情绪
                        sentiment_level = sentiment.get('level', '中性')
                        sentiment_signals = sentiment.get('signals', [])
                        
                        sentiment_emoji_map = {
                            '积极': '🟢', '偏积极': '🟢', 
                            '中性': '🟡', 
                            '偏悲观': '🔴', '悲观': '🔴'
                        }
                        sentiment_emoji = sentiment_emoji_map.get(sentiment_level, '🟡')
                        
                        report_lines.extend([
                            f"💭 **市场情绪**: {sentiment_emoji} {sentiment_level}",
                        ])
                        
                        if sentiment_signals:
                            report_lines.append("   技术信号:")
                            for signal in sentiment_signals[:3]:  # 只显示前3个信号
                                report_lines.append(f"   • {signal}")
                        
                        report_lines.append("")
                        
                        # 操作建议
                        suggestions = self.generate_operation_suggestions(
                            symbol, position_info, technical_data, sentiment, impact_info
                        )
                        
                        if suggestions:
                            report_lines.extend([
                                f"💡 **操作建议**:",
                            ])
                            
                            for suggestion in suggestions:
                                report_lines.append(f"   {suggestion}")
                        
                        report_lines.extend(["", "-" * 60, ""])
                    
                    else:
                        report_lines.extend([
                            "❌ 暂时无法获取股票数据",
                            "",
                            "-" * 60,
                            ""
                        ])
            
            # 整体投资组合建议
            action_plan = macro_report.get('action_plan', {})
            immediate_actions = action_plan.get('priority_1', [])
            medium_actions = action_plan.get('priority_2', [])
            risk_management = action_plan.get('risk_management', [])
            
            report_lines.extend([
                "🎯 **整体投资组合建议**",
                "=" * 40,
                ""
            ])
            
            if immediate_actions:
                report_lines.extend([
                    "🚨 **立即行动建议**:",
                ])
                for i, action in enumerate(immediate_actions, 1):
                    report_lines.append(f"   {i}. {action}")
                report_lines.append("")
            
            if medium_actions:
                report_lines.extend([
                    "📋 **中期调整建议**:",
                ])
                for i, action in enumerate(medium_actions, 1):
                    report_lines.append(f"   {i}. {action}")
                report_lines.append("")
            
            if risk_management:
                report_lines.extend([
                    "🛡️ **风险管理建议**:",
                ])
                for i, suggestion in enumerate(risk_management, 1):
                    report_lines.append(f"   {i}. {suggestion}")
                report_lines.append("")
            
            # 总结
            portfolio_risk = macro_report.get('executive_summary', {}).get('portfolio_risk_level', 'medium')
            risk_emoji_map = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
            risk_emoji = risk_emoji_map.get(portfolio_risk, '🟡')
            
            risk_desc_map = {
                'low': '当前投资组合风险较低，可以保持现有配置',
                'medium': '当前投资组合风险中等，建议适度调整',
                'high': '当前投资组合风险较高，需要立即采取行动'
            }
            risk_desc = risk_desc_map.get(portfolio_risk, '风险等级未知')
            
            report_lines.extend([
                "📝 **每日总结**",
                "=" * 30,
                f"🎯 今日宏观得分: {macro_score:.2f}/1.00 ({int(macro_score*100)}分)",
                f"{risk_emoji} 投资组合风险: {portfolio_risk.upper()}级",
                f"💡 风险描述: {risk_desc}",
                "",
                "📞 **温馨提示**:",
                "• 投资有风险，入市需谨慎",
                "• 建议定期回顾和调整投资组合",
                "• 如有疑问，请咨询专业投资顾问",
                "",
                "=" * 80,
                f"报告生成时间: {current_time}",
                "=" * 80
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"生成每日分析报告失败: {e}")
            return f"❌ 报告生成失败: {str(e)}"
    
    def save_daily_report(self, report: str, filename: Optional[str] = None) -> str:
        """保存每日报告"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"daily_analysis_report_{timestamp}.txt"
            
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            filepath = os.path.join(reports_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"每日报告已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            return ""


def main():
    """主函数"""
    print("🚀 启动增强版每日持股分析系统...")
    
    try:
        # 初始化分析器
        analyzer = EnhancedDailyAnalysis()
        
        # 生成综合分析报告
        print("📊 正在生成每日分析报告...")
        report = analyzer.generate_comprehensive_daily_report()
        
        # 显示报告
        print(report)
        
        # 保存报告
        saved_file = analyzer.save_daily_report(report)
        if saved_file:
            print(f"\n💾 报告已保存到: {saved_file}")
        
        print("\n✅ 每日分析完成！")
        
    except Exception as e:
        print(f"❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 