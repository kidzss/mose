"""
Streamlit分析桥接器
用于将统一股票分析服务整合到Streamlit界面中
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.unified_stock_analyzer import UnifiedStockAnalyzer


class StreamlitAnalysisBridge:
    """Streamlit分析桥接器"""
    
    def __init__(self):
        """初始化桥接器"""
        self.analyzer = UnifiedStockAnalyzer()
    
    def display_comprehensive_analysis(self, symbol: str, force_refresh: bool = False):
        """显示综合分析结果"""
        
        # 显示加载状态
        with st.spinner(f'正在分析 {symbol}...'):
            result = self.analyzer.get_comprehensive_analysis(symbol, force_refresh)
        
        if 'error' in result:
            st.error(f"❌ 分析失败: {result['error']}")
            return
        
        # 显示基本信息
        self._display_basic_info(result)
        
        # 显示市场环境
        self._display_market_environment(result.get('market_environment', {}))
        
        # 显示技术分析
        self._display_technical_analysis(result.get('technical_analysis', {}))
        
        # 显示基本面分析
        self._display_fundamental_analysis(result.get('fundamental_analysis', {}))
        
        # 显示流动性分析
        self._display_liquidity_analysis(result.get('liquidity_analysis', {}))
        
        # 显示智能增强分析
        self._display_enhanced_analysis(result.get('enhanced_analysis', {}))
        
        # 显示股票类型分析
        self._display_stock_type_analysis(result.get('stock_type_analysis', {}))
        
        # 显示右侧交易分析
        self._display_right_side_analysis(result.get('right_side_analysis', {}))
        
        # 显示综合评级
        self._display_comprehensive_rating(result.get('comprehensive_rating', {}))
        
        # 显示交易建议
        self._display_trading_suggestions(result.get('trading_suggestions', {}))
    
    def _display_basic_info(self, result):
        """显示基本信息"""
        symbol = result['symbol']
        basic_info = result.get('basic_info', {})
        
        # 公司基本信息
        company_name = basic_info.get('longName', symbol)
        sector = basic_info.get('sector', 'N/A')
        industry = basic_info.get('industry', 'N/A')
        
        st.markdown(f"### {symbol} | {company_name}")
        st.caption(f"{sector} - {industry}")
        
        # 关键指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = basic_info.get('currentPrice', result.get('current_price', 0))
            price_change = result.get('change_pct', basic_info.get('change_pct', 0))
            delta_color = "inverse" if price_change < 0 else "normal"
            st.metric("当前价格", f"${current_price:.2f}", f"{price_change:+.2f}%", delta_color=delta_color)
        
        with col2:
            market_cap = basic_info.get('marketCap', 0)
            if market_cap > 1e9:
                cap_display = f"${market_cap/1e9:.1f}B"
            elif market_cap > 1e6:
                cap_display = f"${market_cap/1e6:.1f}M"
            else:
                cap_display = f"${market_cap:.0f}"
            st.metric("市值", cap_display)
        
        with col3:
            pe_ratio = basic_info.get('trailingPE', 0)
            st.metric("P/E比率", f"{pe_ratio:.1f}" if pe_ratio and pe_ratio > 0 else "N/A")
        
        with col4:
            beta = basic_info.get('beta', 1)
            st.metric("Beta", f"{beta:.2f}" if beta else "1.00")
    
    def _display_market_environment(self, market_env):
        """显示市场环境"""
        if not market_env or 'error' in market_env:
            return
        
        st.markdown("#### 市场环境")
        
        environment = market_env.get('environment', '数据不足')
        sentiment = market_env.get('market_sentiment', '中性')
        vix_level = market_env.get('vix_level', 20)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("市场趋势", environment)
        
        with col2:
            st.metric("市场情绪", sentiment)
        
        with col3:
            vix_status = "低" if vix_level < 20 else "中" if vix_level < 30 else "高"
            st.metric("恐慌指数", f"{vix_level:.1f} ({vix_status})")
    
    def _display_technical_analysis(self, tech_data):
        """显示技术分析"""
        if not tech_data or 'error' in tech_data:
            return
        
        st.markdown("#### 技术分析")
        
        indicators = tech_data.get('indicators', {})
        
        # 技术指标概览
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = indicators.get('rsi', 50)
            rsi_status = "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常"
            st.metric("RSI", f"{rsi:.1f}", rsi_status)
        
        with col2:
            signal_strength = tech_data.get('signal_strength', 0)
            st.metric("信号强度", f"{signal_strength}/7")
        
        with col3:
            strategy = tech_data.get('strategy', 'N/A')
            strategy_map = {
                'trend_following': '趋势跟踪',
                'value_buying': '价值买入',
                'profit_taking': '获利了结',
                'wait_and_see': '观望等待'
            }
            st.metric("推荐策略", strategy_map.get(strategy, strategy))
        
        with col4:
            tech_score = tech_data.get('score', 0)
            tech_rating = self._score_to_grade(tech_score) if tech_score else 'N/A'
            st.metric("技术评分", f"{tech_score}/100", tech_rating)
        
        # 详细指标
        with st.expander("📊 详细技术指标"):
            tech_df = pd.DataFrame([
                {"指标": "RSI", "数值": f"{indicators.get('rsi', 0):.1f}", "状态": self._get_rsi_status(indicators.get('rsi', 50))},
                {"指标": "MA20", "数值": f"${indicators.get('ma_20', 0):.2f}", "状态": "支撑" if indicators.get('ma_20', 0) < indicators.get('current_price', 0) else "阻力"},
                {"指标": "MA50", "数值": f"${indicators.get('ma_50', 0):.2f}", "状态": "支撑" if indicators.get('ma_50', 0) < indicators.get('current_price', 0) else "阻力"},
                {"指标": "MACD", "数值": f"{indicators.get('macd_line', 0):.3f}", "状态": "多头" if indicators.get('macd_line', 0) > indicators.get('signal_line', 0) else "空头"},
                {"指标": "成交量比", "数值": f"{indicators.get('volume_ratio', 1):.1f}x", "状态": "放量" if indicators.get('volume_ratio', 1) > 1.5 else "正常"},
                {"指标": "52周位置", "数值": f"{indicators.get('position_52w', 50):.1f}%", "状态": self._get_position_status(indicators.get('position_52w', 50))}
            ])
            
            st.dataframe(tech_df, use_container_width=True)
    
    def _display_fundamental_analysis(self, fund_data):
        """显示基本面分析"""
        if not fund_data or 'error' in fund_data:
            return
        
        st.subheader("💼 财务基本面分析")
        
        overall_score = fund_data.get('overall_score', 0.5)
        rating = fund_data.get('rating', '中性')
        
        # 综合评级
        col1, col2 = st.columns([1, 2])
        
        with col1:
            score_color = self._get_score_color(overall_score)
            st.markdown(f"""
            <div style="background-color: {score_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>综合评级: {rating}</h3>
                <p>评分: {overall_score:.2f}/1.0</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 各维度评分
            valuation_score = fund_data.get('valuation', {}).get('score', 0.5)
            profitability_score = fund_data.get('profitability', {}).get('score', 0.5)
            growth_score = fund_data.get('growth', {}).get('score', 0.5)
            health_score = fund_data.get('financial_health', {}).get('score', 0.5)
            
            scores_df = pd.DataFrame({
                '维度': ['估值指标', '盈利能力', '成长性', '财务健康'],
                '评分': [valuation_score, profitability_score, growth_score, health_score],
                '等级': [self._score_to_grade(s) for s in [valuation_score, profitability_score, growth_score, health_score]]
            })
            
            st.dataframe(scores_df, use_container_width=True)
        
        # 详细财务指标
        with st.expander("💰 详细财务分析"):
            self._display_detailed_fundamentals(fund_data)
    
    def _display_liquidity_analysis(self, liquidity_data):
        """显示流动性分析"""
        if not liquidity_data or 'error' in liquidity_data:
            return
        
        st.subheader("💧 流动性风险评估")
        
        score = liquidity_data.get('score', 50)
        risk_level = liquidity_data.get('risk_level', 'MEDIUM')
        market_cap_level = liquidity_data.get('market_cap_level', 'MEDIUM')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("流动性评分", f"{score:.0f}/100")
        
        with col2:
            risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk_level, "🟡")
            st.metric("风险等级", f"{risk_color} {risk_level}")
        
        with col3:
            spread = liquidity_data.get('bid_ask_spread', 0) * 100
            st.metric("买卖价差", f"{spread:.3f}%")
        
        with col4:
            st.metric("市值等级", market_cap_level)
        
        # 分析要点
        analysis_points = liquidity_data.get('analysis_points', [])
        if analysis_points:
            st.markdown("**📋 流动性分析要点:**")
            for point in analysis_points:
                st.write(f"• {point}")
    
    def _display_enhanced_analysis(self, enhanced_data):
        """显示智能增强分析"""
        if not enhanced_data or 'error' in enhanced_data:
            return
        
        st.subheader("🔧 智能增强分析")
        
        overall_score = enhanced_data.get('overall_score', 0.5)
        growth_analysis = enhanced_data.get('growth_analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 成长性分析:**")
            growth_score = growth_analysis.get('score', 0.5)
            growth_rating = growth_analysis.get('rating', '一般')
            
            st.markdown(f"""
            - 评分: {growth_score:.2f}
            - 等级: {growth_rating}
            - 🚀 成长性{'优秀' if growth_score > 0.8 else '良好' if growth_score > 0.6 else '一般'}
            """)
        
        with col2:
            st.markdown("**🏆 行业比较:**")
            industry_comp = enhanced_data.get('industry_comparison', {})
            st.markdown(f"""
            - {industry_comp.get('relative_performance', 'N/A')}
            - {industry_comp.get('comparison', 'N/A')}
            - 🏆 {industry_comp.get('industry_trend', 'N/A')}
            """)
        
        # 智能建议
        smart_suggestions = enhanced_data.get('smart_suggestions', [])
        if smart_suggestions:
            st.markdown("**💡 智能投资建议:**")
            for suggestion in smart_suggestions:
                st.write(f"🚀 {suggestion}")
    
    def _display_stock_type_analysis(self, stock_type_data):
        """显示股票类型分析"""
        if not stock_type_data or 'error' in stock_type_data:
            return
        
        st.subheader("🎯 智能股票类型分析")
        
        stock_type = stock_type_data.get('stock_type', '成长股')
        risk_level = stock_type_data.get('risk_level', 'MEDIUM')
        comprehensive_score = stock_type_data.get('comprehensive_score', 7.9)
        
        # 综合评分展示
        col1, col2 = st.columns([1, 2])
        
        with col1:
            rating_text = "买入" if comprehensive_score >= 8 else "持有" if comprehensive_score >= 6 else "观望"
            rating_color = "#28a745" if comprehensive_score >= 8 else "#ffc107" if comprehensive_score >= 6 else "#dc3545"
            
            st.markdown(f"""
            <div style="background-color: {rating_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>{rating_text} | 综合评分: {comprehensive_score:.1f}/10</h3>
                <p>📊 股票类型: {stock_type} | 风险等级: {risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 各维度评分
            tech_score = stock_type_data.get('technical_score', 8.0)
            fund_score = stock_type_data.get('fundamental_score', 8.5)
            sentiment_score = stock_type_data.get('sentiment_score', 4.0)
            
            scores_df = pd.DataFrame({
                '评分维度': ['技术面评分', '基本面评分', '市场情绪'],
                '评分': [tech_score, fund_score, sentiment_score],
                '权重': ['35%', '55%', '10%']
            })
            
            st.dataframe(scores_df, use_container_width=True)
        
        # 股票特征
        characteristics = stock_type_data.get('stock_characteristics', [])
        if characteristics:
            st.markdown("**📈 股票特征:**")
            char_text = " | ".join(characteristics)
            st.markdown(f"• {char_text}")
    
    def _display_right_side_analysis(self, right_side_data):
        """显示右侧交易分析"""
        if not right_side_data or 'error' in right_side_data:
            return
        
        st.subheader("🎯 右侧交易分析 (防抄底系统)")
        
        decision = right_side_data.get('decision', '观察等待')
        decision_color = right_side_data.get('decision_color', '🟡')
        decision_reason = right_side_data.get('decision_reason', 'N/A')
        score = right_side_data.get('score', 0)
        
        # 决策展示
        col1, col2 = st.columns([1, 2])
        
        with col1:
            bg_color = "#28a745" if "积极买入" in decision else "#ffc107" if "观察等待" in decision else "#dc3545"
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>{decision_color} {decision}</h3>
                <p>评分: {score}/100</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**📋 决策依据:** {decision_reason}")
            
            # 信号确认状态
            trend_confirmed = right_side_data.get('trend_confirmed', False)
            breakout_confirmed = right_side_data.get('breakout_confirmed', False)
            volume_support = right_side_data.get('volume_support', False)
            
            st.markdown(f"""
            **🔍 右侧等待信号:**
            - {'✅' if trend_confirmed else '❌'} 趋势确认: {'已确认' if trend_confirmed else '待确认'}
            - {'✅' if breakout_confirmed else '❌'} 突破确认: {'已确认' if breakout_confirmed else '待确认'}  
            - {'✅' if volume_support else '❌'} 成交量配合: {'充足' if volume_support else '不足'}
            """)
        
        # 核心原则
        core_principles = right_side_data.get('core_principles', [])
        if core_principles:
            st.markdown("**💡 右侧交易核心原则:**")
            for principle in core_principles:
                st.write(f"✅ {principle}")
    
    def _display_comprehensive_rating(self, rating_data):
        """显示综合评级"""
        if not rating_data or 'error' in rating_data:
            return
        
        st.markdown("#### 综合评级")
        
        overall_score = rating_data.get('overall_score', 0.5)
        rating = rating_data.get('rating', '持有')
        confidence = rating_data.get('confidence', 70)
        
        # 综合评级展示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("投资建议", rating, f"评分: {overall_score:.2f}")
        
        with col2:
            st.metric("信心度", f"{confidence}%")
        
        with col3:
            grade = self._score_to_grade(overall_score)
            st.metric("等级", grade)
        
        # 各维度评分
        component_scores = rating_data.get('component_scores', {})
        if component_scores:
            st.markdown("**评分构成:**")
            
            components_col1, components_col2 = st.columns(2)
            
            component_names = {
                'technical': '技术面',
                'fundamental': '基本面', 
                'enhanced': '增强分析',
                'stock_type': '股票类型'
            }
            
            items = list(component_scores.items())
            mid = len(items) // 2
            
            with components_col1:
                for component, score in items[:mid]:
                    name = component_names.get(component, component)
                    st.write(f"• {name}: {score:.2f}")
            
            with components_col2:
                for component, score in items[mid:]:
                    name = component_names.get(component, component)
                    st.write(f"• {name}: {score:.2f}")
    
    def _display_trading_suggestions(self, suggestions_data):
        """显示交易建议"""
        if not suggestions_data or 'error' in suggestions_data:
            return
        
        st.subheader("📋 交易操作建议")
        
        primary_action = suggestions_data.get('primary_action', '持有')
        confidence = suggestions_data.get('confidence', 70)
        
        # 主要建议
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 10px; text-align: center;">
                <h4>🎯 主要操作</h4>
                <p><strong>{primary_action}</strong></p>
                <p>信心度: {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            position_size = suggestions_data.get('position_size', 'N/A')
            holding_period = suggestions_data.get('holding_period', 'N/A')
            
            st.markdown(f"""
            <div style="background-color: #f3e5f5; padding: 1rem; border-radius: 10px; text-align: center;">
                <h4>💼 仓位配置</h4>
                <p><strong>{position_size}</strong></p>
                <p>持有周期: {holding_period}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            entry_timing = suggestions_data.get('entry_timing', 'N/A')
            
            st.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 10px; text-align: center;">
                <h4>⏰ 入场时机</h4>
                <p><strong>{entry_timing}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # 止损止盈
        col1, col2 = st.columns(2)
        
        with col1:
            stop_loss = suggestions_data.get('stop_loss', 0)
            st.metric("🛡️ 建议止损位", f"${stop_loss:.2f}" if stop_loss > 0 else "N/A")
        
        with col2:
            take_profit = suggestions_data.get('take_profit', 0)
            st.metric("💰 建议止盈位", f"${take_profit:.2f}" if take_profit > 0 else "N/A")
        
        # 分析要点
        analysis_points = suggestions_data.get('analysis_points', [])
        if analysis_points:
            st.markdown("**📋 分析要点:**")
            for point in analysis_points:
                st.write(f"📈 {point}")
        
        # 风险警告
        risk_warnings = suggestions_data.get('risk_warnings', [])
        if risk_warnings:
            st.markdown("**⚠️ 风险提示:**")
            for warning in risk_warnings:
                st.write(f"⚠️ {warning}")
        
        # 新增：买入价格指导
        buy_guidance = suggestions_data.get('buy_price_guidance', {})
        if buy_guidance:
            st.subheader("💰 买入价格指导")
            
            current_assessment = buy_guidance.get('current_assessment', '')
            entry_strategy = buy_guidance.get('entry_strategy', '')
            
            if current_assessment:
                st.info(f"📊 **当前评估**: {current_assessment}")
            if entry_strategy:
                st.info(f"🎯 **入场策略**: {entry_strategy}")
            
            optimal_zones = buy_guidance.get('optimal_buy_zones', [])
            if optimal_zones:
                st.markdown("**🎯 最佳买入区间:**")
                
                buy_zones_data = []
                for i, zone in enumerate(optimal_zones, 1):
                    buy_zones_data.append({
                        '序号': f"第{i}档",
                        '价格区间': zone.get('price_range', 'N/A'),
                        '建议仓位': zone.get('allocation', 'N/A'),
                        '买入理由': zone.get('reason', 'N/A')
                    })
                
                if buy_zones_data:
                    buy_df = pd.DataFrame(buy_zones_data)
                    st.dataframe(buy_df, use_container_width=True)
        
        # 新增：波段交易策略
        swing_strategy = suggestions_data.get('swing_trading_strategy', {})
        if swing_strategy:
            # 检查是否为TSLA特殊策略
            strategy_note = swing_strategy.get('strategy_note', '')
            if strategy_note:
                st.subheader("📈 TSLA倒金字塔补仓策略")
                st.success(strategy_note)
            else:
                st.subheader("📈 波段交易策略 (长期持有+波段操作)")
            
            # 策略概述
            core_pct = swing_strategy.get('core_position_pct', 60)
            swing_pct = swing_strategy.get('swing_position_pct', 40)
            total_investment = swing_strategy.get('risk_control', {}).get('total_investment', '')
            
            if total_investment:
                st.info(f"**策略概述**: {core_pct}%核心仓位长期持有，{swing_pct}%波段仓位高抛低吸\n\n💰 **总投资**: {total_investment}")
            else:
                st.info(f"**策略概述**: {core_pct}%核心仓位长期持有，{swing_pct}%波段仓位高抛低吸")
            
            # 加仓策略
            add_positions = swing_strategy.get('add_positions', [])
            if add_positions:
                st.markdown("**💰 加仓策略:**")
                add_data = []
                for i, pos in enumerate(add_positions, 1):
                    add_data.append({
                        '加仓档位': f"第{i}加仓位",
                        '目标价格': f"${pos.get('price', 0):.2f}",
                        '建议股数': pos.get('percentage', 'N/A'),
                        '加仓理由': pos.get('reason', 'N/A')
                    })
                
                if add_data:
                    add_df = pd.DataFrame(add_data)
                    st.dataframe(add_df, use_container_width=True)
            
            # 卖出策略  
            sell_positions = swing_strategy.get('sell_positions', [])
            if sell_positions:
                st.markdown("**📈 波段卖出策略:**")
                sell_data = []
                for i, pos in enumerate(sell_positions, 1):
                    sell_data.append({
                        '卖出档位': f"第{i}卖出位",
                        '目标价格': f"${pos.get('price', 0):.2f}",
                        '减持比例': pos.get('percentage', 'N/A'),
                        '卖出理由': pos.get('reason', 'N/A')
                    })
                
                if sell_data:
                    sell_df = pd.DataFrame(sell_data)
                    st.dataframe(sell_df, use_container_width=True)
            
            # 长期目标
            target_price = swing_strategy.get('target_price', 0)
            upside_potential = swing_strategy.get('upside_potential', 0)
            time_horizon = swing_strategy.get('time_horizon', 'N/A')
            
            if target_price > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 长期目标", f"${target_price:.2f}")
                with col2:
                    st.metric("📈 上涨空间", f"{upside_potential:.1f}%")
                with col3:
                    st.metric("⏰ 时间周期", time_horizon)
            
            # 风险控制
            risk_control = swing_strategy.get('risk_control', {})
            if risk_control:
                st.markdown("**⚠️ 风险控制:**")
                col1, col2 = st.columns(2)
                with col1:
                    stop_loss_pct = risk_control.get('stop_loss_pct', 0)
                    st.write(f"🛡️ 止损幅度: -{stop_loss_pct:.1f}%")
                with col2:
                    position_limit = risk_control.get('position_limit_pct', 0)
                    st.write(f"⚖️ 仓位限制: 不超过组合{position_limit:.0f}%")
                
                # TSLA特殊风险控制信息
                key_support = risk_control.get('key_support', '')
                if key_support:
                    st.warning(f"🔑 **关键支撑**: {key_support}")
                    st.info("💡 **策略原则**: 关键支撑$336不破继续看涨，破位则需重新评估整体策略")
    
    # === 辅助方法 ===
    
    def _get_rsi_status(self, rsi):
        """获取RSI状态"""
        if rsi < 30:
            return "超卖"
        elif rsi > 70:
            return "超买"
        elif rsi < 50:
            return "偏弱"
        else:
            return "偏强"
    
    def _get_position_status(self, position):
        """获取52周位置状态"""
        if position < 25:
            return "低位"
        elif position > 75:
            return "高位"
        else:
            return "中位"
    
    def _get_score_color(self, score):
        """根据评分获取颜色"""
        if score >= 0.8:
            return "#28a745"  # 绿色
        elif score >= 0.6:
            return "#ffc107"  # 黄色
        elif score >= 0.4:
            return "#fd7e14"  # 橙色
        else:
            return "#dc3545"  # 红色
    
    def _score_to_grade(self, score):
        """评分转等级"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "较差"
    
    def _display_detailed_fundamentals(self, fund_data):
        """显示详细基本面数据"""
        
        # 估值指标
        valuation = fund_data.get('valuation', {})
        if valuation:
            st.markdown("**📊 估值指标:**")
            val_details = valuation.get('details', {})
            for key, data in val_details.items():
                if isinstance(data, dict):
                    st.write(f"• {data.get('comment', key)}")
        
        # 盈利能力
        profitability = fund_data.get('profitability', {})
        if profitability:
            st.markdown("**💰 盈利能力:**")
            prof_details = profitability.get('details', {})
            for key, data in prof_details.items():
                if isinstance(data, dict):
                    st.write(f"• {data.get('comment', key)}")
        
        # 成长性
        growth = fund_data.get('growth', {})
        if growth:
            st.markdown("**🚀 成长性:**")
            growth_details = growth.get('details', {})
            for key, data in growth_details.items():
                if isinstance(data, dict):
                    st.write(f"• {data.get('comment', key)}")
        
        # 财务健康度
        health = fund_data.get('financial_health', {})
        if health:
            st.markdown("**🏦 财务健康:**")
            health_details = health.get('details', {})
            for key, data in health_details.items():
                if isinstance(data, dict):
                    st.write(f"• {data.get('comment', key)}")
        
        # 分析师观点
        analyst = fund_data.get('analyst_sentiment', {})
        if analyst:
            st.markdown("**📊 分析师观点:**")
            analyst_details = analyst.get('details', {})
            for key, data in analyst_details.items():
                if isinstance(data, dict):
                    st.write(f"• {data.get('comment', key)}")


# 便捷函数
def display_stock_analysis(symbol: str, force_refresh: bool = False):
    """
    显示股票分析的便捷函数
    
    Args:
        symbol: 股票代码
        force_refresh: 强制刷新缓存
    """
    bridge = StreamlitAnalysisBridge()
    bridge.display_comprehensive_analysis(symbol, force_refresh)


if __name__ == "__main__":
    # 测试代码
    st.title("🎯 股票综合分析测试")
    
    # 测试股票
    test_symbol = st.selectbox("选择测试股票", ["ADBE", "AAPL", "MSFT", "GOOGL"])
    
    if st.button("开始分析"):
        display_stock_analysis(test_symbol) 