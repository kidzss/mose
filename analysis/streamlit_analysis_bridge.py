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
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.unified_stock_analyzer import UnifiedStockAnalyzer


class StreamlitAnalysisBridge:
    """Streamlit分析桥接器"""
    
    def __init__(self):
        """初始化桥接器"""
        self.analyzer = UnifiedStockAnalyzer()
    
    def display_comprehensive_analysis(self, symbol: str, force_refresh: bool = False):
        """
        显示完整的股票综合分析 - 全新8模块专业投资分析系统
        
        📈 技术分析 - 价格走势、技术指标、图表分析
        💰 基本面分析 - 财务指标、估值分析  
        💧 流动性分析 - 成交量、买卖价差
        🎯 买入时机分析 - 基于技术+基本面
        🧠 智能增强分析 - AI辅助判断
        🏷️ 股票类型分析 - 成长股/价值股分类
        ➡️ 右侧交易分析 - 趋势确认
        📊 仓位管理建议 - 具体操作建议
        """
        try:
            # 获取分析结果
            with st.spinner(f"正在获取 {symbol} 的完整分析数据..."):
                result = self.analyzer.get_comprehensive_analysis(symbol)
            
            if 'error' in result:
                st.error(f"分析失败: {result['error']}")
                return
            
            # === 1. 基本信息概览 ===
            self._display_basic_info(result)
            
            # === 2. 市场环境 ===  
            if 'market_environment' in result:
                self._display_market_environment(result['market_environment'])
            
            # 创建8个分析模块的标签页
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📈 技术分析", "💰 基本面分析", "💧 流动性分析", "🎯 买入时机", 
                "🧠 智能分析", "🏷️ 股票类型", "➡️ 右侧交易", "📊 仓位管理"
            ])
            
            # === 模块1: 📈 技术分析 (新整合的完整技术分析) ===
            with tab1:
                self._display_enhanced_technical_analysis(result.get('technical_analysis', {}), symbol)
            
            # === 模块2: 💰 基本面分析 ===
            with tab2:
                self._display_fundamental_analysis(result.get('fundamental_analysis', {}))
            
            # === 模块3: 💧 流动性分析 ===
            with tab3:
                self._display_liquidity_analysis(result.get('liquidity_analysis', {}))
            
            # === 模块4: 🎯 买入时机分析 ===
            with tab4:
                self._display_timing_analysis(result.get('timing_analysis', {}))
            
            # === 模块5: 🧠 智能增强分析 ===
            with tab5:
                self._display_enhanced_analysis(result.get('enhanced_analysis', {}))
            
            # === 模块6: 🏷️ 股票类型分析 ===
            with tab6:
                self._display_stock_type_analysis(result.get('stock_type_analysis', {}))
            
            # === 模块7: ➡️ 右侧交易分析 ===
            with tab7:
                self._display_right_side_analysis(result.get('right_side_analysis', {}))
            
            # === 模块8: 📊 仓位管理建议 ===
            with tab8:
                self._display_position_management_analysis(result, symbol)
            
        except Exception as e:
            st.error(f"分析过程中发生错误: {e}")
            st.info("请检查股票代码或稍后重试")
    
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
    
    def _display_enhanced_technical_analysis(self, tech_data, symbol):
        """
        显示增强版技术分析 - 整合了完整的图表、指标和分析
        这是原技术分析标签页的完整功能整合版本
        """
        st.markdown("### 📈 完整技术分析")
        st.markdown("**价格走势、技术指标、图表分析一站式专业技术分析**")
        
        if not tech_data or 'error' in tech_data:
            st.warning("技术分析数据获取失败，正在尝试获取基础数据...")
            
            # 备用数据获取 - 使用简化的技术分析
            try:
                import yfinance as yf
                stock = yf.Ticker(symbol)
                hist = stock.history(period="6mo")
                
                if not hist.empty:
                    # 计算基础技术指标
                    current_price = hist['Close'].iloc[-1]
                    ma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
                    ma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
                    
                    # 计算RSI - 防止除零错误
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    # 增强除零保护
                    loss_safe = loss.replace(0, 1e-10).fillna(1e-10)
                    gain_safe = gain.fillna(0)
                    rs = gain_safe / loss_safe
                    rsi_series = 100 - (100 / (1 + rs))
                    rsi = rsi_series.iloc[-1] if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50
                    
                    # 显示基础指标
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        # 修复除零错误：确保前一日收盘价不为0
                        if len(hist) >= 2 and hist['Close'].iloc[-2] > 0:
                            change_pct = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100)
                        else:
                            change_pct = 0
                        st.metric("当前价格", f"${current_price:.2f}", f"{change_pct:+.2f}%")
                    with col2:
                        rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
                        st.metric("RSI", f"{rsi:.1f}", rsi_color)
                    with col3:
                        st.metric("20日均线", f"${ma_20:.2f}")
                    with col4:
                        st.metric("50日均线", f"${ma_50:.2f}")
                    
                    # 简化图表
                    self._create_enhanced_stock_chart(symbol, hist)
                    
                else:
                    st.error(f"无法获取 {symbol} 的历史数据")
            except Exception as e:
                st.error(f"备用数据获取失败: {e}")
            return
        
        # === 技术指标概览 ===
        st.markdown("#### 📊 核心技术指标")
        
        indicators = tech_data.get('indicators', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = indicators.get('current_price', 0)
            price_change = indicators.get('price_change_pct', 0)
            st.metric("💰 当前价格", f"${current_price:.2f}", f"{price_change:+.2f}%")
        
        with col2:
            rsi = indicators.get('rsi', 50)
            rsi_status = "超卖🟢" if rsi < 30 else "超买🔴" if rsi > 70 else "正常🟡"
            st.metric("📈 RSI指标", f"{rsi:.1f}", rsi_status)
        
        with col3:
            signal_strength = tech_data.get('signal_strength', 0)
            st.metric("🎯 信号强度", f"{signal_strength}/7")
        
        with col4:
            tech_score = tech_data.get('score', 0)
            tech_rating = self._score_to_grade(tech_score) if tech_score else 'N/A'
            st.metric("⭐ 技术评分", f"{tech_score}/100", tech_rating)
        
        # === 技术分析策略建议 ===
        col1, col2 = st.columns([1, 2])
        
        with col1:
            strategy = tech_data.get('strategy', 'N/A')
            strategy_map = {
                'trend_following': '趋势跟踪',
                'value_buying': '价值买入', 
                'profit_taking': '获利了结',
                'wait_and_see': '观望等待'
            }
            
            strategy_chinese = strategy_map.get(strategy, strategy)
            strategy_color = {
                '趋势跟踪': '#28a745',
                '价值买入': '#17a2b8', 
                '获利了结': '#ffc107',
                '观望等待': '#6c757d'
            }.get(strategy_chinese, '#6c757d')
            
            st.markdown(f"""
            <div style="background-color: {strategy_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>📋 推荐策略</h3>
                <p>{strategy_chinese}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🎯 技术信号详情:**")
            signals = tech_data.get('signals', [])
            if signals:
                for signal in signals:
                    st.write(f"• {signal}")
            else:
                st.info("暂无明确技术信号")
        
        # === 详细技术指标表格 ===
        with st.expander("📊 详细技术指标", expanded=True):
            tech_df = pd.DataFrame([
                {"指标": "RSI", "数值": f"{indicators.get('rsi', 0):.1f}", "状态": self._get_rsi_status(indicators.get('rsi', 50))},
                {"指标": "MA20", "数值": f"${indicators.get('ma_20', 0):.2f}", "状态": "支撑" if indicators.get('ma_20', 0) < indicators.get('current_price', 0) else "阻力"},
                {"指标": "MA50", "数值": f"${indicators.get('ma_50', 0):.2f}", "状态": "支撑" if indicators.get('ma_50', 0) < indicators.get('current_price', 0) else "阻力"},
                {"指标": "MACD", "数值": f"{indicators.get('macd_line', 0):.3f}", "状态": "多头" if indicators.get('macd_line', 0) > indicators.get('signal_line', 0) else "空头"},
                {"指标": "成交量比", "数值": f"{indicators.get('volume_ratio', 1):.1f}x", "状态": "放量" if indicators.get('volume_ratio', 1) > 1.5 else "正常"},
                {"指标": "52周位置", "数值": f"{indicators.get('position_52w', 50):.1f}%", "状态": self._get_position_status(indicators.get('position_52w', 50))}
            ])
            
            st.dataframe(tech_df, use_container_width=True)
        
        # === 专业技术图表 ===
        st.markdown("#### 📈 专业技术图表")
        try:
            # 尝试创建增强图表
            self._create_enhanced_stock_chart(symbol)
        except Exception as e:
            st.warning(f"图表生成遇到问题: {e}")
            st.info("正在尝试简化图表...")
    
    def _create_enhanced_stock_chart(self, symbol, hist_data=None):
        """
        创建增强版股票技术图表
        整合K线、均线、RSI、成交量的完整技术分析图表
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import yfinance as yf
            import numpy as np
            
            # 获取历史数据
            if hist_data is None:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="6mo")
            else:
                hist = hist_data
            
            if hist.empty:
                st.error("无法获取历史数据")
                return
            
            # 创建子图
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=(f'{symbol} 价格走势与技术指标', 'RSI相对强弱指标', '成交量分析'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # === K线图 ===
            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='K线',
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=1, col=1
            )
            
            # === 移动平均线 ===
            if len(hist) >= 20:
                ma_20 = hist['Close'].rolling(20).mean()
                fig.add_trace(
                    go.Scatter(x=hist.index, y=ma_20, mode='lines', name='MA20', 
                              line=dict(color='orange', width=2)),
                    row=1, col=1
                )
            
            if len(hist) >= 50:
                ma_50 = hist['Close'].rolling(50).mean()
                fig.add_trace(
                    go.Scatter(x=hist.index, y=ma_50, mode='lines', name='MA50', 
                              line=dict(color='red', width=2)),
                    row=1, col=1
                )
            
            # === RSI指标 ===
            def calculate_rsi_series(prices, period=14):
                try:
                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
                    
                    # 防止除零错误，将0替换为极小值
                    loss_safe = loss.replace(0, 1e-10)
                    rs = gain / loss_safe
                    rsi = 100 - (100 / (1 + rs))
                    
                    # 处理无效值
                    rsi = rsi.fillna(50)  # 用中性值填充NaN
                    rsi = rsi.replace([np.inf, -np.inf], [100, 0])  # 处理无穷大
                    rsi = rsi.clip(0, 100)  # 确保在0-100范围内
                    
                    return rsi
                except Exception as e:
                    # 如果计算失败，返回中性RSI值
                    return pd.Series([50] * len(prices), index=prices.index)
            
            rsi_series = calculate_rsi_series(hist['Close'])
            fig.add_trace(
                go.Scatter(x=hist.index, y=rsi_series, mode='lines', name='RSI', 
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            
            # RSI参考线
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
            
            # === 成交量 ===
            if 'Volume' in hist.columns:
                # 根据涨跌着色成交量
                colors = ['red' if close < open else 'green' for close, open in zip(hist['Close'], hist['Open'])]
                fig.add_trace(
                    go.Bar(x=hist.index, y=hist['Volume'], name='成交量', 
                          marker_color=colors, opacity=0.7),
                    row=3, col=1
                )
            
            # === 图表美化 ===
            fig.update_layout(
                title=f"📈 {symbol} 专业技术分析图表",
                height=800,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # 设置Y轴标签
            fig.update_yaxes(title_text="价格 ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="成交量", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # === 图表说明 ===
            with st.expander("📖 图表说明"):
                st.markdown("""
                **📈 价格走势图 (上图):**
                - 🟢🔴 K线：绿涨红跌，显示开盘、收盘、最高、最低价
                - 🟠 MA20：20日移动平均线，短期趋势参考
                - 🔴 MA50：50日移动平均线，中期趋势参考
                
                **📊 RSI指标 (中图):**
                - 🟣 RSI线：相对强弱指标，衡量买卖力道
                - 🔴 70线：超买警戒线，价格可能回调
                - 🟢 30线：超卖警戒线，价格可能反弹
                
                **📊 成交量 (下图):**
                - 🟢🔴 成交量柱：交易活跃度，配合价格确认趋势
                """)
                
        except Exception as e:
            st.error(f"图表创建失败: {e}")
            st.info("请检查网络连接或稍后重试")

    def _display_timing_analysis(self, timing_data):
        """显示买入时机分析"""
        if not timing_data or 'error' in timing_data:
            st.warning("买入时机分析数据暂无")
            return
        
        st.markdown("### 🎯 专业买入时机分析")
        st.markdown("**基于技术面+基本面的综合买入时机判断**")
        
        # 时机评级
        timing_score = timing_data.get('score', 50)
        timing_rating = timing_data.get('rating', '观望')
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            rating_color = "#28a745" if timing_rating == '买入' else "#ffc107" if timing_rating == '观望' else "#dc3545"
            st.markdown(f"""
            <div style="background-color: {rating_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>🎯 {timing_rating}</h3>
                <p>时机评分: {timing_score}/100</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = timing_data.get('confidence', 70)
            st.metric("🔮 预测信心", f"{confidence}%")
        
        with col3:
            reasons = timing_data.get('reasons', [])
            if reasons:
                st.markdown("**📋 关键原因:**")
                for reason in reasons:
                    st.write(f"• {reason}")
        
        # 详细分析
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 技术面信号:**")
            tech_signals = timing_data.get('technical_signals', [])
            for signal in tech_signals:
                st.write(f"🔍 {signal}")
        
        with col2:
            st.markdown("**💼 基本面支撑:**")
            fundamental_signals = timing_data.get('fundamental_signals', [])
            for signal in fundamental_signals:
                st.write(f"💰 {signal}")
        
        # 风险提示
        risks = timing_data.get('risks', [])
        if risks:
            st.markdown("**⚠️ 风险提示:**")
            for risk in risks:
                st.warning(f"🚨 {risk}")

    def _display_position_management_analysis(self, analysis_result, symbol):
        """显示仓位管理建议分析"""
        st.markdown("### 📊 智能仓位管理建议")
        st.markdown("**基于实际持仓和当前市场状况的专业仓位管理策略**")
        
        # 读取实际持仓数据
        try:
            with open('portfolio_config.json', 'r', encoding='utf-8') as f:
                portfolio_data = json.load(f)
            
            positions = portfolio_data.get('positions', {})
            portfolio_info = portfolio_data.get('portfolio', {})
            total_value = portfolio_info.get('total_value', 27533.17)
            
            # 获取当前股票的持仓信息
            current_position = positions.get(symbol, {})
            
            if current_position:
                current_shares = current_position.get('shares', 0)
                current_value = current_position.get('current_value', 0)
                current_weight = current_position.get('weight', 0)
                cost_basis = current_position.get('cost_basis', 0)
                unrealized_pnl = current_position.get('unrealized_pnl', 0)
                
                st.markdown("#### 💼 当前持仓状况")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("持仓股数", f"{current_shares}股")
                with col2:
                    st.metric("持仓市值", f"${current_value:,.2f}")
                with col3:
                    st.metric("组合占比", f"{current_weight:.2f}%")
                with col4:
                    pnl_color = "normal" if unrealized_pnl >= 0 else "inverse"
                    st.metric("浮动盈亏", f"${unrealized_pnl:+.2f}", delta_color=pnl_color)
            else:
                st.info(f"📝 当前未持有 {symbol}")
                current_shares = 0
                current_value = 0
                current_weight = 0
        
        except Exception as e:
            st.warning(f"无法读取持仓数据: {e}")
            current_shares = 0
            current_value = 0
            current_weight = 0
        
        # 获取技术分析数据
        basic_info = analysis_result.get('basic_info', {})
        tech_data = analysis_result.get('technical_analysis', {})
        
        # 尝试从多个位置获取当前价格
        current_price = (
            basic_info.get('current_price', 0) or
            basic_info.get('currentPrice', 0) or
            tech_data.get('indicators', {}).get('current_price', 0) or
            tech_data.get('current_price', 0) or
            0
        )
        
        # 如果还是获取不到价格，尝试实时获取
        if current_price <= 0:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
                if current_price <= 0:
                    # 最后尝试从历史数据获取最新价格
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
            except Exception as e:
                print(f"实时获取 {symbol} 价格失败: {e}")
                current_price = 0
        
        rsi = tech_data.get('indicators', {}).get('rsi', 50)
        position_52w = tech_data.get('indicators', {}).get('position_52w', 50)
        
        # 显示关键技术指标
        st.markdown("#### 📊 关键技术指标")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("当前价格", f"${current_price:.2f}")
        with col2:
            rsi_status = "超卖🟢" if rsi < 30 else "超买🔴" if rsi > 70 else "正常🟡"
            st.metric("RSI", f"{rsi:.1f}", rsi_status)
        with col3:
            st.metric("52周位置", f"{position_52w:.1f}%")
        
        # 智能目标仓位建议
        st.markdown("#### 🎯 专业仓位建议")
        
        # 根据技术指标和市场环境计算建议仓位
        if rsi > 75:
            recommended_weight = max(current_weight - 2, 0)  # 超买，建议减仓
            action = "减仓"
            reason = f"RSI={rsi:.1f}严重超买，建议适度减仓"
            risk_level = "🔴 高风险"
        elif rsi > 70:
            recommended_weight = max(current_weight - 1, 0)  # 轻度超买
            action = "观望"
            reason = f"RSI={rsi:.1f}轻度超买，建议观望等待"
            risk_level = "🟡 中等风险"
        elif rsi < 25:
            recommended_weight = min(current_weight + 3, 15)  # 严重超卖，可考虑加仓
            action = "加仓"
            reason = f"RSI={rsi:.1f}严重超卖，存在反弹机会"
            risk_level = "🟢 低风险"
        elif rsi < 30:
            recommended_weight = min(current_weight + 1.5, 12)  # 轻度超卖
            action = "小幅加仓"
            reason = f"RSI={rsi:.1f}超卖，可考虑适度建仓"
            risk_level = "🟢 较低风险"
        elif position_52w < 20:  # 接近年内低点
            recommended_weight = min(current_weight + 2, 12)
            action = "逢低布局"
            reason = f"当前处于52周{position_52w:.1f}%位置，接近低点"
            risk_level = "🟢 较低风险"
        elif position_52w > 80:  # 接近年内高点
            recommended_weight = max(current_weight - 1.5, 0)
            action = "高位减仓"
            reason = f"当前处于52周{position_52w:.1f}%位置，接近高点"
            risk_level = "🟡 中等风险"
        else:
            # 正常区间，根据股票类型给出建议
            sector = current_position.get('sector', 'Unknown')
            if sector == 'Technology':
                recommended_weight = min(current_weight + 0.5, 10)  # 科技股建议控制在10%以下
            elif sector in ['Healthcare', 'Consumer Staples']:
                recommended_weight = min(current_weight + 1, 15)  # 防御性股票可以更高配置
            else:
                recommended_weight = min(current_weight + 0.5, 8)
            action = "适度配置"
            reason = f"{sector}板块，技术指标正常，可适度配置"
            risk_level = "🟡 正常风险"
        
        # 显示建议
        col1, col2 = st.columns([1, 2])
        
        with col1:
            action_color = {
                "减仓": "#dc3545",
                "观望": "#ffc107", 
                "加仓": "#28a745",
                "小幅加仓": "#28a745",
                "逢低布局": "#17a2b8",
                "高位减仓": "#fd7e14",
                "适度配置": "#6c757d"
            }.get(action, "#6c757d")
            
            st.markdown(f"""
            <div style="background-color: {action_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h4>📋 {action}</h4>
                <p>建议仓位: {recommended_weight:.1f}%</p>
                <p>{risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            **📋 分析依据:**
            {reason}
            
            **💡 执行建议:**
            • 当前仓位: {current_weight:.2f}%
            • 建议仓位: {recommended_weight:.1f}%
            • 仓位调整: {recommended_weight - current_weight:+.1f}%
            """)
            
            if abs(recommended_weight - current_weight) < 0.5:
                st.success("✅ 当前仓位配置合理，无需调整")
            elif recommended_weight > current_weight:
                # 修复除零错误：确保current_price不为0
                if current_price > 0:
                    target_shares = int((recommended_weight * total_value / 100) / current_price)
                    add_shares = target_shares - current_shares
                    st.info(f"📈 建议增持约 {add_shares} 股")
                else:
                    st.warning(f"⚠️ 无法获取 {symbol} 的有效价格数据，请稍后重试或检查股票代码")
            else:
                # 修复除零错误：确保current_price不为0
                if current_price > 0:
                    target_shares = int((recommended_weight * total_value / 100) / current_price)
                    reduce_shares = current_shares - target_shares
                    st.warning(f"📉 建议减持约 {reduce_shares} 股")
                else:
                    st.warning(f"⚠️ 无法获取 {symbol} 的有效价格数据，请稍后重试或检查股票代码")
        
        # 风险控制建议
        st.markdown("#### ⚠️ 专业风险控制")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stop_loss_pct = current_position.get('stop_loss_threshold', 0.08) * 100
            stop_loss_price = current_price * (1 - stop_loss_pct/100)
            st.metric("止损位", f"${stop_loss_price:.2f}", f"-{stop_loss_pct:.0f}%")
        
        with col2:
            take_profit_price = current_price * 1.15
            st.metric("目标获利位", f"${take_profit_price:.2f}", "+15%")
        
        with col3:
            max_single_weight = 15.0
            st.metric("最大单股占比", f"{max_single_weight:.0f}%", "风控限制")
        
        # 基于记忆中的TSLA策略给出特殊建议
        if symbol == "TSLA":
            st.markdown("#### 🚗 TSLA特殊策略建议")
            st.info("""
            **📈 倒金字塔加仓策略 (基于历史策略):**
            • 第一批30%($296-300，$825)试探性建仓验证支撑
            • 第二批40%($285-290，$1,100)确认趋势后重仓买入
            • 第三批30%($273-280，$825)极值区域收割风险最低时加码
            • 总资金约$2,750，约占总资产10%
            • 关键支撑$336不破继续看涨
            """)
        
        elif symbol == "AMD":
            st.markdown("#### 💾 AMD特殊策略建议") 
            st.success("""
            **📈 技术形态分析 (基于最新分析):**
            • 光头光脚大阳线，技术形态拐头向上
            • 预期后续继续上涨，目标价位150-160区间
            • 当前策略：继续持有等待，不急于减仓
            • 强势信号确认，技术面支撑明确
            """)
        
        # 市场环境考量
        st.markdown("#### 🌍 市场环境考量")
        market_env = analysis_result.get('market_environment', {})
        if market_env:
            environment = market_env.get('environment', '正常')
            sentiment = market_env.get('market_sentiment', '中性')
            
            if environment == '牛市' and sentiment == '乐观':
                st.success("🐂 当前牛市环境，市场情绪乐观，可适度提高仓位")
            elif environment == '熊市' or sentiment == '悲观':
                st.warning("🐻 当前市场环境谨慎，建议控制仓位，加强风控")
            else:
                st.info("📊 当前市场环境中性，建议按技术指标稳健操作")

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
        
        # 买入价格指导
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
        
        # 波段交易策略
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