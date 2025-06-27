#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业实时交易监控Dashboard
使用Streamlit创建交互式Web界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import sys
import os
from datetime import datetime, timedelta
import time
import yfinance as yf
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

class StreamlitTradingDashboard:
    """Streamlit交易监控仪表板"""
    
    def __init__(self):
        self.data_interface = DataInterface()
        self.yahoo_source = self.data_interface.get_data_source('yahoo')
        
        # 设置页面配置
        st.set_page_config(
            page_title="专业交易监控系统",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 初始化session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'selected_stocks' not in st.session_state:
            st.session_state.selected_stocks = ['AMD', 'NVDA', 'TSLA', 'AAPL', 'MSFT']
    
    def run_dashboard(self):
        """运行主仪表板"""
        # 页面标题
        st.title("🚀 Mose实时交易监控系统")
        st.markdown("---")
        
        # 侧边栏配置
        self._setup_sidebar()
        
        # 主要内容区域
        if st.session_state.get('show_amd_analysis', True):
            self._show_amd_position_analysis()
        
        # 实时监控区域
        self._show_realtime_monitoring()
        
        # 技术分析区域
        self._show_technical_analysis()
        
        # 市场概览
        self._show_market_overview()
        
        # 自动刷新
        if st.session_state.auto_refresh:
            time.sleep(5)
            st.rerun()
    
    def _setup_sidebar(self):
        """设置侧边栏"""
        st.sidebar.header("📊 监控配置")
        
        # 股票选择
        available_stocks = ['AMD', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOG', 'META', 'AMZN']
        st.session_state.selected_stocks = st.sidebar.multiselect(
            "选择监控股票",
            available_stocks,
            default=st.session_state.selected_stocks
        )
        
        # 自动刷新设置
        st.session_state.auto_refresh = st.sidebar.toggle(
            "自动刷新 (5秒)",
            value=st.session_state.auto_refresh
        )
        
        # AMD持仓分析开关
        st.session_state.show_amd_analysis = st.sidebar.toggle(
            "显示AMD持仓分析",
            value=True
        )
        
        # 刷新按钮
        if st.sidebar.button("🔄 立即刷新"):
            st.rerun()
        
        # 显示最后更新时间
        st.sidebar.markdown(f"**最后更新:** {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    def _show_amd_position_analysis(self):
        """显示AMD持仓分析"""
        st.header("💼 AMD持仓分析")
        
        # 持仓信息
        col1, col2, col3, col4 = st.columns(4)
        
        # 获取AMD实时数据
        amd_data = self._get_stock_data('AMD')
        if amd_data is not None:
            current_price = amd_data['current_price']
            
            # 计算持仓数据
            cost_basis = 125.746  # 补仓后的加权平均成本价
            current_gain = 8.96  # 当前盈利%
            recent_add_price = 136.9  # 最近补仓价格
            recent_add_shares = 5  # 最近补仓股数
            
            with col1:
                st.metric("当前价格", f"${current_price:.2f}", f"{amd_data['change_pct']:+.2f}%")
            
            with col2:
                st.metric("成本价", f"${cost_basis:.2f}", "持仓盈利 +8.96%")
            
            with col3:
                st.metric("最近补仓", f"${recent_add_price:.2f}", f"{recent_add_shares}股")
            
            with col4:
                # 计算建议
                if current_price > recent_add_price * 1.02:
                    suggestion = "🟢 考虑减仓"
                elif current_price < recent_add_price * 0.98:
                    suggestion = "🔴 考虑止损"
                else:
                    suggestion = "🟡 持有观望"
                st.metric("操作建议", suggestion)
            
            # AMD决策分析
            self._show_amd_decision_analysis(amd_data, cost_basis, current_gain)
    
    def _show_amd_decision_analysis(self, amd_data: Dict, cost_basis: float, current_gain: float):
        """AMD决策分析"""
        st.subheader("🎯 AMD操作决策分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 技术面分析:**")
            
            rsi = amd_data.get('rsi', 50)
            ma20_position = "上方" if amd_data['current_price'] > amd_data.get('ma20', amd_data['current_price']) else "下方"
            
            # 技术评分
            tech_score = 5  # 基础分
            if 30 <= rsi <= 70:
                tech_score += 2
                rsi_status = "✅ 合理区间"
            elif rsi < 30:
                tech_score += 3
                rsi_status = "🟢 超卖反弹"
            else:
                tech_score -= 1
                rsi_status = "🔴 超买风险"
            
            st.write(f"- RSI: {rsi:.1f} {rsi_status}")
            st.write(f"- 均线位置: {ma20_position}")
            st.write(f"- 技术评分: {tech_score}/10")
        
        with col2:
            st.markdown("**💡 操作建议:**")
            
            # 基于当前情况的具体建议
            current_price = amd_data['current_price']
            
            if current_gain > 10:  # 盈利超过10%
                st.success("🎯 **建议分批减仓**")
                st.write("- 盈利已达8.96%，可考虑落袋为安")
                st.write("- 建议减仓30-50%，保留核心仓位")
            elif current_price < 135:  # 价格回调
                st.warning("⚠️ **谨慎观望**")
                st.write("- 价格回调至关键支撑位")
                st.write("- 观察是否跌破130支撑")
            else:
                st.info("📈 **持有为主**")
                st.write("- 维持当前仓位")
                st.write("- 设置止损位于130以下")
            
            # 风险提示
            st.markdown("**⚠️ 风险提示:**")
            st.write("- 半导体板块波动性大")
            st.write("- 建议设置止损保护利润")
    
    def _show_realtime_monitoring(self):
        """显示实时监控"""
        st.header("⚡ 实时价格监控")
        
        # 获取所有选中股票的数据
        stock_data = {}
        for symbol in st.session_state.selected_stocks:
            data = self._get_stock_data(symbol)
            if data:
                stock_data[symbol] = data
        
        if stock_data:
            # 创建实时价格表格
            df_display = pd.DataFrame({
                '股票': list(stock_data.keys()),
                '当前价格': [f"${data['current_price']:.2f}" for data in stock_data.values()],
                '涨跌幅': [f"{data['change_pct']:+.2f}%" for data in stock_data.values()],
                'RSI': [f"{data.get('rsi', 50):.1f}" for data in stock_data.values()],
                '成交量比': [f"{data.get('volume_ratio', 1):.1f}x" for data in stock_data.values()],
                '信号': [self._get_trading_signal(data) for data in stock_data.values()]
            })
            
            st.dataframe(df_display, use_container_width=True)
            
            # 实时价格图表
            self._create_price_charts(stock_data)
    
    def _show_technical_analysis(self):
        """显示技术分析"""
        st.header("📊 技术分析")
        
        # 选择要分析的股票
        selected_symbol = st.selectbox(
            "选择股票进行详细技术分析",
            st.session_state.selected_stocks,
            index=0 if 'AMD' in st.session_state.selected_stocks else 0
        )
        
        if selected_symbol:
            # 获取历史数据
            historical_data = self._get_historical_data(selected_symbol)
            if historical_data is not None:
                self._create_technical_chart(selected_symbol, historical_data)
    
    def _show_market_overview(self):
        """显示市场概览"""
        st.header("🌍 市场概览")
        
        # 主要指数
        indices = {
            '纳斯达克100': '^NDX',
            '标普500': '^GSPC',
            '道琼斯': '^DJI',
            'VIX恐慌指数': '^VIX'
        }
        
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        
        for i, (name, symbol) in enumerate(indices.items()):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='2d')
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2]
                    change_pct = (current - prev) / prev * 100
                    
                    with cols[i]:
                        st.metric(name, f"{current:.2f}", f"{change_pct:+.2f}%")
            except:
                with cols[i]:
                    st.metric(name, "N/A", "N/A")
    
    def _get_stock_data(self, symbol: str) -> Optional[Dict]:
        """获取股票数据"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='5d', interval='1m')
            
            if data.empty:
                return None
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change_pct = (current_price - prev_price) / prev_price * 100
            
            # 计算技术指标
            closes = data['Close'].values
            if len(closes) >= 14:
                # RSI
                delta = np.diff(closes)
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
            else:
                rsi = 50
            
            # 移动平均
            ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
            
            # 成交量比率
            volumes = data['Volume'].values
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            return {
                'current_price': float(current_price),
                'change_pct': float(change_pct),
                'rsi': float(rsi),
                'ma20': float(ma20),
                'volume_ratio': float(volume_ratio),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"获取{symbol}数据失败: {e}")
            return None
    
    def _get_historical_data(self, symbol: str, period: str = '1mo') -> Optional[pd.DataFrame]:
        """获取历史数据"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"获取{symbol}历史数据失败: {e}")
            return None
    
    def _get_trading_signal(self, data: Dict) -> str:
        """获取交易信号"""
        rsi = data.get('rsi', 50)
        current_price = data['current_price']
        ma20 = data.get('ma20', current_price)
        
        if rsi < 30 and current_price > ma20:
            return "🟢 买入"
        elif rsi > 70 and current_price < ma20:
            return "🔴 卖出"
        elif rsi < 30:
            return "🟡 观望"
        else:
            return "⚪ 中性"
    
    def _create_price_charts(self, stock_data: Dict):
        """创建价格图表"""
        # 选择要显示图表的股票
        chart_symbols = st.multiselect(
            "选择要显示图表的股票",
            list(stock_data.keys()),
            default=list(stock_data.keys())[:3]  # 默认显示前3个
        )
        
        if chart_symbols:
            for symbol in chart_symbols:
                # 获取更详细的历史数据用于图表
                historical_data = self._get_historical_data(symbol, '1d')
                if historical_data is not None:
                    fig = go.Figure()
                    
                    # 添加价格线
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data['Close'],
                        mode='lines',
                        name=f'{symbol} 价格',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # 添加移动平均线
                    ma20 = historical_data['Close'].rolling(20).mean()
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=ma20,
                        mode='lines',
                        name='MA20',
                        line=dict(color='orange', width=1)
                    ))
                    
                    fig.update_layout(
                        title=f'{symbol} 实时价格走势',
                        xaxis_title='时间',
                        yaxis_title='价格 ($)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def _create_technical_chart(self, symbol: str, data: pd.DataFrame):
        """创建技术分析图表"""
        # 计算技术指标
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} 价格走势', 'RSI指标'),
            row_heights=[0.7, 0.3]
        )
        
        # 价格图
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='价格'
        ), row=1, col=1)
        
        # 移动平均线
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            mode='lines',
            name='MA50',
            line=dict(color='purple', width=1)
        ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        # RSI超买超卖线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} 技术分析',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """主函数"""
    dashboard = StreamlitTradingDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()