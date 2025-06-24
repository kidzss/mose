#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版专业投资分析演示系统
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="专业投资分析演示",
    page_icon="📊",
    layout="wide"
)

def get_market_data():
    """获取市场数据"""
    try:
        # VIX
        vix = yf.Ticker('^VIX')
        vix_data = vix.history(period='5d')
        vix_current = vix_data['Close'].iloc[-1] if not vix_data.empty else 17.70
        
        # 标普500
        spx = yf.Ticker('^GSPC')
        spx_data = spx.history(period='5d')
        spx_current = spx_data['Close'].iloc[-1] if not spx_data.empty else 6092
        
        return {
            'vix': vix_current,
            'spx': spx_current,
            'vix_data': vix_data,
            'spx_data': spx_data
        }
    except:
        return {
            'vix': 17.70,
            'spx': 6092,
            'vix_data': None,
            'spx_data': None
        }

def calculate_market_probability(vix, spx):
    """计算市场概率"""
    # 基础概率
    up_prob = 50
    
    # VIX因子
    if vix < 18:
        up_prob += 20
    elif vix > 25:
        up_prob -= 20
    
    # 限制范围
    up_prob = max(20, min(80, up_prob))
    
    return {
        'up_probability': up_prob,
        'down_probability': 100 - up_prob,
        'bullish_target': spx * 1.05,
        'bearish_target': spx * 0.95
    }

def get_stock_data(symbol):
    """获取个股数据"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1mo')
        if data.empty:
            return None
        
        current_price = data['Close'].iloc[-1]
        
        # 计算RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        return {
            'current_price': current_price,
            'rsi': rsi,
            'data': data
        }
    except:
        return None

def calculate_stock_probability(stock_data):
    """计算个股概率"""
    if not stock_data:
        return None
    
    up_prob = 50
    rsi = stock_data['rsi']
    
    # RSI因子
    if rsi < 30:
        up_prob += 20
    elif rsi > 70:
        up_prob -= 20
    
    up_prob = max(20, min(80, up_prob))
    
    return {
        'up_probability': up_prob,
        'down_probability': 100 - up_prob,
        'bullish_target': stock_data['current_price'] * 1.08,
        'bearish_target': stock_data['current_price'] * 0.92
    }

def main():
    st.title("🎯 专业投资分析演示系统")
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.header("📊 配置")
    amd_shares = st.sidebar.number_input("AMD持仓股数", value=0, min_value=0)
    amd_cost = st.sidebar.number_input("AMD成本价", value=125.746, min_value=0.0)
    
    # 主要标签页
    tab1, tab2, tab3 = st.tabs(["📊 市场分析", "🎯 个股分析", "💼 持仓管理"])
    
    with tab1:
        st.header("📊 市场概率分析")
        
        # 获取市场数据
        market_data = get_market_data()
        vix = market_data['vix']
        spx = market_data['spx']
        
        # 显示当前状态
        col1, col2 = st.columns(2)
        with col1:
            st.metric("VIX恐慌指数", f"{vix:.2f}")
        with col2:
            st.metric("标普500", f"{spx:.0f}")
        
        # 计算概率
        market_prob = calculate_market_probability(vix, spx)
        
        # 显示概率预测
        st.subheader("🎯 市场方向概率")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 概率饼图
            fig = go.Figure(data=[
                go.Pie(
                    labels=['上涨概率', '下跌概率'],
                    values=[market_prob['up_probability'], market_prob['down_probability']],
                    marker=dict(colors=['green', 'red'])
                )
            ])
            fig.update_layout(title="市场方向概率")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📊 预测结果:**")
            st.write(f"🟢 上涨概率: **{market_prob['up_probability']:.0f}%**")
            st.write(f"🔴 下跌概率: **{market_prob['down_probability']:.0f}%**")
            st.write(f"🎯 乐观目标: **{market_prob['bullish_target']:.0f}**")
            st.write(f"🎯 悲观目标: **{market_prob['bearish_target']:.0f}**")
        
        # 情景分析
        st.subheader("📋 情景分析")
        
        scenarios = [
            ("🟢 乐观情景 (35%)", "VIX下降至15以下，标普突破6300", "科技股大涨"),
            ("🟡 中性情景 (50%)", "VIX在15-20震荡，标普整理", "个股分化"),
            ("🔴 悲观情景 (15%)", "VIX反弹至25以上，标普回调", "全面调整")
        ]
        
        for name, condition, impact in scenarios:
            with st.expander(name):
                st.write(f"**条件:** {condition}")
                st.write(f"**影响:** {impact}")
    
    with tab2:
        st.header("🎯 AMD个股分析")
        
        # 获取AMD数据
        amd_data = get_stock_data('AMD')
        
        if amd_data:
            current_price = amd_data['current_price']
            rsi = amd_data['rsi']
            
            # 显示当前状态
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("当前价格", f"${current_price:.2f}")
            with col2:
                rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
                st.metric("RSI", f"{rsi:.1f}", rsi_color)
            with col3:
                st.metric("状态", "超卖" if rsi < 30 else "超买" if rsi > 70 else "中性")
            
            # 计算概率
            stock_prob = calculate_stock_probability(amd_data)
            
            if stock_prob:
                st.subheader("🎯 AMD概率预测")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 概率条形图
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['上涨', '下跌'],
                            y=[stock_prob['up_probability'], stock_prob['down_probability']],
                            marker_color=['green', 'red']
                        )
                    ])
                    fig.update_layout(title="AMD方向概率", yaxis_title="概率 (%)")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**📊 预测结果:**")
                    st.write(f"🟢 上涨概率: **{stock_prob['up_probability']:.0f}%**")
                    st.write(f"🔴 下跌概率: **{stock_prob['down_probability']:.0f}%**")
                    st.write(f"🎯 上涨目标: **${stock_prob['bullish_target']:.2f}**")
                    st.write(f"🎯 下跌目标: **${stock_prob['bearish_target']:.2f}**")
                    
                    # 操作建议
                    if stock_prob['up_probability'] > 65:
                        st.success("💡 建议: 考虑买入或持有")
                    elif stock_prob['up_probability'] < 35:
                        st.error("💡 建议: 考虑减仓或观望")
                    else:
                        st.info("💡 建议: 保持中性观望")
        else:
            st.error("无法获取AMD数据")
    
    with tab3:
        st.header("💼 持仓管理")
        
        if amd_shares > 0:
            # AMD持仓分析
            amd_data = get_stock_data('AMD')
            if amd_data:
                current_price = amd_data['current_price']
                
                # 计算盈亏
                total_cost = amd_cost * amd_shares
                current_value = current_price * amd_shares
                unrealized_pnl = current_value - total_cost
                pnl_pct = (unrealized_pnl / total_cost) * 100
                
                st.subheader("💼 AMD持仓详情")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("持仓股数", f"{amd_shares}", "股")
                
                with col2:
                    st.metric("成本价", f"${amd_cost:.2f}")
                
                with col3:
                    st.metric("当前价", f"${current_price:.2f}")
                
                with col4:
                    color = "normal" if pnl_pct >= 0 else "inverse"
                    st.metric("盈亏", f"${unrealized_pnl:.2f}", f"{pnl_pct:+.2f}%")
                
                # 持仓建议
                st.subheader("💡 操作建议")
                
                if pnl_pct > 15:
                    st.success("🎯 建议分批减仓，锁定部分利润")
                elif pnl_pct > 8:
                    st.info("📈 可继续持有，设置止损保护利润")
                elif pnl_pct < -5:
                    st.warning("⚠️ 考虑止损或加仓摊薄成本")
                else:
                    st.info("📊 保持当前仓位，密切关注")
        else:
            st.info("请在侧边栏输入持仓信息")
        
        # 投资组合建议
        st.subheader("📋 资产配置建议")
        
        allocation = {"现金": 20, "债券": 20, "大盘股": 35, "科技股": 25}
        
        fig = go.Figure(data=[
            go.Pie(labels=list(allocation.keys()), values=list(allocation.values()))
        ])
        fig.update_layout(title="平衡型投资者建议配置")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 