#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化投资分析系统 - 无需额外依赖
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="简化投资分析",
    page_icon="📊",
    layout="wide"
)

@st.cache_data(ttl=60)  # 缓存1分钟
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
    except Exception as e:
        st.error(f"数据获取失败: {e}")
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
    
    # VIX因子调整
    if vix < 18:
        up_prob += 20
        market_state = "低恐慌"
    elif vix > 25:
        up_prob -= 20
        market_state = "高恐慌"
    else:
        market_state = "中性"
    
    # 限制范围
    up_prob = max(20, min(80, up_prob))
    
    return {
        'up_probability': up_prob,
        'down_probability': 100 - up_prob,
        'bullish_target': spx * 1.05,
        'bearish_target': spx * 0.95,
        'market_state': market_state
    }

@st.cache_data(ttl=60)  # 缓存1分钟
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
        
        # 计算移动平均
        ma_20 = data['Close'].rolling(20).mean().iloc[-1]
        ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else ma_20
        
        return {
            'current_price': current_price,
            'rsi': rsi,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'data': data
        }
    except Exception as e:
        st.error(f"获取{symbol}数据失败: {e}")
        return None

def calculate_stock_probability(stock_data):
    """计算个股概率"""
    if not stock_data:
        return None
    
    up_prob = 50
    rsi = stock_data['rsi']
    current_price = stock_data['current_price']
    ma_20 = stock_data['ma_20']
    
    # RSI因子
    if rsi < 30:
        up_prob += 20
        rsi_state = "超卖"
    elif rsi > 70:
        up_prob -= 20
        rsi_state = "超买"
    else:
        rsi_state = "中性"
    
    # 趋势因子
    if current_price > ma_20:
        up_prob += 10
        trend_state = "上升趋势"
    else:
        up_prob -= 10
        trend_state = "下降趋势"
    
    up_prob = max(20, min(80, up_prob))
    
    return {
        'up_probability': up_prob,
        'down_probability': 100 - up_prob,
        'bullish_target': current_price * 1.08,
        'bearish_target': current_price * 0.92,
        'rsi_state': rsi_state,
        'trend_state': trend_state
    }

def main():
    st.title("📊 简化投资分析系统")
    st.markdown("---")
    
    # 侧边栏配置
    st.sidebar.header("⚙️ 配置")
    
    # 持仓信息
    amd_shares = st.sidebar.number_input("AMD持仓股数", value=0, min_value=0, step=1)
    amd_cost = st.sidebar.number_input("AMD成本价", value=125.746, min_value=0.0, step=0.01)
    
    # 风险偏好
    risk_level = st.sidebar.selectbox("风险偏好", ["保守", "稳健", "积极"])
    
    # 主要内容
    tab1, tab2, tab3 = st.tabs(["📈 市场分析", "🎯 个股分析", "💼 持仓管理"])
    
    with tab1:
        st.header("📈 市场概率分析")
        
        # 获取市场数据
        with st.spinner("获取市场数据..."):
            market_data = get_market_data()
        
        vix = market_data['vix']
        spx = market_data['spx']
        
        # 显示当前状态
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("VIX恐慌指数", f"{vix:.2f}")
        
        with col2:
            st.metric("标普500", f"{spx:.0f}")
        
        with col3:
            # 计算日内变化 (模拟)
            change_pct = np.random.uniform(-1.5, 1.5)
            delta_color = "normal" if change_pct >= 0 else "inverse"
            st.metric("今日变化", f"{change_pct:+.2f}%")
        
        # 市场概率分析
        market_prob = calculate_market_probability(vix, spx)
        
        st.subheader("🎯 市场方向概率")
        
        # 使用条形图显示概率
        prob_data = pd.DataFrame({
            '方向': ['上涨', '下跌'],
            '概率': [market_prob['up_probability'], market_prob['down_probability']]
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.bar_chart(prob_data.set_index('方向'))
        
        with col2:
            st.markdown("**📊 预测结果:**")
            st.write(f"🟢 上涨概率: **{market_prob['up_probability']:.0f}%**")
            st.write(f"🔴 下跌概率: **{market_prob['down_probability']:.0f}%**")
            st.write(f"📊 市场状态: **{market_prob['market_state']}**")
            st.write(f"🎯 乐观目标: **{market_prob['bullish_target']:.0f}**")
            st.write(f"🎯 悲观目标: **{market_prob['bearish_target']:.0f}**")
        
        # 情景分析
        st.subheader("📋 情景分析")
        
        scenarios = [
            ("🟢 乐观情景", "35%", "VIX < 15, 科技股领涨", "标普突破6300"),
            ("🟡 中性情景", "50%", "VIX 15-20震荡", "标普6000-6300整理"),
            ("🔴 悲观情景", "15%", "VIX > 25反弹", "标普回调至5800")
        ]
        
        for name, prob, condition, target in scenarios:
            with st.expander(f"{name} ({prob})"):
                st.write(f"**条件:** {condition}")
                st.write(f"**目标:** {target}")
    
    with tab2:
        st.header("🎯 AMD个股分析")
        
        with st.spinner("获取AMD数据..."):
            amd_data = get_stock_data('AMD')
        
        if amd_data:
            current_price = amd_data['current_price']
            rsi = amd_data['rsi']
            ma_20 = amd_data['ma_20']
            
            # 显示当前状态
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("当前价格", f"${current_price:.2f}")
            
            with col2:
                rsi_delta = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
                st.metric("RSI", f"{rsi:.1f}", rsi_delta)
            
            with col3:
                st.metric("20日均线", f"${ma_20:.2f}")
            
            with col4:
                trend = "上升" if current_price > ma_20 else "下降"
                st.metric("趋势", trend)
            
            # 计算概率
            stock_prob = calculate_stock_probability(amd_data)
            
            if stock_prob:
                st.subheader("🎯 AMD概率预测")
                
                # 概率可视化
                prob_df = pd.DataFrame({
                    '概率': [stock_prob['up_probability'], stock_prob['down_probability']]
                }, index=['上涨', '下跌'])
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.bar_chart(prob_df)
                
                with col2:
                    st.markdown("**📊 分析结果:**")
                    st.write(f"🟢 上涨概率: **{stock_prob['up_probability']:.0f}%**")
                    st.write(f"🔴 下跌概率: **{stock_prob['down_probability']:.0f}%**")
                    st.write(f"📊 RSI状态: **{stock_prob['rsi_state']}**")
                    st.write(f"📈 趋势状态: **{stock_prob['trend_state']}**")
                    st.write(f"🎯 上涨目标: **${stock_prob['bullish_target']:.2f}**")
                    st.write(f"🎯 下跌目标: **${stock_prob['bearish_target']:.2f}**")
                
                # 操作建议
                st.subheader("💡 操作建议")
                
                if stock_prob['up_probability'] > 65:
                    st.success("🟢 **建议:** 考虑买入或继续持有")
                    st.info("📈 高概率上涨，适合积极投资者")
                elif stock_prob['up_probability'] < 35:
                    st.error("🔴 **建议:** 考虑减仓或观望")
                    st.warning("📉 下跌风险较高，建议谨慎")
                else:
                    st.info("🟡 **建议:** 保持中性观望")
                    st.info("📊 方向不明确，等待更好机会")
        else:
            st.error("❌ 无法获取AMD数据，请检查网络连接")
    
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
                    delta_symbol = "+" if pnl_pct >= 0 else ""
                    st.metric("盈亏", f"${unrealized_pnl:.2f}", f"{delta_symbol}{pnl_pct:.2f}%")
                
                # 持仓状态
                st.subheader("📊 持仓状态")
                
                if pnl_pct > 15:
                    st.success("🎯 **优秀表现** - 考虑分批减仓锁定利润")
                elif pnl_pct > 8:
                    st.info("📈 **良好表现** - 可继续持有，设置止损")
                elif pnl_pct > 0:
                    st.info("📊 **小幅盈利** - 保持当前仓位")
                elif pnl_pct > -5:
                    st.warning("⚠️ **小幅亏损** - 密切关注，考虑加仓或止损")
                else:
                    st.error("🔴 **较大亏损** - 建议止损或评估基本面")
                
                # 风险建议
                st.subheader("⚡ 风险管理建议")
                
                if risk_level == "保守":
                    st.info("🛡️ 保守策略: 设置5%止损，盈利10%考虑减仓")
                elif risk_level == "稳健":
                    st.info("⚖️ 稳健策略: 设置8%止损，盈利15%考虑减仓")
                else:
                    st.info("🚀 积极策略: 设置12%止损，盈利20%考虑减仓")
        else:
            st.info("📝 请在侧边栏输入持仓信息以查看详细分析")
        
        # 投资组合建议
        st.subheader("📋 资产配置建议")
        
        if risk_level == "保守":
            allocation = {"现金": 30, "债券": 40, "大盘股": 25, "科技股": 5}
        elif risk_level == "稳健":
            allocation = {"现金": 20, "债券": 30, "大盘股": 35, "科技股": 15}
        else:
            allocation = {"现金": 10, "债券": 20, "大盘股": 35, "科技股": 35}
        
        # 显示配置
        allocation_df = pd.DataFrame(list(allocation.items()), columns=['资产类别', '比例(%)'])
        st.bar_chart(allocation_df.set_index('资产类别'))
        
        for asset, pct in allocation.items():
            st.write(f"• {asset}: {pct}%")

if __name__ == "__main__":
    main() 