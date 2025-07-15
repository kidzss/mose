#!/usr/bin/env python3
"""
投资组合分散化分析
分析当前持仓结构，提供分散化建议
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def load_portfolio_data():
    """加载投资组合数据"""
    try:
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载投资组合数据失败: {e}")
        return None

def get_stock_data(symbols, period="1y"):
    """获取股票数据"""
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if not hist.empty:
                data[symbol] = {
                    'current_price': hist['Close'].iloc[-1],
                    'price_change': hist['Close'].iloc[-1] - hist['Close'].iloc[-2],
                    'price_change_pct': ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100,
                    'volume': hist['Volume'].iloc[-1],
                    'market_cap': ticker.info.get('marketCap', 0),
                    'pe_ratio': ticker.info.get('trailingPE', 0),
                    'beta': ticker.info.get('beta', 1.0),
                    'sector': ticker.info.get('sector', 'Unknown'),
                    'industry': ticker.info.get('industry', 'Unknown')
                }
        except Exception as e:
            st.warning(f"获取{symbol}数据失败: {e}")
    return data

def analyze_portfolio_diversification(portfolio_data):
    """分析投资组合分散化情况"""
    
    # 当前持仓分析
    positions = portfolio_data.get('positions', {})
    
    # 按行业分类
    sector_allocation = {}
    tech_stocks = []
    healthcare_stocks = []
    financial_stocks = []
    
    for symbol, position in positions.items():
        sector = position.get('sector', 'Unknown')
        weight = position.get('weight', 0)
        
        if sector not in sector_allocation:
            sector_allocation[sector] = 0
        sector_allocation[sector] += weight
        
        if sector == 'Technology':
            tech_stocks.append({
                'symbol': symbol,
                'weight': weight,
                'shares': position.get('shares', 0),
                'cost_basis': position.get('cost_basis', 0)
            })
        elif sector == 'Healthcare':
            healthcare_stocks.append({
                'symbol': symbol,
                'weight': weight,
                'shares': position.get('shares', 0),
                'cost_basis': position.get('cost_basis', 0)
            })
        elif sector == 'Financial':
            financial_stocks.append({
                'symbol': symbol,
                'weight': weight,
                'shares': position.get('shares', 0),
                'cost_basis': position.get('cost_basis', 0)
            })
    
    return {
        'sector_allocation': sector_allocation,
        'tech_stocks': tech_stocks,
        'healthcare_stocks': healthcare_stocks,
        'financial_stocks': financial_stocks,
        'total_tech_weight': sum([stock['weight'] for stock in tech_stocks]),
        'total_healthcare_weight': sum([stock['weight'] for stock in healthcare_stocks]),
        'total_financial_weight': sum([stock['weight'] for stock in financial_stocks])
    }

def analyze_potential_investments(candidates):
    """分析潜在投资机会"""
    results = {}
    
    for symbol in candidates:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                price_52w_high = hist['High'].max()
                price_52w_low = hist['Low'].min()
                price_position = (current_price - price_52w_low) / (price_52w_high - price_52w_low) * 100
                
                # 计算技术指标
                rsi = calculate_rsi(hist['Close'])
                ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                
                # 基本面评分
                pe_ratio = info.get('trailingPE', 0)
                peg_ratio = info.get('pegRatio', 0)
                profit_margins = info.get('profitMargins', 0)
                roe = info.get('returnOnEquity', 0)
                
                # 综合评分
                technical_score = 0
                if current_price > ma_20 > ma_50:
                    technical_score += 30
                if 30 <= rsi <= 70:
                    technical_score += 20
                if price_position < 80:
                    technical_score += 25
                if current_price > ma_20:
                    technical_score += 25
                
                fundamental_score = 0
                if pe_ratio and 10 <= pe_ratio <= 25:
                    fundamental_score += 25
                if peg_ratio and peg_ratio < 2:
                    fundamental_score += 25
                if profit_margins and profit_margins > 0.1:
                    fundamental_score += 25
                if roe and roe > 0.15:
                    fundamental_score += 25
                
                results[symbol] = {
                    'current_price': current_price,
                    'price_change_pct': ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100,
                    'price_position_52w': price_position,
                    'rsi': rsi,
                    'ma_20': ma_20,
                    'ma_50': ma_50,
                    'pe_ratio': pe_ratio,
                    'peg_ratio': peg_ratio,
                    'profit_margins': profit_margins,
                    'roe': roe,
                    'technical_score': technical_score,
                    'fundamental_score': fundamental_score,
                    'total_score': (technical_score + fundamental_score) / 2,
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'beta': info.get('beta', 1.0)
                }
                
        except Exception as e:
            st.warning(f"分析{symbol}失败: {e}")
    
    return results

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def render_diversification_analysis():
    """渲染分散化分析界面"""
    st.title("📊 投资组合分散化分析")
    st.markdown("---")
    
    # 加载数据
    portfolio_data = load_portfolio_data()
    if not portfolio_data:
        st.error("无法加载投资组合数据")
        return
    
    # 分析当前持仓
    analysis = analyze_portfolio_diversification(portfolio_data)
    
    # 显示当前持仓分析
    st.header("📋 当前持仓分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 行业分布")
        sector_data = analysis['sector_allocation']
        
        if sector_data:
            # 创建饼图
            fig = px.pie(
                values=list(sector_data.values()),
                names=list(sector_data.keys()),
                title="行业权重分布"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示具体数据
            for sector, weight in sector_data.items():
                st.write(f"**{sector}**: {weight:.2f}%")
    
    with col2:
        st.subheader("⚠️ 风险分析")
        
        # 科技股集中度风险
        tech_weight = analysis['total_tech_weight']
        st.write(f"**科技股权重**: {tech_weight:.2f}%")
        
        if tech_weight > 50:
            st.error("🚨 科技股集中度过高！建议分散投资")
        elif tech_weight > 40:
            st.warning("⚠️ 科技股权重偏高，建议适当分散")
        else:
            st.success("✅ 科技股权重合理")
        
        # 行业分散度
        sector_count = len(sector_data)
        st.write(f"**覆盖行业数**: {sector_count}")
        
        if sector_count < 3:
            st.warning("⚠️ 行业分散度不足，建议增加行业覆盖")
        else:
            st.success("✅ 行业分散度良好")
    
    st.markdown("---")
    
    # 分散化建议
    st.header("💡 分散化建议")
    
    if tech_weight > 40:
        st.markdown("""
        ### 🎯 主要建议
        
        **1. 降低科技股集中度**
        - 当前科技股权重: {:.2f}%
        - 建议目标: 降至30-35%
        - 可考虑减仓部分AMD或NVDA
        
        **2. 增加防御性配置**
        - 医疗健康股: 建议10-15%
        - 消费股: 建议10-15%
        - 金融股: 建议10-15%
        
        **3. 分批调整策略**
        - 不要一次性大幅调整
        - 分3-6个月逐步调整
        - 利用市场回调机会
        """.format(tech_weight))
    
    st.markdown("---")
    
    # 潜在投资机会分析
    st.header("🔍 潜在投资机会分析")
    
    # 候选股票
    candidates = ['JNJ', 'ABT', 'COST', 'DIS', 'JPM', 'V']
    
    # 获取数据
    with st.spinner("正在分析潜在投资机会..."):
        investment_analysis = analyze_potential_investments(candidates)
    
    if investment_analysis:
        # 按评分排序
        sorted_candidates = sorted(
            investment_analysis.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # 显示分析结果
        for i, (symbol, data) in enumerate(sorted_candidates):
            with st.expander(f"📊 {symbol} - 综合评分: {data['total_score']:.1f}/100", expanded=i<3):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**当前价格**: ${data['current_price']:.2f}")
                    st.write(f"**涨跌幅**: {data['price_change_pct']:+.2f}%")
                    st.write(f"**52周位置**: {data['price_position_52w']:.1f}%")
                    st.write(f"**RSI**: {data['rsi']:.1f}")
                
                with col2:
                    st.write(f"**PE比率**: {data['pe_ratio']:.1f}" if data['pe_ratio'] else "**PE比率**: N/A")
                    st.write(f"**PEG比率**: {data['peg_ratio']:.2f}" if data['peg_ratio'] else "**PEG比率**: N/A")
                    st.write(f"**净利润率**: {data['profit_margins']*100:.1f}%" if data['profit_margins'] else "**净利润率**: N/A")
                    st.write(f"**ROE**: {data['roe']*100:.1f}%" if data['roe'] else "**ROE**: N/A")
                
                with col3:
                    st.write(f"**技术评分**: {data['technical_score']:.1f}/100")
                    st.write(f"**基本面评分**: {data['fundamental_score']:.1f}/100")
                    st.write(f"**行业**: {data['sector']}")
                    st.write(f"**Beta**: {data['beta']:.2f}")
                
                # 投资建议
                if data['total_score'] >= 70:
                    st.success("🎯 **强烈推荐**: 综合评分优秀，建议考虑投资")
                elif data['total_score'] >= 50:
                    st.info("📈 **可以考虑**: 评分良好，可以关注")
                else:
                    st.warning("⚠️ **需要观察**: 评分较低，建议等待更好时机")
    
    st.markdown("---")
    
    # 具体操作建议
    st.header("🎯 具体操作建议")
    
    st.markdown("""
    ### 📅 调整时间表
    
    **第一阶段 (1-2个月)**:
    - 观察市场回调机会
    - 准备减仓部分科技股
    - 研究目标股票基本面
    
    **第二阶段 (2-3个月)**:
    - 开始分批减仓科技股
    - 逐步建仓防御性股票
    - 监控市场环境变化
    
    **第三阶段 (3-6个月)**:
    - 完成投资组合调整
    - 达到目标行业分布
    - 建立新的平衡配置
    
    ### 🎯 推荐投资顺序
    
    1. **JNJ** (医疗健康) - 防御性强，分红稳定
    2. **COST** (消费) - 抗通胀，成长性好
    3. **JPM** (金融) - 银行业龙头，利率受益
    4. **V** (金融) - 支付网络，全球化受益
    5. **ABT** (医疗) - 多元化医疗设备
    6. **DIS** (消费) - 娱乐巨头，估值合理
    """)

if __name__ == "__main__":
    render_diversification_analysis() 