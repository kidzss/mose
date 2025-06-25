#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业实时交易监控系统
Professional Real-time Trading Monitor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入统一分析系统
from analysis.unified_stock_analyzer import UnifiedStockAnalyzer
from analysis.streamlit_analysis_bridge import display_stock_analysis

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="专业交易监控系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-green {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-red {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-yellow {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 全局配置
INDICES = ['^GSPC', '^IXIC', '^DJI', '^VIX']
UPDATE_INTERVAL = 60  # 秒

@st.cache_data(ttl=300)  # 缓存5分钟
def load_portfolio_config():
    """从JSON配置文件加载持仓和观察仓信息"""
    try:
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 提取当前持仓股票 (排除港股和已卖出的股票)
        current_positions = []
        portfolio_info = {}
        
        for symbol, position in config.get('positions', {}).items():
            if (not symbol.endswith('.HK') and 
                position.get('shares', 0) > 0 and 
                position.get('status') != 'SOLD'):
                current_positions.append(symbol)
                portfolio_info[symbol] = {
                    'shares': position.get('shares', 0),
                    'cost_basis': position.get('cost_basis', 0),
                    'weight': position.get('weight', 0),
                    'sector': position.get('sector', 'Unknown'),
                    'stop_loss_threshold': position.get('stop_loss_threshold', 0.08),
                    'investment_amount': position.get('investment_amount', 0)
                }
        
        # 提取观察仓股票
        watchlist_stocks = list(config.get('watchlist', {}).keys())
        
        # 完整监控列表
        all_stocks = current_positions + watchlist_stocks
        
        return {
            'current_positions': current_positions,
            'watchlist_stocks': watchlist_stocks,
            'all_stocks': all_stocks,
            'portfolio_info': portfolio_info,
            'watchlist_info': config.get('watchlist', {}),
            'meta': config.get('meta', {})
        }
        
    except Exception as e:
        st.error(f"配置文件加载失败: {e}")
        # 返回默认配置
        return {
            'current_positions': ['AMD', 'NVDA'],
            'watchlist_stocks': ['MSFT', 'AAPL'],
            'all_stocks': ['AMD', 'NVDA', 'MSFT', 'AAPL'],
            'portfolio_info': {},
            'watchlist_info': {},
            'meta': {}
        }

@st.cache_data(ttl=60)  # 缓存1分钟
def get_realtime_data(symbols):
    """获取实时市场数据"""
    data = {}
    try:
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            # 获取更多历史数据用于RSI计算
            hist = ticker.history(period='3mo', interval='1d')  # 改为3个月数据
            info = ticker.info
            
            if not hist.empty and len(hist) >= 15:  # 确保有足够的数据
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                
                # 计算技术指标
                rsi = calculate_rsi(hist['Close'])
                ma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
                ma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
                
                data[symbol] = {
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                    'rsi': rsi,
                    'ma_20': ma_20,
                    'ma_50': ma_50,
                    'high_52w': info.get('fiftyTwoWeekHigh', current_price),
                    'low_52w': info.get('fiftyTwoWeekLow', current_price),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'hist_data': hist
                }
            else:
                # 默认数据
                data[symbol] = {
                    'price': 0, 'change': 0, 'change_pct': 0, 'volume': 0,
                    'rsi': 50, 'ma_20': 0, 'ma_50': 0,
                    'high_52w': 0, 'low_52w': 0, 'market_cap': 0, 'pe_ratio': 0,
                    'hist_data': pd.DataFrame()
                }
    except Exception as e:
        st.error(f"数据获取错误: {e}")
    
    return data

def calculate_rsi(prices, period=14):
    """计算RSI指标 - 使用标准RSI算法"""
    try:
        if len(prices) < period + 1:
            print(f"数据不足: {len(prices)} < {period + 1}")
            return 50
        
        # 计算价格变化
        delta = prices.diff().dropna()
        
        # 分离涨跌
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # 计算平均涨跌幅 (使用简单移动平均)
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 获取最后一个有效值
        rsi_value = rsi.iloc[-1]
        
        if pd.isna(rsi_value) or np.isinf(rsi_value):
            print(f"RSI计算结果无效: {rsi_value}")
            return 50
        
        print(f"RSI计算成功: {rsi_value:.2f}")
        return float(rsi_value)
        
    except Exception as e:
        print(f"RSI计算错误: {e}")
        return 50

def calculate_technical_signals(data):
    """计算技术信号"""
    signals = {}
    
    for symbol, info in data.items():
        if symbol.startswith('^'):  # 跳过指数
            continue
            
        price = info['price']
        rsi = info['rsi']
        ma_20 = info['ma_20']
        ma_50 = info['ma_50']
        
        # 技术信号评分
        score = 0
        signals_list = []
        
        # RSI信号
        if rsi < 30:
            score += 2
            signals_list.append("RSI超卖")
        elif rsi > 70:
            score -= 2
            signals_list.append("RSI超买")
        elif 30 <= rsi <= 40:
            score += 1
            signals_list.append("RSI偏低")
        elif 60 <= rsi <= 70:
            score -= 1
            signals_list.append("RSI偏高")
        
        # 均线信号
        if price > ma_20 > ma_50:
            score += 2
            signals_list.append("多头排列")
        elif price < ma_20 < ma_50:
            score -= 2
            signals_list.append("空头排列")
        elif price > ma_20:
            score += 1
            signals_list.append("短期强势")
        elif price < ma_20:
            score -= 1
            signals_list.append("短期弱势")
        
        # 52周位置
        pos_52w = (price - info['low_52w']) / (info['high_52w'] - info['low_52w']) * 100 if info['high_52w'] != info['low_52w'] else 50
        
        if pos_52w < 25:
            score += 1
            signals_list.append("接近年低")
        elif pos_52w > 75:
            score -= 1
            signals_list.append("接近年高")
        
        # 综合评级
        if score >= 3:
            rating = "强烈买入"
            color = "#28a745"
        elif score >= 1:
            rating = "买入"
            color = "#28a745"
        elif score <= -3:
            rating = "强烈卖出"
            color = "#dc3545"
        elif score <= -1:
            rating = "卖出"
            color = "#dc3545"
        else:
            rating = "中性"
            color = "#ffc107"
        
        signals[symbol] = {
            'score': score,
            'rating': rating,
            'color': color,
            'signals': signals_list,
            'position_52w': pos_52w
        }
    
    return signals

def create_market_overview_chart(indices_data):
    """创建市场概览图表"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('标普500', '纳斯达克', '道琼斯', 'VIX恐慌指数'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
    names = ['标普500', '纳斯达克', '道琼斯', 'VIX']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, (symbol, name, (row, col)) in enumerate(zip(symbols, names, positions)):
        if symbol in indices_data and not indices_data[symbol]['hist_data'].empty:
            hist = indices_data[symbol]['hist_data']
            
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name=name,
                    line=dict(color='red' if symbol == '^VIX' else 'blue', width=2)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title="市场指数实时监控",
        height=600,
        showlegend=False
    )
    
    return fig

def create_stock_chart(symbol, data):
    """创建个股详细图表"""
    if symbol not in data or data[symbol]['hist_data'].empty:
        return None
    
    hist = data[symbol]['hist_data']
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(f'{symbol} 价格走势', 'RSI指标', '成交量'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='价格'
        ),
        row=1, col=1
    )
    
    # 移动平均线
    if len(hist) >= 20:
        ma_20 = hist['Close'].rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=hist.index, y=ma_20, mode='lines', name='MA20', line=dict(color='orange')),
            row=1, col=1
        )
    
    if len(hist) >= 50:
        ma_50 = hist['Close'].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=hist.index, y=ma_50, mode='lines', name='MA50', line=dict(color='red')),
            row=1, col=1
        )
    
    # RSI
    def calculate_rsi_series(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    rsi_series = calculate_rsi_series(hist['Close'])
    fig.add_trace(
        go.Scatter(x=hist.index, y=rsi_series, mode='lines', name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    # RSI参考线
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # 成交量
    if 'Volume' in hist.columns:
        fig.add_trace(
            go.Bar(x=hist.index, y=hist['Volume'], name='成交量', marker_color='lightblue'),
            row=3, col=1
        )
    
    fig.update_layout(
        title=f"{symbol} 技术分析",
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    # 页面标题
    st.markdown('<div class="main-header">⚡ 专业实时交易监控系统</div>', unsafe_allow_html=True)
    
    # 加载配置信息
    config = load_portfolio_config()
    
    # 实时更新时间显示
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"**🕒 最后更新:** {current_time}")
    
    # 显示投资组合概况
    if config['meta']:
        st.markdown(f"**💼 总资产:** ${config['meta'].get('total_assets', 0):,.2f}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("持仓股票", len(config['current_positions']))
        with col2:
            st.metric("观察股票", len(config['watchlist_stocks']))
        with col3:
            st.metric("总监控", len(config['all_stocks']))
    
    # 侧边栏配置
    st.sidebar.header("🎛️ 交易配置")
    
    # 显示持仓信息
    st.sidebar.subheader("💼 当前持仓")
    for symbol in config['current_positions']:
        info = config['portfolio_info'].get(symbol, {})
        st.sidebar.markdown(f"**{symbol}** - {info.get('shares', 0)}股 @ ${info.get('cost_basis', 0):.2f}")
    
    # 监控股票选择
    st.sidebar.subheader("📊 监控股票")
    
    # 分组显示
    show_positions = st.sidebar.checkbox("显示持仓股票", value=True)
    show_watchlist = st.sidebar.checkbox("显示观察仓股票", value=True)
    
    available_stocks = []
    if show_positions:
        available_stocks.extend(config['current_positions'])
    if show_watchlist:
        available_stocks.extend(config['watchlist_stocks'])
    
    # 默认选择所有持仓股票
    default_selection = config['current_positions'][:5]  # 最多显示5只
    
    selected_stocks = st.sidebar.multiselect(
        "选择监控股票",
        available_stocks,
        default=default_selection
    )
    
    # 持仓设置 - 从配置文件获取
    st.sidebar.subheader("💼 持仓详情")
    portfolio = {}
    
    # 显示实际持仓信息
    for stock in selected_stocks:
        if stock in config['portfolio_info']:
            info = config['portfolio_info'][stock]
            portfolio[stock] = {
                'shares': info['shares'],
                'cost': info['cost_basis'],
                'weight': info['weight'],
                'investment_amount': info['investment_amount']
            }
            
            # 在侧边栏显示详细信息
            with st.sidebar.expander(f"📊 {stock} 详情"):
                st.write(f"**持股数量:** {info['shares']} 股")
                st.write(f"**成本价格:** ${info['cost_basis']:.2f}")
                st.write(f"**投资金额:** ${info['investment_amount']:,.2f}")
                st.write(f"**仓位权重:** {info['weight']:.2f}%")
                st.write(f"**行业板块:** {info['sector']}")
                st.write(f"**止损阈值:** {info['stop_loss_threshold']*100:.1f}%")
    
    # 风险设置
    st.sidebar.subheader("⚠️ 风险管理")
    stop_loss_pct = st.sidebar.slider("止损百分比", 1, 20, 8, 1)
    take_profit_pct = st.sidebar.slider("止盈百分比", 5, 50, 15, 1)
    
    # 自动刷新
    auto_refresh = st.sidebar.checkbox("自动刷新 (60秒)", value=False)
    if auto_refresh:
        st.sidebar.info("页面将每60秒自动刷新")
        time.sleep(1)
        st.rerun()
    
    # 手动刷新按钮
    if st.sidebar.button("🔄 立即刷新数据"):
        st.cache_data.clear()
        st.rerun()
    
    # 获取所有数据
    with st.spinner("获取实时数据..."):
        all_symbols = selected_stocks + INDICES
        market_data = get_realtime_data(all_symbols)
        
        if selected_stocks:
            stock_data = {k: v for k, v in market_data.items() if k in selected_stocks}
            signals = calculate_technical_signals(stock_data)
        else:
            stock_data = {}
            signals = {}
    
    # 主要内容区域 - 新增深度分析标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 市场概览", "📈 监控股票", "🎯 技术分析", "🔬 深度分析", "💼 投资组合"])
    
    with tab1:
        st.header("📊 市场概览")
        
        # 市场指数显示
        col1, col2, col3, col4 = st.columns(4)
        
        indices_info = [
            ('^GSPC', '标普500', col1),
            ('^IXIC', '纳斯达克', col2),
            ('^DJI', '道琼斯', col3),
            ('^VIX', 'VIX恐慌', col4)
        ]
        
        for symbol, name, col in indices_info:
            with col:
                if symbol in market_data:
                    price = market_data[symbol]['price']
                    change_pct = market_data[symbol]['change_pct']
                    
                    st.metric(
                        name,
                        f"{price:.2f}" if price < 1000 else f"{price:.0f}",
                        f"{change_pct:+.2f}%"
                    )
                else:
                    st.metric(name, "N/A", "N/A")
        
        # 市场概览图表
        if any(symbol in market_data for symbol in INDICES):
            indices_data = {k: v for k, v in market_data.items() if k in INDICES}
            market_chart = create_market_overview_chart(indices_data)
            if market_chart:
                st.plotly_chart(market_chart, use_container_width=True)
        
        # 市场情绪分析
        st.subheader("📋 市场情绪分析")
        
        if '^VIX' in market_data:
            vix = market_data['^VIX']['price']
            
            if vix < 15:
                st.markdown('<div class="alert-green">🟢 <b>极低恐慌</b>: 市场过度乐观，注意潜在风险</div>', unsafe_allow_html=True)
            elif vix < 20:
                st.markdown('<div class="alert-green">🟢 <b>低恐慌</b>: 市场相对平静，适合持股</div>', unsafe_allow_html=True)
            elif vix < 25:
                st.markdown('<div class="alert-yellow">🟡 <b>中等恐慌</b>: 市场有所担忧，保持谨慎</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-red">🔴 <b>高恐慌</b>: 市场恐慌情绪浓厚，考虑防御</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("📈 实时监控股票")
        
        if not selected_stocks:
            st.warning("请在侧边栏选择要监控的股票")
        else:
            # 创建监控表格
            monitor_data = []
            
            for symbol in selected_stocks:
                if symbol in stock_data:
                    data = stock_data[symbol]
                    signal = signals.get(symbol, {})
                    
                    monitor_data.append({
                        '股票': symbol,
                        '价格': f"${data['price']:.2f}",
                        '涨跌': f"{data['change']:+.2f}",
                        '涨跌幅': f"{data['change_pct']:+.2f}%",
                        'RSI': f"{data['rsi']:.1f}",
                        '52周位置': f"{signal.get('position_52w', 0):.1f}%",
                        '评级': signal.get('rating', 'N/A'),
                        '市值': f"{data['market_cap']/1e9:.1f}B" if data['market_cap'] > 0 else 'N/A',
                        'PE': f"{data['pe_ratio']:.1f}" if data['pe_ratio'] and data['pe_ratio'] > 0 else 'N/A'
                    })
            
            if monitor_data:
                df = pd.DataFrame(monitor_data)
                st.dataframe(df, use_container_width=True)
            
            # 预警信息
            st.subheader("⚠️ 实时预警")
            
            alerts = []
            for symbol in selected_stocks:
                if symbol in stock_data and symbol in signals:
                    data = stock_data[symbol]
                    signal = signals[symbol]
                    
                    # RSI预警
                    if data['rsi'] < 25:
                        alerts.append(f"🟢 {symbol}: RSI严重超卖 ({data['rsi']:.1f})")
                    elif data['rsi'] > 75:
                        alerts.append(f"🔴 {symbol}: RSI严重超买 ({data['rsi']:.1f})")
                    
                    # 价格预警
                    if signal['position_52w'] < 10:
                        alerts.append(f"🟢 {symbol}: 接近52周最低点")
                    elif signal['position_52w'] > 90:
                        alerts.append(f"🔴 {symbol}: 接近52周最高点")
                    
                    # 强烈买入/卖出信号
                    if signal['rating'] == '强烈买入':
                        alerts.append(f"💚 {symbol}: 强烈买入信号")
                    elif signal['rating'] == '强烈卖出':
                        alerts.append(f"❤️ {symbol}: 强烈卖出信号")
            
            if alerts:
                for alert in alerts:
                    st.markdown(f"• {alert}")
            else:
                st.info("暂无重要预警信息")
    
    with tab3:
        st.header("🎯 技术分析")
        
        if not selected_stocks:
            st.warning("请在侧边栏选择要分析的股票")
        else:
            # 选择要分析的股票
            analysis_stock = st.selectbox("选择要分析的股票", selected_stocks)
            
            if analysis_stock and analysis_stock in stock_data:
                data = stock_data[analysis_stock]
                signal = signals.get(analysis_stock, {})
                
                # 基本信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("当前价格", f"${data['price']:.2f}", f"{data['change_pct']:+.2f}%")
                with col2:
                    rsi_color = "🟢" if data['rsi'] < 30 else "🔴" if data['rsi'] > 70 else "🟡"
                    st.metric("RSI", f"{data['rsi']:.1f}", rsi_color)
                with col3:
                    st.metric("20日均线", f"${data['ma_20']:.2f}")
                with col4:
                    st.metric("50日均线", f"${data['ma_50']:.2f}")
                
                # 技术评级
                st.subheader("📊 技术评级")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    rating_color = signal.get('color', '#ffc107')
                    st.markdown(f"""
                    <div style="background-color: {rating_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h3>{signal.get('rating', 'N/A')}</h3>
                        <p>综合评分: {signal.get('score', 0)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**技术信号:**")
                    for sig in signal.get('signals', []):
                        st.write(f"• {sig}")
                
                # 快速入口到深度分析
                st.info(f"💡 想要查看 {analysis_stock} 的完整分析报告？切换到 **🔬 深度分析** 标签页获取基本面、流动性、智能分析等更多信息！")
                
                # 详细图表
                chart = create_stock_chart(analysis_stock, stock_data)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
    
    with tab4:
        st.header("🔬 深度股票分析")
        st.markdown("**专业级股票综合分析系统** - 集成技术面、基本面、流动性、智能分析等7大维度")
        
        if not selected_stocks:
            st.warning("请在侧边栏选择要分析的股票")
        else:
            # 股票选择器
            deep_analysis_stock = st.selectbox("选择要深度分析的股票", selected_stocks, key="deep_analysis_selector")
            
            if deep_analysis_stock:
                # 初始化统一分析器
                if 'unified_analyzer' not in st.session_state:
                    st.session_state.unified_analyzer = UnifiedStockAnalyzer()
                
                # 刷新分析按钮
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("🔄 刷新分析", key="refresh_deep_analysis"):
                        st.cache_data.clear()
                        st.rerun()
                
                with col2:
                    force_refresh = st.checkbox("强制刷新", help="跳过缓存，重新获取所有数据")
                
                # 显示完整的股票分析
                try:
                    with st.spinner(f"正在进行 {deep_analysis_stock} 的深度分析..."):
                        display_stock_analysis(deep_analysis_stock, force_refresh=force_refresh)
                        
                except Exception as e:
                    st.error(f"分析过程中出现错误: {e}")
                    st.info("请检查股票代码是否正确，或稍后重试")
    
    with tab5:
        st.header("💼 投资组合管理")
        
        if not portfolio:
            st.info("请在侧边栏输入持仓信息")
        else:
            # 计算组合价值
            total_value = 0
            total_cost = 0
            portfolio_data = []
            
            for symbol, pos in portfolio.items():
                if symbol in stock_data:
                    current_price = stock_data[symbol]['price']
                    shares = pos['shares']
                    cost_price = pos['cost']
                    
                    current_value = current_price * shares
                    cost_value = cost_price * shares
                    unrealized_pnl = current_value - cost_value
                    pnl_pct = (unrealized_pnl / cost_value * 100) if cost_value > 0 else 0
                    
                    total_value += current_value
                    total_cost += cost_value
                    
                    # 风险管理
                    stop_loss_price = cost_price * (1 - stop_loss_pct / 100)
                    take_profit_price = cost_price * (1 + take_profit_pct / 100)
                    
                    portfolio_data.append({
                        '股票': symbol,
                        '股数': shares,
                        '成本价': f"${cost_price:.2f}",
                        '当前价': f"${current_price:.2f}",
                        '市值': f"${current_value:.2f}",
                        '盈亏': f"${unrealized_pnl:.2f}",
                        '盈亏率': f"{pnl_pct:+.2f}%",
                        '止损价': f"${stop_loss_price:.2f}",
                        '止盈价': f"${take_profit_price:.2f}"
                    })
            
            # 组合总览
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总市值", f"${total_value:.2f}")
            with col2:
                st.metric("总成本", f"${total_cost:.2f}")
            with col3:
                st.metric("总盈亏", f"${total_pnl:.2f}", f"{total_pnl_pct:+.2f}%")
            with col4:
                win_rate = len([x for x in portfolio_data if '+' in x['盈亏率']]) / len(portfolio_data) * 100 if portfolio_data else 0
                st.metric("胜率", f"{win_rate:.1f}%")
            
            # 持仓详情表格
            if portfolio_data:
                st.subheader("📊 持仓详情")
                df = pd.DataFrame(portfolio_data)
                st.dataframe(df, use_container_width=True)
            
            # 风险预警
            st.subheader("⚠️ 风险预警")
            risk_alerts = []
            
            for symbol, pos in portfolio.items():
                if symbol in stock_data:
                    current_price = stock_data[symbol]['price']
                    cost_price = pos['cost']
                    
                    loss_pct = (current_price - cost_price) / cost_price * 100
                    
                    if loss_pct <= -stop_loss_pct:
                        risk_alerts.append(f"🔴 {symbol}: 达到止损线 ({loss_pct:.1f}%)")
                    elif loss_pct >= take_profit_pct:
                        risk_alerts.append(f"🟢 {symbol}: 达到止盈线 ({loss_pct:.1f}%)")
                    elif loss_pct <= -stop_loss_pct * 0.8:
                        risk_alerts.append(f"🟡 {symbol}: 接近止损线 ({loss_pct:.1f}%)")
            
            if risk_alerts:
                for alert in risk_alerts:
                    st.markdown(f"• {alert}")
            else:
                st.success("✅ 当前无风险预警")
    
    # 页面底部信息
    st.markdown("---")
    st.markdown("**💡 使用说明:** 本系统提供实时市场数据和技术分析，仅供参考，投资有风险，决策需谨慎。")
    st.markdown("**🔬 深度分析:** 集成专业级分析系统，提供技术面、基本面、流动性、智能分析等多维度评估。")

if __name__ == "__main__":
    main() 