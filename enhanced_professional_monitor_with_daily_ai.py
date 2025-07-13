#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版专业实时交易监控系统 - 集成AI每日持股分析
Enhanced Professional Real-time Trading Monitor with AI Daily Holdings Analysis
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
import asyncio
import requests
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入AI每日持股分析模块
from ai_realtime_analyzer import AIRealtimeAnalyzer
from daily_holdings_analysis import DailyHoldingsAnalyzer

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="增强版专业交易监控系统",
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
    .ai-analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .daily-analysis-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# 全局配置
INDICES = ['^GSPC', '^IXIC', '^DJI', '^VIX']
UPDATE_INTERVAL = 60  # 秒

# 初始化AI每日持股分析器
@st.cache_resource
def get_ai_daily_analyzer():
    """获取AI每日持股分析器实例"""
    try:
        return AIRealtimeAnalyzer(use_daily_analysis=True)
    except Exception as e:
        st.error(f"AI每日持股分析器初始化失败: {e}")
        return None

@st.cache_resource
def get_daily_analyzer():
    """获取每日持股分析器实例"""
    try:
        return DailyHoldingsAnalyzer()
    except Exception as e:
        st.error(f"每日持股分析器初始化失败: {e}")
        return None

async def get_enhanced_ai_analysis(symbol, stock_data, portfolio_info=None):
    """获取增强AI分析结果（基于每日持股分析）"""
    try:
        ai_analyzer = get_ai_daily_analyzer()
        if not ai_analyzer:
            return {"error": "AI分析器未初始化"}
        
        if symbol not in stock_data:
            return {"error": "股票数据不可用"}
        
        data = stock_data[symbol]
        
        # 准备市场数据
        market_data = {
            'current_price': data['price'],
            'change_pct': data['change_pct'],
            'volume': data['volume'],
            'rsi': data['rsi'],
            'ma_20': data['ma_20'],
            'ma_50': data['ma_50'],
            'volume_ratio': data.get('volume_ratio', 1.0)
        }
        
        # 添加持仓信息
        if portfolio_info:
            market_data['position_info'] = {
                'shares': portfolio_info.get('shares', 0),
                'cost_basis': portfolio_info.get('cost_basis', 0),
                'weight': portfolio_info.get('weight', 0),
                'sector': portfolio_info.get('sector', 'Unknown')
            }
        
        # 调用AI分析
        result = await ai_analyzer.analyze_market_event(
            symbol=symbol,
            event_type="portfolio_position",
            market_data=market_data,
            analysis_type="comprehensive"
        )
        
        if result.get('success'):
            # 转换结果格式
            action_suggestion = result.get('action_suggestion', {})
            ai_analysis = result.get('ai_analysis', '')
            
            return {
                'recommendation': action_suggestion.get('action', '不明确'),
                'reasoning': action_suggestion.get('reason', '无分析理由'),
                'confidence': '高',  # 基于每日持股分析的AI分析置信度更高
                'risk_warnings': action_suggestion.get('risk_warning', '无风险警告'),
                'technical_analysis': f"RSI: {data['rsi']:.1f}, MA20: ${data['ma_20']:.2f}, MA50: ${data['ma_50']:.2f}",
                'raw_response': ai_analysis,
                'daily_analysis_data': result.get('daily_analysis', {}),
                'model_used': result.get('model_used', 'Unknown')
            }
        else:
            return {"error": result.get('error', 'AI分析失败')}
        
    except Exception as e:
        return {"error": f"AI分析失败: {str(e)}"}

@st.cache_data(ttl=60)  # 缓存1分钟
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
                    'stop_loss': position.get('stop_loss', 0),
                    'target_price': position.get('target_price', 0),
                    'notes': position.get('notes', ''),
                    'technical_analysis': position.get('technical_analysis', {})
                }
        
        # 提取观察仓股票 - 将字典转换为列表
        watchlist_dict = config.get('watchlist', {})
        watchlist = list(watchlist_dict.keys()) if isinstance(watchlist_dict, dict) else watchlist_dict
        
        return {
            'current_positions': current_positions,
            'portfolio_info': portfolio_info,
            'watchlist': watchlist,
            'config': config
        }
    except Exception as e:
        st.error(f"加载配置文件失败: {e}")
        return {
            'current_positions': [],
            'portfolio_info': {},
            'watchlist': [],
            'config': {}
        }

@st.cache_data(ttl=60)  # 缓存1分钟
def get_realtime_data(symbols):
    """获取实时市场数据"""
    data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d', interval='1d')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                # 计算技术指标
                rsi = calculate_rsi(hist['Close'])
                ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                
                # 计算成交量比率
                current_volume = hist['Volume'].iloc[-1]
                avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                data[symbol] = {
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': current_volume,
                    'volume_ratio': volume_ratio,
                    'rsi': rsi,
                    'ma_20': ma_20,
                    'ma_50': ma_50,
                    'high': hist['High'].iloc[-1],
                    'low': hist['Low'].iloc[-1],
                    'open': hist['Open'].iloc[-1]
                }
        except Exception as e:
            st.warning(f"获取 {symbol} 数据失败: {e}")
    
    return data

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    try:
        if len(prices) < period + 1:
            return 50
        
        delta = prices.diff().dropna()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    except:
        return 50

def calculate_technical_signals(data):
    """计算技术信号"""
    signals = {}
    
    for symbol, stock_data in data.items():
        signals[symbol] = {
            'rsi_signal': '超买' if stock_data['rsi'] > 70 else '超卖' if stock_data['rsi'] < 30 else '中性',
            'ma_signal': '多头' if stock_data['price'] > stock_data['ma_20'] > stock_data['ma_50'] else '空头' if stock_data['price'] < stock_data['ma_20'] < stock_data['ma_50'] else '震荡',
            'volume_signal': '放量' if stock_data['volume_ratio'] > 1.5 else '缩量' if stock_data['volume_ratio'] < 0.5 else '正常',
            'trend': '上涨' if stock_data['change_pct'] > 0 else '下跌'
        }
    
    return signals

def create_market_overview_chart(indices_data):
    """创建市场概览图表"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['标普500', '纳斯达克', '道琼斯', 'VIX恐慌指数'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (symbol, data) in enumerate(indices_data.items()):
        if data:
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=[datetime.now()],
                    y=[data['price']],
                    mode='markers+text',
                    text=[f"${data['price']:.2f}"],
                    textposition="top center",
                    name=symbol,
                    marker=dict(size=15, color=colors[i]),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # 添加涨跌幅
            color = 'green' if data['change_pct'] > 0 else 'red'
            fig.add_annotation(
                x=datetime.now(),
                y=data['price'],
                text=f"{data['change_pct']:+.2f}%",
                showarrow=False,
                font=dict(color=color, size=12),
                yshift=30,
                row=row, col=col
            )
    
    fig.update_layout(
        height=400,
        title_text="市场指数概览",
        showlegend=False
    )
    
    return fig

def create_stock_chart(symbol, data):
    """创建个股图表"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f'{symbol} 价格走势', '成交量'],
        row_width=[0.7, 0.3]
    )
    
    # 价格K线图
    fig.add_trace(
        go.Candlestick(
            x=[datetime.now()],
            open=[data['open']],
            high=[data['high']],
            low=[data['low']],
            close=[data['price']],
            name=symbol
        ),
        row=1, col=1
    )
    
    # 均线
    fig.add_trace(
        go.Scatter(
            x=[datetime.now()],
            y=[data['ma_20']],
            mode='markers',
            name='MA20',
            marker=dict(color='orange', size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[datetime.now()],
            y=[data['ma_50']],
            mode='markers',
            name='MA50',
            marker=dict(color='red', size=8)
        ),
        row=1, col=1
    )
    
    # 成交量
    fig.add_trace(
        go.Bar(
            x=[datetime.now()],
            y=[data['volume']],
            name='成交量',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        title_text=f"{symbol} 技术分析",
        xaxis_rangeslider_visible=False
    )
    
    return fig

def get_ollama_models():
    """获取可用的Ollama模型"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models.get('models', [])]
        return []
    except:
        return []

def display_daily_analysis_summary():
    """显示每日持股分析摘要"""
    try:
        daily_analyzer = get_daily_analyzer()
        if not daily_analyzer:
            return
        
        # 获取每日分析数据
        all_symbols = list(daily_analyzer.portfolio.keys()) + daily_analyzer.market_indices + daily_analyzer.watchlist
        all_symbols = list(set(all_symbols))
        
        data = daily_analyzer.get_today_data(all_symbols)
        
        if data:
            # 分析投资组合表现
            portfolio_analysis = daily_analyzer.analyze_portfolio_performance(data)
            
            st.markdown('<div class="daily-analysis-card">', unsafe_allow_html=True)
            st.markdown("### 📊 每日持股分析摘要")
            st.markdown(f"**更新时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 显示组合总览
            if isinstance(portfolio_analysis, list) and len(portfolio_analysis) > 0:
                total_value = sum(item.get('current_value', 0) for item in portfolio_analysis)
                total_cost = sum(item.get('cost_value', 0) for item in portfolio_analysis)
                total_pnl = total_value - total_cost
                total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总市值", f"${total_value:,.2f}")
                with col2:
                    st.metric("总成本", f"${total_cost:,.2f}")
                with col3:
                    st.metric("总盈亏", f"${total_pnl:+,.2f}")
                with col4:
                    st.metric("盈亏率", f"{total_pnl_pct:+.2f}%")
                
                # 显示个股表现
                st.markdown("### 📈 个股表现")
                portfolio_df = []
                for item in portfolio_analysis:
                    if isinstance(item, dict) and 'symbol' in item:
                        portfolio_df.append({
                            '股票': item['symbol'],
                            '现价': f"${item.get('current_price', 0):.2f}",
                            '盈亏': f"${item.get('unrealized_pnl', 0):+,.2f}",
                            '盈亏率': f"{item.get('pnl_pct', 0):+.2f}%",
                            'RSI': f"{item.get('rsi', 0):.1f}",
                            '建议': item.get('suggestion', 'N/A')
                        })
                
                if portfolio_df:
                    df = pd.DataFrame(portfolio_df)
                    st.dataframe(df, use_container_width=True)
            
            # 显示市场环境
            if '^VIX' in data:
                vix_value = data['^VIX']['price']
                vix_analysis = ""
                if vix_value < 15:
                    vix_analysis = "市场恐慌情绪低，风险偏好较高"
                elif vix_value < 25:
                    vix_analysis = "市场恐慌情绪正常"
                else:
                    vix_analysis = "市场恐慌情绪较高，需要谨慎"
                
                st.markdown("### 🌍 市场环境")
                st.info(f"VIX恐慌指数: {vix_value:.2f} - {vix_analysis}")
    
    except Exception as e:
        st.warning(f"获取每日分析摘要失败: {e}")

def main():
    # 页面标题
    st.markdown('<h1 class="main-header">🚀 增强版专业实时交易监控系统</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">集成AI每日持股分析的智能投资决策平台</p>', unsafe_allow_html=True)
    
    # 侧边栏配置
    st.sidebar.header("⚙️ 系统配置")
    
    # 加载配置
    config = load_portfolio_config()
    current_positions = config['current_positions']
    portfolio_info = config['portfolio_info']
    watchlist = config['watchlist']
    
    # 股票选择
    st.sidebar.markdown("### 📊 股票选择")
    
    # 合并所有股票
    all_stocks = list(set(current_positions + watchlist))
    
    if not all_stocks:
        st.sidebar.warning("未找到股票配置，请检查portfolio_config.json文件")
        return
    
    # 多选股票
    selected_stocks = st.sidebar.multiselect(
        "选择要监控的股票",
        all_stocks,
        default=current_positions[:5] if current_positions else all_stocks[:5]
    )
    
    if not selected_stocks:
        st.warning("请在侧边栏选择要监控的股票")
        return
    
    # 获取实时数据
    stock_data = get_realtime_data(selected_stocks)
    
    if not stock_data:
        st.error("无法获取股票数据，请检查网络连接")
        return
    
    # 计算技术信号
    technical_signals = calculate_technical_signals(stock_data)
    
    # 获取指数数据
    indices_data = get_realtime_data(INDICES)
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 市场概览", 
        "📈 监控股票", 
        "💼 投资组合", 
        "🤖 AI每日持股分析", 
        "📋 每日分析摘要",
        "⚙️ 系统状态"
    ])
    
    # 标签页1: 市场概览
    with tab1:
        st.header("📊 市场概览")
        
        # 市场指数图表
        if indices_data:
            fig = create_market_overview_chart(indices_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # 市场指标
        col1, col2, col3, col4 = st.columns(4)
        
        if '^GSPC' in indices_data:
            sp500_data = indices_data['^GSPC']
            with col1:
                st.metric(
                    "标普500", 
                    f"${sp500_data['price']:,.2f}",
                    f"{sp500_data['change_pct']:+.2f}%"
                )
        
        if '^IXIC' in indices_data:
            nasdaq_data = indices_data['^IXIC']
            with col2:
                st.metric(
                    "纳斯达克", 
                    f"${nasdaq_data['price']:,.2f}",
                    f"{nasdaq_data['change_pct']:+.2f}%"
                )
        
        if '^DJI' in indices_data:
            dow_data = indices_data['^DJI']
            with col3:
                st.metric(
                    "道琼斯", 
                    f"${dow_data['price']:,.2f}",
                    f"{dow_data['change_pct']:+.2f}%"
                )
        
        if '^VIX' in indices_data:
            vix_data = indices_data['^VIX']
            with col4:
                st.metric(
                    "VIX恐慌指数", 
                    f"{vix_data['price']:.2f}",
                    f"{vix_data['change_pct']:+.2f}%"
                )
    
    # 标签页2: 监控股票
    with tab2:
        st.header("📈 监控股票")
        
        # 股票数据表格
        stock_df = []
        for symbol in selected_stocks:
            if symbol in stock_data:
                data = stock_data[symbol]
                signals = technical_signals.get(symbol, {})
                
                stock_df.append({
                    '股票': symbol,
                    '现价': f"${data['price']:.2f}",
                    '涨跌幅': f"{data['change_pct']:+.2f}%",
                    '成交量': f"{data['volume']:,}",
                    '成交量比': f"{data['volume_ratio']:.1f}x",
                    'RSI': f"{data['rsi']:.1f}",
                    'MA20': f"${data['ma_20']:.2f}",
                    'MA50': f"${data['ma_50']:.2f}",
                    'RSI信号': signals.get('rsi_signal', 'N/A'),
                    '均线信号': signals.get('ma_signal', 'N/A'),
                    '成交量信号': signals.get('volume_signal', 'N/A')
                })
        
        if stock_df:
            df = pd.DataFrame(stock_df)
            st.dataframe(df, use_container_width=True)
        
        # 个股图表
        if selected_stocks:
            selected_stock = st.selectbox("选择股票查看详细图表", selected_stocks)
            if selected_stock in stock_data:
                fig = create_stock_chart(selected_stock, stock_data[selected_stock])
                st.plotly_chart(fig, use_container_width=True)
    
    # 标签页3: 投资组合
    with tab3:
        st.header("💼 投资组合")
        
        if not current_positions:
            st.info("当前没有持仓")
        else:
            # 投资组合概览
            total_value = 0
            total_cost = 0
            portfolio_df = []
            
            for symbol in current_positions:
                if symbol in stock_data and symbol in portfolio_info:
                    data = stock_data[symbol]
                    info = portfolio_info[symbol]
                    
                    shares = info['shares']
                    cost_basis = info['cost_basis']
                    
                    current_value = data['price'] * shares
                    cost_value = cost_basis * shares
                    unrealized_pnl = current_value - cost_value
                    pnl_pct = (unrealized_pnl / cost_value) * 100 if cost_value > 0 else 0
                    
                    total_value += current_value
                    total_cost += cost_value
                    
                    portfolio_df.append({
                        '股票': symbol,
                        '持股': shares,
                        '成本': f"${cost_basis:.2f}",
                        '现价': f"${data['price']:.2f}",
                        '市值': f"${current_value:,.2f}",
                        '盈亏': f"${unrealized_pnl:+,.2f}",
                        '盈亏率': f"{pnl_pct:+.2f}%",
                        '权重': f"{info['weight']:.1f}%",
                        '行业': info['sector']
                    })
            
            if portfolio_df:
                # 总览指标
                total_pnl = total_value - total_cost
                total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总市值", f"${total_value:,.2f}")
                with col2:
                    st.metric("总成本", f"${total_cost:,.2f}")
                with col3:
                    st.metric("总盈亏", f"${total_pnl:+,.2f}")
                with col4:
                    st.metric("总盈亏率", f"{total_pnl_pct:+.2f}%")
                
                # 持仓详情
                st.markdown("### 📋 持仓详情")
                df = pd.DataFrame(portfolio_df)
                st.dataframe(df, use_container_width=True)
    
    # 标签页4: AI每日持股分析
    with tab4:
        st.header("🤖 AI每日持股分析")
        st.markdown("**基于每日持股分析结果的AI智能诊断系统**")
        
        # 股票分类选择
        st.markdown("### 📊 股票选择")
        
        # 获取持仓股票和观察仓股票
        position_stocks = list(portfolio_info.keys()) if portfolio_info else []
        watchlist_stocks = list(watchlist.keys()) if watchlist else []
        
        # 创建股票选择界面
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💼 持仓股票")
            if position_stocks:
                selected_positions = st.multiselect(
                    "选择持仓股票进行AI分析",
                    position_stocks,
                    default=position_stocks[:3] if len(position_stocks) > 3 else position_stocks,
                    help="选择您当前持有的股票进行AI分析"
                )
            else:
                st.info("当前没有持仓股票")
                selected_positions = []
        
        with col2:
            st.markdown("#### 👀 观察仓股票")
            if watchlist_stocks:
                selected_watchlist = st.multiselect(
                    "选择观察仓股票进行AI分析",
                    watchlist_stocks,
                    default=watchlist_stocks[:3] if len(watchlist_stocks) > 3 else watchlist_stocks,
                    help="选择您关注的观察仓股票进行AI分析"
                )
            else:
                st.info("当前没有观察仓股票")
                selected_watchlist = []
        
        # 合并选中的股票
        all_selected_stocks = selected_positions + selected_watchlist
        
        if not all_selected_stocks:
            st.warning("请选择要分析的股票")
        else:
            # AI分析控制
            st.markdown("### ⚙️ AI分析控制")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                analysis_type = st.selectbox(
                    "分析类型",
                    ["comprehensive", "detailed", "quick"],
                    format_func=lambda x: {
                        "comprehensive": "综合分析",
                        "detailed": "详细分析", 
                        "quick": "快速分析"
                    }[x],
                    help="选择AI分析的深度和详细程度"
                )
            
            with col2:
                if st.button("🔍 批量AI分析", type="primary"):
                    st.session_state.batch_analysis = True
                    st.session_state.analysis_type = analysis_type
            
            with col3:
                if st.button("🔄 刷新数据"):
                    st.rerun()
            
            # 显示市场数据表格
            st.markdown("### 📈 实时市场数据")
            
            market_df = []
            for symbol in all_selected_stocks:
                if symbol in stock_data:
                    data = stock_data[symbol]
                    position_info = portfolio_info.get(symbol, {})
                    watchlist_info = watchlist.get(symbol, {})
                    
                    # 判断是持仓还是观察仓
                    stock_type = "持仓" if symbol in selected_positions else "观察仓"
                    shares = position_info.get('shares', 0)
                    cost_basis = position_info.get('cost_basis', 0)
                    target_price = watchlist_info.get('target_buy_price', 0)
                    
                    if shares > 0:
                        current_value = data['price'] * shares
                        cost_value = cost_basis * shares
                        unrealized_pnl = current_value - cost_value
                        pnl_pct = (unrealized_pnl / cost_value) * 100 if cost_value > 0 else 0
                        
                        market_df.append({
                            '股票': symbol,
                            '类型': stock_type,
                            '现价': f"${data['price']:.2f}",
                            '涨跌幅': f"{data['change_pct']:+.2f}%",
                            '持股': shares,
                            '成本': f"${cost_basis:.2f}",
                            '市值': f"${current_value:,.2f}",
                            '盈亏': f"${unrealized_pnl:+,.2f}",
                            '盈亏率': f"{pnl_pct:+.2f}%",
                            '权重': f"{position_info.get('weight', 0):.1f}%"
                        })
                    else:
                        # 观察仓股票
                        market_df.append({
                            '股票': symbol,
                            '类型': stock_type,
                            '现价': f"${data['price']:.2f}",
                            '涨跌幅': f"{data['change_pct']:+.2f}%",
                            '目标价': f"${target_price:.2f}" if target_price > 0 else "N/A",
                            '价差': f"${data['price'] - target_price:+.2f}" if target_price > 0 else "N/A",
                            '价差率': f"{(data['price'] - target_price) / target_price * 100:+.2f}%" if target_price > 0 else "N/A",
                            '持股': 0,
                            '成本': "N/A",
                            '市值': "N/A",
                            '盈亏': "N/A",
                            '盈亏率': "N/A",
                            '权重': "N/A"
                        })
            
            if market_df:
                df = pd.DataFrame(market_df)
                st.dataframe(df, use_container_width=True)
            
            # AI分析结果
            st.markdown("### 🤖 AI分析结果")
            
            # 检查是否需要进行批量分析
            if st.session_state.get('batch_analysis', False):
                st.session_state.batch_analysis = False
                
                # 为每个选中的股票进行AI分析
                for symbol in all_selected_stocks:
                    if symbol in stock_data:
                        data = stock_data[symbol]
                        
                        # 添加持仓信息
                        position_info = portfolio_info.get(symbol, {})
                        watchlist_info = watchlist.get(symbol, {})
                        
                        if position_info.get('shares', 0) > 0:
                            data['position_info'] = {
                                'shares': position_info.get('shares', 0),
                                'cost_basis': position_info.get('cost_basis', 0),
                                'weight': position_info.get('weight', 0),
                                'sector': position_info.get('sector', 'Unknown')
                            }
                        elif watchlist_info:
                            data['watchlist_info'] = {
                                'target_buy_price': watchlist_info.get('target_buy_price', 0),
                                'reason': watchlist_info.get('reason', ''),
                                'category': watchlist_info.get('category', 'Unknown')
                            }
                        
                        # 执行AI分析
                        with st.spinner(f"正在分析 {symbol}..."):
                            try:
                                # 获取AI分析器
                                ai_analyzer = get_ai_daily_analyzer()
                                
                                # 执行AI分析
                                ai_result = asyncio.run(get_enhanced_ai_analysis(symbol, {symbol: data}, portfolio_info))
                                
                                if ai_result and not ai_result.get('error'):
                                    # 显示AI分析结果
                                    st.success(f"✅ {symbol} AI分析完成")
                                    
                                    # 创建可展开的分析结果
                                    with st.expander(f"📊 {symbol} 详细分析", expanded=True):
                                        # 操作建议
                                        recommendation = ai_result.get('recommendation', 'N/A')
                                        reasoning = ai_result.get('reasoning', '无分析理由')
                                        confidence = ai_result.get('confidence', 'N/A')
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("操作建议", recommendation)
                                        with col2:
                                            st.metric("置信度", confidence)
                                        with col3:
                                            st.metric("技术指标", ai_result.get('technical_analysis', 'N/A'))
                                        
                                        # 详细分析
                                        st.markdown("#### 📋 分析理由")
                                        st.write(reasoning)
                                        
                                        # 风险提示
                                        risk_warnings = ai_result.get('risk_warnings', '无风险警告')
                                        if risk_warnings and risk_warnings != '无风险警告':
                                            st.warning(f"⚠️ **风险提示**: {risk_warnings}")
                                        
                                        # 原始AI响应
                                        raw_response = ai_result.get('raw_response', '')
                                        if raw_response:
                                            st.markdown("#### 🤖 AI原始分析")
                                            st.markdown(raw_response)
                                        
                                        # 每日分析数据
                                        daily_data = ai_result.get('daily_analysis_data', {})
                                        if daily_data:
                                            st.markdown("#### 📊 每日分析数据")
                                            st.json(daily_data)
                                    
                                    # 保存到历史记录
                                    if 'analysis_history' not in st.session_state:
                                        st.session_state.analysis_history = []
                                    
                                    st.session_state.analysis_history.append({
                                        'symbol': symbol,
                                        'timestamp': datetime.now(),
                                        'result': ai_result,
                                        'type': 'position' if symbol in selected_positions else 'watchlist'
                                    })
                                    
                                else:
                                    error_msg = ai_result.get('error', '未知错误') if ai_result else 'AI分析失败'
                                    st.error(f"❌ {symbol} AI分析失败: {error_msg}")
                                    
                            except Exception as e:
                                st.error(f"❌ {symbol} AI分析出错: {e}")
            
            # 显示分析历史记录
            if 'analysis_history' in st.session_state and st.session_state.analysis_history:
                st.markdown("### 📚 分析历史记录")
                
                # 筛选最近的分析记录
                recent_history = st.session_state.analysis_history[-10:]  # 显示最近10条
                
                for record in recent_history:
                    timestamp = record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    symbol = record['symbol']
                    result = record['result']
                    record_type = record['type']
                    
                    # 获取操作建议
                    recommendation = result.get('recommendation', 'N/A') if result else 'N/A'
                    
                    # 显示记录
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.write(f"**{timestamp}** - {symbol}")
                    with col2:
                        st.write(f"类型: {record_type}")
                    with col3:
                        st.write(f"建议: {recommendation}")
                    with col4:
                        if st.button(f"查看详情", key=f"view_{symbol}_{timestamp}"):
                            st.json(result)
                    
                    st.divider()
            
            # 快速分析按钮
            st.markdown("### ⚡ 快速分析")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 分析持仓股票", help="快速分析所有持仓股票"):
                    for symbol in selected_positions:
                        if symbol in stock_data:
                            # 这里可以添加快速分析逻辑
                            st.info(f"快速分析 {symbol} - 功能开发中...")
            
            with col2:
                if st.button("👀 分析观察仓", help="快速分析所有观察仓股票"):
                    for symbol in selected_watchlist:
                        if symbol in stock_data:
                            # 这里可以添加快速分析逻辑
                            st.info(f"快速分析 {symbol} - 功能开发中...")
            
            with col3:
                if st.button("🧹 清空历史", help="清空分析历史记录"):
                    if 'analysis_history' in st.session_state:
                        st.session_state.analysis_history = []
                    st.success("历史记录已清空")
                    st.rerun()
    
    # 标签页5: 每日分析摘要
    with tab5:
        st.header("📋 每日分析摘要")
        display_daily_analysis_summary()
    
    # 标签页6: 系统状态
    with tab6:
        st.header("⚙️ 系统状态")
        
        # 系统信息
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 系统组件状态")
            
            # AI分析器状态
            ai_analyzer = get_ai_daily_analyzer()
            if ai_analyzer:
                st.success("✅ AI每日持股分析器")
            else:
                st.error("❌ AI每日持股分析器")
            
            # 每日分析器状态
            daily_analyzer = get_daily_analyzer()
            if daily_analyzer:
                st.success("✅ 每日持股分析器")
            else:
                st.error("❌ 每日持股分析器")
            
            # 数据获取状态
            if stock_data:
                st.success(f"✅ 实时数据获取 ({len(stock_data)} 只股票)")
            else:
                st.error("❌ 实时数据获取")
        
        with col2:
            st.markdown("### 📊 数据统计")
            
            st.metric("监控股票数", len(selected_stocks))
            st.metric("持仓股票数", len(current_positions))
            st.metric("观察股票数", len(watchlist))
            st.metric("指数监控数", len(INDICES))
        
        # AI模型信息
        st.markdown("### 🤖 AI模型信息")
        models = get_ollama_models()
        if models:
            st.success(f"✅ 检测到 {len(models)} 个可用模型")
            for model in models:
                st.info(f"📋 {model}")
        else:
            st.warning("⚠️ 未检测到可用模型，请确保Ollama服务正在运行")
        
        # 系统时间
        st.markdown("### ⏰ 系统时间")
        st.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 刷新按钮
        if st.button("🔄 刷新系统状态"):
            st.rerun()
    
    # 页面底部信息
    st.markdown("---")
    st.markdown("**💡 使用说明:** 本系统提供实时市场数据和技术分析，集成AI每日持股分析功能，仅供参考，投资有风险，决策需谨慎。")
    st.markdown("**🔬 深度分析:** 集成每日持股分析系统，提供技术面、基本面、流动性、智能分析等多维度评估。")
    st.markdown("**🧠 决策支持:** 基于每日持股分析结果的AI智能诊断，帮助您做出更明智的投资决策。")

    # AI模型选择器
    st.sidebar.markdown("### 🤖 选择AI模型")
    models = get_ollama_models()
    if not models:
        st.sidebar.warning("未检测到本地模型，请先运行ollama serve并下载模型。")
    else:
        default_model = st.session_state.get("ai_model", models[0])
        selected_model = st.sidebar.selectbox("可用模型", models, index=models.index(default_model) if default_model in models else 0)
        st.session_state["ai_model"] = selected_model
        if st.sidebar.button("🔄 刷新模型列表"):
            st.experimental_rerun()
        st.sidebar.info(f"当前模型：{selected_model}")

if __name__ == "__main__":
    main() 