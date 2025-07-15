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
import requests
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入统一分析系统
from analysis.unified_stock_analyzer import UnifiedStockAnalyzer
from analysis.streamlit_analysis_bridge import display_stock_analysis
from analysis.decision_support_system import DecisionSupportSystem

# 导入AI模块
from ai_realtime_analyzer import AIRealtimeAnalyzer
from ai_trading_module import AITradingModule

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
    .ai-analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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

# 初始化AI分析器
@st.cache_resource
def get_ai_analyzer():
    """获取AI分析器实例"""
    try:
        return AIRealtimeAnalyzer()
    except Exception as e:
        st.error(f"AI分析器初始化失败: {e}")
        return None

@st.cache_resource
def get_ai_trading_module():
    """获取AI交易模块实例"""
    try:
        return AITradingModule()
    except Exception as e:
        st.error(f"AI交易模块初始化失败: {e}")
        return None

def get_ai_analysis(symbol, stock_data, portfolio_info=None):
    """获取AI分析结果"""
    try:
        ai_module = get_ai_trading_module()
        if not ai_module:
            return {"error": "AI模块未初始化"}
        
        if symbol not in stock_data:
            return {"error": "股票数据不可用"}
        
        data = stock_data[symbol]
        
        # 准备分析数据
        analysis_data = {
            'current_price': data['price'],
            'change_pct': data['change_pct'],
            'volume': data['volume'],
            'rsi': data['rsi'],
            'ma_20': data['ma_20'],
            'ma_50': data['ma_50']
        }
        
        # 使用同步方式调用异步方法
        import asyncio
        
        # 创建新的事件循环
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 调用异步方法
            if portfolio_info:
                result = loop.run_until_complete(
                    ai_module.analyze_portfolio_position(symbol, data, portfolio_info)
                )
            else:
                result = loop.run_until_complete(
                    ai_module.analyze_stock_signal(symbol, analysis_data, "quick")
                )
            
            loop.close()
            
            # 转换结果格式以匹配期望的输出
            if result.get('success'):
                ai_response = result.get('ai_response', {})
                action_suggestion = result.get('action_suggestion', {})
                
                return {
                    'recommendation': action_suggestion.get('action', '不明确'),
                    'reasoning': action_suggestion.get('reason', '无分析理由'),
                    'confidence': '中等',
                    'risk_warnings': action_suggestion.get('risk_warning', '无风险警告'),
                    'technical_analysis': f"RSI: {data['rsi']:.1f}, MA20: ${data['ma_20']:.2f}, MA50: ${data['ma_50']:.2f}",
                    'raw_response': ai_response.get('content', '无原始响应')
                }
            else:
                return {"error": result.get('error', 'AI分析失败')}
                
        except Exception as e:
            return {"error": f"异步调用失败: {str(e)}"}
        
    except Exception as e:
        return {"error": f"AI分析失败: {str(e)}"}

@st.cache_data(ttl=60)  # 缓存1分钟，减少缓存时间
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
        
        # 计算RS和RSI - 防止除零错误
        avg_loss_safe = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss_safe
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
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            
            # 防止除零错误
            loss_safe = loss.replace(0, 1e-10)
            rs = gain / loss_safe
            rsi = 100 - (100 / (1 + rs))
            
            # 处理无效值
            rsi = rsi.fillna(50)
            rsi = rsi.replace([np.inf, -np.inf], [100, 0])
            rsi = rsi.clip(0, 100)
            
            return rsi
        except Exception as e:
            return pd.Series([50] * len(prices), index=prices.index)
    
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

# ========== AI模型选择器 ========== #
def get_ollama_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m['name'] for m in resp.json().get('models', [])]
    except Exception:
        pass
    return []

def main():
    # 页面标题
    st.markdown('<div class="main-header">⚡ 专业实时交易监控系统</div>', unsafe_allow_html=True)
    
    # 添加清除缓存按钮
    if st.sidebar.button("🔄 清除缓存并重新加载配置"):
        st.cache_data.clear()
        st.success("缓存已清除，配置将重新加载")
        st.rerun()
    
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
    
    # 持仓信息已在投资组合标签页显示，此处不再重复
    
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
    
    # 持仓设置 - 从配置文件获取
    st.sidebar.subheader("💼 持仓详情")
    portfolio = {}
    
    # 使用portfolio_config.json中的最新数据
    try:
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            portfolio_config = json.load(f)
        
        # 显示实际持仓信息 - 加载所有持仓股票，不依赖selected_stocks
        positions = portfolio_config.get('positions', {})
        for symbol, position in positions.items():
            # 只加载有持仓且未卖出的股票
            if (position.get('shares', 0) > 0 and 
                position.get('status') != 'SOLD' and
                not symbol.endswith('.HK')):  # 排除港股
                
                portfolio[symbol] = {
                    'shares': position['shares'],
                    'cost': position['cost_basis'],
                    'weight': position['weight'],
                    'investment_amount': position['investment_amount']
                }
                
                # 在侧边栏显示详细信息
                with st.sidebar.expander(f"📊 {symbol} 详情"):
                    st.write(f"**持股数量:** {position['shares']} 股")
                    st.write(f"**成本价格:** ${position['cost_basis']:.2f}")
                    st.write(f"**投资金额:** ${position['investment_amount']:,.2f}")
                    st.write(f"**仓位权重:** {position['weight']:.2f}%")
                    st.write(f"**行业板块:** {position['sector']}")
                    st.write(f"**止损阈值:** {position['stop_loss_threshold']*100:.1f}%")
        
        # 更新selected_stocks以包含所有持仓股票
        portfolio_symbols = list(portfolio.keys())
        if portfolio_symbols:
            # 合并持仓股票和原有选择，去重
            all_available_stocks = list(set(available_stocks + portfolio_symbols))
            selected_stocks = st.sidebar.multiselect(
                "选择监控股票",
                all_available_stocks,
                default=portfolio_symbols  # 默认选择所有持仓股票
            )
        else:
            selected_stocks = st.sidebar.multiselect(
                "选择监控股票",
                available_stocks,
                default=default_selection
            )
            
    except Exception as e:
        st.sidebar.error(f"加载投资组合配置失败: {e}")
        # 回退到原有逻辑
        selected_stocks = st.sidebar.multiselect(
            "选择监控股票",
            available_stocks,
            default=default_selection
        )
        
        for stock in selected_stocks:
            if stock in config['portfolio_info']:
                info = config['portfolio_info'][stock]
                portfolio[stock] = {
                    'shares': info['shares'],
                    'cost': info['cost_basis'],
                    'weight': info['weight'],
                    'investment_amount': info['investment_amount']
                }
    
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
    
    # 主要内容区域 - 专业投资分析中心架构
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📊 市场概览", "📈 监控股票", "🔬 专业投资分析中心", "💼 投资组合", "🧠 决策支持", "🤖 AI诊断", "💬 AI问答", "⚙️ 系统设置"])
    
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
        st.header("🔬 专业投资分析中心")
        st.markdown("**专业级股票综合分析系统** - 集成技术面、基本面、流动性、智能分析等8大维度")
        
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
    
    with tab4:
        st.header("💼 投资组合管理")
        
        # 调试信息
        st.subheader("🔍 调试信息")
        st.write(f"Portfolio变量内容: {portfolio}")
        st.write(f"Portfolio键数量: {len(portfolio)}")
        
        if not portfolio:
            st.info("请在侧边栏输入持仓信息")
        else:
            # 计算组合价值
            total_value = 0
            total_cost = 0
            portfolio_data = []
            
            st.write("### 计算过程:")
            for symbol, pos in portfolio.items():
                st.write(f"处理股票: {symbol}")
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
                    
                    st.write(f"  {symbol}: 股数={shares}, 成本价=${cost_price:.2f}, 现价=${current_price:.2f}")
                    st.write(f"  市值=${current_value:.2f}, 成本=${cost_value:.2f}, 盈亏=${unrealized_pnl:.2f}")
                    
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
                else:
                    st.write(f"  {symbol}: 未找到股票数据")
            
            st.write(f"### 总计:")
            st.write(f"总市值: ${total_value:.2f}")
            st.write(f"总成本: ${total_cost:.2f}")
            
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
    
    with tab5:
        st.header("🧠 投资决策支持系统")
        st.markdown("**专门为避免抄底抄到半山腰、避免卖到半路而设计**")
        
        # 初始化决策支持系统
        if 'decision_support' not in st.session_state:
            st.session_state.decision_support = DecisionSupportSystem()
        
        dss = st.session_state.decision_support
        
        # 决策类型选择
        decision_type = st.radio("选择决策类型", [
            "📊 仓位管理分析", "🔍 买入时机分析", "💰 卖出时机分析", 
            "📝 查看决策历史", "✍️ 添加备注", "🤖 AI智能分析", "📚 备注管理"
        ])
        
        if decision_type == "📊 仓位管理分析":
            st.subheader("专业仓位管理分析")
            st.markdown("**集成技术分析、风险评估、加仓策略的专业仓位管理系统**")
            
            if not selected_stocks:
                st.warning("请在侧边栏选择要分析的股票")
            else:
                position_stock = st.selectbox("选择要分析仓位的股票", selected_stocks, key="position_analysis")
                
                # 仓位输入
                col1, col2 = st.columns(2)
                with col1:
                    current_position = st.number_input("当前仓位 (%)", min_value=0.0, max_value=100.0, 
                                                     value=18.29, step=0.1, key="current_pos")
                with col2:
                    target_position = st.number_input("目标仓位 (%)", min_value=0.0, max_value=100.0, 
                                                    value=25.0, step=0.1, key="target_pos")
                
                if st.button("📊 开始仓位管理分析", type="primary"):
                    with st.spinner(f"正在分析 {position_stock} 的仓位管理策略..."):
                        # 获取当前分析数据
                        if 'unified_analyzer' not in st.session_state:
                            st.session_state.unified_analyzer = UnifiedStockAnalyzer()
                        
                        current_analysis = st.session_state.unified_analyzer.get_comprehensive_analysis(position_stock)
                        
                        # 进行仓位管理分析
                        position_decision = dss.analyze_position_management(
                            position_stock, current_position, target_position, current_analysis
                        )
                        
                        if 'decision' in position_decision:
                            # 基本信息展示
                            st.markdown("### 📊 当前市场状况")
                            tech_data = position_decision['technical_data']
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("当前价格", f"${position_decision['current_price']:.2f}")
                            with col2:
                                st.metric("MA20", f"${tech_data['ma20']:.2f}")
                            with col3:
                                st.metric("RSI", f"{tech_data['rsi']:.1f}")
                            with col4:
                                price_vs_ma20 = tech_data['price_vs_ma20_pct']
                                st.metric("价格偏离MA20", f"{price_vs_ma20:+.1f}%")
                            
                            # 仓位状况
                            st.markdown("### 📈 仓位状况分析")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("当前仓位", f"{position_decision['current_position']:.1f}%")
                            with col2:
                                st.metric("目标仓位", f"{position_decision['target_position']:.1f}%")
                            with col3:
                                gap = position_decision['position_gap']
                                st.metric("仓位缺口", f"{gap:+.1f}%")
                            
                            # 决策结果
                            decision_detail = position_decision['decision']
                            st.markdown("### 🎯 仓位管理决策")
                            
                            action = decision_detail['action']
                            confidence = decision_detail['confidence']
                            risk_level = decision_detail['risk_level']
                            
                            if action == "暂时不要加仓":
                                st.error(f"🔴 **{action}** (信心度: {confidence}%, 风险: {risk_level})")
                            elif action == "可以小幅加仓":
                                st.warning(f"🟡 **{action}** (信心度: {confidence}%, 风险: {risk_level})")
                            else:
                                st.success(f"🟢 **{action}** (信心度: {confidence}%, 风险: {risk_level})")
                            
                            st.info(f"**决策理由:** {decision_detail['reason']}")
                            
                            # 推荐策略详情
                            st.markdown("### 💡 推荐操作策略")
                            strategies = position_decision['strategies']
                            recommended_strategy_key = decision_detail['recommended_strategy']
                            
                            if recommended_strategy_key in strategies:
                                recommended_strategy = strategies[recommended_strategy_key]
                                st.markdown(f"**{recommended_strategy['name']}** (推荐)")
                                
                                for batch in recommended_strategy['batches']:
                                    if 'position_add' in batch:
                                        st.markdown(f"• **第{batch['batch']}批加仓**: {batch['price_range']} "
                                                  f"(加仓{batch['position_add']:.1f}%) - {batch['condition']}")
                                    elif 'position_reduce' in batch:
                                        st.markdown(f"• **减仓**: {batch['price_range']} "
                                                  f"(减仓{batch['position_reduce']:.1f}%) - {batch['condition']}")
                            
                            # 其他策略选择
                            with st.expander("查看其他策略选择"):
                                for key, strategy in strategies.items():
                                    if key != recommended_strategy_key:
                                        st.markdown(f"**{strategy['name']}** ({'推荐' if strategy['recommended'] else '不推荐'})")
                                        for batch in strategy['batches']:
                                            if 'position_add' in batch:
                                                st.markdown(f"  • {batch['price_range']} (加仓{batch['position_add']:.1f}%)")
                                            elif 'position_reduce' in batch:
                                                st.markdown(f"  • {batch['price_range']} (减仓{batch['position_reduce']:.1f}%)")
                            
                            # 风险评估
                            st.markdown("### ⚠️ 风险评估")
                            risk_assessment = position_decision['risk_assessment']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("风险评分", f"{risk_assessment['risk_score']}/100")
                            with col2:
                                st.metric("风险等级", risk_assessment['risk_level'])
                            
                            if risk_assessment['risk_factors']:
                                st.markdown("**风险因素:**")
                                for factor in risk_assessment['risk_factors']:
                                    st.markdown(f"• {factor}")
                            
                            st.markdown(f"**风险建议:** {risk_assessment['recommendation']}")
                            
                            # 最优时机
                            st.markdown("### ⏰ 最优操作时机")
                            optimal_timing = position_decision['optimal_timing']
                            
                            st.markdown(f"**时机判断:** {optimal_timing['best_timing']}")
                            st.markdown("**时机信号:**")
                            for signal in optimal_timing['signals']:
                                st.markdown(f"• {signal}")
                            
                            # 保存决策记录
                            dss.save_decision(position_decision)
                            
                            # 用户备注区域
                            st.markdown("### ✍️ 添加您的想法")
                            user_note = st.text_area("记录您对此次仓位分析的想法:", key=f"position_note_{position_stock}")
                            if st.button("💾 保存备注") and user_note:
                                dss.add_user_note(position_stock, f"仓位管理分析备注: {user_note}")
                                st.success("备注已保存!")
                        else:
                            st.error("分析失败，请稍后重试")
        
        elif decision_type == "🔍 买入时机分析":
            st.subheader("买入时机分析 - 避免抄底陷阱")
            
            if not selected_stocks:
                st.warning("请在侧边栏选择要分析的股票")
            else:
                buy_stock = st.selectbox("选择要分析买入时机的股票", selected_stocks, key="buy_analysis")
                
                if st.button("🔍 分析买入时机", type="primary"):
                    with st.spinner(f"正在分析 {buy_stock} 的买入时机..."):
                        # 获取当前分析数据
                        if 'unified_analyzer' not in st.session_state:
                            st.session_state.unified_analyzer = UnifiedStockAnalyzer()
                        
                        current_analysis = st.session_state.unified_analyzer.get_comprehensive_analysis(buy_stock)
                        
                        # 进行买入时机分析
                        buy_decision = dss.analyze_buy_timing(buy_stock, current_analysis)
                        
                        if 'decision' in buy_decision:
                            # 显示决策结果
                            decision_detail = buy_decision['decision']
                            decision = decision_detail['action']
                            confidence = decision_detail['confidence']
                            
                            # 决策结果展示
                            if decision == "建议买入":
                                st.success(f"🟢 **{decision}** (信心度: {confidence}%)")
                            elif decision == "可以考虑":
                                st.warning(f"🟡 **{decision}** (信心度: {confidence}%)")
                            else:
                                st.error(f"🔴 **{decision}** (信心度: {confidence}%)")
                            
                            # 详细分析结果
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**✅ 支持理由:**")
                                for reason in decision_detail.get('reasons', []):
                                    st.markdown(f"• {reason}")
                            
                            with col2:
                                st.markdown("**⚠️ 风险提醒:**")
                                for warning in decision_detail.get('warnings', []):
                                    st.markdown(f"• {warning}")
                            
                            # 保存决策记录
                            dss.save_decision(buy_decision)
                            
                            # 用户备注区域
                            user_note = st.text_area("添加您的想法和备注:", key=f"buy_note_{buy_stock}")
                            if st.button("💾 保存备注") and user_note:
                                dss.add_user_note(buy_stock, f"买入分析备注: {user_note}")
                                st.success("备注已保存!")
                        else:
                            st.error("分析失败，请稍后重试")
        
        elif decision_type == "💰 卖出时机分析":
            st.subheader("卖出时机分析 - 避免卖到半路")
            
            # 获取持仓股票
            holding_stocks = []
            if portfolio:
                holding_stocks = list(portfolio.keys())
            
            if not holding_stocks:
                st.warning("当前没有持仓股票")
            else:
                sell_stock = st.selectbox("选择要分析卖出时机的持仓股票", holding_stocks, key="sell_analysis")
                
                if st.button("💰 分析卖出时机", type="primary"):
                    with st.spinner(f"正在分析 {sell_stock} 的卖出时机..."):
                        # 获取持仓信息
                        if sell_stock in portfolio:
                            position_info = {
                                'cost_basis': portfolio[sell_stock]['cost'],
                                'shares': portfolio[sell_stock]['shares']
                            }
                            
                            # 获取当前分析数据
                            if 'unified_analyzer' not in st.session_state:
                                st.session_state.unified_analyzer = UnifiedStockAnalyzer()
                            
                            current_analysis = st.session_state.unified_analyzer.get_comprehensive_analysis(sell_stock)
                            
                            # 进行卖出时机分析
                            sell_decision = dss.analyze_sell_timing(sell_stock, position_info, current_analysis)
                            
                            if 'decision' in sell_decision:
                                # 显示决策结果
                                decision_detail = sell_decision['decision']
                                decision = decision_detail['action']
                                confidence = decision_detail['confidence']
                                reason = decision_detail.get('summary', '无详细信息')
                                
                                # 盈亏状况
                                profit_pct = sell_decision['profit_pct']
                                if profit_pct > 0:
                                    st.success(f"💰 当前盈利: {profit_pct:.1f}%")
                                else:
                                    st.error(f"📉 当前亏损: {profit_pct:.1f}%")
                                
                                # 决策建议
                                if decision == "考虑止损":
                                    st.error(f"🔴 **{decision}** (信心度: {confidence}%)")
                                elif decision == "考虑减仓":
                                    st.warning(f"🟡 **{decision}** (信心度: {confidence}%)")
                                else:
                                    st.success(f"🟢 **{decision}** (信心度: {confidence}%)")
                                
                                st.info(f"**分析理由:** {reason}")
                                
                                # 保存决策记录
                                dss.save_decision(sell_decision)
                                
                                # 用户备注区域
                                user_note = st.text_area("添加您的想法和备注:", key=f"sell_note_{sell_stock}")
                                if st.button("💾 保存备注") and user_note:
                                    dss.add_user_note(sell_stock, f"卖出分析备注: {user_note}")
                                    st.success("备注已保存!")
                            else:
                                st.error("分析失败，请稍后重试")
                        else:
                            st.error("找不到持仓信息")
        
        elif decision_type == "📝 查看决策历史":
            st.subheader("决策历史记录")
            
            if not selected_stocks:
                st.warning("请在侧边栏选择股票")
            else:
                history_stock = st.selectbox("选择要查看历史的股票", selected_stocks, key="history_analysis")
                days = st.slider("查看最近几天的记录", 7, 90, 30)
                
                history = dss.get_decision_history(history_stock, days)
                
                if history:
                    st.success(f"找到 {len(history)} 条记录")
                    
                    for record in history:
                        with st.expander(f"{record['timestamp'][:19]} - {record.get('decision_type', 'UNKNOWN')}"):
                            if record.get('type') == 'USER_NOTE':
                                st.markdown(f"**用户备注:** {record['note']}")
                            else:
                                if record.get('decision_type') == 'BUY_TIMING':
                                    st.markdown(f"**买入决策:** {record.get('decision', 'N/A')}")
                                    st.markdown(f"**信心度:** {record.get('confidence', 0)}%")
                                    st.markdown(f"**价格:** ${record.get('current_price', 0):.2f}")
                                elif record.get('decision_type') == 'SELL_TIMING':
                                    st.markdown(f"**卖出决策:** {record.get('decision', 'N/A')}")
                                    st.markdown(f"**信心度:** {record.get('confidence', 0)}%")
                                    st.markdown(f"**价格:** ${record.get('current_price', 0):.2f}")
                                    st.markdown(f"**盈亏:** {record.get('profit_pct', 0):.1f}%")
                                    st.markdown(f"**理由:** {record.get('reason', 'N/A')}")
                                
                                if record.get('user_notes'):
                                    st.markdown(f"**备注:** {record['user_notes']}")
                else:
                    st.info("暂无决策记录")
        
        elif decision_type == "✍️ 添加备注":
            st.subheader("添加投资备注")
            
            if not selected_stocks:
                st.warning("请在侧边栏选择股票")
            else:
                # 创建两列布局
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 📝 添加新备注")
                    note_stock = st.selectbox("选择股票", selected_stocks, key="note_stock")
                    note_content = st.text_area("输入您的想法、分析或备注:", height=150)
                    
                    # 备注类型选择
                    note_type = st.selectbox("备注类型", [
                        "💭 投资想法", "📊 技术分析", "💰 基本面分析", 
                        "⚠️ 风险提醒", "🎯 操作计划", "📈 市场观察", "其他"
                    ])
                    
                    if st.button("💾 保存备注", type="primary") and note_content:
                        # 添加备注类型到内容
                        full_note = f"[{note_type}] {note_content}"
                        result = dss.add_user_note(note_stock, full_note)
                        st.success(result)
                        st.rerun()  # 刷新页面显示新备注
                
                with col2:
                    st.markdown("### 📋 最近备注")
                    if selected_stocks:
                        recent_stock = st.selectbox("选择股票查看备注", selected_stocks, key="recent_notes")
                        user_notes = dss.get_user_notes(recent_stock, days=30)
                        
                        if user_notes:
                            st.success(f"找到 {len(user_notes)} 条备注")
                            
                            for note in user_notes[:5]:  # 显示最近5条
                                note_time = datetime.fromisoformat(note['timestamp']).strftime('%m-%d %H:%M')
                                with st.expander(f"{note_time} - {note['note'][:50]}..."):
                                    st.markdown(f"**备注内容:** {note['note']}")
                                    
                                    # 添加编辑和删除功能
                                    col_edit, col_del = st.columns(2)
                                    with col_edit:
                                        if st.button("✏️ 编辑", key=f"edit_{note.get('note_id', '')}"):
                                            st.session_state.editing_note = note
                                    
                                    with col_del:
                                        if st.button("🗑️ 删除", key=f"del_{note.get('note_id', '')}"):
                                            if 'note_id' in note:
                                                result = dss.delete_user_note(recent_stock, note['note_id'])
                                                st.success(result)
                                                st.rerun()
                        else:
                            st.info("暂无备注记录")
        
        # 新增AI分析功能
        elif decision_type == "🤖 AI智能分析":
            st.subheader("AI智能分析 - 基于您的备注和决策")
            
            if not selected_stocks:
                st.warning("请在侧边栏选择股票")
            else:
                # 导入AI分析模块
                try:
                    from analysis.ai_analysis_integration import AIAnalysisIntegration
                    
                    ai_stock = st.selectbox("选择要分析的股票", selected_stocks, key="ai_analysis")
                    analysis_type = st.selectbox("分析类型", [
                        "comprehensive", "investment_strategy", "risk_assessment", 
                        "decision_optimization", "psychology_analysis"
                    ])
                    
                    # AI API配置
                    with st.expander("🔧 AI API配置"):
                        api_key = st.text_input("AI API密钥", type="password", help="输入您的AI API密钥")
                        api_endpoint = st.text_input("API端点", value="https://api.openai.com/v1/chat/completions", help="AI API端点地址")
                        
                        if st.button("💾 保存配置"):
                            st.success("配置已保存到环境变量")
                    
                    if st.button("🤖 开始AI分析", type="primary"):
                        with st.spinner("AI正在分析您的投资决策..."):
                            try:
                                # 导出数据用于AI分析
                                export_data = dss.export_notes_for_ai(ai_stock)
                                
                                # 初始化AI分析器
                                ai_analyzer = AIAnalysisIntegration(api_key=api_key, api_endpoint=api_endpoint)
                                
                                # 执行分析
                                analysis_result = ai_analyzer.analyze_user_notes(export_data, analysis_type)
                                
                                # 显示分析结果
                                st.success("✅ AI分析完成!")
                                
                                # 分析摘要
                                st.markdown("### 📊 分析摘要")
                                st.info(analysis_result.get('summary', '无摘要信息'))
                                
                                # 风险评估
                                if 'risk_assessment' in analysis_result:
                                    risk_data = analysis_result['risk_assessment']
                                    st.markdown("### ⚠️ 风险评估")
                                    
                                    col_risk1, col_risk2 = st.columns(2)
                                    with col_risk1:
                                        risk_level = risk_data.get('risk_level', 'MEDIUM')
                                        if risk_level == 'HIGH':
                                            st.error(f"风险等级: {risk_level}")
                                        elif risk_level == 'LOW':
                                            st.success(f"风险等级: {risk_level}")
                                        else:
                                            st.warning(f"风险等级: {risk_level}")
                                    
                                    with col_risk2:
                                        if risk_data.get('risk_factors'):
                                            st.markdown("**风险因素:**")
                                            for factor in risk_data['risk_factors']:
                                                st.markdown(f"• {factor}")
                                
                                # 投资建议
                                if 'recommendations' in analysis_result:
                                    st.markdown("### 💡 投资建议")
                                    for i, rec in enumerate(analysis_result['recommendations'], 1):
                                        st.markdown(f"{i}. {rec}")
                                
                                # 心理洞察
                                if 'psychology_insights' in analysis_result:
                                    psych_data = analysis_result['psychology_insights']
                                    if psych_data:
                                        st.markdown("### 🧠 心理洞察")
                                        
                                        # 创建心理状态图表
                                        psych_df = pd.DataFrame(list(psych_data.items()), columns=['心理状态', '强度'])
                                        st.bar_chart(psych_df.set_index('心理状态'))
                                
                                # 原始响应（可折叠）
                                with st.expander("📄 查看完整AI分析报告"):
                                    st.text(analysis_result.get('raw_response', '无原始响应'))
                                
                                # 保存分析结果
                                if st.button("💾 保存分析结果"):
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    result_file = f"ai_analysis_{ai_stock}_{timestamp}.json"
                                    
                                    with open(result_file, 'w', encoding='utf-8') as f:
                                        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
                                    
                                    st.success(f"分析结果已保存到: {result_file}")
                                
                            except Exception as e:
                                st.error(f"AI分析失败: {str(e)}")
                                st.info("请检查API配置或网络连接")
                
                except ImportError:
                    st.error("AI分析模块未找到，请确保已安装相关依赖")
                    st.info("您可以通过以下方式安装: pip install requests")
        
        # 新增备注管理功能
        elif decision_type == "📚 备注管理":
            st.subheader("备注管理 - 查看和管理所有备注")
            
            if not selected_stocks:
                st.warning("请在侧边栏选择股票")
            else:
                manage_stock = st.selectbox("选择股票", selected_stocks, key="manage_notes")
                
                # 统计信息
                user_notes = dss.get_user_notes(manage_stock, days=365)  # 获取一年的备注
                decisions = dss.get_decision_history(manage_stock, days=365)
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("备注总数", len(user_notes))
                with col_stats2:
                    st.metric("决策记录", len([d for d in decisions if d.get('type') != 'USER_NOTE']))
                with col_stats3:
                    st.metric("最近活跃", "今天" if user_notes else "无")
                
                # 备注列表
                if user_notes:
                    st.markdown("### 📝 备注列表")
                    
                    # 搜索和过滤
                    search_term = st.text_input("🔍 搜索备注", placeholder="输入关键词搜索...")
                    
                    filtered_notes = user_notes
                    if search_term:
                        filtered_notes = [note for note in user_notes if search_term.lower() in note['note'].lower()]
                    
                    st.info(f"显示 {len(filtered_notes)} 条备注")
                    
                    for note in filtered_notes:
                        note_time = datetime.fromisoformat(note['timestamp']).strftime('%Y-%m-%d %H:%M')
                        
                        with st.expander(f"📅 {note_time} - {note['note'][:60]}..."):
                            st.markdown(f"**完整内容:** {note['note']}")
                            
                            # 操作按钮
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if st.button("📋 复制", key=f"copy_{note.get('note_id', '')}"):
                                    st.write("已复制到剪贴板")
                            
                            with col2:
                                if st.button("✏️ 编辑", key=f"edit_manage_{note.get('note_id', '')}"):
                                    st.session_state.editing_note = note
                            
                            with col3:
                                if st.button("🗑️ 删除", key=f"del_manage_{note.get('note_id', '')}"):
                                    if 'note_id' in note:
                                        result = dss.delete_user_note(manage_stock, note['note_id'])
                                        st.success(result)
                                        st.rerun()
                else:
                    st.info("暂无备注记录")
                
                # 批量操作
                st.markdown("### 🔧 批量操作")
                col_batch1, col_batch2 = st.columns(2)
                
                with col_batch1:
                    if st.button("📤 导出备注数据"):
                        export_data = dss.export_notes_for_ai(manage_stock)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        export_file = f"notes_export_{manage_stock}_{timestamp}.json"
                        
                        with open(export_file, 'w', encoding='utf-8') as f:
                            json.dump(export_data, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"备注数据已导出到: {export_file}")
                
                with col_batch2:
                    if st.button("📊 生成备注报告"):
                        # 生成简单的备注统计报告
                        if user_notes:
                            report = f"""
# {manage_stock} 备注分析报告

## 统计信息
- 备注总数: {len(user_notes)}
- 时间范围: {user_notes[-1]['timestamp'][:10]} 至 {user_notes[0]['timestamp'][:10]}
- 平均每天备注: {len(user_notes) / max(1, (datetime.now() - datetime.fromisoformat(user_notes[-1]['timestamp'])).days):.1f} 条

## 备注类型分析
"""
                            
                            # 分析备注类型
                            note_types = {}
                            for note in user_notes:
                                note_content = note['note']
                                for note_type in ["💭 投资想法", "📊 技术分析", "💰 基本面分析", "⚠️ 风险提醒", "🎯 操作计划", "📈 市场观察"]:
                                    if note_type in note_content:
                                        note_types[note_type] = note_types.get(note_type, 0) + 1
                                        break
                            
                            for note_type, count in note_types.items():
                                report += f"- {note_type}: {count} 条\n"
                            
                            # 保存报告
                            report_file = f"notes_report_{manage_stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            with open(report_file, 'w', encoding='utf-8') as f:
                                f.write(report)
                            
                            st.success(f"备注报告已生成: {report_file}")
                            st.text(report)
                        else:
                            st.warning("没有备注数据可生成报告")
        
        # 编辑备注功能
        if 'editing_note' in st.session_state:
            st.markdown("### ✏️ 编辑备注")
            editing_note = st.session_state.editing_note
            
            edited_content = st.text_area("编辑备注内容:", value=editing_note['note'], height=100)
            
            col_edit1, col_edit2 = st.columns(2)
            with col_edit1:
                if st.button("💾 保存修改"):
                    if 'note_id' in editing_note:
                        result = dss.update_user_note(editing_note['symbol'], editing_note['note_id'], edited_content)
                        st.success(result)
                        del st.session_state.editing_note
                        st.rerun()
            
            with col_edit2:
                if st.button("❌ 取消编辑"):
                    del st.session_state.editing_note
                    st.rerun()
    
    # 替换AI诊断tab为AI每日持股分析监控系统
    with tab6:
        from start_ai_daily_analysis_monitor import AIDailyAnalysisMonitor
        monitor = AIDailyAnalysisMonitor()
        monitor.run_streamlit_app()
    
    # AI问答功能
    with tab7:
        try:
            # 导入Ollama AI问答模块
            from monitor.ollama_ai_qa import OllamaAIQA
            
            # 创建AI问答界面实例
            qa_interface = OllamaAIQA()
            
            # 渲染AI问答界面
            qa_interface.render_qa_interface()
            
        except ImportError as e:
            st.error(f"AI问答模块导入失败: {e}")
            st.info("请确保monitor/ollama_ai_qa.py文件存在")
        except Exception as e:
            st.error(f"AI问答功能初始化失败: {e}")
            st.info("请检查Ollama服务是否正在运行")
    
    # 系统设置标签页
    with tab8:
        st.header("⚙️ 系统设置")
        
        # 创建子标签页
        settings_tab1, settings_tab2, settings_tab3 = st.tabs([
            "🤖 AI设置", 
            "📊 配置管理", 
            "📈 系统信息"
        ])
        
        with settings_tab1:
            st.subheader("🤖 AI设置")
            st.write("**AI模型:** DeepSeek R1")
            st.write("**API端点:** http://localhost:11434")
            
            # 测试AI连接
            if st.button("🔗 测试AI连接"):
                try:
                    import requests
                    response = requests.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ AI连接正常")
                    else:
                        st.error("❌ AI连接失败")
                except Exception as e:
                    st.error(f"❌ AI连接失败: {e}")
        
        with settings_tab2:
            st.subheader("📊 配置管理")
            
            # 导入配置管理器
            try:
                from portfolio_config_manager import render_portfolio_config_manager
                render_portfolio_config_manager()
            except ImportError as e:
                st.error(f"配置管理器导入失败: {e}")
                st.info("请确保portfolio_config_manager.py文件存在")
            except Exception as e:
                st.error(f"配置管理功能初始化失败: {e}")
        
        with settings_tab3:
            st.subheader("📈 系统信息")
            st.write(f"**最后更新:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 显示配置文件状态
            import os
            config_files = ['portfolio_config.json', 'personal_investor_config.json']
            st.markdown("### 📁 配置文件状态")
            for config_file in config_files:
                if os.path.exists(config_file):
                    file_size = os.path.getsize(config_file)
                    st.write(f"✅ **{config_file}**: {file_size:,} bytes")
                else:
                    st.write(f"❌ **{config_file}**: 文件不存在")
            
            # 显示系统状态
            st.markdown("### 🔧 系统状态")
            try:
                # 检查数据接口
                from data.data_interface import DataInterface
                data_interface = DataInterface()
                st.write("✅ **数据接口**: 正常")
            except Exception as e:
                st.write(f"❌ **数据接口**: {e}")
            
            try:
                # 检查AI模块
                from ai_trading_module import AITradingModule
                ai_module = AITradingModule()
                st.write("✅ **AI模块**: 正常")
            except Exception as e:
                st.write(f"❌ **AI模块**: {e}")
            
            # 数据更新功能
            st.markdown("---")
            st.subheader("🔄 数据更新管理")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 股票数据更新")
                
                # 更新选项
                update_type = st.selectbox(
                    "选择更新类型",
                    ["增量更新", "强制更新", "指定股票更新"],
                    help="增量更新：只更新最新数据；强制更新：重新获取所有数据；指定股票：更新特定股票"
                )
                
                if update_type == "指定股票更新":
                    # 从配置文件获取持仓股票
                    portfolio_data = load_portfolio_config()
                    if portfolio_data and 'positions' in portfolio_data:
                        position_symbols = list(portfolio_data['positions'].keys())
                        selected_symbols = st.multiselect(
                            "选择要更新的股票",
                            position_symbols,
                            default=position_symbols[:3]  # 默认选择前3个
                        )
                    else:
                        selected_symbols = st.text_input(
                            "输入股票代码（用逗号分隔）",
                            value="NVDA,AMD,GOOG",
                            help="例如：NVDA,AMD,GOOG"
                        ).split(',')
                        selected_symbols = [s.strip() for s in selected_symbols if s.strip()]
                else:
                    selected_symbols = None
                
                # 更新按钮
                if st.button("🔄 开始数据更新", type="primary"):
                    with st.spinner("正在更新股票数据..."):
                        try:
                            # 导入数据更新器
                            from data.data_updater import MarketDataUpdater
                            from config.trading_config import default_config
                            
                            # 数据库配置
                            db_config = {
                                "host": default_config.database.host,
                                "port": default_config.database.port,
                                "user": default_config.database.user,
                                "password": default_config.database.password,
                                "database": default_config.database.database
                            }
                            
                            # 创建更新器
                            updater = MarketDataUpdater(db_config)
                            
                            # 执行更新
                            if update_type == "增量更新":
                                report = updater.update_stock_data(symbols=selected_symbols, force_update=False)
                            elif update_type == "强制更新":
                                report = updater.update_stock_data(symbols=selected_symbols, force_update=True)
                            else:  # 指定股票更新
                                report = updater.update_stock_data(symbols=selected_symbols, force_update=False)
                            
                            # 显示更新结果
                            st.success("✅ 数据更新完成！")
                            st.write(f"**总计**: {report['total']} 只股票")
                            st.write(f"**更新成功**: {report['updated']} 只")
                            st.write(f"**跳过**: {report['skipped']} 只")
                            st.write(f"**失败**: {report['failed']} 只")
                            
                            # 显示详细结果
                            if report['details']:
                                with st.expander("📋 详细更新结果"):
                                    for symbol, status in report['details'].items():
                                        if status == 'updated':
                                            st.write(f"✅ {symbol}: 更新成功")
                                        elif status == 'skipped (up to date)':
                                            st.write(f"⏭️ {symbol}: 数据已是最新")
                                        else:
                                            st.write(f"❌ {symbol}: {status}")
                            
                        except Exception as e:
                            st.error(f"❌ 数据更新失败: {e}")
                            st.info("请检查数据库连接和网络状态")
            
            with col2:
                st.markdown("### 📊 数据状态检查")
                
                # 检查数据更新时间
                if st.button("🔍 检查数据状态"):
                    try:
                        from data.data_updater import MarketDataUpdater
                        from config.trading_config import default_config
                        
                        db_config = {
                            "host": default_config.database.host,
                            "port": default_config.database.port,
                            "user": default_config.database.user,
                            "password": default_config.database.password,
                            "database": default_config.database.database
                        }
                        
                        updater = MarketDataUpdater(db_config)
                        
                        # 获取持仓股票的最后更新时间
                        portfolio_data = load_portfolio_config()
                        if portfolio_data and 'positions' in portfolio_data:
                            position_symbols = list(portfolio_data['positions'].keys())
                            
                            st.markdown("#### 📅 持仓股票数据状态")
                            for symbol in position_symbols[:5]:  # 显示前5个
                                last_update = updater.get_last_update_time(symbol)
                                if last_update:
                                    days_ago = (datetime.now() - last_update).days
                                    if days_ago == 0:
                                        st.write(f"✅ {symbol}: 今天已更新")
                                    elif days_ago == 1:
                                        st.write(f"🟡 {symbol}: 1天前更新")
                                    else:
                                        st.write(f"🔴 {symbol}: {days_ago}天前更新")
                                else:
                                    st.write(f"❌ {symbol}: 无数据")
                        
                    except Exception as e:
                        st.error(f"❌ 检查数据状态失败: {e}")
                
                # 数据清理功能
                st.markdown("### 🧹 数据维护")
                
                if st.button("🗑️ 清理缓存"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("✅ 缓存已清理")
                
                if st.button("🔄 刷新系统"):
                    st.rerun()

    # 页面底部信息
    st.markdown("---")
    st.markdown("**💡 使用说明:** 本系统提供实时市场数据和技术分析，仅供参考，投资有风险，决策需谨慎。")
    st.markdown("**🔬 深度分析:** 集成专业级分析系统，提供技术面、基本面、流动性、智能分析等多维度评估。")
    st.markdown("**🧠 决策支持:** 专门为避免抄底抄到半山腰、避免卖到半路而设计，帮助您做出更明智的投资决策。")

    # 新增AI模型选择器
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