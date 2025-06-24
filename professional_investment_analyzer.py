#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业个人投资分析系统
基于多数据源的概率预测模型
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProfessionalInvestmentAnalyzer:
    """专业投资分析系统"""
    
    def __init__(self):
        # 页面配置
        st.set_page_config(
            page_title="专业投资分析系统",
            page_icon="📊",
            layout="wide"
        )
        
        # 市场指标
        self.market_indices = {
            'VIX': '^VIX',
            'SPX': '^GSPC', 
            'NDX': '^NDX',
            'DJI': '^DJI',
            'DXY': 'DX-Y.NYB',  # 美元指数
            'TNX': '^TNX'       # 10年期国债
        }
        
        # 个股池
        self.stock_universe = ['AMD', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOG', 'META', 'AMZN']
    
    def run_analyzer(self):
        """运行分析系统"""
        st.title("🎯 专业个人投资分析系统")
        st.markdown("---")
        
        # 侧边栏配置
        self._setup_sidebar()
        
        # 主分析区域
        tab1, tab2, tab3, tab4 = st.tabs(["📊 市场概率预测", "🎯 个股分析", "💼 持仓管理", "⚡ 实时监控"])
        
        with tab1:
            self._market_probability_analysis()
        
        with tab2:
            self._individual_stock_analysis()
        
        with tab3:
            self._portfolio_management()
        
        with tab4:
            self._realtime_monitoring()
    
    def _setup_sidebar(self):
        """设置侧边栏"""
        st.sidebar.header("🔧 分析配置")
        
        # 时间范围
        st.session_state.time_horizon = st.sidebar.selectbox(
            "预测时间范围",
            ["1周", "2周", "1个月", "3个月"],
            index=1
        )
        
        # 风险偏好
        st.session_state.risk_preference = st.sidebar.selectbox(
            "风险偏好",
            ["保守", "平衡", "积极"],
            index=1
        )
        
        # 关注股票
        if 'focus_stocks' not in st.session_state:
            st.session_state.focus_stocks = ['AMD', 'NVDA', 'TSLA']
        
        st.session_state.focus_stocks = st.sidebar.multiselect(
            "重点关注股票",
            self.stock_universe,
            default=st.session_state.focus_stocks
        )
        
        # 持仓信息
        st.sidebar.subheader("💼 当前持仓")
        if 'amd_shares' not in st.session_state:
            st.session_state.amd_shares = 0
        if 'amd_cost' not in st.session_state:
            st.session_state.amd_cost = 125.746
            
        st.session_state.amd_shares = st.sidebar.number_input("AMD股数", value=st.session_state.amd_shares, min_value=0)
        st.session_state.amd_cost = st.sidebar.number_input("AMD成本价", value=st.session_state.amd_cost, min_value=0.0)
    
    def _market_probability_analysis(self):
        """市场概率分析"""
        st.header("📊 市场概率预测分析")
        
        # 获取市场数据
        market_data = self._get_market_data()
        
        if market_data:
            # 市场状态分析
            market_state = self._analyze_market_state(market_data)
            
            # 显示市场状态
            self._display_market_state(market_state)
            
            # 概率预测
            predictions = self._generate_market_predictions(market_state)
            self._display_probability_predictions(predictions)
            
            # 市场情景分析
            self._scenario_analysis(market_state)
    
    def _individual_stock_analysis(self):
        """个股分析"""
        st.header("🎯 个股概率分析")
        
        # 选择股票
        selected_stock = st.selectbox(
            "选择分析股票",
            st.session_state.focus_stocks,
            index=0 if 'AMD' in st.session_state.focus_stocks else 0
        )
        
        if selected_stock:
            # 获取个股数据
            stock_data = self._get_stock_data(selected_stock)
            
            if stock_data is not None and not stock_data.empty:
                # 个股状态分析
                stock_state = self._analyze_stock_state(selected_stock, stock_data)
                
                # 显示个股分析
                self._display_stock_analysis(selected_stock, stock_state)
                
                # 个股预测
                stock_predictions = self._generate_stock_predictions(selected_stock, stock_state)
                self._display_stock_predictions(selected_stock, stock_predictions)
    
    def _portfolio_management(self):
        """持仓管理"""
        st.header("💼 智能持仓管理")
        
        if hasattr(st.session_state, 'amd_shares') and st.session_state.amd_shares > 0:
            # AMD持仓分析
            self._amd_portfolio_analysis()
        
        # 投资组合建议
        self._portfolio_recommendations()
    
    def _realtime_monitoring(self):
        """实时监控"""
        st.header("⚡ 实时监控面板")
        
        # 获取实时数据
        realtime_data = self._get_realtime_data()
        
        if realtime_data:
            # 实时指标
            self._display_realtime_metrics(realtime_data)
            
            # 实时图表
            self._create_realtime_charts(realtime_data)
    
    def _get_market_data(self):
        """获取市场数据"""
        market_data = {}
        
        with st.spinner("正在获取市场数据..."):
            for name, symbol in self.market_indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='3mo')
                    if not data.empty:
                        market_data[name] = {
                            'data': data,
                            'current': data['Close'].iloc[-1],
                            'prev': data['Close'].iloc[-2],
                            'change_pct': (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                        }
                except:
                    continue
        
        return market_data
    
    def _analyze_market_state(self, market_data):
        """分析市场状态"""
        state = {}
        
        # VIX分析
        if 'VIX' in market_data:
            vix_current = market_data['VIX']['current']
            vix_data = market_data['VIX']['data']
            vix_30d_avg = vix_data['Close'].tail(30).mean()
            
            state['vix'] = {
                'current': vix_current,
                'level': self._classify_vix_level(vix_current),
                'vs_30d': (vix_current - vix_30d_avg) / vix_30d_avg * 100,
                'trend': 'down' if market_data['VIX']['change_pct'] < 0 else 'up'
            }
        
        # 标普500分析
        if 'SPX' in market_data:
            spx_current = market_data['SPX']['current']
            spx_data = market_data['SPX']['data']
            spx_ma20 = spx_data['Close'].rolling(20).mean().iloc[-1]
            spx_ma50 = spx_data['Close'].rolling(50).mean().iloc[-1]
            
            state['spx'] = {
                'current': spx_current,
                'ma20': spx_ma20,
                'ma50': spx_ma50,
                'trend': 'bullish' if spx_current > spx_ma20 > spx_ma50 else 'bearish',
                'momentum': market_data['SPX']['change_pct']
            }
        
        # 美元指数分析
        if 'DXY' in market_data:
            dxy_current = market_data['DXY']['current']
            state['dxy'] = {
                'current': dxy_current,
                'trend': 'up' if market_data['DXY']['change_pct'] > 0 else 'down',
                'strength': 'strong' if dxy_current > 105 else 'weak'
            }
        
        # 综合市场情绪
        state['overall_sentiment'] = self._calculate_market_sentiment(state)
        
        return state
    
    def _classify_vix_level(self, vix):
        """VIX水平分类"""
        if vix < 15:
            return "极低恐慌"
        elif vix < 20:
            return "低恐慌"
        elif vix < 30:
            return "中等恐慌"
        else:
            return "高恐慌"
    
    def _calculate_market_sentiment(self, state):
        """计算综合市场情绪"""
        sentiment_score = 50  # 中性基准
        
        # VIX影响
        if 'vix' in state:
            if state['vix']['current'] < 18:
                sentiment_score += 15
            elif state['vix']['current'] > 25:
                sentiment_score -= 15
        
        # 标普趋势影响
        if 'spx' in state:
            if state['spx']['trend'] == 'bullish':
                sentiment_score += 10
            else:
                sentiment_score -= 10
        
        # 限制在0-100范围
        sentiment_score = max(0, min(100, sentiment_score))
        
        if sentiment_score > 70:
            return "极度乐观"
        elif sentiment_score > 55:
            return "乐观"
        elif sentiment_score > 45:
            return "中性"
        elif sentiment_score > 30:
            return "悲观"
        else:
            return "极度悲观"
    
    def _display_market_state(self, market_state):
        """显示市场状态"""
        st.subheader("🌍 当前市场状态")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'vix' in market_state:
                vix_info = market_state['vix']
                st.metric(
                    "VIX恐慌指数",
                    f"{vix_info['current']:.2f}",
                    f"{vix_info['vs_30d']:+.1f}% vs 30日均值"
                )
                st.caption(f"水平: {vix_info['level']}")
        
        with col2:
            if 'spx' in market_state:
                spx_info = market_state['spx']
                st.metric(
                    "标普500",
                    f"{spx_info['current']:.0f}",
                    f"{spx_info['momentum']:+.2f}%"
                )
                st.caption(f"趋势: {spx_info['trend']}")
        
        with col3:
            if 'dxy' in market_state:
                dxy_info = market_state['dxy']
                st.metric(
                    "美元指数",
                    f"{dxy_info['current']:.2f}",
                    f"{'↑' if dxy_info['trend'] == 'up' else '↓'}"
                )
                st.caption(f"强度: {dxy_info['strength']}")
        
        with col4:
            sentiment = market_state['overall_sentiment']
            emoji = {"极度乐观": "🚀", "乐观": "📈", "中性": "➡️", "悲观": "📉", "极度悲观": "🔻"}
            st.metric(
                "市场情绪",
                sentiment,
                emoji.get(sentiment, "➡️")
            )
    
    def _generate_market_predictions(self, market_state):
        """生成市场预测"""
        predictions = {}
        
        # 基于VIX和标普的预测逻辑
        if 'vix' in market_state and 'spx' in market_state:
            vix_current = market_state['vix']['current']
            spx_current = market_state['spx']['current']
            
            # 上涨概率计算
            up_prob = 50  # 基准概率
            
            # VIX因子
            if vix_current < 18:
                up_prob += 20
            elif vix_current > 25:
                up_prob -= 20
            
            # 趋势因子
            if market_state['spx']['trend'] == 'bullish':
                up_prob += 15
            else:
                up_prob -= 15
            
            # 限制概率范围
            up_prob = max(15, min(85, up_prob))
            
            predictions['market'] = {
                'up_probability': up_prob,
                'down_probability': 100 - up_prob,
                'targets': {
                    'bullish': spx_current * 1.05,
                    'bearish': spx_current * 0.95
                }
            }
        
        return predictions
    
    def _display_probability_predictions(self, predictions):
        """显示概率预测"""
        st.subheader("🎯 概率预测")
        
        if 'market' in predictions:
            market_pred = predictions['market']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 概率饼图
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['上涨概率', '下跌概率'],
                        values=[market_pred['up_probability'], market_pred['down_probability']],
                        marker=dict(colors=['#00ff00', '#ff0000'])
                    )
                ])
                fig.update_layout(title="市场方向概率")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📊 预测详情:**")
                st.write(f"🟢 上涨概率: **{market_pred['up_probability']:.0f}%**")
                st.write(f"🔴 下跌概率: **{market_pred['down_probability']:.0f}%**")
                st.write(f"🎯 乐观目标: **{market_pred['targets']['bullish']:.0f}**")
                st.write(f"🎯 悲观目标: **{market_pred['targets']['bearish']:.0f}**")
    
    def _scenario_analysis(self, market_state):
        """情景分析"""
        st.subheader("📋 情景分析")
        
        scenarios = [
            {
                "name": "🟢 乐观情景",
                "probability": "35%",
                "conditions": [
                    "VIX继续下降至15以下",
                    "标普突破6300点",
                    "科技股领涨"
                ],
                "impact": "AMD等科技股大幅上涨"
            },
            {
                "name": "🟡 中性情景", 
                "probability": "50%",
                "conditions": [
                    "VIX在15-20区间震荡",
                    "标普在6000-6300整理",
                    "市场等待催化剂"
                ],
                "impact": "个股分化，精选为王"
            },
            {
                "name": "🔴 悲观情景",
                "probability": "15%",
                "conditions": [
                    "VIX反弹至25以上",
                    "标普回调至5800-6000",
                    "风险事件冲击"
                ],
                "impact": "全面回调，现金为王"
            }
        ]
        
        for scenario in scenarios:
            with st.expander(f"{scenario['name']} (概率: {scenario['probability']})"):
                st.write("**触发条件:**")
                for condition in scenario['conditions']:
                    st.write(f"• {condition}")
                st.write(f"**对投资的影响:** {scenario['impact']}")
    
    def _get_stock_data(self, symbol):
        """获取个股数据"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='3mo')
            return data
        except:
            return None
    
    def _analyze_stock_state(self, symbol, data):
        """分析个股状态"""
        current_price = data['Close'].iloc[-1]
        
        # 技术指标
        rsi = self._calculate_rsi(data['Close'])
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma50 = data['Close'].rolling(50).mean().iloc[-1]
        
        # 波动性
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        
        # 价格位置
        high_52w = data['High'].max()
        low_52w = data['Low'].min()
        price_position = (current_price - low_52w) / (high_52w - low_52w) * 100
        
        return {
            'current_price': current_price,
            'rsi': rsi,
            'ma20': ma20,
            'ma50': ma50,
            'volatility': volatility,
            'price_position': price_position,
            'trend': 'bullish' if current_price > ma20 > ma50 else 'bearish'
        }
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).iloc[-1]
    
    def _display_stock_analysis(self, symbol, stock_state):
        """显示个股分析"""
        st.subheader(f"📈 {symbol} 技术分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("当前价格", f"${stock_state['current_price']:.2f}")
        
        with col2:
            rsi_color = "🟢" if stock_state['rsi'] < 30 else "🔴" if stock_state['rsi'] > 70 else "🟡"
            st.metric("RSI", f"{stock_state['rsi']:.1f}", rsi_color)
        
        with col3:
            st.metric("价格位置", f"{stock_state['price_position']:.1f}%", "52周区间")
        
        with col4:
            st.metric("年化波动率", f"{stock_state['volatility']:.1f}%")
    
    def _generate_stock_predictions(self, symbol, stock_state):
        """生成个股预测"""
        # 基于技术指标的概率计算
        up_prob = 50
        
        # RSI因子
        if stock_state['rsi'] < 30:
            up_prob += 20
        elif stock_state['rsi'] > 70:
            up_prob -= 20
        
        # 趋势因子
        if stock_state['trend'] == 'bullish':
            up_prob += 15
        else:
            up_prob -= 15
        
        # 价格位置因子
        if stock_state['price_position'] < 30:
            up_prob += 10
        elif stock_state['price_position'] > 80:
            up_prob -= 10
        
        up_prob = max(20, min(80, up_prob))
        
        return {
            'up_probability': up_prob,
            'down_probability': 100 - up_prob,
            'targets': {
                'bullish': stock_state['current_price'] * 1.08,
                'bearish': stock_state['current_price'] * 0.92
            }
        }
    
    def _display_stock_predictions(self, symbol, predictions):
        """显示个股预测"""
        st.subheader(f"🎯 {symbol} 概率预测")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 概率条形图
            fig = go.Figure(data=[
                go.Bar(
                    x=['上涨', '下跌'],
                    y=[predictions['up_probability'], predictions['down_probability']],
                    marker_color=['green', 'red']
                )
            ])
            fig.update_layout(title=f"{symbol} 方向概率", yaxis_title="概率 (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📊 预测结果:**")
            st.write(f"🟢 上涨概率: **{predictions['up_probability']:.0f}%**")
            st.write(f"🔴 下跌概率: **{predictions['down_probability']:.0f}%**")
            st.write(f"🎯 上涨目标: **${predictions['targets']['bullish']:.2f}**")
            st.write(f"🎯 下跌目标: **${predictions['targets']['bearish']:.2f}**")
            
            # 操作建议
            if predictions['up_probability'] > 65:
                st.success("💡 建议: 考虑买入或持有")
            elif predictions['up_probability'] < 35:
                st.error("💡 建议: 考虑减仓或观望")
            else:
                st.info("💡 建议: 保持中性，等待更明确信号")
    
    def _amd_portfolio_analysis(self):
        """AMD持仓分析"""
        st.subheader("💼 AMD持仓分析")
        
        # 获取AMD数据
        amd_data = self._get_stock_data('AMD')
        if amd_data:
            current_price = amd_data['Close'].iloc[-1]
            cost_basis = st.session_state.amd_cost
            shares = st.session_state.amd_shares
            
            # 计算盈亏
            total_cost = cost_basis * shares
            current_value = current_price * shares
            unrealized_pnl = current_value - total_cost
            pnl_pct = (unrealized_pnl / total_cost) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("持仓股数", f"{shares}", "股")
            
            with col2:
                st.metric("成本价", f"${cost_basis:.2f}")
            
            with col3:
                st.metric("当前价", f"${current_price:.2f}")
            
            with col4:
                st.metric("盈亏", f"${unrealized_pnl:.2f}", f"{pnl_pct:+.2f}%")
            
            # 持仓建议
            self._portfolio_recommendations_amd(current_price, cost_basis, pnl_pct)
    
    def _portfolio_recommendations_amd(self, current_price, cost_basis, pnl_pct):
        """AMD持仓建议"""
        st.markdown("**💡 持仓操作建议:**")
        
        if pnl_pct > 15:
            st.success("🎯 建议分批减仓，锁定部分利润")
        elif pnl_pct > 8:
            st.info("📈 可继续持有，设置止损保护利润")
        elif pnl_pct < -5:
            st.warning("⚠️ 考虑止损或加仓摊薄成本")
        else:
            st.info("📊 保持当前仓位，密切关注")
    
    def _portfolio_recommendations(self):
        """投资组合建议"""
        st.subheader("📋 投资组合建议")
        
        risk_pref = st.session_state.risk_preference
        
        if risk_pref == "保守":
            allocation = {"现金": 40, "债券": 30, "大盘股": 20, "科技股": 10}
        elif risk_pref == "平衡":
            allocation = {"现金": 20, "债券": 20, "大盘股": 35, "科技股": 25}
        else:  # 积极
            allocation = {"现金": 10, "债券": 10, "大盘股": 30, "科技股": 50}
        
        # 显示资产配置
        fig = go.Figure(data=[
            go.Pie(labels=list(allocation.keys()), values=list(allocation.values()))
        ])
        fig.update_layout(title=f"{risk_pref}型投资者建议配置")
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_realtime_data(self):
        """获取实时数据"""
        return self._get_market_data()  # 复用市场数据获取
    
    def _display_realtime_metrics(self, data):
        """显示实时指标"""
        st.subheader("📊 实时市场指标")
        
        cols = st.columns(len(data))
        
        for i, (name, info) in enumerate(data.items()):
            with cols[i]:
                color = "🟢" if info['change_pct'] > 0 else "🔴" if info['change_pct'] < 0 else "⚪"
                st.metric(
                    name,
                    f"{info['current']:.2f}",
                    f"{color} {info['change_pct']:+.2f}%"
                )
    
    def _create_realtime_charts(self, data):
        """创建实时图表"""
        st.subheader("📈 实时走势图")
        
        # 选择要显示的指标
        selected_indices = st.multiselect(
            "选择显示的指标",
            list(data.keys()),
            default=['VIX', 'SPX']
        )
        
        if selected_indices:
            for index in selected_indices:
                if index in data:
                    index_data = data[index]['data']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=index_data.index,
                        y=index_data['Close'],
                        mode='lines',
                        name=index,
                        line=dict(width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{index} 走势图",
                        xaxis_title="时间",
                        yaxis_title="价格",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def main():
    """主函数"""
    analyzer = ProfessionalInvestmentAnalyzer()
    analyzer.run_analyzer()

if __name__ == "__main__":
    main()