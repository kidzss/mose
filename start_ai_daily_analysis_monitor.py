#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI每日持股分析监控系统
AI Daily Holdings Analysis Monitor System
"""

import asyncio
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
import warnings
warnings.filterwarnings('ignore')

# 导入AI分析器
from ai_realtime_analyzer import AIRealtimeAnalyzer
from daily_holdings_analysis import DailyHoldingsAnalyzer

class AIDailyAnalysisMonitor:
    """AI每日持股分析监控系统"""
    
    def __init__(self):
        """初始化监控系统"""
        self.ai_analyzer = AIRealtimeAnalyzer(use_daily_analysis=True)
        self.daily_analyzer = DailyHoldingsAnalyzer()
        self.analysis_history = []
        
    def load_portfolio_config(self):
        """加载投资组合配置"""
        try:
            with open('portfolio_config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"无法读取portfolio_config.json: {e}")
            return {}
    
    def get_real_time_data(self, symbols):
        """获取实时数据"""
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='60d', interval='1d')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    # 计算技术指标
                    technical_indicators = self._calculate_technical_indicators(hist)
                    
                    # 获取财务数据
                    financial_data = self._get_financial_data(info)
                    
                    # 构建完整的市场数据
                    data[symbol] = {
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': hist['Volume'].iloc[-1],
                        'market_cap': info.get('marketCap', 0),
                        'technical_indicators': technical_indicators,
                        'financial_data': financial_data,
                        'company_info': {
                            'name': info.get('longName', ''),
                            'sector': info.get('sector', ''),
                            'industry': info.get('industry', ''),
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'pb_ratio': info.get('priceToBook', 0),
                            'dividend_yield': info.get('dividendYield', 0)
                        }
                    }
            except Exception as e:
                st.warning(f"获取 {symbol} 数据失败: {e}")
        
        return data
    
    def _calculate_technical_indicators(self, hist):
        """计算技术指标"""
        try:
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 移动平均线
            ma20 = hist['Close'].rolling(window=20).mean()
            ma50 = hist['Close'].rolling(window=50).mean()
            
            # MACD
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal
            
            # 布林带
            bb_middle = hist['Close'].rolling(window=20).mean()
            bb_std = hist['Close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # 成交量指标
            volume_ma20 = hist['Volume'].rolling(window=20).mean()
            volume_ratio = hist['Volume'].iloc[-1] / volume_ma20.iloc[-1] if volume_ma20.iloc[-1] > 0 else 1
            
            # 波动率
            returns = hist['Close'].pct_change()
            volatility = returns.rolling(window=20).std() * (252 ** 0.5)
            
            return {
                'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
                'ma20': ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else hist['Close'].iloc[-1],
                'ma50': ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else hist['Close'].iloc[-1],
                'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
                'macd_signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0,
                'macd_hist': macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0,
                'bb_upper': bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else hist['Close'].iloc[-1],
                'bb_lower': bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else hist['Close'].iloc[-1],
                'volume_ratio': volume_ratio,
                'volatility': volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.2,
                'trend': 'up' if hist['Close'].iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1] else 'down' if hist['Close'].iloc[-1] < ma20.iloc[-1] < ma50.iloc[-1] else 'sideways'
            }
        except Exception as e:
            st.warning(f"计算技术指标失败: {e}")
            return {}
    
    def _get_financial_data(self, info):
        """获取财务数据"""
        try:
            return {
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'profit_margins': info.get('profitMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'roa': info.get('returnOnAssets', 0),
                'roe': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'free_cashflow': info.get('freeCashflow', 0),
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0)
            }
        except Exception as e:
            st.warning(f"获取财务数据失败: {e}")
            return {}
    
    async def analyze_stock_with_ai(self, symbol, market_data, analysis_type):
        """使用AI分析股票"""
        try:
            # 构建完整的分析数据
            analysis_data = self._build_comprehensive_analysis_data(symbol, market_data)
            
            # 调用AI分析器
            result = await self.ai_analyzer.analyze_market_event(
                symbol=symbol,
                event_type="portfolio_position",
                market_data=analysis_data,
                analysis_type=analysis_type
            )
            
            if result['success']:
                return result
            else:
                return None
        except Exception as e:
            st.error(f"AI分析失败: {e}")
            return None
    
    def _build_comprehensive_analysis_data(self, symbol, market_data):
        """构建综合分析数据"""
        try:
            # 基础市场数据
            current_price = market_data.get('price', 0)
            change_pct = market_data.get('change_pct', 0)
            volume = market_data.get('volume', 0)
            
            # 技术指标
            tech_indicators = market_data.get('technical_indicators', {})
            rsi = tech_indicators.get('rsi', 50)
            ma20 = tech_indicators.get('ma20', current_price)
            ma50 = tech_indicators.get('ma50', current_price)
            macd = tech_indicators.get('macd', 0)
            macd_signal = tech_indicators.get('macd_signal', 0)
            volume_ratio = tech_indicators.get('volume_ratio', 1)
            volatility = tech_indicators.get('volatility', 0.2)
            trend = tech_indicators.get('trend', 'sideways')
            
            # 财务数据
            financial = market_data.get('financial_data', {})
            pe_ratio = financial.get('pe_ratio', 0)
            peg_ratio = financial.get('peg_ratio', 0)
            pb_ratio = financial.get('pb_ratio', 0)
            roe = financial.get('roe', 0)
            profit_margins = financial.get('profit_margins', 0)
            revenue_growth = financial.get('revenue_growth', 0)
            
            # 公司信息
            company_info = market_data.get('company_info', {})
            company_name = company_info.get('name', symbol)
            sector = company_info.get('sector', 'Unknown')
            industry = company_info.get('industry', 'Unknown')
            market_cap = company_info.get('market_cap', 0)
            
            # 持仓信息（如果有）
            position_info = market_data.get('position_info', {})
            watchlist_info = market_data.get('watchlist_info', {})
            
            # 构建分析数据
            analysis_data = {
                'symbol': symbol,
                'current_price': current_price,
                'change_pct': change_pct,
                'volume': volume,
                'market_cap': market_cap,
                
                # 技术分析
                'technical_analysis': {
                    'rsi': rsi,
                    'ma20': ma20,
                    'ma50': ma50,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'volume_ratio': volume_ratio,
                    'volatility': volatility,
                    'trend': trend,
                    'price_vs_ma20': (current_price - ma20) / ma20 * 100 if ma20 > 0 else 0,
                    'price_vs_ma50': (current_price - ma50) / ma50 * 100 if ma50 > 0 else 0
                },
                
                # 财务分析
                'financial_analysis': {
                    'pe_ratio': pe_ratio,
                    'peg_ratio': peg_ratio,
                    'pb_ratio': pb_ratio,
                    'roe': roe,
                    'profit_margins': profit_margins,
                    'revenue_growth': revenue_growth,
                    'valuation_grade': self._grade_valuation(pe_ratio, peg_ratio, pb_ratio),
                    'profitability_grade': self._grade_profitability(roe, profit_margins),
                    'growth_grade': self._grade_growth(revenue_growth)
                },
                
                # 公司信息
                'company_info': {
                    'name': company_name,
                    'sector': sector,
                    'industry': industry,
                    'market_cap': market_cap,
                    'market_cap_category': self._categorize_market_cap(market_cap)
                },
                
                # 持仓分析（如果有）
                'position_analysis': position_info if position_info else None,
                'watchlist_analysis': watchlist_info if watchlist_info else None,
                
                # 市场环境
                'market_environment': {
                    'trend_strength': self._calculate_trend_strength(trend, current_price, ma20, ma50),
                    'volume_analysis': self._analyze_volume(volume_ratio),
                    'volatility_assessment': self._assess_volatility(volatility),
                    'overall_sentiment': self._calculate_sentiment(rsi, macd, trend, change_pct)
                }
            }
            
            return analysis_data
            
        except Exception as e:
            st.error(f"构建分析数据失败: {e}")
            return market_data
    
    def _grade_valuation(self, pe_ratio, peg_ratio, pb_ratio):
        """评估估值等级"""
        score = 0
        
        if pe_ratio > 0 and pe_ratio < 15:
            score += 2
        elif pe_ratio > 0 and pe_ratio < 25:
            score += 1
            
        if peg_ratio > 0 and peg_ratio < 1:
            score += 2
        elif peg_ratio > 0 and peg_ratio < 1.5:
            score += 1
            
        if pb_ratio > 0 and pb_ratio < 3:
            score += 1
            
        if score >= 4:
            return "优秀"
        elif score >= 2:
            return "良好"
        else:
            return "一般"
    
    def _grade_profitability(self, roe, profit_margins):
        """评估盈利能力等级"""
        score = 0
        
        if roe > 15:
            score += 2
        elif roe > 10:
            score += 1
            
        if profit_margins > 0.2:
            score += 2
        elif profit_margins > 0.1:
            score += 1
            
        if score >= 3:
            return "优秀"
        elif score >= 1:
            return "良好"
        else:
            return "一般"
    
    def _grade_growth(self, revenue_growth):
        """评估成长性等级"""
        if revenue_growth > 0.2:
            return "优秀"
        elif revenue_growth > 0.1:
            return "良好"
        elif revenue_growth > 0:
            return "一般"
        else:
            return "较差"
    
    def _categorize_market_cap(self, market_cap):
        """分类市值"""
        if market_cap > 100000000000:  # 1000亿
            return "LARGE"
        elif market_cap > 10000000000:  # 100亿
            return "MID"
        else:
            return "SMALL"
    
    def _calculate_trend_strength(self, trend, current_price, ma20, ma50):
        """计算趋势强度"""
        if trend == 'up' and current_price > ma20 > ma50:
            return "强势上升趋势"
        elif trend == 'up':
            return "上升趋势"
        elif trend == 'down' and current_price < ma20 < ma50:
            return "强势下降趋势"
        elif trend == 'down':
            return "下降趋势"
        else:
            return "震荡趋势"
    
    def _analyze_volume(self, volume_ratio):
        """分析成交量"""
        if volume_ratio > 2:
            return "放量"
        elif volume_ratio > 1.5:
            return "量增"
        elif volume_ratio < 0.7:
            return "缩量"
        else:
            return "正常"
    
    def _assess_volatility(self, volatility):
        """评估波动率"""
        if volatility > 0.4:
            return "高波动"
        elif volatility > 0.25:
            return "中等波动"
        else:
            return "低波动"
    
    def _calculate_sentiment(self, rsi, macd, trend, change_pct):
        """计算市场情绪"""
        score = 0
        
        # RSI贡献
        if rsi > 70:
            score += 1
        elif rsi < 30:
            score -= 1
            
        # MACD贡献
        if macd > 0:
            score += 1
        else:
            score -= 1
            
        # 趋势贡献
        if trend == 'up':
            score += 1
        elif trend == 'down':
            score -= 1
            
        # 价格变化贡献
        if change_pct > 2:
            score += 1
        elif change_pct < -2:
            score -= 1
            
        if score >= 2:
            return "积极"
        elif score <= -2:
            return "消极"
        else:
            return "中性"
    
    def run_streamlit_app(self):
        """运行Streamlit应用"""
        st.set_page_config(
            page_title="AI每日持股分析监控",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 AI每日持股分析监控系统")
        st.markdown("---")
        
        # 侧边栏配置
        st.sidebar.header("⚙️ 系统配置")
        
        # 加载投资组合
        portfolio_config = self.load_portfolio_config()
        positions = portfolio_config.get('positions', {})
        watchlist = portfolio_config.get('watchlist', {})
        
        if not positions and not watchlist:
            st.error("未找到投资组合配置")
            return
        
        # 股票选择界面
        st.sidebar.markdown("### 📊 股票选择")
        
        # 获取持仓股票和观察仓股票
        position_symbols = list(positions.keys()) if positions else []
        watchlist_symbols = list(watchlist.keys()) if watchlist else []
        
        # 持仓股票选择
        st.sidebar.markdown("#### 💼 持仓股票")
        if position_symbols:
            selected_positions = st.sidebar.multiselect(
                "选择持仓股票进行AI分析",
                position_symbols,
                default=position_symbols[:3] if len(position_symbols) > 3 else position_symbols,
                help="选择您当前持有的股票进行AI分析"
            )
        else:
            st.sidebar.info("当前没有持仓股票")
            selected_positions = []
        
        # 观察仓股票选择
        st.sidebar.markdown("#### 👀 观察仓股票")
        if watchlist_symbols:
            selected_watchlist = st.sidebar.multiselect(
                "选择观察仓股票进行AI分析",
                watchlist_symbols,
                default=watchlist_symbols[:3] if len(watchlist_symbols) > 3 else watchlist_symbols,
                help="选择您关注的观察仓股票进行AI分析"
            )
        else:
            st.sidebar.info("当前没有观察仓股票")
            selected_watchlist = []
        
        # 合并选中的股票
        all_selected_symbols = selected_positions + selected_watchlist
        
        if not all_selected_symbols:
            st.warning("请选择要分析的股票")
            return
        
        # AI分析控制
        enable_ai = st.sidebar.checkbox("启用AI分析", value=True)
        analysis_interval = st.sidebar.slider("分析间隔(秒)", 30, 300, 60)
        
        # 自动刷新
        if st.sidebar.button("🔄 刷新数据"):
            st.rerun()
        
        # 主界面
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 实时市场数据")
            
            if all_selected_symbols:
                # 获取实时数据
                real_time_data = self.get_real_time_data(all_selected_symbols)
                
                if real_time_data:
                    # 显示市场数据表格
                    market_df = []
                    for symbol, data in real_time_data.items():
                        # 判断是持仓还是观察仓
                        stock_type = "持仓" if symbol in selected_positions else "观察仓"
                        
                        position = positions.get(symbol, {})
                        watchlist_info = watchlist.get(symbol, {})
                        
                        shares = position.get('shares', 0)
                        cost_basis = position.get('cost_basis', 0)
                        target_price = watchlist_info.get('target_buy_price', 0)
                        
                        if shares > 0:
                            # 持仓股票
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
                                '盈亏率': f"{pnl_pct:+.2f}%"
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
                                '盈亏率': "N/A"
                            })
                    
                    if market_df:
                        df = pd.DataFrame(market_df)
                        st.dataframe(df, use_container_width=True)
                
                # AI分析结果
                if enable_ai and all_selected_symbols:
                    st.subheader("🤖 AI分析结果")
                    
                    # AI分析控制面板
                    st.markdown("### ⚙️ AI分析控制")
                    
                    # 分析类型选择
                    analysis_types = {
                        "comprehensive": "综合分析",
                        "detailed": "详细分析", 
                        "quick": "快速分析"
                    }
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        selected_analysis_type = st.selectbox(
                            "选择分析类型",
                            options=list(analysis_types.keys()),
                            format_func=lambda x: analysis_types[x],
                            help="选择AI分析的深度和详细程度"
                        )
                    
                    with col2:
                        # 股票选择下拉列表
                        stock_options = [f"{symbol} ({'持仓' if symbol in selected_positions else '观察仓'})" for symbol in all_selected_symbols]
                        selected_stock_display = st.selectbox(
                            "选择要分析的股票",
                            options=stock_options,
                            help="选择要进行AI分析的股票"
                        )
                        
                        # 提取股票代码
                        selected_stock = selected_stock_display.split(" (")[0] if selected_stock_display else None
                    
                    with col3:
                        # 分析按钮
                        analyze_button = st.button("🔍 开始AI分析", type="primary", help="点击开始AI分析")
                    
                    # 执行AI分析
                    if analyze_button and selected_stock and selected_stock in real_time_data:
                        market_data = real_time_data[selected_stock]
                        
                        # 添加持仓信息
                        position = positions.get(selected_stock, {})
                        watchlist_info = watchlist.get(selected_stock, {})
                        
                        if position.get('shares', 0) > 0:
                            market_data['position_info'] = {
                                'shares': position.get('shares', 0),
                                'cost_basis': position.get('cost_basis', 0),
                                'weight': position.get('weight', 0),
                                'sector': position.get('sector', 'Unknown')
                            }
                        elif watchlist_info:
                            market_data['watchlist_info'] = {
                                'target_buy_price': watchlist_info.get('target_buy_price', 0),
                                'reason': watchlist_info.get('reason', ''),
                                'category': watchlist_info.get('category', 'Unknown')
                            }
                        
                        # 执行AI分析
                        with st.spinner(f"正在分析 {selected_stock}..."):
                            ai_result = asyncio.run(self.analyze_stock_with_ai(selected_stock, market_data, selected_analysis_type))
                            
                            if ai_result:
                                # 显示AI分析结果
                                st.success(f"✅ {selected_stock} AI分析完成")
                                
                                # 创建分析结果卡片
                                with st.container():
                                    st.markdown("### 📊 分析结果")
                                    
                                    # 操作建议
                                    action_suggestion = ai_result.get('action_suggestion', {})
                                    action = action_suggestion.get('action', 'N/A')
                                    reason = action_suggestion.get('reason', '无分析理由')
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("操作建议", action)
                                    with col2:
                                        st.metric("分析类型", analysis_types[selected_analysis_type])
                                    with col3:
                                        st.metric("股票类型", "持仓" if selected_stock in selected_positions else "观察仓")
                                    
                                    # 分析理由
                                    st.markdown("#### 📋 分析理由")
                                    st.info(reason)
                                    
                                    # 详细分析
                                    with st.expander(f"📊 {selected_stock} 详细分析", expanded=True):
                                        st.markdown(ai_result.get('ai_analysis', '无分析内容'))
                                    
                                    # 风险提示
                                    risk_warning = action_suggestion.get('risk_warning', '')
                                    if risk_warning:
                                        st.warning(f"⚠️ **风险提示**: {risk_warning}")
                                
                                # 保存到历史记录
                                self.analysis_history.append({
                                    'symbol': selected_stock,
                                    'timestamp': datetime.now(),
                                    'result': ai_result,
                                    'type': 'position' if selected_stock in selected_positions else 'watchlist',
                                    'analysis_type': selected_analysis_type
                                })
                                
                                st.success("✅ 分析结果已保存到历史记录")
                            else:
                                st.error(f"❌ {selected_stock} AI分析失败")
                    
                    # 显示最近的分析结果
                    if self.analysis_history:
                        st.markdown("### 📚 最近分析结果")
                        
                        # 筛选最近的分析记录
                        recent_history = self.analysis_history[-5:]  # 显示最近5条
                        
                        for record in recent_history:
                            timestamp = record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                            symbol = record['symbol']
                            result = record['result']
                            record_type = record['type']
                            analysis_type = record.get('analysis_type', 'unknown')
                            
                            # 获取操作建议
                            action_suggestion = result.get('action_suggestion', {})
                            action = action_suggestion.get('action', 'N/A')
                            
                            # 显示记录
                            with st.expander(f"📅 {timestamp} - {symbol} ({record_type}) - {action}", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**股票**: {symbol}")
                                    st.write(f"**类型**: {record_type}")
                                with col2:
                                    st.write(f"**分析类型**: {analysis_types.get(analysis_type, analysis_type)}")
                                    st.write(f"**操作建议**: {action}")
                                with col3:
                                    st.write(f"**分析时间**: {timestamp}")
                                
                                # 显示分析理由
                                reason = action_suggestion.get('reason', '无分析理由')
                                st.markdown(f"**分析理由**: {reason}")
                                
                                # 显示详细分析（可折叠）
                                ai_analysis = result.get('ai_analysis', '无分析内容')
                                if ai_analysis and len(ai_analysis) > 100:
                                    with st.expander("查看完整分析内容", expanded=False):
                                        st.markdown(ai_analysis)
                                elif ai_analysis:
                                    st.markdown(f"**分析内容**: {ai_analysis}")
        
        with col2:
            # 分析历史
            if self.analysis_history:
                st.subheader("📚 分析历史")
                for record in self.analysis_history[-5:]:  # 显示最近5条
                    timestamp = record['timestamp'].strftime('%H:%M:%S')
                    symbol = record['symbol']
                    action = record['result'].get('action_suggestion', {}).get('action', 'N/A')
                    record_type = record.get('type', 'unknown')
                    st.write(f"**{timestamp}** - {symbol} ({record_type}): {action}")
            else:
                st.info("暂无分析历史记录")
        
        # 页脚
        st.markdown("---")
        st.markdown(f"*最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

def main():
    """主函数"""
    monitor = AIDailyAnalysisMonitor()
    monitor.run_streamlit_app()

if __name__ == "__main__":
    main() 