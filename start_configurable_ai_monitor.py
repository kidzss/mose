#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可配置AI每日持股分析监控系统
Configurable AI Daily Holdings Analysis Monitor System
支持选择持仓股票和观察仓股票进行AI分析
"""

import asyncio
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# 导入AI分析器
from ai_realtime_analyzer import AIRealtimeAnalyzer
from daily_holdings_analysis import DailyHoldingsAnalyzer

class ConfigurableAIMonitor:
    """可配置AI每日持股分析监控系统"""
    
    def __init__(self):
        """初始化监控系统"""
        self.ai_analyzer = AIRealtimeAnalyzer(use_daily_analysis=True)
        self.daily_analyzer = DailyHoldingsAnalyzer()
        self.analysis_history = []
        
    def load_portfolio_config(self):
        """加载投资组合配置"""
        config_paths = [
            'portfolio_config.json',
            'config/portfolio_config.json',
            'config/portfolio_config_latest.json'
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        print(f"✅ 成功加载配置文件: {path}")
                        return config
                except Exception as e:
                    print(f"⚠️ 无法读取配置文件 {path}: {e}")
                    continue
        
        st.error("无法找到有效的投资组合配置文件")
        return {}
    
    def get_real_time_data(self, symbols):
        """获取实时数据"""
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='5d', interval='1d')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    data[symbol] = {
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': hist['Volume'].iloc[-1],
                        'market_cap': info.get('marketCap', 0)
                    }
            except Exception as e:
                st.warning(f"获取 {symbol} 数据失败: {e}")
        
        return data
    
    async def analyze_stock_with_ai(self, symbol, market_data, analysis_type="comprehensive"):
        """使用AI分析股票"""
        try:
            result = await self.ai_analyzer.analyze_market_event(
                symbol=symbol,
                event_type="portfolio_position",
                market_data=market_data,
                analysis_type=analysis_type
            )
            
            if result['success']:
                return result
            else:
                return None
        except Exception as e:
            st.error(f"AI分析失败: {e}")
            return None
    
    def run_streamlit_app(self):
        """运行Streamlit应用"""
        st.set_page_config(
            page_title="可配置AI每日持股分析监控",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎯 可配置AI每日持股分析监控系统")
        st.markdown("**支持选择持仓股票和观察仓股票进行AI智能分析**")
        st.markdown("---")
        
        # 侧边栏配置
        st.sidebar.header("⚙️ 系统配置")
        
        # 加载投资组合
        portfolio_config = self.load_portfolio_config()
        
        if not portfolio_config:
            st.error("未找到投资组合配置")
            return
        
        # 获取持仓和观察仓股票
        positions = portfolio_config.get('positions', {})
        watchlist = portfolio_config.get('watchlist', {})
        
        position_symbols = list(positions.keys()) if positions else []
        watchlist_symbols = list(watchlist.keys()) if watchlist else []
        
        # 股票选择界面
        st.sidebar.markdown("### 📊 股票选择")
        
        # 持仓股票选择
        st.sidebar.markdown("#### 💼 持仓股票")
        if position_symbols:
            selected_positions = st.sidebar.multiselect(
                "选择持仓股票",
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
                "选择观察仓股票",
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
            st.warning("请在侧边栏选择要分析的股票")
            return
        
        # AI分析控制
        st.sidebar.markdown("### 🤖 AI分析设置")
        enable_ai = st.sidebar.checkbox("启用AI分析", value=True)
        
        if enable_ai:
            analysis_type = st.sidebar.selectbox(
                "分析类型",
                ["comprehensive", "detailed", "quick"],
                format_func=lambda x: {
                    "comprehensive": "综合分析",
                    "detailed": "详细分析", 
                    "quick": "快速分析"
                }[x]
            )
            
            analysis_interval = st.sidebar.slider("分析间隔(秒)", 30, 300, 60)
        
        # 自动刷新
        if st.sidebar.button("🔄 刷新数据"):
            st.rerun()
        
        # 主界面
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 实时市场数据")
            
            # 获取实时数据
            real_time_data = self.get_real_time_data(all_selected_symbols)
            
            if real_time_data:
                # 创建数据表格
                market_df = []
                for symbol in all_selected_symbols:
                    if symbol in real_time_data:
                        data = real_time_data[symbol]
                        
                        # 判断是持仓还是观察仓
                        stock_type = "持仓" if symbol in selected_positions else "观察仓"
                        
                        # 获取持仓信息
                        position_info = positions.get(symbol, {})
                        watchlist_info = watchlist.get(symbol, {})
                        
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
                if enable_ai and all_selected_symbols:
                    st.subheader("🤖 AI分析结果")
                    
                    # 批量分析按钮
                    if st.button("🔍 批量AI分析", type="primary"):
                        # 为每个选中的股票进行AI分析
                        for symbol in all_selected_symbols:
                            if symbol in real_time_data:
                                market_data = real_time_data[symbol]
                                
                                # 添加持仓信息
                                position_info = positions.get(symbol, {})
                                watchlist_info = watchlist.get(symbol, {})
                                
                                if position_info.get('shares', 0) > 0:
                                    market_data['position_info'] = {
                                        'shares': position_info.get('shares', 0),
                                        'cost_basis': position_info.get('cost_basis', 0),
                                        'weight': position_info.get('weight', 0),
                                        'sector': position_info.get('sector', 'Unknown')
                                    }
                                elif watchlist_info:
                                    market_data['watchlist_info'] = {
                                        'target_buy_price': watchlist_info.get('target_buy_price', 0),
                                        'reason': watchlist_info.get('reason', ''),
                                        'category': watchlist_info.get('category', 'Unknown')
                                    }
                                
                                # 创建AI分析按钮
                                with st.spinner(f"正在分析 {symbol}..."):
                                    ai_result = asyncio.run(self.analyze_stock_with_ai(symbol, market_data, analysis_type))
                                    
                                    if ai_result:
                                        # 显示AI分析结果
                                        st.success(f"✅ {symbol} AI分析完成")
                                        
                                        # 操作建议
                                        action_suggestion = ai_result.get('action_suggestion', {})
                                        st.info(f"**操作建议**: {action_suggestion.get('action', 'N/A')}")
                                        
                                        # 详细分析
                                        with st.expander(f"📊 {symbol} 详细分析", expanded=True):
                                            st.markdown(ai_result.get('ai_analysis', '无分析内容'))
                                        
                                        # 保存到历史记录
                                        self.analysis_history.append({
                                            'symbol': symbol,
                                            'timestamp': datetime.now(),
                                            'result': ai_result,
                                            'type': 'position' if symbol in selected_positions else 'watchlist'
                                        })
                                    else:
                                        st.error(f"❌ {symbol} AI分析失败")
            else:
                st.error("无法获取实时数据")
        
        with col2:
            st.subheader("📋 投资组合概览")
            
            if positions:
                total_value = 0
                total_cost = 0
                portfolio_summary = []
                
                for symbol, info in positions.items():
                    shares = info.get('shares', 0)
                    cost_basis = info.get('cost_basis', 0)
                    
                    if shares > 0:
                        cost_value = cost_basis * shares
                        total_cost += cost_value
                        
                        # 获取实时价格
                        if symbol in real_time_data:
                            current_price = real_time_data[symbol]['price']
                            current_value = current_price * shares
                            total_value += current_value
                            
                            unrealized_pnl = current_value - cost_value
                            pnl_pct = (unrealized_pnl / cost_value) * 100 if cost_value > 0 else 0
                            
                            portfolio_summary.append({
                                '股票': symbol,
                                '权重': f"{info.get('weight', 0):.1f}%",
                                '盈亏率': f"{pnl_pct:+.2f}%"
                            })
                
                # 显示组合总览
                if total_cost > 0:
                    total_pnl = total_value - total_cost
                    total_pnl_pct = (total_pnl / total_cost) * 100
                    
                    st.metric("总市值", f"${total_value:,.2f}")
                    st.metric("总成本", f"${total_cost:,.2f}")
                    st.metric("总盈亏", f"${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
                
                # 显示个股摘要
                if portfolio_summary:
                    summary_df = pd.DataFrame(portfolio_summary)
                    st.dataframe(summary_df, use_container_width=True)
            
            # 分析历史记录
            if self.analysis_history:
                st.subheader("📚 分析历史")
                for record in self.analysis_history[-5:]:  # 显示最近5条
                    timestamp = record['timestamp'].strftime('%H:%M:%S')
                    symbol = record['symbol']
                    action = record['result'].get('action_suggestion', {}).get('action', 'N/A')
                    record_type = record['type']
                    st.write(f"**{timestamp}** - {symbol} ({record_type}): {action}")

def main():
    """主函数"""
    monitor = ConfigurableAIMonitor()
    monitor.run_streamlit_app()

if __name__ == "__main__":
    main() 