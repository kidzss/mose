#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模块集成示例
展示如何将AI分析模块集成到专业实时交易监控系统
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入AI交易模块
from ai_trading_module import AITradingModule, AITradingUI

# 导入现有的监控系统（示例）
try:
    from intraday_realtime_monitor import RealtimeIntradayMonitor
except ImportError:
    print("⚠️ 未找到intraday_realtime_monitor模块，将使用模拟数据")

class AIEnhancedMonitor:
    """AI增强的实时监控系统"""
    
    def __init__(self):
        """初始化AI增强监控系统"""
        self.ai_module = AITradingModule()
        self.monitor = None  # 实际的监控系统实例
        
        # 尝试初始化现有监控系统
        try:
            self.monitor = RealtimeIntradayMonitor()
            print("✅ 成功集成现有监控系统")
        except Exception as e:
            print(f"⚠️ 无法初始化现有监控系统: {e}")
            print("将使用模拟数据进行演示")
        
        print("🤖 AI增强监控系统初始化完成")
    
    async def enhanced_monitor_loop(self):
        """AI增强的监控循环"""
        print("🚀 启动AI增强监控循环...")
        
        # 模拟股票数据
        mock_stocks = [
            {
                'symbol': 'NVDA',
                'current_price': 155.02,
                'change_pct': 2.5,
                'volume': 15000000,
                'rsi': 65,
                'macd': 'bullish'
            },
            {
                'symbol': 'AMD',
                'current_price': 59.19,
                'change_pct': -1.2,
                'volume': 12000000,
                'rsi': 75,
                'macd': 'bearish',
                'bollinger_position': 'upper_band'
            },
            {
                'symbol': 'TSLA',
                'current_price': 296.50,
                'change_pct': 3.8,
                'volume': 18000000,
                'rsi': 70,
                'macd': 'bullish',
                'volume_ratio': 2.1
            }
        ]
        
        while True:
            try:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - AI增强监控运行中...")
                
                # 处理每个股票
                for stock_data in mock_stocks:
                    await self._process_stock_with_ai(stock_data)
                
                # 显示AI分析摘要
                await self._show_ai_summary()
                
                # 等待下一次更新
                await asyncio.sleep(60)  # 每分钟更新一次
                
            except KeyboardInterrupt:
                print("\n⏹️ 用户停止监控")
                break
            except Exception as e:
                print(f"❌ 监控循环出错: {e}")
                await asyncio.sleep(10)
    
    async def _process_stock_with_ai(self, stock_data: Dict):
        """使用AI处理单个股票"""
        symbol = stock_data['symbol']
        
        print(f"\n📊 处理 {symbol}...")
        
        # 1. 基础监控逻辑（原有功能）
        await self._basic_monitoring(symbol, stock_data)
        
        # 2. AI分析（新增功能）
        await self._ai_analysis(symbol, stock_data)
        
        # 3. 智能警报（新增功能）
        await self._smart_alerts(symbol, stock_data)
    
    async def _basic_monitoring(self, symbol: str, stock_data: Dict):
        """基础监控逻辑"""
        # 这里可以集成原有的监控逻辑
        price = stock_data['current_price']
        change_pct = stock_data['change_pct']
        
        print(f"  📈 {symbol}: ${price:.2f} ({change_pct:+.2f}%)")
        
        # 基础信号检测
        if abs(change_pct) > 3:
            print(f"  ⚠️ {symbol} 价格波动较大")
        
        if stock_data.get('rsi', 50) > 70:
            print(f"  🔴 {symbol} RSI超买")
        elif stock_data.get('rsi', 50) < 30:
            print(f"  🟢 {symbol} RSI超卖")
    
    async def _ai_analysis(self, symbol: str, stock_data: Dict):
        """AI分析"""
        try:
            # 根据数据特征选择分析类型
            if any(key in stock_data for key in ['rsi', 'macd', 'bollinger_position']):
                analysis_type = "comprehensive"
            elif abs(stock_data.get('change_pct', 0)) > 2:
                analysis_type = "detailed"
            else:
                analysis_type = "quick"
            
            # 调用AI分析
            result = await self.ai_module.analyze_stock_signal(
                symbol, stock_data, analysis_type
            )
            
            if result.get('success'):
                action_suggestion = result.get('action_suggestion', {})
                action = action_suggestion.get('action', '不明确')
                reason = action_suggestion.get('reason', '无')
                
                print(f"  🤖 AI建议: {action} - {reason}")
                
                # 根据AI建议执行相应操作
                await self._execute_ai_suggestion(symbol, action_suggestion)
            else:
                print(f"  ❌ AI分析失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"  ❌ AI分析出错: {e}")
    
    async def _execute_ai_suggestion(self, symbol: str, action_suggestion: Dict):
        """执行AI建议"""
        action = action_suggestion.get('action', '')
        
        if action == '止损':
            print(f"  🚨 执行止损操作: {symbol}")
            # 这里可以集成实际的止损逻辑
            await self._send_alert(symbol, "止损", action_suggestion.get('reason', ''))
            
        elif action == '止盈':
            print(f"  💰 执行止盈操作: {symbol}")
            # 这里可以集成实际的止盈逻辑
            await self._send_alert(symbol, "止盈", action_suggestion.get('reason', ''))
            
        elif action == '减仓':
            print(f"  📉 执行减仓操作: {symbol}")
            # 这里可以集成实际的减仓逻辑
            await self._send_alert(symbol, "减仓", action_suggestion.get('reason', ''))
            
        elif action == '加仓':
            print(f"  📈 执行加仓操作: {symbol}")
            # 这里可以集成实际的加仓逻辑
            
        elif action == '观望':
            print(f"  👀 保持观望: {symbol}")
    
    async def _smart_alerts(self, symbol: str, stock_data: Dict):
        """智能警报"""
        # 获取AI警报
        alerts = self.ai_module.get_alerts(symbol, limit=1)
        
        for alert in alerts:
            if alert.get('priority') == 'high':
                print(f"  🚨 高优先级警报: {symbol} - {alert.get('action')}")
                await self._send_alert(
                    symbol, 
                    alert.get('action', ''), 
                    alert.get('reason', ''),
                    priority='high'
                )
    
    async def _send_alert(self, symbol: str, action: str, reason: str, priority: str = 'medium'):
        """发送警报"""
        # 这里可以集成邮件、微信、钉钉等通知方式
        alert_msg = f"🚨 {symbol} {action}: {reason}"
        
        if priority == 'high':
            print(f"  🔴 高优先级警报: {alert_msg}")
        else:
            print(f"  🟡 普通警报: {alert_msg}")
    
    async def _show_ai_summary(self):
        """显示AI分析摘要"""
        summary = self.ai_module.get_analysis_summary()
        
        if summary.get('total_analyses', 0) > 0:
            print(f"\n📋 AI分析摘要:")
            print(f"  总分析次数: {summary.get('total_analyses', 0)}")
            print(f"  成功率: {summary.get('success_rate', 0):.1%}")
            print(f"  分析股票: {list(summary.get('symbol_stats', {}).keys())}")
            
            # 显示最新警报
            alerts = self.ai_module.get_alerts(limit=3)
            if alerts:
                print(f"  最新警报: {len(alerts)} 条")
                for alert in alerts[-3:]:
                    print(f"    - {alert['symbol']}: {alert['action']}")

# Streamlit集成示例
def create_streamlit_ai_dashboard():
    """创建Streamlit AI仪表板"""
    import streamlit as st
    
    st.set_page_config(
        page_title="AI增强交易监控系统",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI增强交易监控系统")
    
    # 初始化AI模块
    if 'ai_module' not in st.session_state:
        st.session_state.ai_module = AITradingModule()
    
    # 侧边栏配置
    st.sidebar.header("🤖 AI配置")
    
    # 分析类型选择
    analysis_type = st.sidebar.selectbox(
        "默认分析类型",
        ["quick", "detailed", "comprehensive"],
        format_func=lambda x: {"quick": "快速分析", "detailed": "详细分析", "comprehensive": "综合分析"}[x]
    )
    
    # 自动分析开关
    auto_analysis = st.sidebar.toggle("自动AI分析", value=True)
    
    # 主界面
    tab1, tab2, tab3 = st.tabs(["📊 实时监控", "🤖 AI分析", "🚨 智能警报"])
    
    with tab1:
        st.header("📊 实时监控")
        
        # 模拟实时数据
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("NVDA", "$155.02", "+2.5%")
        with col2:
            st.metric("AMD", "$59.19", "-1.2%")
        with col3:
            st.metric("TSLA", "$313.17", "-1.18%")
        
        # 手动AI分析按钮
        if st.button("🤖 手动AI分析", type="primary"):
            # 这里可以触发AI分析
            st.success("AI分析已触发")
    
    with tab2:
        st.header("🤖 AI分析")
        
        # 显示最近分析结果
        recent_analyses = st.session_state.ai_module.get_recent_analysis(limit=10)
        
        if recent_analyses:
            for analysis in recent_analyses:
                if analysis.get('success'):
                    action_suggestion = analysis.get('action_suggestion', {})
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.write(f"**{analysis.get('symbol', 'Unknown')}**")
                    with col2:
                        st.write(f"建议: {action_suggestion.get('action', '不明确')}")
                        st.write(f"理由: {action_suggestion.get('reason', '无')}")
                    with col3:
                        st.write(f"时间: {analysis.get('timestamp', '')[:19]}")
                    
                    st.divider()
        else:
            st.info("暂无AI分析记录")
    
    with tab3:
        st.header("🚨 智能警报")
        
        # 显示警报
        alerts = st.session_state.ai_module.get_alerts(limit=10)
        
        if alerts:
            for alert in alerts:
                priority_color = "🔴" if alert.get('priority') == 'high' else "🟡"
                st.write(f"{priority_color} **{alert['symbol']}** - {alert['action']}")
                st.write(f"原因: {alert['reason']}")
                st.write(f"时间: {alert['timestamp'][:19]}")
                st.divider()
        else:
            st.info("暂无警报")

# 测试函数
async def test_integration():
    """测试集成功能"""
    print("🚀 测试AI模块集成...")
    
    # 创建AI增强监控系统
    enhanced_monitor = AIEnhancedMonitor()
    
    # 运行监控循环（运行30秒后停止）
    try:
        await asyncio.wait_for(enhanced_monitor.enhanced_monitor_loop(), timeout=30)
    except asyncio.TimeoutError:
        print("\n⏰ 测试完成（30秒超时）")
    
    print("✅ 集成测试完成")

if __name__ == "__main__":
    # 运行集成测试
    asyncio.run(test_integration()) 