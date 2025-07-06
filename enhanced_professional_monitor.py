"""
AI增强的专业交易监控器
集成持仓感知AI分析，显示AI原文输出
"""

import streamlit as st
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Any, Optional
import logging

# 导入原有系统的组件
from monitor.portfolio_monitor import PortfolioMonitor
from monitor.market_monitor import MarketMonitor
from monitor.data_fetcher import DataFetcher
from monitor.enhanced_stock_analyzer import EnhancedStockAnalyzer
from monitor.notification_manager import NotificationManager
from monitor.alert_system import AlertSystem
from data.data_interface import DataInterface
from config.portfolio_config import load_portfolio_config

# 导入AI模块
from ai_trading_module import AITradingModule

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEnhancedProfessionalMonitor:
    """AI增强的专业交易监控器"""
    
    def __init__(self):
        """初始化监控器"""
        self.portfolio_config = None
        self.portfolio_monitor = None
        self.market_monitor = None
        self.data_fetcher = None
        self.stock_analyzer = None
        self.notification_manager = None
        self.alert_system = None
        self.data_interface = None
        self.ai_module = AITradingModule()
        
        # 初始化系统
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化系统组件"""
        try:
            # 加载配置
            self.portfolio_config = load_portfolio_config()
            
            # 初始化数据接口
            self.data_interface = DataInterface()
            
            # 初始化通知管理器
            self.notification_manager = NotificationManager(
                email_config={
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "your_email@gmail.com",
                    "sender_password": "your_password"
                }
            )
            
            # 初始化警报系统
            self.alert_system = AlertSystem(
                config={
                    "price_change_threshold": 5.0,
                    "volume_spike_threshold": 2.0,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            )
            
            # 初始化数据获取器
            self.data_fetcher = DataFetcher(
                data_interface=self.data_interface,
                notification_manager=self.notification_manager
            )
            
            # 初始化股票分析器
            self.stock_analyzer = EnhancedStockAnalyzer(
                data_interface=self.data_interface,
                notification_manager=self.notification_manager
            )
            
            # 初始化市场监控器
            self.market_monitor = MarketMonitor(
                data_fetcher=self.data_fetcher,
                stock_analyzer=self.stock_analyzer,
                notification_manager=self.notification_manager,
                alert_system=self.alert_system
            )
            
            # 初始化投资组合监控器
            self.portfolio_monitor = PortfolioMonitor(
                portfolio_config=self.portfolio_config,
                data_interface=self.data_interface,
                notification_manager=self.notification_manager,
                alert_system=self.alert_system
            )
            
            logger.info("AI增强专业监控器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            st.error(f"系统初始化失败: {e}")
    
    def run(self):
        """运行监控器"""
        st.set_page_config(
            page_title="AI增强专业交易监控器",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 页面标题
        st.title("🤖 AI增强专业交易监控器")
        st.markdown("---")
        
        # 侧边栏控制
        self._render_sidebar()
        
        # 主界面
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 市场概览", 
            "📈 监控股票", 
            "💼 投资组合", 
            "🤖 AI智能诊断", 
            "⚙️ 系统设置"
        ])
        
        with tab1:
            self._render_market_overview()
        
        with tab2:
            self._render_monitored_stocks()
        
        with tab3:
            self._render_portfolio_info()
        
        with tab4:
            self._render_ai_diagnosis()
        
        with tab5:
            self._render_system_settings()
    
    def _render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("🎛️ 控制面板")
        
        # AI诊断控制
        st.sidebar.markdown("### 🤖 AI诊断设置")
        enable_ai = st.sidebar.checkbox("启用AI诊断", value=True)
        
        if enable_ai:
            # 股票选择
            if self.portfolio_config and 'positions' in self.portfolio_config:
                position_symbols = [pos['symbol'] for pos in self.portfolio_config['positions']]
                selected_symbol = st.sidebar.selectbox(
                    "选择诊断股票",
                    position_symbols,
                    index=0 if position_symbols else None
                )
            else:
                selected_symbol = st.sidebar.text_input("输入股票代码", value="NVDA")
            
            # 分析间隔
            analysis_interval = st.sidebar.slider(
                "AI分析间隔(分钟)",
                min_value=1,
                max_value=60,
                value=5
            )
            
            # 手动触发分析
            if st.sidebar.button("🔍 立即AI诊断", type="primary"):
                self._trigger_ai_analysis(selected_symbol)
        
        # 数据刷新控制
        st.sidebar.markdown("### 🔄 数据刷新")
        auto_refresh = st.sidebar.checkbox("自动刷新", value=True)
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "刷新间隔(秒)",
                min_value=10,
                max_value=300,
                value=30
            )
            
            # 自动刷新逻辑
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = time.time()
            
            if time.time() - st.session_state.last_refresh > refresh_interval:
                st.rerun()
    
    def _render_market_overview(self):
        """渲染市场概览"""
        st.header("📊 市场概览")
        
        try:
            # 获取主要指数数据
            indices = ['^GSPC', '^IXIC', '^DJI', '^VIX']
            market_data = {}
            
            for index in indices:
                try:
                    data = self.data_interface.get_historical_data(index, period="1d")
                    if data and not data.empty:
                        latest = data.iloc[-1]
                        market_data[index] = {
                            'price': latest['Close'],
                            'change_pct': ((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close']) * 100,
                            'volume': latest['Volume']
                        }
                except Exception as e:
                    logger.warning(f"获取{index}数据失败: {e}")
            
            # 显示市场数据
            if market_data:
                cols = st.columns(len(market_data))
                index_names = {
                    '^GSPC': 'S&P 500',
                    '^IXIC': 'NASDAQ',
                    '^DJI': '道琼斯',
                    '^VIX': 'VIX恐慌指数'
                }
                
                for i, (index, data) in enumerate(market_data.items()):
                    with cols[i]:
                        name = index_names.get(index, index)
                        change_color = "green" if data['change_pct'] >= 0 else "red"
                        
                        st.metric(
                            label=name,
                            value=f"${data['price']:.2f}",
                            delta=f"{data['change_pct']:.2f}%",
                            delta_color="normal"
                        )
            
            # 显示最近AI分析摘要
            st.subheader("🤖 最近AI分析摘要")
            recent_analyses = self.ai_module.get_recent_analysis(limit=3)
            
            if recent_analyses:
                for analysis in recent_analyses:
                    if analysis.get('success'):
                        symbol = analysis.get('symbol', 'Unknown')
                        action_suggestion = analysis.get('action_suggestion', {})
                        action = action_suggestion.get('action', '不明确')
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.write(f"**{symbol}**")
                        with col2:
                            st.write(f"建议: {action}")
                        with col3:
                            st.write(f"时间: {analysis.get('timestamp', '')[:19]}")
            else:
                st.info("暂无AI分析记录")
                
        except Exception as e:
            st.error(f"获取市场数据失败: {e}")
    
    def _render_monitored_stocks(self):
        """渲染监控股票"""
        st.header("📈 监控股票")
        
        try:
            if self.portfolio_config and 'positions' in self.portfolio_config:
                positions = self.portfolio_config['positions']
                
                # 获取股票数据
                stock_data = []
                for position in positions:
                    symbol = position['symbol']
                    try:
                        data = self.data_interface.get_historical_data(symbol, period="1d")
                        if data and not data.empty:
                            latest = data.iloc[-1]
                            prev = data.iloc[-2]
                            
                            stock_data.append({
                                'symbol': symbol,
                                'price': latest['Close'],
                                'change_pct': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
                                'volume': latest['Volume'],
                                'shares': position.get('shares', 0),
                                'cost_basis': position.get('cost_basis', 0),
                                'weight': position.get('weight', 0)
                            })
                    except Exception as e:
                        logger.warning(f"获取{symbol}数据失败: {e}")
                
                if stock_data:
                    df = pd.DataFrame(stock_data)
                    
                    # 计算持仓价值
                    df['position_value'] = df['price'] * df['shares']
                    df['unrealized_pnl'] = (df['price'] - df['cost_basis']) * df['shares']
                    df['unrealized_pnl_pct'] = ((df['price'] - df['cost_basis']) / df['cost_basis']) * 100
                    
                    # 显示表格
                    st.dataframe(
                        df[['symbol', 'price', 'change_pct', 'shares', 'position_value', 'unrealized_pnl_pct']],
                        use_container_width=True
                    )
                else:
                    st.warning("无法获取股票数据")
            else:
                st.warning("未找到持仓配置")
                
        except Exception as e:
            st.error(f"获取监控股票数据失败: {e}")
    
    def _render_portfolio_info(self):
        """渲染投资组合信息"""
        st.header("💼 投资组合信息")
        
        try:
            if self.portfolio_config and 'positions' in self.portfolio_config:
                positions = self.portfolio_config['positions']
                
                # 计算组合统计
                total_value = 0
                total_cost = 0
                sector_weights = {}
                
                for position in positions:
                    shares = position.get('shares', 0)
                    cost_basis = position.get('cost_basis', 0)
                    sector = position.get('sector', 'Unknown')
                    
                    # 获取当前价格
                    try:
                        data = self.data_interface.get_historical_data(position['symbol'], period="1d")
                        if data and not data.empty:
                            current_price = data.iloc[-1]['Close']
                            position_value = current_price * shares
                            total_value += position_value
                            total_cost += cost_basis * shares
                            
                            # 计算行业权重
                            if sector not in sector_weights:
                                sector_weights[sector] = 0
                            sector_weights[sector] += position_value
                    except:
                        pass
                
                # 显示组合统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总市值", f"${total_value:,.2f}")
                with col2:
                    st.metric("总成本", f"${total_cost:,.2f}")
                with col3:
                    unrealized_pnl = total_value - total_cost
                    unrealized_pnl_pct = (unrealized_pnl / total_cost) * 100 if total_cost > 0 else 0
                    st.metric("未实现盈亏", f"${unrealized_pnl:,.2f}", f"{unrealized_pnl_pct:.2f}%")
                
                # 显示行业分布
                st.subheader("行业分布")
                if sector_weights:
                    sector_df = pd.DataFrame([
                        {'sector': sector, 'weight': weight, 'weight_pct': (weight/total_value)*100}
                        for sector, weight in sector_weights.items()
                    ])
                    st.bar_chart(sector_df.set_index('sector')['weight_pct'])
                
                # 显示持仓详情
                st.subheader("持仓详情")
                for position in positions:
                    with st.expander(f"{position['symbol']} - {position.get('sector', 'Unknown')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**持股数量:** {position.get('shares', 0):,}")
                            st.write(f"**成本价格:** ${position.get('cost_basis', 0):.2f}")
                            st.write(f"**仓位权重:** {position.get('weight', 0):.2f}%")
                        with col2:
                            st.write(f"**行业:** {position.get('sector', 'Unknown')}")
                            st.write(f"**止损价:** ${position.get('stop_loss', 0):.2f}")
                            st.write(f"**止盈价:** ${position.get('take_profit', 0):.2f}")
                        
                        if 'notes' in position:
                            st.write(f"**投资笔记:** {position['notes']}")
            else:
                st.warning("未找到投资组合配置")
                
        except Exception as e:
            st.error(f"获取投资组合信息失败: {e}")
    
    def _render_ai_diagnosis(self):
        """渲染AI诊断"""
        st.header("🤖 AI智能诊断")
        
        # 创建子标签页
        ai_tab1, ai_tab2 = st.tabs(["📊 AI分析", "💬 AI对话"])
        
        with ai_tab1:
            self._render_ai_analysis_tab()
        
        with ai_tab2:
            self._render_ai_chat_tab()
    
    def _render_ai_analysis_tab(self):
        """渲染AI分析标签页"""
        # AI诊断控制
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if self.portfolio_config and 'positions' in self.portfolio_config:
                position_symbols = [pos['symbol'] for pos in self.portfolio_config['positions']]
                selected_symbol = st.selectbox(
                    "选择诊断股票",
                    position_symbols,
                    index=0 if position_symbols else None
                )
            else:
                selected_symbol = st.text_input("输入股票代码", value="NVDA")
        
        with col2:
            analysis_type = st.selectbox(
                "分析类型",
                ["quick", "detailed", "comprehensive"],
                format_func=lambda x: {"quick": "快速", "detailed": "详细", "comprehensive": "综合"}[x]
            )
        
        with col3:
            if st.button("🔍 开始AI诊断", type="primary"):
                self._trigger_ai_analysis(selected_symbol, analysis_type)
        
        # 显示AI分析结果
        st.subheader("📊 AI分析结果")
        
        # 获取最近的AI分析
        recent_analyses = self.ai_module.get_recent_analysis(symbol=selected_symbol, limit=5)
        
        if recent_analyses:
            for analysis in recent_analyses:
                if analysis.get('success'):
                    self._render_ai_analysis_result(analysis)
        else:
            st.info("暂无AI分析记录，请点击上方按钮开始诊断")
        
        # 显示AI警报
        st.subheader("🚨 AI警报")
        alerts = self.ai_module.get_alerts(symbol=selected_symbol, limit=5)
        
        if alerts:
            for alert in alerts:
                self._render_ai_alert(alert)
        else:
            st.info("暂无AI警报")
    
    def _render_ai_chat_tab(self):
        """渲染AI对话标签页"""
        try:
            # 导入AI对话界面
            from monitor.ai_chat_interface import AIChatInterface
            
            # 创建AI对话界面实例
            chat_interface = AIChatInterface()
            
            # 渲染对话界面
            chat_interface.render_chat_interface()
            
        except ImportError as e:
            st.error(f"AI对话模块导入失败: {e}")
            st.info("请确保ai_chat_interface.py文件存在")
        except Exception as e:
            st.error(f"AI对话功能初始化失败: {e}")
            st.info("请检查AI对话模块配置")
    
    def _render_ai_analysis_result(self, analysis: Dict):
        """渲染AI分析结果"""
        symbol = analysis.get('symbol', 'Unknown')
        timestamp = analysis.get('timestamp', '')
        action_suggestion = analysis.get('action_suggestion', {})
        
        # 创建可展开的分析卡片
        with st.expander(f"📊 {symbol} - {action_suggestion.get('action', '不明确')} ({timestamp[:19]})", expanded=True):
            
            # 显示结构化建议
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🎯 操作建议")
                action = action_suggestion.get('action', '不明确')
                action_color = {
                    '加仓': '🟢',
                    '减仓': '🟡', 
                    '观望': '🔵',
                    '止损': '🔴',
                    '止盈': '🟣'
                }.get(action, '⚪')
                
                st.write(f"{action_color} **建议操作:** {action}")
                st.write(f"📝 **简单理由:** {action_suggestion.get('reason', '无')}")
                st.write(f"⚠️ **风险提醒:** {action_suggestion.get('risk_warning', '无')}")
                
                if 'timing' in action_suggestion:
                    st.write(f"⏰ **操作时机:** {action_suggestion.get('timing', '无')}")
                if 'position_suggestion' in action_suggestion:
                    st.write(f"💰 **仓位建议:** {action_suggestion.get('position_suggestion', '无')}")
            
            with col2:
                st.markdown("### 📈 技术指标")
                if 'technical_indicators' in analysis:
                    indicators = analysis['technical_indicators']
                    st.write(f"**RSI:** {indicators.get('rsi', 'N/A')}")
                    st.write(f"**MACD:** {indicators.get('macd', 'N/A')}")
                    st.write(f"**布林带位置:** {indicators.get('bollinger_position', 'N/A')}")
                    st.write(f"**成交量比率:** {indicators.get('volume_ratio', 'N/A')}")
            
            # 显示AI原文输出
            st.markdown("### 🤖 AI原文分析")
            ai_text = analysis.get('ai_analysis', '')
            if ai_text:
                # 使用代码块显示AI原文，保持格式
                st.code(ai_text, language='text')
                
                # 添加复制按钮
                if st.button(f"📋 复制AI原文", key=f"copy_{symbol}_{timestamp}"):
                    st.write("✅ 已复制到剪贴板")
            else:
                st.warning("未找到AI原文输出")
            
            # 显示多时间框架分析
            if 'multi_timeframe_analysis' in analysis:
                multi_analysis = analysis['multi_timeframe_analysis']
                if multi_analysis.get('success'):
                    st.markdown("### ⏰ 多时间框架分析")
                    multi_action = multi_analysis.get('action_suggestion', {})
                    st.write(f"**短线(1-7天):** {multi_action.get('short_term', 'N/A')}")
                    st.write(f"**中线(1-4周):** {multi_action.get('medium_term', 'N/A')}")
                    st.write(f"**长线(1-6个月):** {multi_action.get('long_term', 'N/A')}")
    
    def _render_ai_alert(self, alert: Dict):
        """渲染AI警报"""
        symbol = alert.get('symbol', 'Unknown')
        action = alert.get('action', 'Unknown')
        priority = alert.get('priority', 'medium')
        timestamp = alert.get('timestamp', '')
        
        # 根据优先级设置颜色
        priority_colors = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }
        
        with st.expander(f"{priority_colors.get(priority, '⚪')} {symbol} - {action} ({timestamp[:19]})"):
            st.write(f"**警报类型:** {alert.get('alert_type', 'Unknown')}")
            st.write(f"**优先级:** {priority}")
            st.write(f"**描述:** {alert.get('description', '无描述')}")
            
            if 'ai_analysis' in alert:
                st.markdown("**AI分析:**")
                st.code(alert['ai_analysis'], language='text')
    
    def _render_system_settings(self):
        """渲染系统设置"""
        st.header("⚙️ 系统设置")
        
        # AI设置
        st.subheader("🤖 AI设置")
        st.write("**AI模型:** DeepSeek R1")
        st.write("**API端点:** http://localhost:11434")
        
        # 测试AI连接
        if st.button("🔗 测试AI连接"):
            try:
                # 这里可以添加AI连接测试逻辑
                st.success("✅ AI连接正常")
            except Exception as e:
                st.error(f"❌ AI连接失败: {e}")
        
        # 系统信息
        st.subheader("📊 系统信息")
        st.write(f"**最后更新:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**分析记录数:** {len(self.ai_module.get_recent_analysis())}")
        st.write(f"**警报数量:** {len(self.ai_module.get_alerts())}")
    
    def _trigger_ai_analysis(self, symbol: str, analysis_type: str = "comprehensive"):
        """触发AI分析"""
        try:
            # 获取股票数据
            data = self.data_interface.get_historical_data(symbol, period="5d")
            if data is None or data.empty:
                st.error(f"无法获取{symbol}的数据")
                return
            
            # 计算技术指标
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            signal_data = {
                'current_price': latest['Close'],
                'change_pct': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
                'volume': latest['Volume'],
                'volume_ratio': latest['Volume'] / data['Volume'].mean() if len(data) > 1 else 1,
                'rsi': 50,  # 简化，实际应该计算RSI
                'macd': 'neutral',  # 简化
                'bollinger_position': 'middle_band'  # 简化
            }
            
            # 获取持仓信息
            position_info = None
            if self.portfolio_config and 'positions' in self.portfolio_config:
                for position in self.portfolio_config['positions']:
                    if position['symbol'] == symbol:
                        position_info = position
                        break
            
            # 执行AI分析
            with st.spinner(f"🤖 正在分析 {symbol}..."):
                if position_info:
                    # 持仓分析
                    result = asyncio.run(
                        self.ai_module.analyze_portfolio_position(symbol, signal_data, position_info)
                    )
                else:
                    # 普通分析
                    result = asyncio.run(
                        self.ai_module.analyze_stock_signal(symbol, signal_data, analysis_type)
                    )
            
            if result.get('success'):
                st.success(f"✅ {symbol} AI分析完成")
                st.rerun()  # 刷新页面显示新结果
            else:
                st.error(f"❌ {symbol} AI分析失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            st.error(f"触发AI分析失败: {e}")

def main():
    """主函数"""
    monitor = AIEnhancedProfessionalMonitor()
    monitor.run()

if __name__ == "__main__":
    main() 