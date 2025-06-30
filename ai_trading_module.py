#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI交易分析模块
集成到专业实时交易监控系统
提供智能化的市场分析和交易建议
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import streamlit as st

# 导入AI实时分析器
from ai_realtime_analyzer import AIRealtimeAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AITradingModule:
    """AI交易分析模块"""
    
    def __init__(self):
        """初始化AI交易模块"""
        self.ai_analyzer = AIRealtimeAnalyzer()
        self.analysis_cache = {}  # 缓存分析结果
        self.alert_history = []   # 警报历史
        
        logger.info("🤖 AI交易分析模块初始化完成")
    
    async def analyze_stock_signal(self, 
                                 symbol: str, 
                                 signal_data: Dict,
                                 analysis_type: str = "quick",
                                 position_info: Dict = None) -> Dict[str, Any]:
        """
        分析股票信号（支持持仓感知）
        
        Args:
            symbol: 股票代码
            signal_data: 信号数据
            analysis_type: 分析类型 (quick, detailed, comprehensive)
            position_info: 持仓信息，包含shares, cost_basis, weight等
            
        Returns:
            分析结果
        """
        try:
            # 构建市场数据
            market_data = self._build_market_data(signal_data)
            
            # 确定事件类型
            event_type = self._determine_event_type(signal_data)
            
            # 如果有持仓信息，添加到分析中
            if position_info:
                market_data['position_info'] = position_info
                # 计算持仓相关指标
                current_price = signal_data.get('current_price', 0)
                cost_basis = position_info.get('cost_basis', 0)
                shares = position_info.get('shares', 0)
                
                if current_price > 0 and cost_basis > 0:
                    # 计算盈亏
                    unrealized_pnl = (current_price - cost_basis) * shares
                    unrealized_pnl_pct = ((current_price - cost_basis) / cost_basis) * 100
                    
                    market_data['position_metrics'] = {
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': unrealized_pnl_pct,
                        'current_price': current_price,
                        'cost_basis': cost_basis,
                        'shares': shares,
                        'position_value': current_price * shares
                    }
            
            # 调用AI分析
            result = await self.ai_analyzer.analyze_market_event(
                symbol, event_type, market_data, analysis_type
            )
            
            # 如果有持仓信息，进行多时间框架分析
            if position_info and result.get('success'):
                multi_timeframe_analysis = await self._perform_multi_timeframe_analysis(
                    symbol, signal_data, position_info
                )
                result['multi_timeframe_analysis'] = multi_timeframe_analysis
            
            # 缓存结果
            cache_key = f"{symbol}_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.analysis_cache[cache_key] = result
            
            # 检查是否需要警报
            if result.get('success'):
                await self._check_alerts(symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票信号失败: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_market_data(self, signal_data: Dict) -> Dict:
        """构建市场数据"""
        market_data = {}
        
        # 基础价格数据
        if 'current_price' in signal_data:
            market_data['current_price'] = signal_data['current_price']
        if 'change_pct' in signal_data:
            market_data['change_pct'] = signal_data['change_pct']
        if 'volume' in signal_data:
            market_data['volume'] = signal_data['volume']
        
        # 技术指标
        if 'rsi' in signal_data:
            market_data['rsi'] = signal_data['rsi']
        if 'macd' in signal_data:
            market_data['macd'] = signal_data['macd']
        if 'bollinger_position' in signal_data:
            market_data['bollinger_position'] = signal_data['bollinger_position']
        
        # 技术信号
        technical_signals = {}
        for key in ['rsi', 'macd', 'bollinger_position', 'ma_position', 'volume_ratio']:
            if key in signal_data:
                technical_signals[key] = signal_data[key]
        
        if technical_signals:
            market_data['technical_signals'] = technical_signals
        
        return market_data
    
    def _determine_event_type(self, signal_data: Dict) -> str:
        """确定事件类型"""
        # 价格相关事件
        if 'change_pct' in signal_data:
            change_pct = abs(signal_data['change_pct'])
            if change_pct > 5:
                return "price_spike"
            elif change_pct > 2:
                return "price_movement"
            else:
                return "price_alert"
        
        # 成交量相关事件
        if 'volume_ratio' in signal_data and signal_data['volume_ratio'] > 2:
            return "volume_spike"
        
        # 技术信号事件
        if any(key in signal_data for key in ['rsi', 'macd', 'bollinger_position']):
            return "technical_signal"
        
        return "market_event"
    
    async def _check_alerts(self, symbol: str, analysis_result: Dict):
        """检查是否需要警报"""
        action_suggestion = analysis_result.get('action_suggestion', {})
        action = action_suggestion.get('action', '')
        
        # 定义警报条件
        alert_conditions = {
            '止损': '需要立即止损',
            '止盈': '建议获利了结',
            '减仓': '建议减仓控制风险'
        }
        
        if action in alert_conditions:
            alert = {
                'symbol': symbol,
                'action': action,
                'reason': action_suggestion.get('reason', ''),
                'risk_warning': action_suggestion.get('risk_warning', ''),
                'timestamp': datetime.now().isoformat(),
                'priority': 'high' if action == '止损' else 'medium'
            }
            
            self.alert_history.append(alert)
            logger.warning(f"🚨 AI警报: {symbol} - {action} - {alert['reason']}")
    
    def get_recent_analysis(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """获取最近的分析结果"""
        history = self.ai_analyzer.get_analysis_history(symbol, limit)
        return history
    
    def get_alerts(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """获取警报历史"""
        alerts = self.alert_history
        
        if symbol:
            alerts = [alert for alert in alerts if alert.get('symbol') == symbol]
        
        return alerts[-limit:] if limit > 0 else alerts
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        return self.ai_analyzer.get_analysis_summary()
    
    async def _perform_multi_timeframe_analysis(self, symbol: str, signal_data: Dict, position_info: Dict) -> Dict:
        """执行多时间框架分析（短线、中线、长线）"""
        try:
            # 构建多时间框架分析提示
            current_price = signal_data.get('current_price', 0)
            cost_basis = position_info.get('cost_basis', 0)
            shares = position_info.get('shares', 0)
            weight = position_info.get('weight', 0)
            sector = position_info.get('sector', 'Unknown')
            
            # 计算持仓指标
            unrealized_pnl_pct = ((current_price - cost_basis) / cost_basis) * 100 if cost_basis > 0 else 0
            position_value = current_price * shares
            
            # 构建分析数据
            analysis_data = {
                'current_price': current_price,
                'change_pct': signal_data.get('change_pct', 0),
                'volume_ratio': signal_data.get('volume_ratio', 1),
                'rsi': signal_data.get('rsi', 50),
                'macd': signal_data.get('macd', 'neutral'),
                'bollinger_position': signal_data.get('bollinger_position', 'middle_band'),
                'position_info': {
                    'shares': shares,
                    'cost_basis': cost_basis,
                    'weight': weight,
                    'sector': sector,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'position_value': position_value
                }
            }
            
            # 调用AI进行多时间框架分析
            result = await self.ai_analyzer.analyze_market_event(
                symbol, 
                "multi_timeframe_analysis", 
                analysis_data, 
                "comprehensive"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"多时间框架分析失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_portfolio_position(self, symbol: str, stock_data: Dict, position_info: Dict) -> Dict[str, Any]:
        """
        专门分析持仓股票
        
        Args:
            symbol: 股票代码
            stock_data: 股票数据
            position_info: 持仓信息
            
        Returns:
            持仓分析结果
        """
        try:
            # 构建持仓分析数据
            position_analysis_data = {
                'current_price': stock_data.get('price', 0),
                'change_pct': stock_data.get('change_pct', 0),
                'volume': stock_data.get('volume', 0),
                'volume_ratio': stock_data.get('volume_ratio', 1),
                'rsi': stock_data.get('rsi', 50),
                'macd': stock_data.get('macd', 'neutral'),
                'bollinger_position': stock_data.get('bollinger_position', 'middle_band'),
                'ma_20': stock_data.get('ma_20', 0),
                'ma_50': stock_data.get('ma_50', 0),
                'position_info': position_info
            }
            
            # 计算持仓指标
            current_price = stock_data.get('price', 0)
            cost_basis = position_info.get('cost_basis', 0)
            shares = position_info.get('shares', 0)
            
            if current_price > 0 and cost_basis > 0:
                unrealized_pnl = (current_price - cost_basis) * shares
                unrealized_pnl_pct = ((current_price - cost_basis) / cost_basis) * 100
                position_value = current_price * shares
                
                position_analysis_data['position_metrics'] = {
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'position_value': position_value,
                    'cost_basis': cost_basis,
                    'shares': shares
                }
            
            # 调用AI分析
            result = await self.ai_analyzer.analyze_market_event(
                symbol, "portfolio_position", position_analysis_data, "comprehensive"
            )
            
            # 添加多时间框架分析
            if result.get('success'):
                multi_timeframe = await self._perform_multi_timeframe_analysis(
                    symbol, stock_data, position_info
                )
                result['multi_timeframe_analysis'] = multi_timeframe
            
            return result
            
        except Exception as e:
            logger.error(f"持仓分析失败: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Streamlit界面集成
class AITradingUI:
    """AI交易分析UI组件"""
    
    def __init__(self):
        """初始化UI组件"""
        self.ai_module = AITradingModule()
    
    def render_ai_analysis_panel(self):
        """渲染AI分析面板"""
        st.markdown("## 🤖 AI智能分析")
        
        # 分析类型选择
        col1, col2 = st.columns(2)
        with col1:
            analysis_type = st.selectbox(
                "分析类型",
                ["quick", "detailed", "comprehensive"],
                format_func=lambda x: {"quick": "快速分析", "detailed": "详细分析", "comprehensive": "综合分析"}[x]
            )
        
        with col2:
            if st.button("🔄 刷新AI分析", type="primary"):
                st.rerun()
        
        # 显示最近分析结果
        st.markdown("### 📊 最近分析结果")
        recent_analyses = self.ai_module.get_recent_analysis(limit=5)
        
        if recent_analyses:
            for analysis in recent_analyses:
                if analysis.get('success'):
                    self._render_analysis_card(analysis)
        else:
            st.info("暂无分析记录")
        
        # 显示警报
        st.markdown("### 🚨 AI警报")
        alerts = self.ai_module.get_alerts(limit=5)
        
        if alerts:
            for alert in alerts:
                self._render_alert_card(alert)
        else:
            st.info("暂无警报")
    
    def _render_analysis_card(self, analysis: Dict):
        """渲染分析卡片"""
        symbol = analysis.get('symbol', 'Unknown')
        action_suggestion = analysis.get('action_suggestion', {})
        action = action_suggestion.get('action', '不明确')
        
        # 根据操作类型设置颜色
        color_map = {
            '加仓': 'green',
            '减仓': 'orange',
            '观望': 'blue',
            '止损': 'red',
            '止盈': 'purple'
        }
        
        color = color_map.get(action, 'gray')
        
        with st.container():
            st.markdown(f"""
            <div style="border: 1px solid {color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                <h4>{symbol} - {action}</h4>
                <p><strong>理由:</strong> {action_suggestion.get('reason', '无')}</p>
                <p><strong>风险:</strong> {action_suggestion.get('risk_warning', '无')}</p>
                <p><strong>时间:</strong> {analysis.get('timestamp', '')[:19]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_alert_card(self, alert: Dict):
        """渲染警报卡片"""
        symbol = alert.get('symbol', 'Unknown')
        action = alert.get('action', 'Unknown')
        priority = alert.get('priority', 'medium')
        
        # 根据优先级设置颜色
        color = 'red' if priority == 'high' else 'orange'
        
        with st.container():
            st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 10px; border-radius: 5px; margin: 5px 0; background-color: #fff3cd;">
                <h4>🚨 {symbol} - {action}</h4>
                <p><strong>原因:</strong> {alert.get('reason', '无')}</p>
                <p><strong>风险:</strong> {alert.get('risk_warning', '无')}</p>
                <p><strong>时间:</strong> {alert.get('timestamp', '')[:19]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    async def analyze_current_stock(self, symbol: str, market_data: Dict):
        """分析当前股票"""
        try:
            with st.spinner(f"🤖 AI正在分析 {symbol}..."):
                result = await self.ai_module.analyze_stock_signal(
                    symbol, market_data, "quick"
                )
            
            if result.get('success'):
                st.success(f"✅ {symbol} AI分析完成")
                
                # 显示分析结果
                action_suggestion = result.get('action_suggestion', {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("建议操作", action_suggestion.get('action', '不明确'))
                
                with col2:
                    st.metric("风险等级", "高" if "风险" in action_suggestion.get('risk_warning', '') else "中")
                
                with col3:
                    st.metric("分析时间", result.get('timestamp', '')[:19])
                
                # 显示详细建议
                st.markdown("### 📋 详细建议")
                st.write(f"**简单理由:** {action_suggestion.get('reason', '无')}")
                st.write(f"**风险提醒:** {action_suggestion.get('risk_warning', '无')}")
                
                if 'timing' in action_suggestion:
                    st.write(f"**操作时机:** {action_suggestion.get('timing', '无')}")
                
                if 'position_suggestion' in action_suggestion:
                    st.write(f"**仓位建议:** {action_suggestion.get('position_suggestion', '无')}")
                
                # 显示完整AI分析
                with st.expander("📄 查看完整AI分析"):
                    st.text(result.get('ai_analysis', '无'))
            else:
                st.error(f"❌ {symbol} AI分析失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            st.error(f"❌ 分析过程出错: {e}")

# 便捷函数
async def quick_ai_analysis(symbol: str, price: float, change_pct: float) -> Dict[str, Any]:
    """快速AI分析"""
    ai_module = AITradingModule()
    signal_data = {
        'current_price': price,
        'change_pct': change_pct
    }
    return await ai_module.analyze_stock_signal(symbol, signal_data, "quick")

# 测试函数
async def test_ai_trading_module():
    """测试AI交易模块"""
    print("🚀 测试AI交易模块...")
    
    ai_module = AITradingModule()
    
    # 测试价格信号分析
    print("\n📊 测试价格信号分析...")
    signal_data = {
        'current_price': 155.02,
        'change_pct': 2.5,
        'volume': 15000000,
        'rsi': 65
    }
    
    result = await ai_module.analyze_stock_signal("NVDA", signal_data, "quick")
    if result.get('success'):
        print("✅ 价格信号分析成功")
        action_suggestion = result.get('action_suggestion', {})
        print(f"建议操作: {action_suggestion.get('action', '不明确')}")
        print(f"简单理由: {action_suggestion.get('reason', '无')}")
        print(f"风险提醒: {action_suggestion.get('risk_warning', '无')}")
    else:
        print(f"❌ 价格信号分析失败: {result.get('error')}")
    
    # 测试技术信号分析
    print("\n🎯 测试技术信号分析...")
    tech_signal = {
        'current_price': 59.19,
        'rsi': 75,
        'macd': 'bearish',
        'bollinger_position': 'upper_band',
        'volume_ratio': 2.5
    }
    
    result2 = await ai_module.analyze_stock_signal("AMD", tech_signal, "comprehensive")
    if result2.get('success'):
        print("✅ 技术信号分析成功")
        action_suggestion = result2.get('action_suggestion', {})
        print(f"建议操作: {action_suggestion.get('action', '不明确')}")
        print(f"简单理由: {action_suggestion.get('reason', '无')}")
        print(f"风险提醒: {action_suggestion.get('risk_warning', '无')}")
        if 'timing' in action_suggestion:
            print(f"操作时机: {action_suggestion.get('timing', '无')}")
        if 'position_suggestion' in action_suggestion:
            print(f"仓位建议: {action_suggestion.get('position_suggestion', '无')}")
    else:
        print(f"❌ 技术信号分析失败: {result2.get('error')}")
    
    # 显示分析摘要
    summary = ai_module.get_analysis_summary()
    print(f"\n📋 分析摘要: {summary}")
    
    # 显示警报
    alerts = ai_module.get_alerts()
    if alerts:
        print(f"\n🚨 警报数量: {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert['symbol']}: {alert['action']} - {alert['reason']}")

if __name__ == "__main__":
    asyncio.run(test_ai_trading_module()) 