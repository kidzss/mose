"""
测试AI增强监控器的显示功能
验证AI原文是否正确显示
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
from ai_trading_module import AITradingModule

def test_ai_display():
    """测试AI显示功能"""
    st.title("🤖 AI显示功能测试")
    
    # 初始化AI模块
    ai_module = AITradingModule()
    
    # 测试数据
    test_symbol = "NVDA"
    test_data = {
        'current_price': 155.02,
        'change_pct': 2.5,
        'volume': 50000000,
        'volume_ratio': 1.2,
        'rsi': 65,
        'macd': 'bullish',
        'bollinger_position': 'middle_band'
    }
    
    st.write(f"📊 测试股票: {test_symbol}")
    st.json(test_data)
    
    # 手动触发AI分析
    if st.button("🔍 开始AI分析", type="primary"):
        with st.spinner("🤖 正在分析..."):
            # 执行AI分析
            result = asyncio.run(
                ai_module.analyze_stock_signal(test_symbol, test_data, "comprehensive")
            )
            
            if result.get('success'):
                st.success("✅ AI分析完成")
                
                # 显示结构化建议
                action_suggestion = result.get('action_suggestion', {})
                st.markdown("### 🎯 结构化建议")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**建议操作:** {action_suggestion.get('action', '不明确')}")
                    st.write(f"**简单理由:** {action_suggestion.get('reason', '无')}")
                with col2:
                    st.write(f"**风险提醒:** {action_suggestion.get('risk_warning', '无')}")
                
                # 显示AI原文
                st.markdown("### 🤖 AI原文分析")
                ai_text = result.get('ai_analysis', '')
                if ai_text:
                    # 使用代码块显示AI原文
                    st.code(ai_text, language='text')
                    
                    # 显示原文长度
                    st.info(f"AI原文长度: {len(ai_text)} 字符")
                    
                    # 显示原文前200字符预览
                    st.markdown("### 📝 原文预览")
                    st.text(ai_text[:200] + "..." if len(ai_text) > 200 else ai_text)
                else:
                    st.error("❌ 未找到AI原文")
                
                # 显示完整结果
                st.markdown("### 📊 完整分析结果")
                st.json(result)
            else:
                st.error(f"❌ AI分析失败: {result.get('error', '未知错误')}")
    
    # 显示最近分析历史
    st.markdown("### 📈 最近分析历史")
    recent_analyses = ai_module.get_recent_analysis(limit=5)
    
    if recent_analyses:
        for i, analysis in enumerate(recent_analyses):
            if analysis.get('success'):
                symbol = analysis.get('symbol', 'Unknown')
                action = analysis.get('action_suggestion', {}).get('action', '不明确')
                has_ai_text = bool(analysis.get('ai_analysis', ''))
                ai_text_length = len(analysis.get('ai_analysis', ''))
                
                with st.expander(f"{i+1}. {symbol} - {action} (原文长度: {ai_text_length})"):
                    st.write(f"**时间:** {analysis.get('timestamp', '')[:19]}")
                    st.write(f"**有AI原文:** {'✅' if has_ai_text else '❌'}")
                    
                    if has_ai_text:
                        ai_text = analysis.get('ai_analysis', '')
                        st.code(ai_text, language='text')
    else:
        st.info("暂无分析历史")

if __name__ == "__main__":
    test_ai_display() 