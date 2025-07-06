#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI对话界面功能
验证AI对话模块是否正常工作
"""

import sys
import os
import streamlit as st
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_ai_chat_interface():
    """测试AI对话界面功能"""
    st.title("🤖 AI对话界面测试")
    st.markdown("**测试AI对话功能是否正常工作**")
    
    try:
        # 导入AI对话界面
        from monitor.ai_chat_interface import AIChatInterface
        
        # 创建AI对话界面实例
        chat_interface = AIChatInterface()
        
        # 测试基本功能
        st.success("✅ AI对话界面模块导入成功")
        
        # 测试初始化
        if hasattr(chat_interface, 'chat_history'):
            st.success("✅ 对话历史初始化成功")
        else:
            st.error("❌ 对话历史初始化失败")
        
        # 测试会话状态
        if 'chat_messages' in st.session_state:
            st.success("✅ 会话状态初始化成功")
        else:
            st.error("❌ 会话状态初始化失败")
        
        # 测试模拟AI回复
        test_message = "请帮我分析一下NVDA的当前走势"
        test_response = chat_interface._generate_mock_ai_response(test_message, "stock_analysis")
        
        if test_response and 'content' in test_response:
            st.success("✅ 模拟AI回复生成成功")
            
            # 显示测试回复
            with st.expander("📊 查看测试AI回复", expanded=False):
                st.markdown("**测试消息:**")
                st.write(test_message)
                st.markdown("**AI回复:**")
                st.write(test_response['content'])
                
                if 'analysis_result' in test_response:
                    st.markdown("**分析结果:**")
                    st.json(test_response['analysis_result'])
        else:
            st.error("❌ 模拟AI回复生成失败")
        
        # 测试快速问题处理
        quick_questions = [
            "portfolio_health",
            "market_trend", 
            "risk_management"
        ]
        
        st.markdown("### 📋 快速问题测试")
        for question_type in quick_questions:
            if hasattr(chat_interface, '_process_quick_question'):
                st.success(f"✅ 快速问题处理函数存在: {question_type}")
            else:
                st.error(f"❌ 快速问题处理函数不存在: {question_type}")
        
        # 测试导出功能
        if hasattr(chat_interface, '_export_chat_history'):
            st.success("✅ 导出功能存在")
        else:
            st.error("❌ 导出功能不存在")
        
        # 显示功能说明
        st.markdown("### 🎯 功能说明")
        st.markdown("""
        **AI对话功能包括：**
        
        1. **💬 实时对话**
           - 支持多种消息类型
           - 智能AI回复生成
           - 对话历史记录
        
        2. **⚡ 快速操作**
           - 预设问题模板
           - 股票快速分析
           - 市场数据查看
        
        3. **📊 分析结果**
           - 结构化分析显示
           - 风险评估
           - 投资建议
        
        4. **📥 数据管理**
           - 对话历史导出
           - 分析结果保存
           - 会话状态管理
        """)
        
        # 测试集成到主系统
        st.markdown("### 🔗 系统集成测试")
        
        try:
            # 模拟主系统集成
            from enhanced_professional_monitor import AIEnhancedProfessionalMonitor
            
            # 创建监控器实例
            monitor = AIEnhancedProfessionalMonitor()
            
            if hasattr(monitor, '_render_ai_chat_tab'):
                st.success("✅ AI对话标签页渲染函数存在")
            else:
                st.error("❌ AI对话标签页渲染函数不存在")
                
        except ImportError as e:
            st.warning(f"⚠️ 主系统模块导入失败: {e}")
        except Exception as e:
            st.error(f"❌ 系统集成测试失败: {e}")
        
        st.success("🎉 AI对话界面测试完成！")
        
    except ImportError as e:
        st.error(f"❌ AI对话界面模块导入失败: {e}")
        st.info("请确保monitor/ai_chat_interface.py文件存在")
    except Exception as e:
        st.error(f"❌ AI对话界面测试失败: {e}")
        st.info("请检查模块配置和依赖")

def main():
    """主函数"""
    test_ai_chat_interface()

if __name__ == "__main__":
    main() 