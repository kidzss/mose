#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动AI对话功能测试
"""

import streamlit as st
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """主函数"""
    st.set_page_config(
        page_title="AI对话功能测试",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI对话功能测试")
    st.markdown("**测试AI对话界面功能**")
    
    # 创建标签页
    tab1, tab2 = st.tabs(["🧪 功能测试", "💬 对话演示"])
    
    with tab1:
        # 导入并运行测试
        try:
            from test_ai_chat_interface import test_ai_chat_interface
            test_ai_chat_interface()
        except Exception as e:
            st.error(f"测试运行失败: {e}")
    
    with tab2:
        # 直接演示AI对话功能
        try:
            from monitor.ai_chat_interface import AIChatInterface
            
            st.markdown("### 💬 AI对话演示")
            st.markdown("**与AI助手进行实时对话，获取投资建议和分析**")
            
            # 创建AI对话界面实例
            chat_interface = AIChatInterface()
            
            # 渲染对话界面
            chat_interface.render_chat_interface()
            
        except ImportError as e:
            st.error(f"AI对话模块导入失败: {e}")
            st.info("请确保monitor/ai_chat_interface.py文件存在")
        except Exception as e:
            st.error(f"AI对话演示失败: {e}")

if __name__ == "__main__":
    main() 