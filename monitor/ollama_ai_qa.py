#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama AI问答模块
使用本地Ollama模型进行AI对话交流
"""

import streamlit as st
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaAIQA:
    """Ollama AI问答界面"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """初始化Ollama AI问答界面"""
        self.base_url = base_url
        self.chat_history = []
        self.max_history = 50  # 最大对话历史记录数
        
        # 初始化会话状态
        if 'ollama_chat_messages' not in st.session_state:
            st.session_state.ollama_chat_messages = []
        if 'ollama_model' not in st.session_state:
            st.session_state.ollama_model = "llama3.2"
    
    def get_available_models(self) -> List[str]:
        """获取可用的Ollama模型列表"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                logger.error(f"获取模型列表失败: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"连接Ollama失败: {e}")
            return []
    
    def test_connection(self) -> bool:
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama连接测试失败: {e}")
            return False
    
    def generate_response(self, message: str, model: str = None, max_retries: int = 2) -> Dict:
        """生成AI回复，支持重试机制"""
        if not model:
            model = st.session_state.ollama_model
        
        for attempt in range(max_retries + 1):
            try:
                # 根据用户设置和消息长度动态调整超时时间
                timeout_mode = st.session_state.get('timeout_mode', 'auto')
                message_length = len(message)
                
                if timeout_mode == "auto":
                    # 自动模式：根据消息长度调整
                    if message_length > 1000:
                        timeout = 300  # 大消息使用5分钟超时
                    elif message_length > 500:
                        timeout = 180  # 中等消息使用3分钟超时
                    else:
                        timeout = 120  # 小消息使用2分钟超时
                elif timeout_mode == "short":
                    timeout = 120
                elif timeout_mode == "medium":
                    timeout = 180
                elif timeout_mode == "long":
                    timeout = 300
                elif timeout_mode == "extended":
                    timeout = 600  # 超长模式：10分钟
                else:
                    timeout = 180  # 默认3分钟
                
                # 构建请求数据
                request_data = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": message
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 4000  # 增加最大token数
                    }
                }
                
                # 发送请求
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=request_data,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'success': True,
                        'content': result.get('message', {}).get('content', ''),
                        'model': model,
                        'timestamp': datetime.now().isoformat(),
                        'attempt': attempt + 1
                    }
                else:
                    return {
                        'success': False,
                        'error': f"API请求失败: {response.status_code}",
                        'content': ''
                    }
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    logger.warning(f"第{attempt + 1}次尝试超时，正在重试...")
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    logger.error(f"所有{max_retries + 1}次尝试都超时")
                    return {
                        'success': False,
                        'error': f"请求超时 (已重试{max_retries + 1}次，每次{timeout}秒)",
                        'content': ''
                    }
            except Exception as e:
                logger.error(f"生成AI回复失败: {e}")
                return {
                    'success': False,
                    'error': f"生成回复失败: {str(e)}",
                    'content': ''
                }
    
    def render_qa_interface(self):
        """渲染AI问答界面"""
        st.header("💬 AI智能问答")
        st.markdown("**与本地Ollama AI进行实时对话，获取投资建议和分析**")
        
        # 检查Ollama连接
        if not self.test_connection():
            st.error("❌ 无法连接到Ollama服务")
            st.info("请确保Ollama服务正在运行: `ollama serve`")
            st.info("如果还没有安装Ollama，请访问: https://ollama.ai")
            return
        
        # 创建两列布局
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_chat_area()
        
        with col2:
            self._render_quick_actions()
    
    def _render_chat_area(self):
        """渲染对话区域"""
        st.markdown("### 💬 对话区域")
        
        # 显示对话历史
        self._display_chat_history()
        
        # 用户输入区域
        st.markdown("### 📝 发送消息")
        
        # 消息类型选择
        message_type = st.selectbox(
            "消息类型",
            ["general", "stock_analysis", "portfolio_review", "risk_assessment", "strategy_discussion", "market_analysis"],
            format_func=lambda x: {
                "general": "💬 一般咨询",
                "stock_analysis": "📈 股票分析", 
                "portfolio_review": "💼 投资组合回顾",
                "risk_assessment": "⚠️ 风险评估",
                "strategy_discussion": "🎯 策略讨论",
                "market_analysis": "🌍 市场分析"
            }[x],
            help="选择您要咨询的类型"
        )
        
        # 用户输入
        user_message = st.text_area(
            "输入您的消息:",
            placeholder="例如：请帮我分析一下NVDA的当前走势，我持有100股，成本价是$150...",
            height=150,
            help="详细描述您的问题，AI会根据您的描述提供个性化建议。长消息会自动延长处理时间。"
        )
        
        # 显示消息长度提示
        if user_message:
            message_length = len(user_message)
            if message_length > 1000:
                st.warning(f"⚠️ 消息较长 ({message_length} 字符)，预计需要 5 分钟处理时间")
            elif message_length > 500:
                st.info(f"ℹ️ 消息中等长度 ({message_length} 字符)，预计需要 3 分钟处理时间")
            else:
                st.success(f"✅ 消息长度适中 ({message_length} 字符)，预计需要 2 分钟处理时间")
        
        # 发送按钮
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📤 发送消息", type="primary", use_container_width=True):
                if user_message.strip():
                    self._process_user_message(user_message, message_type)
                else:
                    st.warning("请输入消息内容")
        
        with col2:
            if st.button("🔄 清空对话", use_container_width=True):
                st.session_state.ollama_chat_messages = []
                st.rerun()
        
        with col3:
            if st.button("📥 导出对话", use_container_width=True):
                self._export_chat_history()
    
    def _render_quick_actions(self):
        """渲染快速操作区域"""
        st.markdown("### ⚡ 快速操作")
        
        # 模型选择
        st.markdown("#### 🤖 AI模型")
        available_models = self.get_available_models()
        
        if available_models:
            selected_model = st.selectbox(
                "选择AI模型",
                available_models,
                index=available_models.index(st.session_state.ollama_model) if st.session_state.ollama_model in available_models else 0,
                help="选择要使用的Ollama模型"
            )
            st.session_state.ollama_model = selected_model
            st.success(f"✅ 当前模型: {selected_model}")
        else:
            st.warning("⚠️ 未检测到可用模型")
            st.info("请先下载模型: `ollama pull llama3.2`")
        
        # 超时设置
        st.markdown("#### ⏱️ 超时设置")
        timeout_mode = st.selectbox(
            "超时模式",
            ["auto", "short", "medium", "long", "extended"],
            index=0,
            format_func=lambda x: {
                "auto": "🔄 自动 (根据消息长度)",
                "short": "⚡ 快速 (2分钟)",
                "medium": "⏱️ 标准 (3分钟)", 
                "long": "🐌 长时 (5分钟)",
                "extended": "⏰ 超长 (10分钟)"
            }[x],
            help="选择AI回复的超时时间，老电脑建议使用长时或超长模式"
        )
        
        # 保存超时设置到会话状态
        if 'timeout_mode' not in st.session_state:
            st.session_state.timeout_mode = "auto"
        st.session_state.timeout_mode = timeout_mode
        
        # 快速问题模板
        st.markdown("#### 📋 快速问题")
        
        quick_questions = {
            "portfolio_health": "💼 我的投资组合健康状况如何？",
            "market_trend": "📈 当前市场趋势分析",
            "risk_management": "⚠️ 风险控制建议",
            "entry_timing": "⏰ 最佳买入时机分析",
            "exit_strategy": "🚪 止损止盈策略建议",
            "ai_models": "🤖 推荐哪些AI模型用于投资分析？"
        }
        
        for key, question in quick_questions.items():
            if st.button(question, key=f"ollama_quick_{key}", use_container_width=True):
                self._process_quick_question(key, question)
        
        # 连接状态
        st.markdown("#### 🔗 连接状态")
        if self.test_connection():
            st.success("✅ Ollama连接正常")
        else:
            st.error("❌ Ollama连接失败")
        
        # 模型信息
        if available_models:
            st.markdown("#### 📊 模型信息")
            st.write(f"**可用模型数量:** {len(available_models)}")
            st.write(f"**当前模型:** {st.session_state.ollama_model}")
    
    def _display_chat_history(self):
        """显示对话历史"""
        st.markdown("#### 📜 对话历史")
        
        if not st.session_state.ollama_chat_messages:
            st.info("暂无对话记录，开始您的第一次对话吧！")
            return
        
        # 显示对话记录
        for i, message in enumerate(st.session_state.ollama_chat_messages):
            if message['role'] == 'user':
                # 用户消息
                with st.chat_message("user"):
                    st.write(f"**{message['timestamp']}**")
                    st.write(message['content'])
                    
                    # 显示消息类型标签
                    message_type = message.get('type', 'general')
                    type_labels = {
                        "general": "💬 一般咨询",
                        "stock_analysis": "📈 股票分析",
                        "portfolio_review": "💼 投资组合回顾", 
                        "risk_assessment": "⚠️ 风险评估",
                        "strategy_discussion": "🎯 策略讨论",
                        "market_analysis": "🌍 市场分析"
                    }
                    st.caption(type_labels.get(message_type, "💬 一般咨询"))
            
            elif message['role'] == 'assistant':
                # AI回复
                with st.chat_message("assistant"):
                    st.write(f"**{message['timestamp']}**")
                    st.write(message['content'])
                    
                    # 显示模型信息
                    if 'model' in message:
                        st.caption(f"🤖 模型: {message['model']}")
                    
                    # 显示重试信息
                    if 'attempt' in message and message['attempt'] > 1:
                        st.info(f"🔄 重试次数: {message['attempt']} 次")
                    
                    # 如果有错误信息，显示错误
                    if 'error' in message:
                        st.error(f"❌ 错误: {message['error']}")
        
        # 滚动到最新消息
        st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
    
    def _process_user_message(self, message: str, message_type: str):
        """处理用户消息"""
        try:
            # 添加用户消息到历史
            user_message_data = {
                'role': 'user',
                'content': message,
                'type': message_type,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            st.session_state.ollama_chat_messages.append(user_message_data)
            
            # 生成AI回复
            message_length = len(message)
            timeout_mode = st.session_state.get('timeout_mode', 'auto')
            
            # 根据超时模式显示不同的进度提示
            if timeout_mode == "auto":
                if message_length > 1000:
                    progress_text = "🤖 AI正在处理长消息，请耐心等待 (预计5分钟)..."
                elif message_length > 500:
                    progress_text = "🤖 AI正在处理中等长度消息，请稍候 (预计3分钟)..."
                else:
                    progress_text = "🤖 AI正在思考中..."
            elif timeout_mode == "short":
                progress_text = "🤖 AI正在快速处理中 (2分钟超时)..."
            elif timeout_mode == "medium":
                progress_text = "🤖 AI正在处理中 (3分钟超时)..."
            elif timeout_mode == "long":
                progress_text = "🤖 AI正在深度处理中 (5分钟超时)..."
            elif timeout_mode == "extended":
                progress_text = "🤖 AI正在超长处理中 (10分钟超时)..."
            else:
                progress_text = "🤖 AI正在思考中..."
            
            with st.spinner(progress_text):
                ai_response = self.generate_response(message)
            
            # 添加AI回复到历史
            assistant_message_data = {
                'role': 'assistant',
                'content': ai_response.get('content', '抱歉，AI暂时无法回复'),
                'model': ai_response.get('model', 'unknown'),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            
            if not ai_response.get('success'):
                assistant_message_data['error'] = ai_response.get('error', '未知错误')
            
            st.session_state.ollama_chat_messages.append(assistant_message_data)
            
            # 限制历史记录数量
            if len(st.session_state.ollama_chat_messages) > self.max_history:
                st.session_state.ollama_chat_messages = st.session_state.ollama_chat_messages[-self.max_history:]
            
            st.rerun()
            
        except Exception as e:
            logger.error(f"处理用户消息失败: {e}")
            st.error(f"处理消息失败: {e}")
    
    def _process_quick_question(self, question_type: str, question: str):
        """处理快速问题"""
        # 根据问题类型生成相应的用户消息
        question_messages = {
            "portfolio_health": "请帮我分析一下我的投资组合健康状况，包括风险分散、行业配置、整体表现等方面。",
            "market_trend": "请分析当前市场趋势，包括主要指数走势、市场情绪、宏观经济环境等。",
            "risk_management": "请为我提供风险控制建议，包括仓位管理、止损策略、风险分散等。",
            "entry_timing": "请分析当前是否是好的买入时机，考虑市场环境、个股估值、技术面等因素。",
            "exit_strategy": "请为我提供止损止盈策略建议，包括具体的操作时机和仓位控制。",
            "ai_models": "请推荐一些适合用于投资分析的AI模型，并说明它们的特点和适用场景。"
        }
        
        message = question_messages.get(question_type, question)
        self._process_user_message(message, "general")
    
    def _export_chat_history(self):
        """导出对话历史"""
        try:
            if not st.session_state.ollama_chat_messages:
                st.warning("暂无对话记录可导出")
                return
            
            # 生成导出文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ollama_chat_history_{timestamp}.json"
            
            # 准备导出数据
            export_data = {
                'export_time': datetime.now().isoformat(),
                'total_messages': len(st.session_state.ollama_chat_messages),
                'model_used': st.session_state.ollama_model,
                'messages': st.session_state.ollama_chat_messages
            }
            
            # 创建下载按钮
            st.download_button(
                label="📥 下载对话记录",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=filename,
                mime="application/json"
            )
            
            st.success(f"对话记录已准备下载: {filename}")
            
        except Exception as e:
            logger.error(f"导出对话历史失败: {e}")
            st.error(f"导出失败: {e}")


def main():
    """主函数 - 用于测试"""
    st.title("Ollama AI问答测试")
    
    qa_interface = OllamaAIQA()
    qa_interface.render_qa_interface()


if __name__ == "__main__":
    main() 