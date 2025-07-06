#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI对话界面模块
允许用户与AI进行实时对话，发送投资相关信息并获得AI分析
"""

import streamlit as st
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIChatInterface:
    """AI对话界面"""
    
    def __init__(self):
        """初始化AI对话界面"""
        self.chat_history = []
        self.max_history = 50  # 最大对话历史记录数
        
        # 初始化会话状态
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'ai_analysis_results' not in st.session_state:
            st.session_state.ai_analysis_results = []
    
    def render_chat_interface(self):
        """渲染AI对话界面"""
        st.header("💬 AI投资助手对话")
        st.markdown("**与AI助手进行实时对话，获取投资建议和分析**")
        
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
            ["general", "stock_analysis", "portfolio_review", "risk_assessment", "strategy_discussion"],
            format_func=lambda x: {
                "general": "💬 一般咨询",
                "stock_analysis": "📈 股票分析", 
                "portfolio_review": "💼 投资组合回顾",
                "risk_assessment": "⚠️ 风险评估",
                "strategy_discussion": "🎯 策略讨论"
            }[x],
            help="选择您要咨询的类型"
        )
        
        # 用户输入
        user_message = st.text_area(
            "输入您的消息:",
            placeholder="例如：请帮我分析一下NVDA的当前走势，我持有100股，成本价是$150...",
            height=120,
            help="详细描述您的问题，AI会根据您的描述提供个性化建议"
        )
        
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
                st.session_state.chat_messages = []
                st.session_state.ai_analysis_results = []
                st.rerun()
        
        with col3:
            if st.button("📥 导出对话", use_container_width=True):
                self._export_chat_history()
    
    def _render_quick_actions(self):
        """渲染快速操作区域"""
        st.markdown("### ⚡ 快速操作")
        
        # 快速问题模板
        st.markdown("#### 📋 快速问题")
        
        quick_questions = {
            "portfolio_health": "💼 我的投资组合健康状况如何？",
            "market_trend": "📈 当前市场趋势分析",
            "risk_management": "⚠️ 风险控制建议",
            "entry_timing": "⏰ 最佳买入时机分析",
            "exit_strategy": "🚪 止损止盈策略建议"
        }
        
        for key, question in quick_questions.items():
            if st.button(question, key=f"quick_{key}", use_container_width=True):
                self._process_quick_question(key, question)
        
        # 股票分析快速入口
        st.markdown("#### 📊 股票分析")
        
        # 获取持仓股票
        portfolio_config = self._get_portfolio_config()
        if portfolio_config and 'positions' in portfolio_config:
            position_symbols = [pos['symbol'] for pos in portfolio_config['positions']]
            
            if position_symbols:
                selected_stock = st.selectbox(
                    "选择股票进行快速分析",
                    position_symbols,
                    help="选择您持有的股票进行快速分析"
                )
                
                if st.button(f"🔍 分析 {selected_stock}", use_container_width=True):
                    self._process_stock_analysis(selected_stock)
        
        # 市场数据快速查看
        st.markdown("#### 📈 市场数据")
        
        if st.button("🌍 全球市场概览", use_container_width=True):
            self._process_market_overview()
        
        if st.button("💰 热门股票分析", use_container_width=True):
            self._process_hot_stocks_analysis()
    
    def _display_chat_history(self):
        """显示对话历史"""
        st.markdown("#### 📜 对话历史")
        
        if not st.session_state.chat_messages:
            st.info("暂无对话记录，开始您的第一次对话吧！")
            return
        
        # 显示对话记录
        for i, message in enumerate(st.session_state.chat_messages):
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
                        "strategy_discussion": "🎯 策略讨论"
                    }
                    st.caption(type_labels.get(message_type, "💬 一般咨询"))
            
            elif message['role'] == 'assistant':
                # AI回复
                with st.chat_message("assistant"):
                    st.write(f"**{message['timestamp']}**")
                    st.write(message['content'])
                    
                    # 如果有分析结果，显示详细信息
                    if 'analysis_result' in message:
                        with st.expander("📊 查看详细分析", expanded=False):
                            self._display_analysis_result(message['analysis_result'])
        
        # 滚动到最新消息
        st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
    
    def _display_analysis_result(self, analysis_result: Dict):
        """显示分析结果"""
        if not analysis_result:
            return
        
        # 显示结构化分析
        if 'structured_analysis' in analysis_result:
            structured = analysis_result['structured_analysis']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 主要建议**")
                st.write(structured.get('main_recommendation', '无'))
                
                st.markdown("**⚠️ 风险提示**")
                st.write(structured.get('risk_warning', '无'))
            
            with col2:
                st.markdown("**📈 技术分析**")
                st.write(structured.get('technical_analysis', '无'))
                
                st.markdown("**💰 操作建议**")
                st.write(structured.get('action_suggestion', '无'))
        
        # 显示详细分析
        if 'detailed_analysis' in analysis_result:
            st.markdown("**📋 详细分析**")
            st.write(analysis_result['detailed_analysis'])
        
        # 显示置信度
        if 'confidence' in analysis_result:
            confidence = analysis_result['confidence']
            st.progress(confidence / 100)
            st.caption(f"分析置信度: {confidence}%")
    
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
            st.session_state.chat_messages.append(user_message_data)
            
            # 生成AI回复
            ai_response = self._generate_ai_response(message, message_type)
            
            # 添加AI回复到历史
            assistant_message_data = {
                'role': 'assistant',
                'content': ai_response['content'],
                'analysis_result': ai_response.get('analysis_result'),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            st.session_state.chat_messages.append(assistant_message_data)
            
            # 限制历史记录数量
            if len(st.session_state.chat_messages) > self.max_history:
                st.session_state.chat_messages = st.session_state.chat_messages[-self.max_history:]
            
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
            "exit_strategy": "请为我提供止损止盈策略建议，包括具体的操作时机和仓位控制。"
        }
        
        message = question_messages.get(question_type, question)
        self._process_user_message(message, "general")
    
    def _process_stock_analysis(self, symbol: str):
        """处理股票分析请求"""
        message = f"请帮我分析一下{symbol}的当前情况，包括技术面、基本面、市场表现等，并给出具体的投资建议。"
        self._process_user_message(message, "stock_analysis")
    
    def _process_market_overview(self):
        """处理市场概览请求"""
        message = "请提供全球市场概览，包括主要指数表现、市场热点、风险提示等。"
        self._process_user_message(message, "general")
    
    def _process_hot_stocks_analysis(self):
        """处理热门股票分析请求"""
        message = "请分析当前市场上的热门股票，包括上涨原因、投资机会、风险提示等。"
        self._process_user_message(message, "stock_analysis")
    
    def _generate_ai_response(self, user_message: str, message_type: str) -> Dict:
        """生成AI回复"""
        try:
            # 这里应该调用实际的AI API
            # 目前使用模拟回复
            ai_response = self._generate_mock_ai_response(user_message, message_type)
            return ai_response
            
        except Exception as e:
            logger.error(f"生成AI回复失败: {e}")
            return {
                'content': f"抱歉，AI分析暂时不可用。错误信息: {e}",
                'analysis_result': None
            }
    
    def _generate_mock_ai_response(self, user_message: str, message_type: str) -> Dict:
        """生成模拟AI回复"""
        # 根据消息类型生成不同的回复
        if message_type == "stock_analysis":
            return self._generate_stock_analysis_response(user_message)
        elif message_type == "portfolio_review":
            return self._generate_portfolio_review_response(user_message)
        elif message_type == "risk_assessment":
            return self._generate_risk_assessment_response(user_message)
        elif message_type == "strategy_discussion":
            return self._generate_strategy_discussion_response(user_message)
        else:
            return self._generate_general_response(user_message)
    
    def _generate_stock_analysis_response(self, user_message: str) -> Dict:
        """生成股票分析回复"""
        # 提取股票代码（简单实现）
        symbols = ['NVDA', 'AMD', 'GOOG', 'TSLA', 'AAPL', 'MSFT']
        found_symbols = [s for s in symbols if s.lower() in user_message.lower()]
        
        if found_symbols:
            symbol = found_symbols[0]
            return {
                'content': f"""
基于您的问题，我来分析一下{symbol}的当前情况：

**📈 技术面分析：**
- 当前趋势：上升趋势
- RSI指标：65（中性偏强）
- 成交量：较前期有所增加
- 支撑位：$150
- 阻力位：$180

**💰 基本面分析：**
- 估值水平：合理
- 成长性：良好
- 盈利能力：优秀

**🎯 投资建议：**
- 短期：观望为主，等待回调机会
- 中期：可以考虑分批建仓
- 长期：具有投资价值

**⚠️ 风险提示：**
- 市场波动可能加大
- 注意控制仓位
- 设置合理止损位
                """,
                'analysis_result': {
                    'structured_analysis': {
                        'main_recommendation': '观望为主，等待回调机会',
                        'risk_warning': '市场波动可能加大，注意控制仓位',
                        'technical_analysis': '上升趋势，RSI中性偏强',
                        'action_suggestion': '可以考虑分批建仓'
                    },
                    'detailed_analysis': f'对{symbol}的详细分析包括技术面、基本面和投资建议',
                    'confidence': 75
                }
            }
        else:
            return {
                'content': """
我来为您分析股票投资：

**📊 一般性建议：**
1. 关注技术面和基本面结合
2. 控制单股仓位不超过总资产的10%
3. 设置明确的止损位
4. 保持理性投资心态

**🎯 操作策略：**
- 分批建仓，避免一次性满仓
- 定期回顾和调整投资组合
- 关注市场整体环境变化

**⚠️ 风险控制：**
- 不要追高，等待回调机会
- 设置止损，控制下行风险
- 分散投资，降低集中度风险
                """,
                'analysis_result': {
                    'structured_analysis': {
                        'main_recommendation': '关注技术面和基本面结合',
                        'risk_warning': '不要追高，设置止损',
                        'technical_analysis': '需要具体股票代码进行详细分析',
                        'action_suggestion': '分批建仓，控制仓位'
                    },
                    'detailed_analysis': '提供了一般性的股票投资建议和风险控制策略',
                    'confidence': 60
                }
            }
    
    def _generate_portfolio_review_response(self, user_message: str) -> Dict:
        """生成投资组合回顾回复"""
        return {
            'content': """
**💼 投资组合健康度分析：**

**📊 整体评估：**
- 投资组合表现：良好
- 风险分散度：中等
- 行业配置：需要优化

**🎯 改进建议：**
1. 增加防御性行业配置
2. 控制科技股集中度
3. 考虑添加债券或现金配置
4. 定期再平衡投资组合

**⚠️ 风险提示：**
- 当前科技股占比过高
- 需要关注市场波动风险
- 建议增加现金储备

**📈 具体建议：**
- 可以考虑减仓部分高估值股票
- 增加消费、医疗等防御性行业
- 设置更严格的止损策略
            """,
            'analysis_result': {
                'structured_analysis': {
                    'main_recommendation': '增加防御性行业配置，控制科技股集中度',
                    'risk_warning': '科技股占比过高，需要关注市场波动风险',
                    'technical_analysis': '投资组合整体表现良好，但需要优化配置',
                    'action_suggestion': '减仓部分高估值股票，增加防御性行业'
                },
                'detailed_analysis': '对投资组合进行了全面的健康度分析，包括风险分散、行业配置等方面',
                'confidence': 80
            }
        }
    
    def _generate_risk_assessment_response(self, user_message: str) -> Dict:
        """生成风险评估回复"""
        return {
            'content': """
**⚠️ 风险评估报告：**

**🔍 当前风险状况：**
- 市场风险：中等
- 个股风险：中等偏高
- 流动性风险：低
- 集中度风险：高

**🎯 风险控制建议：**
1. **仓位管理：**
   - 单股仓位不超过10%
   - 行业集中度不超过30%
   - 保持20%现金储备

2. **止损策略：**
   - 设置8-10%止损位
   - 严格执行止损纪律
   - 避免情绪化操作

3. **分散投资：**
   - 增加不同行业配置
   - 考虑添加债券或ETF
   - 定期再平衡

**📊 具体措施：**
- 立即减仓高集中度股票
- 设置自动止损单
- 制定详细的退出策略
            """,
            'analysis_result': {
                'structured_analysis': {
                    'main_recommendation': '立即减仓高集中度股票，设置8-10%止损位',
                    'risk_warning': '集中度风险较高，需要立即分散投资',
                    'technical_analysis': '市场风险中等，个股风险中等偏高',
                    'action_suggestion': '保持20%现金储备，严格执行止损纪律'
                },
                'detailed_analysis': '提供了全面的风险评估和具体的风险控制措施',
                'confidence': 85
            }
        }
    
    def _generate_strategy_discussion_response(self, user_message: str) -> Dict:
        """生成策略讨论回复"""
        return {
            'content': """
**🎯 投资策略讨论：**

**📈 当前策略分析：**
- 策略类型：成长型投资
- 风险偏好：中等偏高
- 投资期限：中长期
- 操作频率：中等

**💡 策略优化建议：**
1. **价值投资结合：**
   - 在成长股中寻找价值洼地
   - 关注基本面良好的公司
   - 避免追高估值股票

2. **技术分析辅助：**
   - 结合技术指标判断买卖时机
   - 关注成交量变化
   - 设置多重技术支撑位

3. **情绪管理：**
   - 保持理性投资心态
   - 避免追涨杀跌
   - 制定明确的投资计划

**📊 具体实施：**
- 建立投资日记，记录决策过程
- 定期回顾和总结投资表现
- 持续学习和改进投资策略
            """,
            'analysis_result': {
                'structured_analysis': {
                    'main_recommendation': '价值投资结合成长投资，避免追高估值股票',
                    'risk_warning': '需要控制风险偏好，避免追涨杀跌',
                    'technical_analysis': '结合技术分析判断买卖时机',
                    'action_suggestion': '建立投资日记，定期回顾和总结'
                },
                'detailed_analysis': '对投资策略进行了深入讨论，提供了具体的优化建议',
                'confidence': 70
            }
        }
    
    def _generate_general_response(self, user_message: str) -> Dict:
        """生成一般性回复"""
        return {
            'content': """
感谢您的咨询！我是您的AI投资助手，可以为您提供：

**📊 主要服务：**
- 股票技术面和基本面分析
- 投资组合健康度评估
- 风险评估和控制建议
- 投资策略讨论和优化
- 市场趋势和热点分析

**💡 使用建议：**
- 提供具体的股票代码获得更准确的分析
- 详细描述您的投资情况和需求
- 定期咨询以跟踪投资表现

**⚠️ 重要提醒：**
- 我的建议仅供参考，不构成投资建议
- 投资有风险，决策需谨慎
- 建议结合专业投资顾问的意见

有什么具体的问题需要我帮助分析吗？
            """,
            'analysis_result': {
                'structured_analysis': {
                    'main_recommendation': '提供具体的股票代码获得更准确的分析',
                    'risk_warning': '投资有风险，决策需谨慎',
                    'technical_analysis': '可以提供技术面和基本面分析',
                    'action_suggestion': '定期咨询以跟踪投资表现'
                },
                'detailed_analysis': '介绍了AI投资助手的主要服务和使用建议',
                'confidence': 90
            }
        }
    
    def _get_portfolio_config(self) -> Optional[Dict]:
        """获取投资组合配置"""
        try:
            # 这里应该从实际的配置文件中读取
            # 目前返回None，表示没有配置
            return None
        except Exception as e:
            logger.error(f"获取投资组合配置失败: {e}")
            return None
    
    def _export_chat_history(self):
        """导出对话历史"""
        try:
            if not st.session_state.chat_messages:
                st.warning("暂无对话记录可导出")
                return
            
            # 生成导出文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ai_chat_history_{timestamp}.json"
            
            # 准备导出数据
            export_data = {
                'export_time': datetime.now().isoformat(),
                'total_messages': len(st.session_state.chat_messages),
                'messages': st.session_state.chat_messages,
                'analysis_results': st.session_state.ai_analysis_results
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
    st.title("AI投资助手对话测试")
    
    chat_interface = AIChatInterface()
    chat_interface.render_chat_interface()


if __name__ == "__main__":
    main() 