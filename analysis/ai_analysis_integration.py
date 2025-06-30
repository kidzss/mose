#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI分析集成模块
AI Analysis Integration Module

用于将用户备注和投资决策数据发送给AI API进行分析
提供智能投资建议和决策优化
"""

import json
import os
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AIAnalysisIntegration:
    """AI分析集成类"""
    
    def __init__(self, api_key: str = None, api_endpoint: str = None):
        """
        初始化AI分析集成
        
        Args:
            api_key: AI API密钥
            api_endpoint: AI API端点
        """
        self.api_key = api_key or os.getenv('AI_API_KEY')
        self.api_endpoint = api_endpoint or os.getenv('AI_API_ENDPOINT')
        
        # 默认配置
        self.default_config = {
            'model': 'gpt-4',  # 默认模型
            'max_tokens': 2000,
            'temperature': 0.7,
            'timeout': 30
        }
        
        # 分析模板
        self.analysis_templates = {
            'investment_strategy': self._get_investment_strategy_template(),
            'risk_assessment': self._get_risk_assessment_template(),
            'decision_optimization': self._get_decision_optimization_template(),
            'psychology_analysis': self._get_psychology_analysis_template()
        }
        
        logger.info("🤖 AI分析集成模块初始化完成")
    
    def analyze_user_notes(self, notes_data: Dict, analysis_type: str = 'comprehensive') -> Dict:
        """
        分析用户备注
        
        Args:
            notes_data: 用户备注数据
            analysis_type: 分析类型 ('comprehensive', 'strategy', 'risk', 'psychology')
            
        Returns:
            分析结果字典
        """
        try:
            if not self.api_key or not self.api_endpoint:
                return self._generate_mock_analysis(notes_data, analysis_type)
            
            # 准备分析提示词
            prompt = self._prepare_analysis_prompt(notes_data, analysis_type)
            
            # 调用AI API
            response = self._call_ai_api(prompt, analysis_type)
            
            # 解析响应
            analysis_result = self._parse_ai_response(response, analysis_type)
            
            # 保存分析结果
            self._save_analysis_result(notes_data, analysis_result, analysis_type)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return self._generate_error_response(str(e))
    
    def _prepare_analysis_prompt(self, notes_data: Dict, analysis_type: str) -> str:
        """准备分析提示词"""
        
        base_prompt = f"""
# 投资决策AI分析请求

## 分析类型: {analysis_type}
## 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 用户数据概览
- 分析股票数量: {len(notes_data.get('notes', []))}
- 用户备注总数: {notes_data.get('summary', {}).get('total_notes', 0)}
- 系统决策总数: {notes_data.get('summary', {}).get('total_decisions', 0)}
- 分析时间范围: 最近90天

## 详细数据
"""
        
        # 添加股票详细数据
        for symbol_data in notes_data.get('notes', []):
            symbol = symbol_data.get('symbol', 'UNKNOWN')
            user_notes = symbol_data.get('user_notes', [])
            decisions = symbol_data.get('decisions', [])
            
            base_prompt += f"""
### {symbol} 分析
**用户备注 ({len(user_notes)} 条):**
"""
            
            for note in user_notes[:5]:  # 最多显示5条备注
                note_time = datetime.fromisoformat(note['timestamp']).strftime('%m-%d %H:%M')
                base_prompt += f"- {note_time}: {note['note']}\n"
            
            base_prompt += f"""
**系统决策 ({len(decisions)} 条):**
"""
            
            for decision in decisions[:3]:  # 最多显示3条决策
                decision_time = datetime.fromisoformat(decision['timestamp']).strftime('%m-%d %H:%M')
                decision_type = decision.get('decision_type', 'UNKNOWN')
                action = decision.get('decision', {}).get('action', 'N/A')
                base_prompt += f"- {decision_time} ({decision_type}): {action}\n"
        
        # 添加分析模板
        if analysis_type in self.analysis_templates:
            base_prompt += self.analysis_templates[analysis_type]
        else:
            base_prompt += self.analysis_templates['investment_strategy']
        
        return base_prompt
    
    def _call_ai_api(self, prompt: str, analysis_type: str) -> Dict:
        """调用AI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.default_config['model'],
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是一位专业的投资顾问和心理学专家，擅长分析投资者的决策模式和心理状态。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': self.default_config['max_tokens'],
                'temperature': self.default_config['temperature']
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=self.default_config['timeout']
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API调用失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"AI API调用失败: {e}")
            raise
    
    def _parse_ai_response(self, response: Dict, analysis_type: str) -> Dict:
        """解析AI响应"""
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # 尝试解析结构化响应
            try:
                # 查找JSON格式的响应
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    structured_data = json.loads(json_str)
                else:
                    structured_data = {}
            except:
                structured_data = {}
            
            return {
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'raw_response': content,
                'structured_data': structured_data,
                'summary': self._extract_summary(content),
                'recommendations': self._extract_recommendations(content),
                'risk_assessment': self._extract_risk_assessment(content),
                'psychology_insights': self._extract_psychology_insights(content)
            }
            
        except Exception as e:
            logger.error(f"解析AI响应失败: {e}")
            return {'error': f'解析响应失败: {str(e)}'}
    
    def _extract_summary(self, content: str) -> str:
        """提取摘要"""
        # 简单的关键词提取
        if '摘要' in content or '总结' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '摘要' in line or '总结' in line:
                    return '\n'.join(lines[i:i+3])
        return content[:200] + '...' if len(content) > 200 else content
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """提取建议"""
        recommendations = []
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['建议', '推荐', '应该', '可以']):
                recommendations.append(line.strip())
        return recommendations[:5]  # 最多5条建议
    
    def _extract_risk_assessment(self, content: str) -> Dict:
        """提取风险评估"""
        risk_level = 'MEDIUM'
        risk_factors = []
        
        if '高风险' in content or '危险' in content:
            risk_level = 'HIGH'
        elif '低风险' in content or '安全' in content:
            risk_level = 'LOW'
        
        # 提取风险因素
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['风险', '危险', '问题', '担忧']):
                risk_factors.append(line.strip())
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors[:3]  # 最多3个风险因素
        }
    
    def _extract_psychology_insights(self, content: str) -> Dict:
        """提取心理洞察"""
        psychology_keywords = {
            'fear': ['恐惧', '害怕', '担心', '焦虑'],
            'greed': ['贪婪', '追涨', '过度自信'],
            'patience': ['耐心', '等待', '冷静'],
            'impulse': ['冲动', '情绪化', '非理性']
        }
        
        insights = {}
        for category, keywords in psychology_keywords.items():
            count = sum(1 for keyword in keywords if keyword in content)
            if count > 0:
                insights[category] = count
        
        return insights
    
    def _save_analysis_result(self, notes_data: Dict, analysis_result: Dict, analysis_type: str):
        """保存分析结果"""
        try:
            result_file = f"ai_analysis_results_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat(),
                    'input_data': notes_data,
                    'analysis_result': analysis_result
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"AI分析结果已保存到: {result_file}")
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
    
    def _generate_mock_analysis(self, notes_data: Dict, analysis_type: str) -> Dict:
        """生成模拟分析结果（当API不可用时）"""
        return {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'status': 'mock_analysis',
            'summary': '基于用户备注的模拟分析结果',
            'recommendations': [
                '建议保持投资记录的习惯',
                '定期回顾投资决策',
                '控制单股仓位风险',
                '设置明确的止损位',
                '保持理性投资心态'
            ],
            'risk_assessment': {
                'risk_level': 'MEDIUM',
                'risk_factors': ['需要更多数据进行分析', '建议接入真实AI API']
            },
            'psychology_insights': {
                'patience': 2,
                'fear': 1
            },
            'note': '这是模拟分析结果，建议配置真实的AI API以获得更准确的分析'
        }
    
    def _generate_error_response(self, error_msg: str) -> Dict:
        """生成错误响应"""
        return {
            'analysis_type': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': error_msg,
            'summary': '分析过程中出现错误',
            'recommendations': ['请检查网络连接', '确认API配置正确', '稍后重试']
        }
    
    def _get_investment_strategy_template(self) -> str:
        """获取投资策略分析模板"""
        return """
## 分析要求
请基于以上数据提供以下分析：

1. **投资策略分析**
   - 识别用户的主要投资策略和风格
   - 评估策略的有效性和一致性
   - 提供策略优化建议

2. **决策模式分析**
   - 分析用户的决策时机选择
   - 评估决策的理性程度
   - 识别决策中的偏见和错误

3. **风险控制评估**
   - 评估用户的风险控制意识
   - 识别潜在的风险点
   - 提供风险控制改进建议

4. **投资心理分析**
   - 分析用户的心理状态
   - 识别情绪化决策的迹象
   - 提供心理调节建议

请提供结构化的分析报告，包含具体的建议和可执行的改进措施。
"""
    
    def _get_risk_assessment_template(self) -> str:
        """获取风险评估模板"""
        return """
## 风险评估要求
请重点分析以下风险方面：

1. **投资组合风险**
   - 集中度风险
   - 相关性风险
   - 流动性风险

2. **决策风险**
   - 时机选择风险
   - 仓位控制风险
   - 止损执行风险

3. **心理风险**
   - 情绪化决策风险
   - 过度自信风险
   - 恐惧贪婪风险

请提供详细的风险评估报告和风险缓解建议。
"""
    
    def _get_decision_optimization_template(self) -> str:
        """获取决策优化模板"""
        return """
## 决策优化要求
请分析以下决策优化方面：

1. **买入决策优化**
   - 时机选择改进
   - 价格控制策略
   - 分批建仓建议

2. **卖出决策优化**
   - 获利了结策略
   - 止损优化
   - 仓位管理改进

3. **持仓管理优化**
   - 仓位调整策略
   - 再平衡建议
   - 风险控制优化

请提供具体的决策优化建议和操作指导。
"""
    
    def _get_psychology_analysis_template(self) -> str:
        """获取心理分析模板"""
        return """
## 心理分析要求
请重点分析以下心理方面：

1. **投资心理状态**
   - 恐惧和贪婪程度
   - 耐心和冲动倾向
   - 自信和谨慎平衡

2. **情绪管理能力**
   - 市场波动时的情绪反应
   - 亏损时的心理承受能力
   - 盈利时的心理状态

3. **决策心理模式**
   - 从众心理倾向
   - 锚定效应影响
   - 确认偏误表现

请提供详细的心理分析报告和心理健康建议。
"""

    def _get_available_models(self):
        """获取可用的模型列表"""
        try:
            response = requests.get(f"{self.api_endpoint.replace('/v1/chat/completions', '')}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            return []
        except Exception as e:
            logger.warning(f"无法获取模型列表: {e}")
            return []
    
    def _get_best_model(self, preferred_model=None):
        """获取最佳可用模型"""
        available_models = self._get_available_models()
        
        if not available_models:
            return self.model
        
        # 如果指定了首选模型，检查是否可用
        if preferred_model:
            for model in available_models:
                if preferred_model in model:
                    return model
        
        # 按优先级查找模型
        model_priorities = [
            'deepseek-r1',
            'llama2',
            'qwen2.5-coder',
            'codellama'
        ]
        
        for priority in model_priorities:
            for model in available_models:
                if priority in model:
                    return model
        
        # 如果都没找到，返回第一个可用模型
        return available_models[0]

    def test_connection(self) -> Dict[str, Any]:
        """
        测试AI连接
        
        Returns:
            连接测试结果
        """
        try:
            # 测试API端点连接
            if self.api_endpoint and 'localhost' in self.api_endpoint:
                # 本地Ollama连接测试
                response = requests.get(
                    f"{self.api_endpoint.replace('/v1/chat/completions', '')}/api/tags", 
                    timeout=10
                )
                if response.status_code == 200:
                    models = response.json()
                    available_models = [model['name'] for model in models.get('models', [])]
                    return {
                        "success": True,
                        "endpoint": self.api_endpoint,
                        "available_models": available_models,
                        "message": f"连接成功，发现 {len(available_models)} 个模型"
                    }
                else:
                    return {
                        "success": False,
                        "endpoint": self.api_endpoint,
                        "error": f"API响应异常: {response.status_code}"
                    }
            else:
                # 云端API连接测试
                if not self.api_key:
                    return {
                        "success": False,
                        "endpoint": self.api_endpoint,
                        "error": "缺少API密钥"
                    }
                
                # 简单的连接测试
                return {
                    "success": True,
                    "endpoint": self.api_endpoint,
                    "message": "云端API配置正确"
                }
                
        except Exception as e:
            return {
                "success": False,
                "endpoint": self.api_endpoint,
                "error": f"连接失败: {str(e)}"
            }

    def _send_request(self, model: str, prompt: str) -> Optional[str]:
        """发送请求到AI API"""
        try:
            if 'localhost' in self.api_endpoint:
                # 本地Ollama请求
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2000
                    }
                }
                
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.error(f"Ollama API请求失败: {response.status_code}")
                    return None
            else:
                # 云端API请求
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'model': self.default_config['model'],
                    'messages': [
                        {
                            'role': 'system',
                            'content': '你是一位专业的投资顾问和心理学专家，擅长分析投资者的决策模式和心理状态。'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'max_tokens': self.default_config['max_tokens'],
                    'temperature': self.default_config['temperature']
                }
                
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.default_config['timeout']
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.error(f"云端API请求失败: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"发送请求失败: {e}")
            return None

    def _build_analysis_prompt(self, symbol: str, notes: List[str], analysis_type: str) -> str:
        """构建分析提示"""
        base_prompt = f"""
# 投资分析请求

## 股票信息
- 股票代码: {symbol}
- 分析类型: {analysis_type}
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 用户备注
"""
        
        for i, note in enumerate(notes, 1):
            base_prompt += f"{i}. {note}\n"
        
        # 添加分析模板
        if analysis_type in self.analysis_templates:
            base_prompt += self.analysis_templates[analysis_type]
        else:
            base_prompt += self.analysis_templates['investment_strategy']
        
        return base_prompt

    def analyze_investment_notes(self, symbol: str, notes: List[str], analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        分析投资备注
        
        Args:
            symbol: 股票代码
            notes: 备注列表
            analysis_type: 分析类型
            
        Returns:
            分析结果
        """
        try:
            # 获取最佳可用模型
            best_model = self._get_best_model(self.model)
            logger.info(f"使用模型: {best_model}")
            
            # 构建分析提示
            prompt = self._build_analysis_prompt(symbol, notes, analysis_type)
            
            # 发送请求
            response = self._send_request(best_model, prompt)
            
            if response:
                return {
                    "success": True,
                    "symbol": symbol,
                    "analysis_type": analysis_type,
                    "model_used": best_model,
                    "analysis": response,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "AI分析失败",
                    "symbol": symbol,
                    "analysis_type": analysis_type
                }
                
        except Exception as e:
            logger.error(f"分析投资备注时出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "analysis_type": analysis_type
            }

# 便捷函数
def analyze_investment_notes(notes_data: Dict, api_key: str = None) -> Dict:
    """分析投资备注的便捷函数"""
    ai_analyzer = AIAnalysisIntegration(api_key=api_key)
    return ai_analyzer.analyze_user_notes(notes_data, 'comprehensive')


if __name__ == "__main__":
    # 测试代码
    test_data = {
        'notes': [
            {
                'symbol': 'AAPL',
                'user_notes': [
                    {
                        'timestamp': '2025-01-15T10:30:00',
                        'note': '苹果股价回调，考虑加仓'
                    },
                    {
                        'timestamp': '2025-01-14T15:20:00',
                        'note': '技术面看好，但担心估值过高'
                    }
                ],
                'decisions': [
                    {
                        'timestamp': '2025-01-15T09:00:00',
                        'decision_type': 'BUY_TIMING',
                        'decision': {'action': '建议买入', 'confidence': 75}
                    }
                ]
            }
        ],
        'summary': {
            'total_symbols': 1,
            'total_notes': 2,
            'total_decisions': 1
        }
    }
    
    ai_analyzer = AIAnalysisIntegration()
    result = ai_analyzer.analyze_user_notes(test_data, 'comprehensive')
    
    print("AI分析结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2)) 