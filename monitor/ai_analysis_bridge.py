#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI分析桥接模块
AI Analysis Bridge Module

封装Ollama API调用，为实时交易监控系统提供统一的AI分析接口
"""

import json
import os
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketEvent:
    """市场事件数据结构"""
    symbol: str
    event_type: str  # 'price_alert', 'volume_spike', 'technical_signal', 'news_event'
    timestamp: str
    price: Optional[float] = None
    volume: Optional[float] = None
    change_percent: Optional[float] = None
    technical_indicators: Optional[Dict] = None
    news_content: Optional[str] = None
    user_notes: Optional[List[str]] = None
    portfolio_context: Optional[Dict] = None

@dataclass
class AIAnalysisResult:
    """AI分析结果数据结构"""
    success: bool
    model_used: str
    analysis_type: str
    analysis_content: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    action_suggestions: List[str]
    confidence_score: float
    timestamp: str
    error_message: Optional[str] = None

class AIAnalysisBridge:
    """AI分析桥接类"""
    
    def __init__(self, 
                 api_endpoint: str = "http://localhost:11434/v1/chat/completions",
                 default_model: str = "deepseek-r1:latest",
                 timeout: int = 3000):
        """
        初始化AI分析桥接
        
        Args:
            api_endpoint: Ollama API端点
            default_model: 默认模型名称
            timeout: 请求超时时间
        """
        self.api_endpoint = api_endpoint
        self.default_model = default_model
        self.timeout = timeout
        
        # 分析模板
        self.analysis_templates = {
            "market_event": self._get_market_event_template(),
            "risk_assessment": self._get_risk_assessment_template(),
            "strategy_suggestion": self._get_strategy_suggestion_template(),
            "psychology_analysis": self._get_psychology_analysis_template(),
            "comprehensive": self._get_comprehensive_template()
        }
    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            response = requests.get(
                f"{self.api_endpoint.replace('/v1/chat/completions', '')}/api/tags", 
                timeout=5
            )
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            return []
        except Exception as e:
            logger.warning(f"无法获取模型列表: {e}")
            return []
    
    def get_best_model(self, preferred_model: Optional[str] = None) -> str:
        """获取最佳可用模型"""
        available_models = self.get_available_models()
        
        if not available_models:
            return self.default_model
        
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
    
    def analyze_market_event(self, 
                           event: MarketEvent, 
                           analysis_type: str = "comprehensive",
                           model_name: Optional[str] = None) -> AIAnalysisResult:
        """
        分析市场事件
        
        Args:
            event: 市场事件数据
            analysis_type: 分析类型
            model_name: 指定模型名称
            
        Returns:
            AI分析结果
        """
        try:
            # 获取最佳模型
            best_model = model_name or self.get_best_model()
            logger.info(f"使用模型 {best_model} 分析事件: {event.symbol} - {event.event_type}")
            
            # 构建分析提示
            prompt = self._build_analysis_prompt(event, analysis_type)
            
            # 发送请求
            response = self._send_request(best_model, prompt)
            
            if response:
                # 解析AI响应
                parsed_result = self._parse_ai_response(response, analysis_type)
                
                return AIAnalysisResult(
                    success=True,
                    model_used=best_model,
                    analysis_type=analysis_type,
                    analysis_content=parsed_result.get('analysis', response),
                    risk_level=parsed_result.get('risk_level', 'medium'),
                    action_suggestions=parsed_result.get('suggestions', []),
                    confidence_score=parsed_result.get('confidence', 0.7),
                    timestamp=datetime.now().isoformat()
                )
            else:
                return AIAnalysisResult(
                    success=False,
                    model_used=best_model,
                    analysis_type=analysis_type,
                    analysis_content="",
                    risk_level="unknown",
                    action_suggestions=[],
                    confidence_score=0.0,
                    timestamp=datetime.now().isoformat(),
                    error_message="AI分析请求失败"
                )
                
        except Exception as e:
            logger.error(f"分析市场事件时出错: {e}")
            return AIAnalysisResult(
                success=False,
                model_used=model_name or self.default_model,
                analysis_type=analysis_type,
                analysis_content="",
                risk_level="unknown",
                action_suggestions=[],
                confidence_score=0.0,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
    
    def _build_analysis_prompt(self, event: MarketEvent, analysis_type: str) -> str:
        """构建分析提示"""
        template = self.analysis_templates.get(analysis_type, self.analysis_templates["comprehensive"])
        
        # 构建事件描述
        event_desc = f"""
股票代码: {event.symbol}
事件类型: {event.event_type}
时间: {event.timestamp}
"""
        
        if event.price:
            event_desc += f"当前价格: ${event.price}\n"
        if event.volume:
            event_desc += f"成交量: {event.volume}\n"
        if event.change_percent:
            event_desc += f"涨跌幅: {event.change_percent}%\n"
        if event.technical_indicators:
            event_desc += f"技术指标: {json.dumps(event.technical_indicators, ensure_ascii=False)}\n"
        if event.news_content:
            event_desc += f"相关新闻: {event.news_content}\n"
        if event.user_notes:
            event_desc += f"用户备注: {'; '.join(event.user_notes)}\n"
        if event.portfolio_context:
            event_desc += f"投资组合上下文: {json.dumps(event.portfolio_context, ensure_ascii=False)}\n"
        
        return template.format(event_description=event_desc)
    
    def _send_request(self, model: str, prompt: str) -> Optional[str]:
        """发送请求到Ollama API"""
        try:
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
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "max_tokens": 1000,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"发送请求时出错: {e}")
            return None
    
    def _parse_ai_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """解析AI响应"""
        try:
            # 尝试解析结构化响应
            if "风险等级:" in response:
                risk_level = "high" if "高风险" in response else "medium" if "中风险" in response else "low"
            else:
                risk_level = "medium"
            
            # 提取建议
            suggestions = []
            if "建议:" in response:
                suggestion_text = response.split("建议:")[1].split("\n")[0]
                suggestions = [s.strip() for s in suggestion_text.split(";") if s.strip()]
            
            return {
                "analysis": response,
                "risk_level": risk_level,
                "suggestions": suggestions,
                "confidence": 0.8
            }
        except Exception as e:
            logger.warning(f"解析AI响应时出错: {e}")
            return {
                "analysis": response,
                "risk_level": "medium",
                "suggestions": [],
                "confidence": 0.7
            }
    
    def _get_market_event_template(self) -> str:
        """市场事件分析模板"""
        return """作为专业的投资分析师，请分析以下市场事件：

{event_description}

请提供：
1. 事件影响评估
2. 短期和长期影响分析
3. 风险提示
4. 操作建议

请用中文回答，保持专业、客观、实用。"""

    def _get_risk_assessment_template(self) -> str:
        """风险评估模板"""
        return """作为风险管理专家，请评估以下投资风险：

{event_description}

请提供：
1. 风险等级评估（低/中/高/极高）
2. 主要风险因素
3. 风险控制措施
4. 预警信号

请用中文回答，重点关注风险识别和控制。"""

    def _get_strategy_suggestion_template(self) -> str:
        """策略建议模板"""
        return """作为投资策略顾问，请提供策略建议：

{event_description}

请提供：
1. 当前市场环境分析
2. 投资策略建议
3. 仓位管理建议
4. 时机选择建议

请用中文回答，提供具体可操作的建议。"""

    def _get_psychology_analysis_template(self) -> str:
        """心理分析模板"""
        return """作为投资心理学专家，请分析投资心理状态：

{event_description}

请提供：
1. 当前心理状态评估
2. 情绪管理建议
3. 决策偏差识别
4. 心理调适方法

请用中文回答，关注投资心理和情绪管理。"""

    def _get_comprehensive_template(self) -> str:
        """综合分析模板"""
        return """分析以下投资事件：

{event_description}

请简要分析：
1. 基本面和技术面
2. 风险评估
3. 操作建议

用中文回答，简洁实用。"""

# 便捷函数
def create_market_event(symbol: str, 
                       event_type: str, 
                       **kwargs) -> MarketEvent:
    """创建市场事件"""
    return MarketEvent(
        symbol=symbol,
        event_type=event_type,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def analyze_event_with_bridge(event: MarketEvent, 
                             analysis_type: str = "comprehensive",
                             model_name: Optional[str] = None) -> AIAnalysisResult:
    """使用桥接模块分析事件"""
    bridge = AIAnalysisBridge()
    return bridge.analyze_market_event(event, analysis_type, model_name) 