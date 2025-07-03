#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI日志装饰器
自动记录AI交互的输入和输出
"""

import functools
import inspect
from typing import Callable, Any, Dict, Optional
from .ai_logger import AILogger, log_ai_input, log_ai_output, log_ai_error

def log_ai_interaction(model: Optional[str] = None, 
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None):
    """装饰器：自动记录AI交互
    
    Args:
        model: AI模型名称
        temperature: 温度参数
        max_tokens: 最大token数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 提取参数
            prompt = bound_args.arguments.get('prompt', '')
            context = bound_args.arguments.get('context', {})
            
            # 记录输入
            interaction_id = log_ai_input(
                prompt=prompt,
                context=context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                function_name=func.__name__,
                args=args,
                kwargs=kwargs
            )
            
            try:
                # 调用原函数
                result = func(*args, **kwargs)
                
                # 记录输出
                log_ai_output(
                    interaction_id=interaction_id,
                    response=str(result),
                    model=model,
                    function_name=func.__name__
                )
                
                return result
                
            except Exception as e:
                # 记录错误
                log_ai_error(
                    interaction_id=interaction_id,
                    error=e,
                    error_type="function_error",
                    function_name=func.__name__
                )
                raise
        
        return wrapper
    return decorator

def log_ai_call(prompt_param: str = 'prompt', 
                context_param: str = 'context',
                response_param: str = 'response'):
    """装饰器：记录AI调用，适用于包装AI API调用的函数
    
    Args:
        prompt_param: 提示词参数名
        context_param: 上下文参数名
        response_param: 响应参数名
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取参数
            prompt = kwargs.get(prompt_param, '')
            context = kwargs.get(context_param, {})
            
            # 记录输入
            interaction_id = log_ai_input(
                prompt=prompt,
                context=context,
                function_name=func.__name__,
                args=args,
                kwargs=kwargs
            )
            
            try:
                # 调用原函数
                result = func(*args, **kwargs)
                
                # 提取响应
                if isinstance(result, dict):
                    response = result.get(response_param, str(result))
                else:
                    response = str(result)
                
                # 记录输出
                log_ai_output(
                    interaction_id=interaction_id,
                    response=response,
                    function_name=func.__name__,
                    result=result
                )
                
                return result
                
            except Exception as e:
                # 记录错误
                log_ai_error(
                    interaction_id=interaction_id,
                    error=e,
                    error_type="api_error",
                    function_name=func.__name__
                )
                raise
        
        return wrapper
    return decorator

class AILoggerMixin:
    """AI日志混入类，为类添加AI日志功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ai_logger = AILogger()
        self._interaction_count = 0
    
    def log_ai_input(self, prompt: str, **kwargs) -> str:
        """记录AI输入"""
        self._interaction_count += 1
        return self.ai_logger.log_ai_input(
            prompt=prompt,
            context={"class": self.__class__.__name__, "interaction_count": self._interaction_count},
            **kwargs
        )
    
    def log_ai_output(self, interaction_id: str, response: str, **kwargs) -> None:
        """记录AI输出"""
        self.ai_logger.log_ai_output(interaction_id, response, **kwargs)
    
    def log_ai_error(self, interaction_id: str, error: Exception, **kwargs) -> None:
        """记录AI错误"""
        self.ai_logger.log_ai_error(interaction_id, error, **kwargs)
    
    def get_ai_stats(self) -> Dict[str, Any]:
        """获取AI统计信息"""
        return {
            "class": self.__class__.__name__,
            "interaction_count": self._interaction_count,
            "session_summary": self.ai_logger.get_session_summary()
        }

# 使用示例
if __name__ == "__main__":
    # 示例1：使用装饰器记录AI函数调用
    @log_ai_interaction(model="gpt-4", temperature=0.7)
    def analyze_stock(symbol: str, prompt: str, context: Dict = None) -> str:
        """分析股票的函数"""
        # 模拟AI分析
        return f"分析结果：{symbol} 目前表现良好，建议持有。"
    
    # 调用函数
    result = analyze_stock("AAPL", "请分析AAPL的走势", {"timeframe": "daily"})
    print("分析结果:", result)
    
    # 示例2：使用混入类
    class StockAnalyzer(AILoggerMixin):
        def __init__(self):
            super().__init__()
            self.name = "StockAnalyzer"
        
        def analyze(self, symbol: str, prompt: str) -> str:
            # 记录输入
            interaction_id = self.log_ai_input(
                prompt=prompt,
                context={"symbol": symbol, "analyzer": self.name}
            )
            
            try:
                # 模拟分析
                result = f"AI分析结果：{symbol} 当前趋势向上"
                
                # 记录输出
                self.log_ai_output(interaction_id, result)
                
                return result
                
            except Exception as e:
                # 记录错误
                self.log_ai_error(interaction_id, e, error_type="analysis_error")
                raise
    
    # 使用混入类
    analyzer = StockAnalyzer()
    result = analyzer.analyze("TSLA", "分析TSLA的技术指标")
    print("混入类结果:", result)
    
    # 获取统计信息
    stats = analyzer.get_ai_stats()
    print("统计信息:", stats) 