#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI日志记录器
记录发送给AI的所有输入信息，便于调试和分析
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class AILogger:
    """AI日志记录器，记录所有AI交互信息"""
    
    def __init__(self, log_dir: str = "logs/ai_interactions"):
        """初始化AI日志记录器
        
        Args:
            log_dir: 日志文件存储目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志格式
        self.logger = logging.getLogger("AILogger")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = self.log_dir / f"ai_interactions_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 记录会话信息
        self.session_id = self._generate_session_id()
        self.interaction_count = 0
        
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"session_{int(time.time())}_{os.getpid()}"
    
    def log_ai_input(self, 
                    prompt: str, 
                    context: Optional[Dict[str, Any]] = None,
                    model: Optional[str] = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    **kwargs) -> str:
        """记录发送给AI的输入信息
        
        Args:
            prompt: 发送给AI的提示词
            context: 上下文信息
            model: 使用的AI模型
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            交互ID
        """
        self.interaction_count += 1
        interaction_id = f"{self.session_id}_interaction_{self.interaction_count}"
        
        # 构建日志信息
        log_data = {
            "interaction_id": interaction_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "context": context or {},
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "other_params": kwargs
        }
        
        # 记录到日志文件
        self.logger.info(f"AI Input - ID: {interaction_id}")
        self.logger.info(f"Prompt: {prompt}")
        if context:
            self.logger.info(f"Context: {json.dumps(context, ensure_ascii=False, indent=2)}")
        if model:
            self.logger.info(f"Model: {model}")
        if temperature is not None:
            self.logger.info(f"Temperature: {temperature}")
        if max_tokens:
            self.logger.info(f"Max Tokens: {max_tokens}")
        if kwargs:
            self.logger.info(f"Other Params: {kwargs}")
        self.logger.info("-" * 80)
        
        # 保存详细JSON文件
        self._save_detailed_log(interaction_id, log_data, "input")
        
        return interaction_id
    
    def log_ai_output(self, 
                     interaction_id: str,
                     response: str,
                     model: Optional[str] = None,
                     usage: Optional[Dict[str, Any]] = None,
                     **kwargs) -> None:
        """记录AI的输出信息
        
        Args:
            interaction_id: 交互ID
            response: AI的响应内容
            model: 使用的AI模型
            usage: 使用统计信息
            **kwargs: 其他参数
        """
        # 构建日志信息
        log_data = {
            "interaction_id": interaction_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "model": model,
            "usage": usage or {},
            "other_params": kwargs
        }
        
        # 记录到日志文件
        self.logger.info(f"AI Output - ID: {interaction_id}")
        self.logger.info(f"Response: {response}")
        if model:
            self.logger.info(f"Model: {model}")
        if usage:
            self.logger.info(f"Usage: {usage}")
        if kwargs:
            self.logger.info(f"Other Params: {kwargs}")
        self.logger.info("=" * 80)
        
        # 保存详细JSON文件
        self._save_detailed_log(interaction_id, log_data, "output")
    
    def log_ai_error(self, 
                    interaction_id: str,
                    error: Exception,
                    error_type: str = "unknown",
                    **kwargs) -> None:
        """记录AI交互错误
        
        Args:
            interaction_id: 交互ID
            error: 错误对象
            error_type: 错误类型
            **kwargs: 其他参数
        """
        # 构建日志信息
        log_data = {
            "interaction_id": interaction_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "error_class": type(error).__name__,
            "other_params": kwargs
        }
        
        # 记录到日志文件
        self.logger.error(f"AI Error - ID: {interaction_id}")
        self.logger.error(f"Error Type: {error_type}")
        self.logger.error(f"Error Message: {error}")
        self.logger.error(f"Error Class: {type(error).__name__}")
        if kwargs:
            self.logger.error(f"Other Params: {kwargs}")
        self.logger.error("=" * 80)
        
        # 保存详细JSON文件
        self._save_detailed_log(interaction_id, log_data, "error")
    
    def _save_detailed_log(self, interaction_id: str, data: Dict[str, Any], log_type: str) -> None:
        """保存详细的JSON日志文件
        
        Args:
            interaction_id: 交互ID
            data: 日志数据
            log_type: 日志类型 (input/output/error)
        """
        try:
            # 创建日期目录
            date_dir = self.log_dir / datetime.now().strftime('%Y%m%d')
            date_dir.mkdir(exist_ok=True)
            
            # 保存JSON文件
            json_file = date_dir / f"{interaction_id}_{log_type}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"保存详细日志失败: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要
        
        Returns:
            会话摘要信息
        """
        return {
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "start_time": datetime.now().isoformat(),
            "log_dir": str(self.log_dir)
        }
    
    def search_interactions(self, 
                          keyword: Optional[str] = None,
                          date: Optional[str] = None,
                          model: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索交互记录
        
        Args:
            keyword: 关键词搜索
            date: 日期过滤 (YYYYMMDD格式)
            model: 模型过滤
            
        Returns:
            匹配的交互记录列表
        """
        results = []
        
        # 确定搜索目录
        if date:
            search_dir = self.log_dir / date
        else:
            search_dir = self.log_dir
        
        if not search_dir.exists():
            return results
        
        # 搜索JSON文件
        for json_file in search_dir.glob("*_input.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 应用过滤条件
                if keyword and keyword.lower() not in data.get('prompt', '').lower():
                    continue
                    
                if model and data.get('model') != model:
                    continue
                
                results.append(data)
                
            except Exception as e:
                self.logger.error(f"读取日志文件失败 {json_file}: {e}")
        
        return results

# 全局AI日志记录器实例
ai_logger = AILogger()

# 便捷函数
def log_ai_input(prompt: str, **kwargs) -> str:
    """记录AI输入"""
    return ai_logger.log_ai_input(prompt, **kwargs)

def log_ai_output(interaction_id: str, response: str, **kwargs) -> None:
    """记录AI输出"""
    ai_logger.log_ai_output(interaction_id, response, **kwargs)

def log_ai_error(interaction_id: str, error: Exception, **kwargs) -> None:
    """记录AI错误"""
    ai_logger.log_ai_error(interaction_id, error, **kwargs)

# 使用示例
if __name__ == "__main__":
    # 创建日志记录器
    logger = AILogger()
    
    # 记录AI输入
    interaction_id = logger.log_ai_input(
        prompt="请分析一下AAPL的股票走势",
        context={"symbol": "AAPL", "timeframe": "daily"},
        model="gpt-4",
        temperature=0.7
    )
    
    # 记录AI输出
    logger.log_ai_output(
        interaction_id=interaction_id,
        response="根据分析，AAPL目前处于上升趋势...",
        model="gpt-4",
        usage={"prompt_tokens": 50, "completion_tokens": 100}
    )
    
    # 记录错误
    try:
        raise Exception("网络连接失败")
    except Exception as e:
        logger.log_ai_error(
            interaction_id=interaction_id,
            error=e,
            error_type="network_error"
        )
    
    # 获取会话摘要
    summary = logger.get_session_summary()
    print("会话摘要:", summary) 