# AI日志记录器使用指南

## 概述

AI日志记录器可以帮助你记录所有发送给AI的输入信息，便于调试、分析和优化AI交互。它提供了多种使用方式，从简单的函数调用到自动化的装饰器和混入类。

## 功能特性

- ✅ **完整记录**：记录AI输入、输出和错误
- ✅ **多种格式**：文本日志和JSON详细记录
- ✅ **搜索功能**：按关键词、日期、模型搜索
- ✅ **自动装饰器**：一键为函数添加日志功能
- ✅ **混入类**：为类添加AI日志功能
- ✅ **统计信息**：提供详细的交互统计

## 快速开始

### 1. 基础使用

```python
from utils.ai_logger import AILogger

# 创建日志记录器
logger = AILogger()

# 记录AI输入
interaction_id = logger.log_ai_input(
    prompt="请分析AAPL的股票走势",
    context={"symbol": "AAPL", "timeframe": "daily"},
    model="gpt-4",
    temperature=0.7
)

# 记录AI输出
logger.log_ai_output(
    interaction_id=interaction_id,
    response="AAPL目前处于上升趋势...",
    model="gpt-4"
)
```

### 2. 使用便捷函数

```python
from utils.ai_logger import log_ai_input, log_ai_output, log_ai_error

# 记录输入
interaction_id = log_ai_input(
    prompt="分析市场情绪",
    context={"market": "US", "sector": "tech"}
)

# 记录输出
log_ai_output(interaction_id, "市场情绪乐观...")

# 记录错误
try:
    # AI调用
    pass
except Exception as e:
    log_ai_error(interaction_id, e, error_type="api_error")
```

### 3. 使用装饰器

```python
from utils.ai_logger_decorator import log_ai_interaction

@log_ai_interaction(model="gpt-4", temperature=0.8)
def analyze_stock(symbol: str, prompt: str, context: dict = None) -> str:
    """分析股票的函数"""
    # 你的AI分析逻辑
    return f"分析结果：{symbol} 表现良好"
```

### 4. 使用混入类

```python
from utils.ai_logger_decorator import AILoggerMixin

class StockAnalyzer(AILoggerMixin):
    def analyze(self, symbol: str, prompt: str) -> str:
        # 记录输入
        interaction_id = self.log_ai_input(
            prompt=prompt,
            context={"symbol": symbol}
        )
        
        try:
            # AI分析逻辑
            result = f"分析结果：{symbol} 趋势向上"
            
            # 记录输出
            self.log_ai_output(interaction_id, result)
            return result
            
        except Exception as e:
            # 记录错误
            self.log_ai_error(interaction_id, e)
            raise
```

## 日志文件结构

```
logs/
└── ai_interactions/
    ├── ai_interactions_20241230.log          # 文本日志
    └── 20241230/                             # 详细JSON记录
        ├── session_1234567890_1234_interaction_1_input.json
        ├── session_1234567890_1234_interaction_1_output.json
        └── session_1234567890_1234_interaction_1_error.json
```

## 日志内容示例

### 输入日志 (input.json)
```json
{
  "interaction_id": "session_1234567890_1234_interaction_1",
  "session_id": "session_1234567890_1234",
  "timestamp": "2024-12-30T10:30:00",
  "prompt": "请分析AAPL的股票走势",
  "context": {
    "symbol": "AAPL",
    "timeframe": "daily"
  },
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### 输出日志 (output.json)
```json
{
  "interaction_id": "session_1234567890_1234_interaction_1",
  "session_id": "session_1234567890_1234",
  "timestamp": "2024-12-30T10:30:05",
  "response": "AAPL目前处于上升趋势...",
  "model": "gpt-4",
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  }
}
```

## 搜索和分析

### 搜索功能

```python
logger = AILogger()

# 按关键词搜索
results = logger.search_interactions(keyword="AAPL")

# 按日期搜索
results = logger.search_interactions(date="20241230")

# 按模型搜索
results = logger.search_interactions(model="gpt-4")

# 组合搜索
results = logger.search_interactions(
    keyword="技术分析",
    date="20241230",
    model="gpt-4"
)
```

### 统计信息

```python
# 获取会话摘要
summary = logger.get_session_summary()
print(f"会话ID: {summary['session_id']}")
print(f"交互次数: {summary['interaction_count']}")

# 获取混入类统计
analyzer = StockAnalyzer()
stats = analyzer.get_ai_stats()
print(f"类名: {stats['class']}")
print(f"交互次数: {stats['interaction_count']}")
```

## 集成到现有系统

### 1. 在ai_realtime_analyzer.py中集成

```python
from utils.ai_logger import AILogger

class AIRealtimeAnalyzer:
    def __init__(self):
        self.ai_logger = AILogger()
    
    def analyze_stock(self, symbol: str, data: dict) -> str:
        # 构建提示词
        prompt = f"请分析{symbol}的技术指标：{data}"
        
        # 记录输入
        interaction_id = self.ai_logger.log_ai_input(
            prompt=prompt,
            context={"symbol": symbol, "data": data}
        )
        
        try:
            # 调用AI
            response = self.call_ai(prompt)
            
            # 记录输出
            self.ai_logger.log_ai_output(interaction_id, response)
            
            return response
            
        except Exception as e:
            # 记录错误
            self.ai_logger.log_ai_error(interaction_id, e)
            raise
```

### 2. 在监控系统中集成

```python
from utils.ai_logger_decorator import log_ai_interaction

@log_ai_interaction(model="gpt-4")
def monitor_market_conditions(market_data: dict, prompt: str) -> str:
    """监控市场条件"""
    # 你的监控逻辑
    return "市场条件分析结果..."
```

## 配置选项

### 自定义日志目录

```python
# 指定自定义日志目录
logger = AILogger(log_dir="custom_logs/ai")

# 使用相对路径
logger = AILogger(log_dir="./logs/ai_interactions")
```

### 日志级别

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)  # 显示所有日志
logging.basicConfig(level=logging.INFO)   # 只显示信息级别以上
logging.basicConfig(level=logging.WARNING) # 只显示警告级别以上
```

## 最佳实践

### 1. 提示词优化

```python
# 好的提示词
prompt = """
请分析以下股票的技术指标：

股票代码：{symbol}
当前价格：${price}
成交量：{volume}
52周最高：${high_52w}
52周最低：${low_52w}

请提供：
1. 技术分析
2. 支撑阻力位
3. 交易建议
"""

# 记录时包含完整上下文
logger.log_ai_input(
    prompt=prompt.format(**stock_data),
    context=stock_data
)
```

### 2. 错误处理

```python
try:
    # AI调用
    response = call_ai(prompt)
    logger.log_ai_output(interaction_id, response)
    
except Exception as e:
    # 记录详细错误信息
    logger.log_ai_error(
        interaction_id=interaction_id,
        error=e,
        error_type="api_error",
        additional_info={"prompt": prompt, "context": context}
    )
    raise
```

### 3. 性能优化

```python
# 批量处理时使用会话
logger = AILogger()

for symbol in symbols:
    interaction_id = logger.log_ai_input(
        prompt=f"分析{symbol}",
        context={"symbol": symbol}
    )
    # 处理逻辑...
```

## 故障排除

### 常见问题

1. **日志文件权限错误**
   ```python
   # 确保日志目录存在且有写权限
   import os
   os.makedirs("logs/ai_interactions", exist_ok=True)
   ```

2. **JSON序列化错误**
   ```python
   # 确保数据可以被JSON序列化
   import json
   try:
       json.dumps(data)
   except TypeError:
       # 处理不可序列化的数据
       data = str(data)
   ```

3. **日志文件过大**
   ```python
   # 定期清理旧日志
   import shutil
   from datetime import datetime, timedelta
   
   # 删除7天前的日志
   cutoff_date = datetime.now() - timedelta(days=7)
   # 清理逻辑...
   ```

## 测试

运行测试脚本验证功能：

```bash
python test_ai_logger.py
```

这将创建示例日志文件并演示所有功能。

## 总结

AI日志记录器提供了完整的AI交互记录功能，帮助你：

1. **调试AI调用**：查看具体的输入和输出
2. **优化提示词**：分析哪些提示词效果更好
3. **监控性能**：跟踪AI调用的成功率和响应时间
4. **合规要求**：满足某些行业的审计要求
5. **成本控制**：监控AI API的使用情况

通过合理使用这些功能，你可以显著提升AI系统的可靠性和效率。 