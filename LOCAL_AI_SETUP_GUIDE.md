# 🆓 免费AI分析设置指南

## 🎯 目标
为投资决策支持系统提供免费的AI分析功能，无需支付API费用。

## 🚀 方案一：Ollama + 本地模型 (推荐)

### 1. 安装Ollama
```bash
# Windows (使用WSL或PowerShell)
winget install Ollama.Ollama

# 或者下载安装包
# https://ollama.ai/download
```

### 2. 下载模型
```bash
# 下载Llama2模型 (免费)
ollama pull llama2

# 下载更强大的模型
ollama pull llama2:13b
ollama pull codellama
ollama pull mistral
```

### 3. 启动Ollama服务
```bash
# 启动Ollama服务
ollama serve

# 测试模型
ollama run llama2 "Hello, how are you?"
```

### 4. 配置环境变量
```bash
# 设置本地API
export AI_API_KEY=""
export AI_API_ENDPOINT="http://localhost:11434/v1/chat/completions"
export AI_MODEL="llama2"
```

## 🚀 方案二：Hugging Face Inference API (有免费额度)

### 1. 注册Hugging Face
- 访问 https://huggingface.co/
- 注册免费账户
- 获取API Token

### 2. 配置环境变量
```bash
export AI_API_KEY="hf_your_token_here"
export AI_API_ENDPOINT="https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
```

### 3. 免费额度
- **免费用户**: 30,000 requests/month
- **Pro用户**: 更多请求和更快的推理速度

## 🚀 方案三：Google Colab + 免费GPU

### 1. 使用Google Colab
```python
# 在Colab中运行本地模型
!pip install transformers torch
!pip install gradio

# 加载模型
from transformers import AutoTokenizer, AutoModelForCausalLM
```

### 2. 免费资源
- **免费GPU**: Tesla T4 (12GB)
- **运行时间**: 12小时/天
- **存储**: 15GB

## 🔧 系统配置更新

### 更新AI配置文件
```json
{
  "ai_api": {
    "local_ollama": {
      "endpoint": "http://localhost:11434/v1/chat/completions",
      "model": "llama2",
      "max_tokens": 2000,
      "temperature": 0.7,
      "timeout": 60
    },
    "huggingface": {
      "endpoint": "https://api-inference.huggingface.co/models/",
      "model": "meta-llama/Llama-2-7b-chat-hf",
      "max_tokens": 2000,
      "temperature": 0.7,
      "timeout": 30
    }
  }
}
```

## 📊 性能对比

| 方案 | 成本 | 速度 | 质量 | 隐私 | 推荐度 |
|------|------|------|------|------|--------|
| Ollama本地 | 免费 | 中等 | 良好 | 高 | ⭐⭐⭐⭐⭐ |
| Hugging Face | 免费额度 | 快 | 良好 | 中等 | ⭐⭐⭐⭐ |
| OpenAI | 付费 | 很快 | 优秀 | 低 | ⭐⭐⭐ |
| Anthropic | 付费 | 很快 | 优秀 | 低 | ⭐⭐⭐ |

## 🛠️ 快速开始

### 1. 选择Ollama方案
```bash
# 安装Ollama
winget install Ollama.Ollama

# 下载模型
ollama pull llama2

# 启动服务
ollama serve
```

### 2. 更新环境变量
```bash
export AI_API_KEY=""
export AI_API_ENDPOINT="http://localhost:11434/v1/chat/completions"
export AI_MODEL="llama2"
```

### 3. 测试连接
```python
# 测试AI连接
python -c "
from analysis.ai_analysis_integration import AIAnalysisIntegration
ai = AIAnalysisIntegration()
result = ai.test_connection()
print(result)
"
```

## 💡 使用建议

### 1. 开发阶段
- 使用Ollama本地模型进行开发和测试
- 无需担心API费用和网络问题

### 2. 生产环境
- 可以考虑付费API获得更好的性能
- 或者使用更强大的本地模型

### 3. 混合使用
- 简单分析使用本地模型
- 复杂分析使用云端API

## 🔍 故障排除

### Ollama常见问题
```bash
# 检查服务状态
ollama list

# 重启服务
ollama serve

# 检查端口
netstat -an | findstr 11434
```

### 模型下载问题
```bash
# 清理缓存
ollama rm llama2
ollama pull llama2

# 使用镜像
ollama pull llama2:latest
```

## 📈 性能优化

### 1. 模型选择
- **llama2:7b**: 快速，适合简单分析
- **llama2:13b**: 平衡，推荐使用
- **codellama**: 适合代码分析

### 2. 硬件要求
- **CPU**: 至少4核心
- **内存**: 至少8GB RAM
- **存储**: 至少10GB可用空间

### 3. 网络优化
- 使用本地模型避免网络延迟
- 配置代理加速模型下载 