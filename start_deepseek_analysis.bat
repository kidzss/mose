@echo off
chcp 65001 >nul
echo 🚀 DeepSeek投资分析系统启动器
echo ==========================================

REM 设置环境变量
set AI_API_KEY=
set AI_API_ENDPOINT=http://localhost:11434/v1/chat/completions
set AI_MODEL=deepseek-r1
set OLLAMA_MODELS=E:\ollama_models

echo 🔧 环境变量设置完成

REM 检查Ollama服务
echo 🔍 检查Ollama服务状态...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Ollama服务未运行，正在启动...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)

REM 启动投资分析系统
echo 🎯 启动投资分析系统...
echo 💡 系统将在浏览器中打开: http://localhost:8501
echo 💡 按 Ctrl+C 停止服务
echo.

python start_deepseek_analysis.py

pause 