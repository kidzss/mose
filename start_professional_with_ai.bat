@echo off
chcp 65001 >nul
echo ⚡ 专业实时交易监控系统 (AI增强版)
echo ========================================

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

REM 激活conda环境
echo 📋 正在激活conda环境 (openbb)...
call conda activate openbb
if %errorlevel% neq 0 (
    echo ❌ conda环境激活失败！
    echo 请确保已安装conda并配置了openbb环境
    pause
    exit /b 1
)
echo ✅ conda环境激活成功

REM 检查端口占用
echo 🔍 检查端口8501是否被占用...
netstat -an | findstr :8501 >nul
if %errorlevel% equ 0 (
    echo ⚠️ 端口8501已被占用，正在终止旧进程...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
        taskkill /PID %%a /F >nul 2>&1
    )
    timeout /t 3 /nobreak >nul
)

echo 🚀 启动专业实时交易监控系统 (AI增强版)...
echo 💡 系统将在浏览器中打开: http://localhost:8501
echo 💡 按 Ctrl+C 停止服务
echo.
echo ⚡ 系统功能:
echo    • 📊 市场概览: 实时指数和情绪分析
echo    • 📈 监控股票: 实时价格和技术指标
echo    • 🔬 专业投资分析中心: 深度技术分析
echo    • 💼 投资组合: 持仓管理和盈亏分析
echo    • 🧠 决策支持: 智能决策辅助系统
echo    • 🤖 AI诊断: 新增AI智能分析模块
echo.

python -m streamlit run professional_trading_monitor.py --server.port 8501 --server.address=localhost --server.headless true

echo.
echo ⚡ 专业实时交易监控系统已停止
pause 