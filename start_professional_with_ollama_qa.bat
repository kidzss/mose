@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   专业实时交易监控系统 - Ollama AI问答版
echo   Professional Trading Monitor with Ollama AI QA
echo ========================================
echo.

echo 🚀 正在启动专业实时交易监控系统...
echo.

echo 📋 系统功能:
echo    • 📊 市场概览 - 实时市场数据监控
echo    • 📈 监控股票 - 个股技术分析
echo    • 🔬 专业投资分析中心 - 深度股票分析
echo    • 💼 投资组合 - 投资组合管理
echo    • 🧠 决策支持 - 智能决策辅助
echo    • 🤖 AI诊断 - AI每日持股分析
echo    • 💬 AI问答 - 本地Ollama AI对话 ✨新功能
echo.

echo ⚠️  重要提示:
echo    • 请确保Ollama服务正在运行: ollama serve
echo    • 确保已下载AI模型: ollama pull deepseek-r1
echo    • 系统将在 http://localhost:8501 启动
echo.

echo 🔧 检查Ollama服务状态...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama服务未运行或无法连接
    echo 💡 请先启动Ollama服务: ollama serve
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Ollama服务连接正常
)

echo.
echo 🌐 启动Streamlit应用...
echo.

python -m streamlit run professional_trading_monitor.py --server.port 8501 --server.address=localhost --server.headless true

echo.
echo 🎉 系统启动完成！
echo 📱 请在浏览器中访问: http://localhost:8501
echo 💬 点击"AI问答"标签页开始与AI对话
echo.
pause 