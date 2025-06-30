@echo off
chcp 65001 >nul

REM 设置openbb环境路径
set OPENBB_PYTHON=C:\Users\a\.conda\envs\openbb\python.exe
set OPENBB_PIP=C:\Users\a\.conda\envs\openbb\Scripts\pip.exe

REM 检查并杀死占用8503端口的进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8503 ^| findstr LISTENING') do (
    echo 发现占用8503端口的进程，PID=%%a，正在终止...
    taskkill /PID %%a /F
    timeout /t 2 >nul
)

echo.
echo ========================================
echo    AI每日持股分析监控系统启动器
echo    AI Daily Holdings Analysis Monitor
echo ========================================
echo.

echo 🚀 正在启动AI每日持股分析监控系统...
echo 📊 端口: 8503
echo 🤖 AI模型: deepseek-r1:latest
echo 📈 数据源: 每日持股分析 + 实时市场数据
echo 🐍 Python环境: openbb
echo.

echo ⏳ 请稍候，系统正在初始化...
echo.

REM 使用openbb环境的Python启动Streamlit应用
"%OPENBB_PYTHON%" -m streamlit run start_ai_daily_analysis_monitor.py --server.port 8503 --server.headless true

echo.
echo ✅ 系统已启动完成！
echo 🌐 请在浏览器中访问: http://localhost:8503
echo.
echo 💡 使用说明:
echo    - 在侧边栏选择要分析的股票
echo    - 点击"分析"按钮进行AI诊断
echo    - 查看实时市场数据和投资组合概览
echo.
echo ⚠️  按 Ctrl+C 停止系统
echo.
pause 