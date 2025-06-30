@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   可配置AI每日持股分析监控系统
echo   Configurable AI Daily Holdings Analysis Monitor
echo ========================================
echo.
echo 🚀 正在启动可配置AI监控系统...
echo.

call conda activate openbb
if %errorlevel% neq 0 (
    echo ❌ conda环境激活失败！
    pause
    exit /b 1
)

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python环境
    echo 请确保已安装Python并添加到PATH环境变量
    pause
    exit /b 1
)

REM 检查Streamlit
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 警告: 未找到Streamlit，正在安装...
    pip install streamlit
    if errorlevel 1 (
        echo ❌ 错误: Streamlit安装失败
        pause
        exit /b 1
    )
)

REM 检查必要的Python包
echo 📦 检查依赖包...
python -c "import yfinance, pandas, asyncio" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 警告: 缺少必要的依赖包，正在安装...
    pip install yfinance pandas asyncio
)

echo.
echo ✅ 环境检查完成
echo.
echo 🌐 启动Web界面...
echo 📱 请在浏览器中访问: http://localhost:8504
echo.
echo 💡 使用说明:
echo    1. 在侧边栏选择要分析的持仓股票和观察仓股票
echo    2. 选择AI分析类型（综合分析/详细分析/快速分析）
echo    3. 点击"批量AI分析"开始分析
echo    4. 查看分析结果和历史记录
echo.
echo ⏹️  按 Ctrl+C 停止服务
echo.

REM 启动Streamlit应用
streamlit run start_configurable_ai_monitor.py --server.port 8505 --server.headless true

echo.
echo 👋 服务已停止
pause 