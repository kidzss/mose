@echo off
chcp 65001 >nul
echo ========================================
echo 🤖 持仓感知AI分析系统启动器
echo ========================================
echo.

:MENU
echo 请选择操作:
echo [1] 🚀 启动增强专业监控器 (持仓感知AI)
echo [2] 🧪 运行持仓AI分析测试
echo [3] 🔄 重置监控系统 (杀死旧进程并重启)
echo [4] 🛑 停止监控系统
echo [5] 📊 查看运行状态
echo [0] ❌ 退出
echo.
set /p choice=请输入选择 (0-5): 

if "%choice%"=="1" goto START_MONITOR
if "%choice%"=="2" goto RUN_TEST
if "%choice%"=="3" goto RESET
if "%choice%"=="4" goto STOP
if "%choice%"=="5" goto STATUS
if "%choice%"=="0" goto EXIT
echo ❌ 无效选择，请重新输入
echo.
goto MENU

:START_MONITOR
echo.
echo 🚀 启动增强专业监控器 (持仓感知AI)...
call :ACTIVATE_ENV
if %errorlevel% neq 0 goto MENU

call :SHOW_MONITOR_INFO
echo 🔍 检查端口8501是否被占用...
netstat -an | findstr :8501 >nul
if %errorlevel% equ 0 (
    echo ⚠️  端口8501已被占用，建议选择重置选项
    echo.
    goto MENU
)

echo ✅ 端口空闲，开始启动...
python -m streamlit run enhanced_professional_monitor.py --server.port 8501 --server.address=localhost --server.headless true
echo.
echo 🤖 持仓感知AI监控系统已停止
goto MENU

:RUN_TEST
echo.
echo 🧪 运行持仓AI分析测试...
call :ACTIVATE_ENV
if %errorlevel% neq 0 goto MENU

echo 🔍 检查Ollama服务状态...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Ollama服务未运行，正在启动...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)

echo 🧪 开始运行持仓感知AI分析测试...
python test_position_aware_ai.py
echo.
echo ✅ 测试完成
pause
goto MENU

:RESET
echo.
echo 🔄 重置监控系统...
call :ACTIVATE_ENV
if %errorlevel% neq 0 goto MENU

echo 🔍 查找并终止现有的监控进程...
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq *streamlit*" /NH 2^>nul') do (
    echo 终止进程 PID: %%i
    taskkill /PID %%i /F >nul 2>&1
)

REM 终止占用8501端口的进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    echo 终止占用端口8501的进程 PID: %%a
    taskkill /PID %%a /F >nul 2>&1
)

echo ⏱️  等待3秒后重启...
timeout /t 3 /nobreak >nul

call :SHOW_MONITOR_INFO
echo ✅ 开始重新启动...
python -m streamlit run enhanced_professional_monitor.py --server.port 8501 --server.headless true
echo.
echo 🤖 持仓感知AI监控系统已停止
goto MENU

:STOP
echo.
echo 🛑 停止监控系统...
echo 🔍 查找并终止现有的监控进程...

REM 终止streamlit相关进程
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /NH 2^>nul') do (
    tasklist /FI "PID eq %%i" /V /NH 2>nul | findstr streamlit >nul
    if %errorlevel% equ 0 (
        echo 终止Streamlit进程 PID: %%i
        taskkill /PID %%i /F >nul 2>&1
    )
)

REM 终止占用8501端口的进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    echo 终止占用端口8501的进程 PID: %%a
    taskkill /PID %%a /F >nul 2>&1
)

echo ✅ 监控系统已停止
echo.
goto MENU

:STATUS
echo.
echo 📊 当前系统状态:
echo ==========================================
echo 🔍 检查Python进程:
tasklist /FI "IMAGENAME eq python.exe" /NH 2>nul | find "python.exe" >nul
if %errorlevel% equ 0 (
    echo ✅ 发现Python进程
    tasklist /FI "IMAGENAME eq python.exe" /NH
) else (
    echo ❌ 未发现Python进程
)

echo.
echo 🔍 检查端口8501状态:
netstat -an | findstr :8501 >nul
if %errorlevel% equ 0 (
    echo ✅ 端口8501正在监听
    netstat -ano | findstr :8501
) else (
    echo ❌ 端口8501未被占用
)

echo.
echo 🔍 检查Ollama服务:
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ollama服务运行正常
) else (
    echo ❌ Ollama服务未运行
)

echo.
echo 🌐 如果系统运行中，请访问: http://localhost:8501
echo ==========================================
echo.
goto MENU

:ACTIVATE_ENV
echo 📋 正在激活conda环境 (openbb)...
call conda activate openbb
if %errorlevel% neq 0 (
    echo ❌ conda环境激活失败！
    echo 请确保已安装conda并配置了openbb环境
    pause
    exit /b 1
)
echo ✅ conda环境激活成功
exit /b 0

:SHOW_MONITOR_INFO
echo 💡 持仓感知AI分析系统将提供智能投资建议
echo.
echo 🤖 系统功能:
echo    • 📊 持仓感知分析: 自动读取portfolio_config.json中的持仓信息
echo    • ⏰ 多时间框架分析: 短线(1-7天)、中线(1-4周)、长线(1-6个月)
echo    • 💰 盈亏计算: 实时计算未实现盈亏和盈亏率
echo    • 🎯 智能建议: 减仓/加仓/持有建议，包含具体理由
echo    • ⚠️  风险提醒: 基于持仓成本的风险评估
echo    • 📈 实时数据: 实时股价、技术指标、成交量分析
echo    • 🔄 自动更新: 可配置的分析间隔，持续监控
echo.
echo 📋 持仓信息包含:
echo    • 持股数量、成本价格、仓位权重
echo    • 行业板块、止损止盈设置
echo    • 投资笔记和交易记录
echo.
echo 🌐 访问地址: http://localhost:8501
echo 📱 请在浏览器中打开上述地址使用系统
echo.
echo ⚠️  重要提示:
echo    • 确保portfolio_config.json文件存在且格式正确
echo    • 需要Ollama服务运行以支持AI分析
echo    • 系统会自动识别持仓股票并提供针对性建议
echo    • 数据每5-60分钟自动更新(可配置)
echo.
echo 🛑 要停止服务，请按 Ctrl+C 或选择停止选项
echo ========================================
echo.
exit /b 0

:EXIT
echo.
echo 👋 感谢使用持仓感知AI分析系统！
echo ========================================
exit 