@echo off
chcp 65001 >nul
echo ========================================
echo ⚡ 实时交易监控系统启动器
echo ========================================
echo.

:MENU
echo 请选择操作:
echo [1] 🚀 启动监控系统
echo [2] 🔄 重置监控系统 (杀死旧进程并重启)
echo [3] 🛑 停止监控系统
echo [4] 📊 查看运行状态
echo [0] ❌ 退出
echo.
set /p choice=请输入选择 (0-4): 

if "%choice%"=="1" goto START
if "%choice%"=="2" goto RESET
if "%choice%"=="3" goto STOP
if "%choice%"=="4" goto STATUS
if "%choice%"=="0" goto EXIT
echo ❌ 无效选择，请重新输入
echo.
goto MENU

:START
echo.
echo 🚀 启动专业实时交易监控系统...
call :ACTIVATE_ENV
if %errorlevel% neq 0 goto MENU

call :SHOW_INFO
echo 🔍 检查端口8501是否被占用...
netstat -an | findstr :8501 >nul
if %errorlevel% equ 0 (
    echo ⚠️  端口8501已被占用，建议选择重置选项
    echo.
    goto MENU
)

echo ✅ 端口空闲，开始启动...
python -m streamlit run professional_trading_monitor.py --server.port 8501 --server.address=localhost --server.headless true
echo.
echo 📈 实时交易监控系统已停止
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

call :SHOW_INFO
echo ✅ 开始重新启动...
python -m streamlit run professional_trading_monitor.py --server.port 8501 --server.headless true
echo.
echo 📈 实时交易监控系统已停止
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

:SHOW_INFO
echo 💡 系统将提供实时股价监控和交易信号分析
echo.
echo 📊 系统功能:
echo    • 智能配置读取: 自动从portfolio_config.json读取持仓信息
echo    • 实时股价监控: 动态监控您的持仓股票和观察仓
echo    • 技术指标分析: RSI, 移动平均线, 成交量分析
echo    • 交易信号提醒: 超买超卖信号自动检测
echo    • 市场概况显示: 主要指数实时数据
echo    • 风险管理提醒: 基于您的实际持仓止损设置
echo.
echo 🌐 访问地址: http://localhost:8501
echo 📱 请在浏览器中打开上述地址使用系统
echo.
echo ⚠️  重要提示:
echo    • 系统启动后会自动打开浏览器
echo    • 数据每60秒自动更新
echo    • 确保网络连接正常以获取实时数据
echo.
echo 🛑 要停止服务，请按 Ctrl+C 或选择停止选项
echo ========================================
echo.
exit /b 0

:EXIT
echo.
echo 👋 感谢使用实时交易监控系统！
echo ========================================
exit 