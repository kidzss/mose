@echo off
chcp 65001 >nul
echo ========================================
echo 🖥️  实时交易终端监控系统
echo ========================================
echo.

:MENU
echo 请选择操作:
echo [1] 🚀 启动终端监控
echo [2] 🔄 重置终端监控 (杀死旧进程并重启)
echo [3] 🛑 停止终端监控
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
echo 🚀 启动实时交易终端监控...
call :ACTIVATE_ENV
if %errorlevel% neq 0 goto MENU

call :SHOW_INFO
echo ✅ 开始启动终端监控...
python intraday_realtime_monitor.py
echo.
echo 📊 实时交易监控已停止
goto MENU

:RESET
echo.
echo 🔄 重置终端监控系统...
call :ACTIVATE_ENV
if %errorlevel% neq 0 goto MENU

echo 🔍 查找并终止现有的监控进程...
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /NH 2^>nul') do (
    tasklist /FI "PID eq %%i" /V /NH 2>nul | findstr intraday_realtime_monitor >nul
    if %errorlevel% equ 0 (
        echo 终止终端监控进程 PID: %%i
        taskkill /PID %%i /F >nul 2>&1
    )
)

echo ⏱️  等待3秒后重启...
timeout /t 3 /nobreak >nul

call :SHOW_INFO
echo ✅ 开始重新启动终端监控...
python intraday_realtime_monitor.py
echo.
echo 📊 实时交易监控已停止
goto MENU

:STOP
echo.
echo 🛑 停止终端监控系统...
echo 🔍 查找并终止现有的监控进程...

for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /NH 2^>nul') do (
    tasklist /FI "PID eq %%i" /V /NH 2>nul | findstr intraday_realtime_monitor >nul
    if %errorlevel% equ 0 (
        echo 终止终端监控进程 PID: %%i
        taskkill /PID %%i /F >nul 2>&1
    )
)

echo ✅ 终端监控系统已停止
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
echo 🔍 检查终端监控进程:
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /NH 2^>nul') do (
    tasklist /FI "PID eq %%i" /V /NH 2>nul | findstr intraday_realtime_monitor >nul
    if %errorlevel% equ 0 (
        echo ✅ 终端监控正在运行 PID: %%i
    )
)

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
echo 💡 系统将在控制台显示实时股价和交易信号
echo.
echo 📊 监控功能:
echo    • 实时价格更新: 每60秒刷新
echo    • RSI超买超卖提醒
echo    • 均线突破信号
echo    • 成交量异常检测
echo    • 自动风险提醒
echo.
echo 📈 监控配置:
echo    • 自动读取: portfolio_config.json中的持仓和观察仓
echo    • 持仓股票: 优先监控您的实际持仓股票
echo    • 观察仓股票: 监控潜在买入机会
echo    • 智能分析: 基于您的成本价和止损设置进行风险提醒
echo.
echo 🛑 要停止监控，请按 Ctrl+C 或选择停止选项
echo ========================================
echo.
exit /b 0

:EXIT
echo.
echo 👋 感谢使用实时交易终端监控系统！
echo ========================================
exit 