@echo off
echo Starting Trading Monitor System...

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 设置日志文件
set LOG_FILE=trading_monitor.log

:: 启动监控系统
echo Starting monitoring system...
python -u run_monitor.py >> %LOG_FILE% 2>&1

:: 保存进程ID
echo %ERRORLEVEL% > monitor.pid

echo Monitoring system started, PID: %ERRORLEVEL%
echo Log file: %LOG_FILE%

:: 定义日志文件路径
set MONITOR_LOG=monitor.log
set MARKET_BOTTOM_LOG=market_bottom.log

:: 启动监控
:start_monitor
echo Starting stock monitor...
start /B python -m monitor.examples.stock_monitor_example > %MONITOR_LOG% 2>&1
echo Stock monitor started.

echo Starting market bottom analysis...
start /B python -m monitor.examples.market_bottom_analysis > %MARKET_BOTTOM_LOG% 2>&1
echo Market bottom analysis started.

echo All services started successfully!
goto menu

:: 停止监控
:stop_monitor
echo Stopping all services...

:: 查找并停止stock monitor
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| find "monitor.examples.stock_monitor_example"') do (
    taskkill /F /PID %%a
    echo Stock monitor stopped (PID: %%a)
)

:: 查找并停止market bottom analysis
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| find "monitor.examples.market_bottom_analysis"') do (
    taskkill /F /PID %%a
    echo Market bottom analysis stopped (PID: %%a)
)

goto menu

:: 查看日志
:view_logs
echo Viewing logs...
echo 1. Stock Monitor Log
echo 2. Market Bottom Analysis Log
echo 3. Both Logs
set /p choice=Enter your choice (1-3): 

if "%choice%"=="1" (
    type %MONITOR_LOG%
) else if "%choice%"=="2" (
    type %MARKET_BOTTOM_LOG%
) else if "%choice%"=="3" (
    type %MONITOR_LOG% %MARKET_BOTTOM_LOG%
) else (
    echo Invalid choice
)

goto menu

:: 检查服务状态
:check_status
echo Checking service status...

:: 检查stock monitor
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| find "monitor.examples.stock_monitor_example"') do (
    echo Stock monitor is running (PID: %%a)
)

:: 检查market bottom analysis
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| find "monitor.examples.market_bottom_analysis"') do (
    echo Market bottom analysis is running (PID: %%a)
)

goto menu

:: 主菜单
:menu
echo.
echo === Market Monitor Control Panel ===
echo 1. Start all services
echo 2. Stop all services
echo 3. View logs
echo 4. Check status
echo 5. Exit
set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" (
    goto start_monitor
) else if "%choice%"=="2" (
    goto stop_monitor
) else if "%choice%"=="3" (
    goto view_logs
) else if "%choice%"=="4" (
    goto check_status
) else if "%choice%"=="5" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice
)

goto menu 