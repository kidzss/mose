@echo off
chcp 65001 >nul
echo.
echo 🚀 快速启动增强版专业监控系统
echo.

:: 检查端口占用并释放
netstat -ano | findstr :8503 >nul
if %errorlevel% equ 0 (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8503') do (
        taskkill /f /pid %%a >nul 2>&1
    )
    timeout /t 1 >nul
)

:: 启动系统
where openbb >nul 2>&1
if %errorlevel% equ 0 (
    openbb python -m streamlit run enhanced_professional_monitor_with_daily_ai.py --server.port 8503 --server.headless true
) else (
    python -m streamlit run enhanced_professional_monitor_with_daily_ai.py --server.port 8503 --server.headless true
) 