@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   可配置AI监控系统测试脚本
echo   Configurable AI Monitor Test Script
echo ========================================
echo.
echo 🧪 开始测试可配置AI监控系统...
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python环境
    pause
    exit /b 1
)

REM 运行测试脚本
echo 📋 运行系统测试...
python test_configurable_ai_monitor.py

echo.
echo ✅ 测试完成
pause 