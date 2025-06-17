@echo off
chcp 65001 >nul
echo ========================================
echo 📊 持股分析定时调度器启动器
echo ========================================
echo.

echo 📋 正在激活conda环境 (openbb)...
call conda activate openbb
if %errorlevel% neq 0 (
    echo ❌ conda环境激活失败！
    echo 请确保已安装conda并配置了openbb环境
    pause
    exit /b 1
)

echo ✅ conda环境激活成功
echo.

echo 🕐 启动持股分析定时调度器...
echo 💡 系统将按计划自动运行持股分析并发送报告
echo.
echo 📅 调度计划:
echo    • 每日分析: 周一至周五 16:30 (美股收盘后30分钟)
echo    • 每周分析: 每周日 20:00
echo    • 每月分析: 每月第一个周日 20:00
echo.
echo 📧 报告将发送至: kidzss@gmail.com
echo 🛑 要停止服务，请按 Ctrl+C
echo ========================================
echo.

python portfolio_analysis_scheduler.py

echo.
echo 📊 持股分析调度器已停止
echo ========================================
pause 