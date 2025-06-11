@echo off
chcp 65001 >nul
echo ================================================================
echo 📅 智能日报定时任务启动器
echo ================================================================
echo.

echo 🔍 检查当前目录...
if not exist "smart_daily_email_sender.py" (
    echo ❌ 错误：请在 monitor 目录下运行此脚本
    pause
    exit /b 1
)

echo ✅ 找到智能日报系统文件
echo.

echo 🚀 启动智能日报定时任务...
echo ⏰ 系统将在美股收盘后30分钟自动发送邮件
echo 📅 只在交易日执行
echo 📧 按 Ctrl+C 可停止程序
echo.

python smart_daily_email_sender.py

echo.
echo 👋 程序已停止
pause 