@echo off
chcp 65001 >nul
title 智能日报系统 - OpenBB环境
echo ================================================================
echo 📅 智能日报定时任务启动器 (OpenBB环境)
echo ================================================================
echo.

echo 🔍 切换到项目根目录...
cd /d "%~dp0\.."
if not exist "monitor\smart_daily_email_sender.py" (
    echo ❌ 错误：未找到智能日报系统文件
    echo 当前目录：%cd%
    pause
    exit /b 1
)

echo ✅ 找到智能日报系统文件
echo 📂 当前路径：%cd%
echo.

echo 🐍 正在激活 openbb 环境...
call conda info --envs | findstr "openbb" >nul
if errorlevel 1 (
    echo ❌ 错误：未找到 openbb 环境
    echo 💡 请先安装 openbb 环境：
    echo    conda create -n openbb python=3.9
    echo    conda activate openbb
    echo    pip install openbb
    pause
    exit /b 1
)

call conda activate openbb
if errorlevel 1 (
    echo ❌ 错误：无法激活 openbb 环境
    pause
    exit /b 1
)

echo ✅ openbb 环境已成功激活
python --version
echo.

echo 🚀 启动智能日报定时任务...
echo ⏰ 系统将在美股收盘后30分钟自动发送邮件 (04:30北京时间)
echo 📅 只在交易日执行 (周一至周五)
echo 📧 按 Ctrl+C 可停止程序
echo ================================================================
echo.

python monitor\smart_daily_email_sender.py

echo.
echo ================================================================
echo 👋 智能日报系统已停止运行
echo ================================================================
pause 