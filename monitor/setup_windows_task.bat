@echo off
chcp 65001 >nul
echo ================================================================
echo 🔧 Windows任务计划程序设置
echo ================================================================
echo.

set "TASK_NAME=智能股票日报系统"
set "SCRIPT_PATH=%~dp0\..\monitor\smart_daily_email_sender.py"
set "CONDA_PATH=C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat"

echo 🔍 检查任务是否已存在...
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠️  任务已存在，正在删除旧任务...
    schtasks /delete /tn "%TASK_NAME%" /f
)

echo 📅 创建新的定时任务...
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "cmd /c \"cd /d %~dp0\.. && conda activate openbb && python monitor\smart_daily_email_sender.py\"" ^
    /sc daily ^
    /st 04:30 ^
    /ru SYSTEM ^
    /rl HIGHEST ^
    /f

if %errorlevel% equ 0 (
    echo ✅ 任务创建成功！
    echo.
    echo 📋 任务信息：
    echo    任务名称: %TASK_NAME%
    echo    执行时间: 每天 04:30 (交易日)
    echo    运行权限: 系统级别
    echo.
    echo 🎯 管理命令：
    echo    查看任务: schtasks /query /tn "%TASK_NAME%"
    echo    启动任务: schtasks /run /tn "%TASK_NAME%"
    echo    删除任务: schtasks /delete /tn "%TASK_NAME%" /f
    echo.
    echo 💡 任务已设置完成，系统会自动在每天04:30执行！
) else (
    echo ❌ 任务创建失败，请以管理员身份运行此脚本
)

echo.
pause 