@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

REM 创建VBS脚本来隐藏运行
echo Set WshShell = CreateObject("WScript.Shell") > "%temp%\run_hidden.vbs"
echo WshShell.Run "cmd /c """"cd /d %cd% && conda activate openbb && python monitor\smart_daily_email_sender.py""""", 0 >> "%temp%\run_hidden.vbs"

REM 执行VBS脚本（隐藏窗口）
cscript //nologo "%temp%\run_hidden.vbs"

REM 清理临时文件
del "%temp%\run_hidden.vbs"

echo 智能日报系统已在后台启动！
timeout /t 3 