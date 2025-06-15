@echo off
chcp 65001 >nul
echo ================================
echo    日常交易助手启动
echo ================================
echo.

cd /d "E:\python_project\mose"
python daily_trading_assistant.py

echo.
echo ================================
echo    分析完成，按任意键关闭
echo ================================
pause 