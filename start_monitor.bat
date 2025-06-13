@echo off
cd /d "%~dp0"
echo.
echo ==========================================
echo 🚀 启动持仓分析报告监控服务
echo ==========================================
echo.
echo 📊 服务功能：
echo    ✓ 每个交易日美股收盘后30分钟自动发送报告
echo    ✓ 自动更新股票数据
echo    ✓ 生成专业持仓分析报告
echo    ✓ 发送邮件到您的邮箱
echo.
echo 🕐 定时发送时间：
echo    - 夏令时（3-10月）：北京时间 04:30
echo    - 冬令时（11-2月）：北京时间 05:30
echo.
echo 📧 邮件将发送到：kidzss@gmail.com
echo.
echo 🛑 要停止服务，请按 Ctrl+C
echo ==========================================
echo.

python start_monitor.py

echo.
echo 👋 监控服务已停止
pause 