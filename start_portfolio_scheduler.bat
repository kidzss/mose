@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ==========================================
echo 📊 持股分析定时调度器服务
echo ==========================================
echo.
echo 🎯 服务功能：
echo    ✓ 每日持股简报（交易日 16:30）
echo    ✓ 每周深度分析（每周日 20:00）
echo    ✓ 每月组合优化（每月第一个周日 20:00）
echo    ✓ 自动发送邮件报告
echo.
echo 📊 分析内容：
echo    • 持仓股票实时表现分析
echo    • 盈亏统计和风险评估
echo    • 技术指标和市场情绪
echo    • 个性化投资建议
echo    • 投资组合优化建议
echo.
echo 📈 持仓股票：
echo    - AMD, GOOGL, PFE, NVDA, TSLA, ADBE
echo    - 观察股票：MSFT, EOG, PHM, CF
echo.
echo 📧 邮件将发送到：kidzss@gmail.com
echo.
echo 🛑 要停止服务，请按 Ctrl+C
echo ==========================================
echo.

python portfolio_analysis_scheduler.py

echo.
echo 👋 持股分析调度服务已停止
pause 