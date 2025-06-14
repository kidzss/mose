@echo off
cd /d "%~dp0"
echo.
echo ==========================================
echo 🚀 个人投资者自动化股票推荐系统
echo ==========================================
echo.
echo 📊 系统功能：
echo    ✓ 每周自动筛选优质股票
echo    ✓ 每月深度分析投资组合
echo    ✓ 季度策略调整建议
echo    ✓ 自动更新市场数据
echo    ✓ 个性化投资建议邮件
echo.
echo ⏰ 定时安排：
echo    - 每周筛选：每周日 20:00
echo    - 每月分析：每月第一个周日 20:00
echo    - 季度调整：每季度第一个周日 20:00
echo.
echo 📧 邮件将发送到：kidzss@gmail.com
echo 🎯 风险偏好：中等风险
echo 💰 最大仓位：20%%
echo.
echo 🛑 要停止服务，请按 Ctrl+C
echo ==========================================
echo.

python personal_investor_automation.py

echo.
echo 👋 自动化服务已停止
pause 