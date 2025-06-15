@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ==========================================
echo 🚀 Personal Investor Automation System
echo ==========================================
echo.
echo 📊 System Features:
echo    ✓ Weekly stock screening
echo    ✓ Monthly portfolio analysis  
echo    ✓ Quarterly strategy adjustment
echo    ✓ Automatic market data updates
echo    ✓ Personalized investment emails
echo.
echo ⏰ Schedule:
echo    - Weekly screening: Sunday 20:00
echo    - Monthly analysis: First Sunday 20:00
echo    - Quarterly adjustment: First Sunday 20:00
echo.
echo 📧 Email will be sent to: kidzss@gmail.com
echo 🎯 Risk preference: Moderate risk
echo 💰 Max position: 20%%
echo.
echo 🛑 Press Ctrl+C to stop service
echo ==========================================
echo.

python personal_investor_automation.py

echo.
echo 👋 Automation service stopped
pause 