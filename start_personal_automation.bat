@echo off
chcp 65001 >nul
echo ========================================
echo 🚀 个人投资自动化系统启动器
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

echo 📊 启动个人投资自动化系统...
echo 💡 系统将自动筛选股票并发送投资建议邮件
echo.

python personal_investor_automation.py

echo.
echo 📈 个人投资自动化系统运行完成
echo ========================================
pause 
