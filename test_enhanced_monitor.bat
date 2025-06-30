@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   增强版专业监控系统测试
echo   Enhanced Monitor System Test
echo ========================================
echo.

echo 🧪 开始测试增强版专业实时交易监控系统...
echo.

:: 直接使用openbb环境运行测试
echo 📋 使用openbb Python环境运行测试...
python test_enhanced_monitor.py

echo.
echo ========================================
echo   测试完成！
echo ========================================
echo.
echo 📋 如果所有测试都通过，您可以:
echo    1. 运行 start_enhanced_professional_monitor.bat 启动系统
echo    2. 访问 http://localhost:8503 使用系统
echo.
echo ⚠️  如果测试失败，请检查:
echo    - Ollama服务是否正在运行
echo    - portfolio_config.json 配置是否正确
echo    - 网络连接是否正常
echo.
pause 