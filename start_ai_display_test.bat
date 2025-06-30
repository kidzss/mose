@echo off
chcp 65001 >nul
echo ========================================
echo 🤖 AI原文显示测试启动器
echo ========================================
echo.

echo 💡 这个测试将验证AI原文是否正确显示
echo.
echo 🤖 系统功能:
echo    • 📊 测试AI分析功能
echo    • 📝 显示AI原文输出
echo    • 🎯 验证结构化建议
echo    • 📈 查看分析历史
echo.
echo 🌐 访问地址: http://localhost:8502
echo 📱 请在浏览器中打开上述地址使用系统
echo.
echo ⚠️  重要提示:
echo    • 确保Ollama服务运行以支持AI分析
echo    • 系统会显示完整的AI原文分析过程
echo    • 可以验证AI是如何得出建议的
echo.
echo 🛑 要停止服务，请按 Ctrl+C
echo ========================================
echo.

call conda activate openbb
if %errorlevel% neq 0 (
    echo ❌ conda环境激活失败！
    pause
    exit /b 1
)

echo ✅ conda环境激活成功
echo 🚀 启动AI显示测试...
python -m streamlit run test_ai_monitor_display.py --server.port 8502 --server.address=localhost --server.headless true

echo.
echo 🤖 AI显示测试已停止
pause 