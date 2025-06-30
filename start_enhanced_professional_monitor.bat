@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   增强版专业实时交易监控系统启动器
echo   Enhanced Professional Trading Monitor
echo ========================================
echo.

:: 检查端口8503是否被占用
echo 🔍 检查端口8503占用情况...
netstat -ano | findstr :8503 >nul
if %errorlevel% equ 0 (
    echo ⚠️  端口8503已被占用，正在终止占用进程...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8503') do (
        taskkill /f /pid %%a >nul 2>&1
    )
    timeout /t 2 >nul
    echo ✅ 端口8503已释放
) else (
    echo ✅ 端口8503可用
)

echo.
echo 🚀 启动增强版专业实时交易监控系统...
echo.

:: 直接使用openbb环境启动
call conda activate openbb
echo 📋 使用openbb Python环境启动系统...
echo.
python -m streamlit run enhanced_professional_monitor_with_daily_ai.py --server.port 8503 --server.headless true

echo.
echo ========================================
echo   系统启动完成！
echo ========================================
echo.
echo 📊 系统功能:
echo    • 📊 市场概览: 实时指数监控
echo    • 📈 监控股票: 技术指标分析
echo    • 💼 投资组合: 持仓管理
echo    • 🤖 AI每日持股分析: 基于每日持股分析的AI诊断
echo    • 📋 每日分析摘要: 投资组合表现
echo    • ⚙️ 系统状态: 组件监控
echo.
echo 🌐 访问地址: http://localhost:8503
echo.
echo 💡 使用说明:
echo    - 在侧边栏选择要监控的股票
echo    - 点击"开始AI每日持股分析"进行智能诊断
echo    - 查看每日分析摘要了解投资组合表现
echo    - 系统会自动刷新数据
echo.
echo ⚠️  注意事项:
echo    - 确保Ollama服务正在运行
echo    - 确保portfolio_config.json配置正确
echo    - 投资有风险，决策需谨慎
echo.
pause 