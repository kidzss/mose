@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ==========================================
echo 持股分析报告生成任务
echo ==========================================
echo.
echo 任务功能：
echo    - 分析当前持仓股票表现
echo    - 生成盈亏统计报告
echo    - 技术指标和市场情绪分析
echo    - 自动发送邮件报告
echo.
echo 分析股票：
echo    - 持仓: AMD, GOOGL, PFE, NVDA, TSLA, ADBE
echo    - 观察: MSFT, EOG, PHM, CF
echo.
echo 邮件发送到: kidzss@gmail.com
echo 运行时间: %date% %time%
echo.
echo ==========================================
echo 开始执行持股分析...
echo ==========================================
echo.

python run_portfolio_analysis.py

echo.
if %ERRORLEVEL% EQU 0 (
    echo ==========================================
    echo 持股分析任务执行成功！
    echo ==========================================
    echo.
    echo 邮件已发送，请检查您的邮箱
    echo 分析报告包含：
    echo    - 持仓股票详细分析
    echo    - 盈亏情况统计 
    echo    - 技术指标分析
    echo    - 市场环境评估
    echo    - 投资建议
    echo.
) else (
    echo ==========================================
    echo 持股分析任务执行失败！
    echo ==========================================
    echo.
    echo 请检查以下可能的原因：
    echo    - 网络连接是否正常
    echo    - 邮件配置是否正确
    echo    - 数据源是否可用
    echo    - Python环境是否配置正确
    echo.
    echo 建议：
    echo    - 检查错误日志信息
    echo    - 手动运行 python run_portfolio_analysis.py
    echo    - 联系技术支持
    echo.
)

echo 任务完成时间: %date% %time%
echo.
echo 按任意键关闭窗口...
pause >nul 