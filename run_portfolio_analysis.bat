@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo.
echo ==========================================
echo Portfolio Analysis Report Generator
echo ==========================================
echo.
echo Functions:
echo    - Update latest market data automatically
echo    - Analyze current portfolio performance with macro analysis
echo    - Generate comprehensive technical indicators analysis
echo    - Smart market environment assessment
echo    - Send intelligent email report automatically
echo.
echo Email to: kidzss@gmail.com
echo Run time: %date% %time%
echo.
echo ==========================================
echo Starting portfolio analysis...
echo ==========================================
echo.

echo [1/2] 📊 Updating market data...
echo -----------------------------------------------
python data/data_updater.py
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Warning: Data update encountered issues, but will continue with analysis
    echo    - May use existing data
    echo    - Check network connectivity
    echo    - Verify data source accessibility
    echo.
)

echo [2/2] 📈 Running portfolio analysis...
echo -----------------------------------------------
python run_portfolio_analysis.py

echo.
if %ERRORLEVEL% EQU 0 (
    echo ==========================================
    echo Portfolio analysis completed successfully!
    echo ==========================================
    echo.
    echo Email sent, please check your mailbox
    echo Analysis report contains:
    echo    - Detailed portfolio stock analysis
    echo    - Profit/loss statistics
    echo    - Technical indicator analysis
    echo    - Market environment assessment
    echo    - Investment recommendations
    echo.
) else (
    echo ==========================================
    echo Portfolio analysis failed!
    echo ==========================================
    echo.
    echo Please check the following possible causes:
    echo    - Network connection status
    echo    - Email configuration correctness
    echo    - Data source availability
    echo    - Python environment configuration
    echo.
    echo Suggestions:
    echo    - Check error log information
    echo    - Run manually: python run_portfolio_analysis.py
    echo    - Contact technical support
    echo.
)

echo Task completion time: %date% %time%
echo.
echo Press any key to close window...
pause >nul 