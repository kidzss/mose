#!/bin/bash

# 设置Python环境
if [ ! -d "venv" ]; then
    echo "Error: Python virtual environment not found. Please create it first."
    exit 1
fi
source venv/bin/activate

# 设置日志文件
LOG_DIR="logs"
mkdir -p $LOG_DIR
MONITOR_LOG="$LOG_DIR/monitor.log"
MARKET_BOTTOM_LOG="$LOG_DIR/market_bottom.log"

# 启动监控
start_monitor() {
    echo "Starting stock monitor..."
    if ! nohup python -m monitor.examples.stock_monitor_example > $MONITOR_LOG 2>&1 & then
        echo "Error: Failed to start stock monitor"
        return 1
    fi
    STOCK_PID=$!
    echo $STOCK_PID > "$LOG_DIR/stock_monitor.pid"
    echo "Stock monitor started. PID: $STOCK_PID"
    
    echo "Starting market bottom analysis..."
    if ! nohup python -m monitor.examples.market_bottom_analysis > $MARKET_BOTTOM_LOG 2>&1 & then
        echo "Error: Failed to start market bottom analysis"
        kill $STOCK_PID 2>/dev/null
        return 1
    fi
    MARKET_PID=$!
    echo $MARKET_PID > "$LOG_DIR/market_bottom.pid"
    echo "Market bottom analysis started. PID: $MARKET_PID"
    
    echo "All services started successfully!"
}

# 停止监控
stop_monitor() {
    echo "Stopping all services..."
    
    # 停止stock monitor
    if [ -f "$LOG_DIR/stock_monitor.pid" ]; then
        STOCK_PID=$(cat "$LOG_DIR/stock_monitor.pid")
        if kill $STOCK_PID 2>/dev/null; then
            echo "Stock monitor stopped (PID: $STOCK_PID)"
        else
            echo "Stock monitor was not running"
        fi
        rm "$LOG_DIR/stock_monitor.pid"
    else
        echo "Stock monitor is not running"
    fi
    
    # 停止market bottom analysis
    if [ -f "$LOG_DIR/market_bottom.pid" ]; then
        MARKET_PID=$(cat "$LOG_DIR/market_bottom.pid")
        if kill $MARKET_PID 2>/dev/null; then
            echo "Market bottom analysis stopped (PID: $MARKET_PID)"
        else
            echo "Market bottom analysis was not running"
        fi
        rm "$LOG_DIR/market_bottom.pid"
    else
        echo "Market bottom analysis is not running"
    fi
}

# 重启监控
restart_monitor() {
    echo "Restarting all services..."
    stop_monitor
    # 等待进程完全停止
    sleep 2
    start_monitor
    echo "All services have been restarted!"
}

# 查看日志
view_logs() {
    echo "Viewing logs..."
    echo "1. Stock Monitor Log"
    echo "2. Market Bottom Analysis Log"
    echo "3. Both Logs"
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            tail -f $MONITOR_LOG
            ;;
        2)
            tail -f $MARKET_BOTTOM_LOG
            ;;
        3)
            tail -f $MONITOR_LOG $MARKET_BOTTOM_LOG
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
}

# 检查服务状态
check_status() {
    echo "Checking service status..."
    
    # 检查stock monitor
    if [ -f "$LOG_DIR/stock_monitor.pid" ]; then
        STOCK_PID=$(cat "$LOG_DIR/stock_monitor.pid")
        if ps -p $STOCK_PID > /dev/null; then
            echo "Stock monitor is running (PID: $STOCK_PID)"
        else
            echo "Stock monitor is not running (PID file exists but process is dead)"
            rm "$LOG_DIR/stock_monitor.pid"
        fi
    else
        echo "Stock monitor is not running"
    fi
    
    # 检查market bottom analysis
    if [ -f "$LOG_DIR/market_bottom.pid" ]; then
        MARKET_PID=$(cat "$LOG_DIR/market_bottom.pid")
        if ps -p $MARKET_PID > /dev/null; then
            echo "Market bottom analysis is running (PID: $MARKET_PID)"
        else
            echo "Market bottom analysis is not running (PID file exists but process is dead)"
            rm "$LOG_DIR/market_bottom.pid"
        fi
    else
        echo "Market bottom analysis is not running"
    fi
}

# 主菜单
main_menu() {
    echo "=== Market Monitor Control Panel ==="
    echo "1. Start all services"
    echo "2. Stop all services"
    echo "3. Restart all services"
    echo "4. View logs"
    echo "5. Check status"
    echo "6. Exit"
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1)
            start_monitor
            ;;
        2)
            stop_monitor
            ;;
        3)
            restart_monitor
            ;;
        4)
            view_logs
            ;;
        5)
            check_status
            ;;
        6)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    main_menu
}

# 启动主菜单
main_menu 