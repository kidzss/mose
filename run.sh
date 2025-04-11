#!/bin/bash

# 定义日志文件路径
MONITOR_LOG="monitor.log"
MARKET_BOTTOM_LOG="market_bottom.log"

# 启动监控
start_monitor() {
    echo "Starting stock monitor..."
    nohup python -m monitor.examples.stock_monitor_example > $MONITOR_LOG 2>&1 &
    echo "Stock monitor started. PID: $!"
    
    echo "Starting market bottom analysis..."
    nohup python -m monitor.examples.market_bottom_analysis > $MARKET_BOTTOM_LOG 2>&1 &
    echo "Market bottom analysis started. PID: $!"
    
    echo "All services started successfully!"
}

# 停止监控
stop_monitor() {
    echo "Stopping all services..."
    
    # 查找并停止stock monitor
    STOCK_PID=$(ps aux | grep "python -m monitor.examples.stock_monitor_example" | grep -v grep | awk '{print $2}')
    if [ ! -z "$STOCK_PID" ]; then
        kill $STOCK_PID
        echo "Stock monitor stopped (PID: $STOCK_PID)"
    else
        echo "Stock monitor is not running"
    fi
    
    # 查找并停止market bottom analysis
    MARKET_PID=$(ps aux | grep "python -m monitor.examples.market_bottom_analysis" | grep -v grep | awk '{print $2}')
    if [ ! -z "$MARKET_PID" ]; then
        kill $MARKET_PID
        echo "Market bottom analysis stopped (PID: $MARKET_PID)"
    else
        echo "Market bottom analysis is not running"
    fi
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
    STOCK_PID=$(ps aux | grep "python -m monitor.examples.stock_monitor_example" | grep -v grep | awk '{print $2}')
    if [ ! -z "$STOCK_PID" ]; then
        echo "Stock monitor is running (PID: $STOCK_PID)"
    else
        echo "Stock monitor is not running"
    fi
    
    # 检查market bottom analysis
    MARKET_PID=$(ps aux | grep "python -m monitor.examples.market_bottom_analysis" | grep -v grep | awk '{print $2}')
    if [ ! -z "$MARKET_PID" ]; then
        echo "Market bottom analysis is running (PID: $MARKET_PID)"
    else
        echo "Market bottom analysis is not running"
    fi
}

# 主菜单
main_menu() {
    echo "=== Market Monitor Control Panel ==="
    echo "1. Start all services"
    echo "2. Stop all services"
    echo "3. View logs"
    echo "4. Check status"
    echo "5. Exit"
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            start_monitor
            ;;
        2)
            stop_monitor
            ;;
        3)
            view_logs
            ;;
        4)
            check_status
            ;;
        5)
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

# 设置脚本可执行权限
chmod +x run.sh

# 启动主菜单
main_menu 