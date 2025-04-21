import logging
from monitor.strategy_monitor import StrategyMonitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("strategy_monitor.log"),
        logging.StreamHandler()
    ]
)

def main():
    # 要监控的股票列表
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # 科技股
        'JPM', 'BAC', 'GS',  # 金融股
        'NVDA', 'AMD', 'INTC',  # 半导体
        'TSLA', 'GM', 'F'  # 汽车
    ]
    
    try:
        # 初始化策略监控器
        monitor = StrategyMonitor()
        
        # 启动监控
        monitor.start_monitoring(symbols)
        
        # 保持程序运行
        while True:
            cmd = input("输入 'q' 退出监控: ")
            if cmd.lower() == 'q':
                break
                
    except KeyboardInterrupt:
        print("\n接收到退出信号")
    finally:
        # 停止监控
        monitor.stop_monitoring()
        print("监控已停止")

if __name__ == '__main__':
    main() 