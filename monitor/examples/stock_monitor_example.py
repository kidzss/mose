import logging
import time
from pathlib import Path
import json

from monitor.data_fetcher import DataFetcher
from monitor.strategy_manager import StrategyManager
from monitor.report_generator import ReportGenerator
from monitor.market_monitor import MarketMonitor
from monitor.stock_manager import StockManager
from monitor.stock_monitor import StockMonitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # 创建组件
    data_fetcher = DataFetcher()
    strategy_manager = StrategyManager()
    report_generator = ReportGenerator()
    market_monitor = MarketMonitor()
    stock_manager = StockManager(data_fetcher=data_fetcher)
    
    # 加载配置
    config_path = Path(__file__).parent.parent / 'config' / 'monitor_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建监控器
    monitor = StockMonitor(
        data_fetcher=data_fetcher,
        strategy_manager=strategy_manager,
        report_generator=report_generator,
        market_monitor=market_monitor,
        stock_manager=stock_manager,
        check_interval=300,  # 5分钟检查一次
        max_alerts=100,
        mode='dev',
        config=config
    )
    
    try:
        # 启动监控
        monitor.start_monitoring()
        
        # 运行一段时间
        while True:
            # 获取警报
            alerts = monitor.get_alerts()
            if alerts:
                print("\n收到新的警报:")
                for alert in alerts:
                    print(f"股票: {alert['symbol']}")
                    print(f"类型: {alert['type']}")
                    print(f"时间: {alert['timestamp']}")
                    print("报告:", alert['report'])
                    print("-" * 50)
            
            # 清除已处理的警报
            monitor.clear_alerts()
            
            # 等待一段时间
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n正在停止监控...")
    finally:
        # 停止监控
        monitor.stop_monitoring()

if __name__ == "__main__":
    main() 