import time
import logging
import asyncio
from monitor.stock_monitor import StockMonitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    # 创建监控器实例，设置刷新间隔为60秒
    monitor = StockMonitor(refresh_interval=60)
    
    print("开始监控股票...")
    print(f"监控的股票: {', '.join(monitor.monitored_stocks)}")
    print(f"观察列表: {', '.join(monitor.watchlist_stocks)}")
    print(f"刷新间隔: {monitor.refresh_interval}秒")
    print("按 Ctrl+C 停止监控")
    
    try:
        # 执行监控
        await monitor.monitor_stocks()
    except KeyboardInterrupt:
        print("\n停止监控...")

if __name__ == "__main__":
    asyncio.run(main()) 