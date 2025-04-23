import logging
import os
import json
from monitor.data_fetcher import DataFetcher
from monitor.stock_manager import StockManager
from monitor.stock_monitor import StockMonitor
from monitor.alert_system import AlertSystem
from monitor.strategy_manager import StrategyManager
from monitor.report_generator import ReportGenerator
from monitor.market_monitor import MarketMonitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'monitor_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 初始化组件
        data_fetcher = DataFetcher(config)
        stock_manager = StockManager(config)
        alert_system = AlertSystem(config)
        strategy_manager = StrategyManager(config)
        report_generator = ReportGenerator(config)
        market_monitor = MarketMonitor(config)
        
        # 创建监控器
        monitor = StockMonitor(
            data_fetcher=data_fetcher,
            strategy_manager=strategy_manager,
            report_generator=report_generator,
            market_monitor=market_monitor,
            stock_manager=stock_manager,
            check_interval=300,  # 5分钟检查一次
            max_alerts=100,
            mode='prod',
            config=config
        )
        
        # 运行监控
        monitor.start_monitoring()
        
    except Exception as e:
        logger.error(f"监控系统运行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 