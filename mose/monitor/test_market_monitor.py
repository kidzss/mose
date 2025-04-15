import logging
import sys
from monitor.market_monitor import MarketMonitor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_market_environment_analysis():
    """测试市场环境分析"""
    try:
        # 初始化市场监控器
        monitor = MarketMonitor()
        
        # 获取市场数据
        market_data = monitor._get_market_data()
        print("\n市场数据:")
        print(market_data)
        
        # 分析市场环境
        environment = monitor._analyze_market_environment()
        print("\n市场环境分析结果:")
        for key, value in environment.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise

if __name__ == "__main__":
    test_market_environment_analysis() 