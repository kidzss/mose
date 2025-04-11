import sys
import os
import logging
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_loader import DataLoader
from strategy import NiuniuStrategy
from strategy_optimizer.parameter_optimizer import ParameterOptimizer

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 加载历史数据
        data_loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 使用一年的数据
        
        data = data_loader.load_historical_data(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"加载了 {len(data)} 条历史数据")
        
        # 创建策略实例
        strategy = NiuniuStrategy()
        
        # 创建优化器实例
        optimizer = ParameterOptimizer(strategy, data)
        
        # 开始优化
        logger.info("开始参数优化...")
        best_params = optimizer.optimize_parameters()
        
        # 输出最佳参数
        logger.info("优化完成！最佳参数：")
        for param, value in best_params.items():
            logger.info(f"{param}: {value}")
            
    except Exception as e:
        logger.error(f"优化过程中出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 