import pandas as pd
import numpy as np
import logging
from strategy_optimizer.data_processors.data_processor import DataProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_momentum_strategy():
    # 初始化数据处理器
    data_processor = DataProcessor()
    
    # 获取一个样本股票的数据
    symbol = "AAPL"  # 使用苹果股票作为测试
    start_date = "2023-01-01"
    end_date = "2024-03-13"
    
    logger.info(f"获取 {symbol} 的数据，时间范围: {start_date} 到 {end_date}")
    
    # 获取股票数据
    df = data_processor.get_stock_data(symbol, start_date, end_date)
    
    if df is None or len(df) == 0:
        logger.error("无法获取股票数据")
        return
    
    logger.info(f"获取到 {len(df)} 条数据记录")
    
    # 打印原始DataFrame的列
    logger.info("\n原始DataFrame的列:")
    logger.info(df.columns.tolist())
    
    # 计算动量策略信号
    df, signals = data_processor._calculate_momentum_signal(df)
    
    if signals is None:
        logger.error("计算动量策略信号失败")
        return
    
    # 再次打印DataFrame的列，看看是否有变化
    logger.info("\n计算信号后DataFrame的列:")
    logger.info(df.columns.tolist())
    
    # 检查信号的基本统计信息
    signal_stats = {
        '信号长度': len(signals),
        '唯一值': np.unique(signals),
        '买入信号数量': np.sum(signals == 1),
        '卖出信号数量': np.sum(signals == -1),
        '持仓不变数量': np.sum(signals == 0)
    }
    
    logger.info("\n动量策略信号统计:")
    for key, value in signal_stats.items():
        logger.info(f"{key}: {value}")
    
    # 检查必需的列是否存在
    required_columns = ['high_20', 'low_10', 'SMA_200', 'RSI', 'MACD', 'MACD_signal']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"缺少必需的列: {missing_columns}")
    else:
        logger.info("所有必需的列都已正确计算")
        
        # 打印每个必需列的基本统计信息
        for col in required_columns:
            stats = {
                '均值': df[col].mean(),
                '标准差': df[col].std(),
                '最小值': df[col].min(),
                '最大值': df[col].max(),
                'NaN数量': df[col].isna().sum()
            }
            logger.info(f"\n{col} 统计信息:")
            for stat_key, stat_value in stats.items():
                logger.info(f"{stat_key}: {stat_value}")

if __name__ == "__main__":
    test_momentum_strategy() 