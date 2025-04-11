#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成测试 - 策略信号到模型训练的完整工作流
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

# 导入项目模块
try:
    from strategy_optimizer.extractors.strategy_signal_extractor import StrategySignalExtractor
    from strategy_optimizer.data_processors.market_state_analyzer import MarketStateAnalyzer
    from strategy_optimizer.models.conditional_xgboost import ConditionalXGBoostCombiner
    from strategy_optimizer.evaluation.report_generator import StrategyReportGenerator
    from strategy.bollinger_bands_strategy import BollingerBandsStrategy
    from strategy.rsi_strategy import RSIStrategy
    from strategy.macd_strategy import MACDStrategy
    from strategy.golden_cross_strategy import GoldenCrossStrategy
    from strategy.custom_strategy import CustomStrategy
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)

def load_test_data(filepath='test_price_data.csv'):
    """
    加载测试数据
    
    参数:
        filepath: 测试数据文件路径
        
    返回:
        DataFrame，包含OHLCV数据
    """
    logger.info(f"加载测试数据: {filepath}")
    try:
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # 确保列名标准化
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        data.rename(columns={k: v for k, v in column_map.items() if k in data.columns}, inplace=True)
        
        # 确保包含所有必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"数据缺少必要的列: {missing_columns}")
        
        return data
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

def prepare_future_returns(data, period=5):
    """
    准备未来收益率数据
    
    参数:
        data: 价格数据
        period: 未来收益周期（天数）
        
    返回:
        Series，未来period天的收益率
    """
    logger.info(f"计算未来{period}天收益率")
    future_returns = data['close'].pct_change(period).shift(-period)
    return future_returns

def main():
    """主函数"""
    logger.info("开始信号整合测试")
    
    # 步骤1: 加载测试数据
    try:
        data = load_test_data()
        logger.info(f"加载了 {len(data)} 条数据记录")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return
    
    # 创建输出目录
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 初始化多个策略
    logger.info("初始化交易策略")
    strategies = [
        BollingerBandsStrategy(),
        RSIStrategy(),
        MACDStrategy(),
        GoldenCrossStrategy(),
        CustomStrategy()
    ]
    
    # 步骤3: 使用策略信号提取器提取信号
    logger.info("使用策略信号提取器提取信号")
    extractor = StrategySignalExtractor()
    signals_df = extractor.extract_signals_from_strategies(strategies, data)
    
    # 输出提取的信号维度
    logger.info(f"提取了 {signals_df.shape[1]} 个信号特征，数据形状: {signals_df.shape}")
    
    # 查看信号重要性排名
    importance_scores = extractor.rank_signals_by_importance()
    top_signals = list(importance_scores.keys())[:10]  # 前10个重要信号
    logger.info(f"最重要的10个信号: {top_signals}")
    
    # 步骤4: 使用市场状态分析器分析市场状态
    logger.info("使用市场状态分析器分析市场状态")
    analyzer = MarketStateAnalyzer(data)
    market_state_df = analyzer.analyze_market_state()
    
    # 输出市场状态分布
    market_state_counts = market_state_df['market_state'].value_counts()
    logger.info(f"市场状态分布:\n{market_state_counts}")
    
    # 步骤5: 准备训练数据
    logger.info("准备训练数据")
    future_returns = prepare_future_returns(data)
    
    # 合并信号和市场状态
    X = signals_df.copy()
    y = future_returns
    
    # 处理NaN值
    X = X.fillna(0)
    mask = ~(y.isna())
    X = X[mask]
    y = y[mask]
    market_state = market_state_df['market_state'][mask]
    
    # 划分训练集和测试集
    logger.info("划分训练集和测试集")
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    market_state_train = market_state.iloc[:train_size]
    market_state_test = market_state.iloc[train_size:]
    
    # 步骤6: 训练条件XGBoost模型
    logger.info("训练条件XGBoost模型")
    model = ConditionalXGBoostCombiner()
    model.fit(X_train, y_train, market_state_train)
    
    # 步骤7: 评估模型并生成报告
    logger.info("评估模型并生成报告")
    report_generator = StrategyReportGenerator(output_dir=output_dir)
    
    # 生成模型性能报告
    performance = report_generator.generate_model_performance_report(
        model, X_train, y_train, X_test, y_test, 
        market_state_train, market_state_test
    )
    
    # 生成特征重要性报告
    feature_importance = report_generator.generate_feature_importance_report(model, X)
    
    # 生成市场状态分析
    market_analysis = report_generator.generate_market_state_analysis(
        market_state, y
    )
    
    # 生成策略比较报告
    strategies_for_comparison = [
        {"name": "条件XGBoost模型", "model": model}
    ]
    comparison = report_generator.generate_strategy_comparison_report(
        strategies_for_comparison, X_test, y_test, market_state_test
    )
    
    # 打印主要结果
    logger.info("\n===== 主要评估结果 =====")
    logger.info(f"模型整体性能 - RMSE: {performance['rmse']:.4f}, 方向准确率: {performance['direction_accuracy']:.4f}")
    
    if 'market_state' in performance:
        logger.info("\n按市场状态划分的性能:")
        for state, perf in performance['market_state'].items():
            logger.info(f"  状态 {state}: 样本数 {perf['count']}, 方向准确率: {perf['direction_accuracy']:.4f}")
    
    logger.info("\n===== 完成信号整合测试 =====")
    
if __name__ == "__main__":
    main() 