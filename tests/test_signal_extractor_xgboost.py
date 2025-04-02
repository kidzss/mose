#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整测试脚本：测试SignalExtractor和XGBoostSignalCombiner功能

此脚本提供了对SignalExtractor和XGBoostSignalCombiner的全面测试，
包括数据加载、信号提取、特征工程、模型训练、性能评估、特征重要性分析等。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir))
sys.path.insert(0, root_dir)

# 导入项目模块
try:
    from strategy_optimizer.utils.signal_extractor import SignalExtractor
    from strategy_optimizer.utils.data_generator import DataGenerator
    from strategy_optimizer.models.xgboost_model import XGBoostSignalCombiner
    from strategy_optimizer.utils.evaluation import evaluate_strategy
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)

def generate_test_data(n_samples=500, save_path=None):
    """生成测试数据"""
    logger.info("生成测试数据...")
    
    # 使用DataGenerator生成合成数据
    generator = DataGenerator(seed=42)
    signals, returns = generator.generate_synthetic_data(
        n_samples=n_samples,
        n_signals=5,
        signal_strength={0: 0.7, 1: 0.5, 2: 0.3},
        noise_level=0.2,
        start_date="2020-01-01"
    )
    
    # 基于收益率生成价格数据
    initial_price = 100.0
    prices = (1 + returns).cumprod() * initial_price
    
    # 创建OHLCV数据
    price_data = pd.DataFrame({
        'open': prices * (1 - np.random.randn(len(prices)) * 0.005),
        'high': 0.0,
        'low': 0.0,
        'close': prices,
        'volume': np.abs(np.random.randn(len(prices)) * 1000000 + 500000)
    }, index=returns.index)
    
    # 填充high和low
    for i in range(len(price_data)):
        idx = price_data.index[i]
        price_data.loc[idx, 'high'] = max(price_data.loc[idx, 'open'], price_data.loc[idx, 'close']) * (1 + abs(np.random.randn()) * 0.008)
        price_data.loc[idx, 'low'] = min(price_data.loc[idx, 'open'], price_data.loc[idx, 'close']) * (1 - abs(np.random.randn()) * 0.008)
    
    # 如果提供了保存路径，保存数据
    if save_path:
        price_data.to_csv(save_path)
        logger.info(f"数据已保存到 {save_path}")
    
    # 计算目标变量：未来的1天收益率
    target_returns = price_data['close'].pct_change().shift(-1)
    
    logger.info(f"生成数据完成: {len(price_data)}条记录，日期范围从{price_data.index[0]}到{price_data.index[-1]}")
    
    return price_data, target_returns

def test_signal_extractor(price_data, returns):
    """测试SignalExtractor功能"""
    logger.info("\n====== 测试SignalExtractor功能 ======")
    
    # 初始化信号提取器
    extractor = SignalExtractor(price_data)
    logger.info("SignalExtractor初始化成功")
    
    # 测试各种信号提取方法
    logger.info("提取趋势信号...")
    trend_signals = extractor.extract_trend_signals()
    logger.info(f"- 提取了{len(trend_signals.columns)}个趋势信号")
    
    logger.info("提取动量信号...")
    momentum_signals = extractor.extract_momentum_signals()
    logger.info(f"- 提取了{len(momentum_signals.columns)}个动量信号")
    
    logger.info("提取波动率信号...")
    volatility_signals = extractor.extract_volatility_signals()
    logger.info(f"- 提取了{len(volatility_signals.columns)}个波动率信号")
    
    logger.info("提取成交量信号...")
    volume_signals = extractor.extract_volume_signals()
    logger.info(f"- 提取了{len(volume_signals.columns)}个成交量信号")
    
    logger.info("提取支撑阻力信号...")
    support_resistance_signals = extractor.extract_support_resistance_signals()
    logger.info(f"- 提取了{len(support_resistance_signals.columns)}个支撑阻力信号")
    
    # 获取所有信号
    all_signals = extractor.get_signals()
    logger.info(f"总共提取了{len(all_signals.columns)}个信号")
    
    # 测试信号元数据
    metadata = extractor.metadata
    categories = set(meta['category'] for meta in metadata.values())
    logger.info(f"信号类别: {', '.join(categories)}")
    
    # 测试信号标准化
    normalized_signals = extractor.normalize_signals(method="zscore")
    logger.info(f"信号标准化完成，使用方法: zscore")
    
    # 测试信号相关性排序
    correlation_df = extractor.rank_signals_by_correlation(returns)
    logger.info("信号与目标相关性排序:")
    for i, row in correlation_df.head(5).iterrows():
        logger.info(f"  {row['signal']}: 相关性 = {row['correlation']:.4f}")
    
    # 测试获取顶部信号
    top_signals = extractor.get_top_signals(returns, n=10)
    logger.info(f"获取了排名前10的信号")
    
    # 获取信号数据统计
    signal_stats = all_signals.describe()
    logger.info(f"信号数据统计完成，统计量维度: {signal_stats.shape}")
    
    # 测试特征有效性：查看NaN的比例
    nan_percent = all_signals.isna().mean()
    logger.info(f"平均NaN比例: {nan_percent.mean():.4f}")
    
    return extractor, all_signals

def test_feature_engineering(extractor, all_signals, returns):
    """测试特征工程功能"""
    logger.info("\n====== 测试特征工程功能 ======")
    
    # 1. 标准化信号
    logger.info("标准化信号...")
    normalized_signals = extractor.normalize_signals(method="zscore")
    
    # 2. 去除包含过多NaN值的特征
    threshold = 0.95  # 要求至少95%的数据是非NaN
    valid_columns = normalized_signals.columns[normalized_signals.notna().mean() >= threshold]
    cleaned_signals = normalized_signals[valid_columns].copy()
    logger.info(f"移除高缺失值特征后，剩余{len(cleaned_signals.columns)}个特征")
    
    # 3. 填充剩余的NaN值
    cleaned_signals.fillna(0, inplace=True)
    logger.info("NaN值已填充")
    
    # 4. 基于相关性筛选特征
    correlation_df = extractor.rank_signals_by_correlation(returns)
    
    # 5. 处理极端值
    def clip_extreme_values(df, min_quantile=0.01, max_quantile=0.99):
        df_clipped = df.copy()
        for col in df_clipped.columns:
            min_val = df_clipped[col].quantile(min_quantile)
            max_val = df_clipped[col].quantile(max_quantile)
            df_clipped[col] = df_clipped[col].clip(min_val, max_val)
        return df_clipped
    
    cleaned_signals = clip_extreme_values(cleaned_signals)
    logger.info("极端值已处理")
    
    logger.info(f"特征工程完成，最终特征数量: {len(cleaned_signals.columns)}")
    logger.info(f"特征样本数量: {len(cleaned_signals)}")
    
    # 打印相关性最高的前5个特征
    logger.info("与收益率相关性最高的前5个特征:")
    for i, row in correlation_df.head(5).iterrows():
        logger.info(f"  {row['signal']}: 相关性 = {row['correlation']:.4f}")
    
    return cleaned_signals

def test_xgboost_model(features, returns, test_size=0.3):
    """测试XGBoostSignalCombiner模型"""
    logger.info("\n====== 测试XGBoostSignalCombiner模型 ======")
    
    # 确保特征和目标变量的索引一致
    common_index = features.index.intersection(returns.index)
    X = features.loc[common_index]
    y = returns.loc[common_index]
    
    # 确保没有NaN值
    # 检查目标变量中的NaN值
    if y.isna().any():
        logger.info(f"目标变量中有{y.isna().sum()}个NaN值，将被移除")
        valid_idx = ~y.isna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
    
    # 检查特征中的NaN值
    if X.isna().any().any():
        logger.info(f"特征中有NaN值，将填充为0")
        X = X.fillna(0)
    
    # 划分训练集和测试集 (按时间顺序)
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    logger.info(f"划分数据集: 训练集 {len(X_train)}条 ({train_size/len(X)*100:.1f}%), 测试集 {len(X_test)}条 ({test_size*100:.1f}%)")
    
    # 初始化XGBoost模型
    model = XGBoostSignalCombiner(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    logger.info("XGBoost模型初始化完成")
    
    # 训练模型
    logger.info("开始训练模型...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    logger.info("模型训练完成")
    
    # 评估模型性能
    train_perf = model.train_performance
    test_perf = model.test_performance
    
    logger.info("\n模型性能评估结果:")
    logger.info("训练集性能:")
    for metric in ['sign_accuracy', 'sharpe_ratio', 'annual_return', 'max_drawdown', 'r2']:
        logger.info(f"- {metric}: {train_perf[metric]:.4f}")
    
    logger.info("\n测试集性能:")
    for metric in ['sign_accuracy', 'sharpe_ratio', 'annual_return', 'max_drawdown', 'r2']:
        logger.info(f"- {metric}: {test_perf[metric]:.4f}")
    
    # 获取特征重要性
    importance = model.get_feature_importance(plot=False)
    
    logger.info("\n特征重要性 (前10个):")
    for i, row in importance.head(10).iterrows():
        logger.info(f"- {row['feature']}: {row['importance']:.4f}")
    
    # 测试模型预测
    y_pred = model.predict(X_test)
    logger.info(f"模型预测完成，预测结果个数: {len(y_pred)}")
    
    # 测试模型交叉验证
    logger.info("\n执行时间序列交叉验证...")
    cv_results = model.cross_validate(X, y, n_splits=5, metrics=['neg_mean_squared_error', 'r2'])
    
    logger.info("交叉验证结果:")
    for metric, (mean, std) in cv_results.items():
        logger.info(f"- {metric}: {mean:.4f} ± {std:.4f}")
    
    # 绘制性能可视化
    plt.figure(figsize=(12, 10))
    
    # 绘制预测vs实际值
    plt.subplot(2, 1, 1)
    plt.plot(y_test.index, y_test.values, label='实际收益率', color='blue', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='预测收益率', color='red', alpha=0.7)
    plt.title('预测vs实际收益率')
    plt.ylabel('收益率')
    plt.legend()
    plt.grid(True)
    
    # 绘制累积收益对比
    plt.subplot(2, 1, 2)
    cum_actual = (1 + y_test).cumprod() - 1
    
    # 计算模型信号产生的累积收益
    position = np.sign(y_pred)
    strategy_returns = position * y_test
    cum_strategy = (1 + strategy_returns).cumprod() - 1
    
    # 买入持有基准
    plt.plot(y_test.index, cum_actual, label='买入持有', color='blue')
    plt.plot(y_test.index, cum_strategy, label='模型策略', color='green')
    plt.title('累积收益对比')
    plt.xlabel('日期')
    plt.ylabel('累积收益率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_model_performance.png')
    logger.info("性能图表已保存为 'test_model_performance.png'")
    
    # 保存和加载模型
    model_path = "../test_xgboost_model.json"
    model.save_model(model_path)
    logger.info(f"模型已保存到 {model_path}")
    
    # 创建新模型实例，加载已保存的模型
    new_model = XGBoostSignalCombiner()
    new_model.load_model(model_path)
    logger.info("模型加载成功")
    
    # 验证加载后的模型预测结果
    y_pred_new = new_model.predict(X_test)
    is_equal = np.allclose(y_pred, y_pred_new)
    logger.info(f"加载后的模型预测与原模型一致: {is_equal}")
    
    return model, X_test, y_test

def run_backtest(model, X_test, y_test):
    """运行回测"""
    logger.info("\n====== 运行回测 ======")
    
    # 执行回测，不同的交易成本
    for cost in [0.0, 0.001, 0.003]:
        logger.info(f"\n交易成本: {cost*100:.1f}%")
        backtest_result = model.backtest(X_test, y_test, transaction_cost=cost)
        
        metrics_to_show = [
            'sharpe_ratio', 'annual_return', 'max_drawdown', 
            'win_rate', 'profit_factor', 'recovery_factor'
        ]
        
        for metric in metrics_to_show:
            if metric in backtest_result:
                logger.info(f"- {metric}: {backtest_result[metric]:.4f}")
    
    # 绘制回测结果
    plt.figure(figsize=(12, 8))
    
    # 绘制回测的累积收益
    cum_returns = backtest_result['cumulative_returns']
    plt.plot(cum_returns.index, cum_returns, label='策略累积收益', color='green')
    
    # 基准收益
    cum_hold = (1 + y_test).cumprod() - 1
    plt.plot(cum_hold.index, cum_hold, label='买入持有', color='blue', alpha=0.7)
    
    plt.title('策略回测结果')
    plt.xlabel('日期')
    plt.ylabel('累积收益率')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_backtest_performance.png')
    logger.info("回测图表已保存为 'test_backtest_performance.png'")
    
    # 计算月度收益率
    if isinstance(backtest_result['strategy_returns'], pd.Series) and len(backtest_result['strategy_returns']) > 0:
        monthly_returns = backtest_result['strategy_returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        logger.info("\n月度收益率:")
        logger.info(monthly_returns.tail(5))
        
        # 计算胜率
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        if total_months > 0:
            win_rate = positive_months / total_months
            logger.info(f"月度胜率: {win_rate:.2%} ({positive_months}/{total_months})")
    
    return backtest_result

def main():
    """主函数"""
    try:
        logger.info("============== 开始测试 ==============")
        
        # 设置随机种子保持结果一致
        np.random.seed(42)
        
        # 生成测试数据
        test_data_path = "test_price_data.csv"
        price_data, returns = generate_test_data(n_samples=500, save_path=test_data_path)
        
        # 测试信号提取
        extractor, all_signals = test_signal_extractor(price_data, returns)
        
        # 测试特征工程
        features = test_feature_engineering(extractor, all_signals, returns)
        
        # 测试XGBoost模型
        model, X_test, y_test = test_xgboost_model(features, returns)
        
        # 运行回测
        backtest_result = run_backtest(model, X_test, y_test)
        
        logger.info("============== 测试成功完成 ==============")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 