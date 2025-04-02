import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入我们创建的工具
from strategy_optimizer.utils.normalization import normalize_signals
from strategy_optimizer.utils.evaluation import evaluate_strategy, plot_strategy_performance
from strategy_optimizer.utils.time_series_cv import TimeSeriesCV
from strategy_optimizer.utils.signal_optimizer import SignalOptimizer, optimize_weights_grid_search

# 生成模拟数据
def generate_test_data(n_samples=500, n_signals=5, seed=42):
    np.random.seed(seed)
    
    # 生成日期
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_samples)]
    
    # 生成模拟信号 - 一些随机信号，但有一定的预测性
    signals = np.random.normal(0, 1, (n_samples, n_signals))
    
    # 生成模拟收益率
    # 假设第一个信号最有效，后面的信号效果依次减弱
    weights = np.array([0.5, 0.3, 0.1, -0.2, 0.1])
    noise = np.random.normal(0, 0.02, n_samples)
    
    # 目标收益 = 信号加权和 + 噪声
    returns = np.dot(signals, weights) + noise
    
    # 转换为pandas对象
    signals_df = pd.DataFrame(signals, index=dates, 
                            columns=[f'Signal_{i+1}' for i in range(n_signals)])
    returns_series = pd.Series(returns, index=dates, name='returns')
    
    return signals_df, returns_series

# 主测试函数
def test_signal_optimization():
    print("生成测试数据...")
    signals, returns = generate_test_data()
    
    print(f"信号数量: {signals.shape[1]}")
    print(f"样本数量: {signals.shape[0]}")
    
    # 测试标准化
    print("\n测试信号标准化...")
    normalized_signals = normalize_signals(signals, method="zscore", window=20)
    print(f"标准化后的信号范围: [{normalized_signals.min().min():.2f}, {normalized_signals.max().max():.2f}]")
    
    # 测试时间序列交叉验证
    print("\n测试时间序列交叉验证...")
    tscv = TimeSeriesCV(n_splits=5, test_size=100)
    
    # 转换为numpy数组用于交叉验证
    features = np.zeros((signals.shape[0], 1, 1))  # 模拟特征，但我们不会使用它
    signals_np = signals.values
    returns_np = returns.values
    
    # 打印交叉验证划分
    fold = 1
    for train_data, test_data in tscv.split(features, signals_np, returns_np):
        train_size = len(train_data['signals'])
        test_size = len(test_data['signals'])
        print(f"折 {fold}: 训练集大小={train_size}, 测试集大小={test_size}")
        fold += 1
    
    # 测试信号优化
    print("\n测试信号优化...")
    optimizer = SignalOptimizer(
        method="sharpe",
        normalize=True,
        normalize_method="zscore",
        normalize_window=20,
        weights_constraint="unit_sum",
        allow_short=True
    )
    
    # 优化权重
    weights = optimizer.optimize(signals, returns, verbose=True)
    
    print("\n优化结果:")
    print(weights)
    
    # 获取表现指标
    performance = optimizer.get_performance()
    print("\n表现指标:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    # 绘制权重图
    optimizer.plot_weights(figsize=(10, 6))
    plt.show()
    
    # 测试评估指标和可视化
    print("\n测试策略评估...")
    
    # 使用优化权重生成组合信号
    combined_signal = optimizer.predict(signals)
    positions = np.sign(combined_signal)
    
    # 计算策略收益
    strategy_returns = returns * positions
    
    # 绘制策略表现
    plot_strategy_performance(strategy_returns, returns, dates=signals.index, title="优化策略表现")
    
    # 测试网格搜索
    print("\n测试参数网格搜索...")
    param_grid = {
        'method': ['sharpe', 'sortino'],
        'normalize_method': ['zscore', 'minmax'],
        'allow_short': [True, False]
    }
    
    # 网格搜索找到最佳参数和权重
    best_params, best_weights, best_performance = optimize_weights_grid_search(
        signals, returns, param_grid, cv=3, n_jobs=1, verbose=True
    )
    
    print("\n网格搜索最佳参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
        
    print("\n网格搜索最佳权重:")
    print(best_weights)
    
    print("\n网格搜索最佳表现:")
    for metric, value in best_performance.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")

# 运行测试
if __name__ == "__main__":
    test_signal_optimization()