# 优化器 (Optimizers)

## 概述

优化器子模块提供了一系列工具，用于优化交易策略的参数和信号组合。它是策略优化器模块的核心部分，通过系统化的方法寻找最佳策略配置，以提高交易性能。

## 主要组件

### ParameterOptimizer

`ParameterOptimizer` 类专注于寻找交易策略的最佳参数组合。

#### 主要功能

1. **参数空间搜索**：支持网格搜索、随机搜索和贝叶斯优化等方法
2. **交叉验证**：通过时间序列交叉验证评估参数稳定性
3. **性能评估**：基于多种指标（夏普比率、最大回撤、胜率等）评估参数组合
4. **过拟合检测**：检测并防止参数过拟合历史数据

#### 使用示例

```python
from strategy_optimizer.optimizers.parameter_optimizer import ParameterOptimizer
from strategy.strategies import RSIStrategy

# 创建参数优化器
optimizer = ParameterOptimizer(optimization_method="bayesian")

# 定义参数搜索空间
param_space = {
    "period": [7, 14, 21, 28], 
    "overbought": [70, 75, 80],
    "oversold": [20, 25, 30]
}

# 执行优化
best_params, performance_metrics = optimizer.optimize(
    strategy_class=RSIStrategy,
    param_space=param_space,
    data=price_data,
    target_metric="sharpe_ratio",
    iterations=100,
    cv_folds=5  # 5折时间序列交叉验证
)

# 创建优化后的策略
optimized_strategy = RSIStrategy(**best_params)

# 获取参数重要性
param_importance = optimizer.get_parameter_importance()

# 可视化优化过程
optimizer.plot_optimization_history()
```

### SignalOptimizer

`SignalOptimizer` 类专注于优化多个交易信号的组合方式，找到最佳的信号加权方案。

#### 主要功能

1. **信号权重优化**：寻找最佳的信号组合权重
2. **市场状态感知**：根据不同市场状态调整信号权重
3. **特征选择**：识别并保留最有预测价值的信号
4. **模型选择**：支持多种机器学习模型（随机森林、XGBoost、神经网络等）

#### 使用示例

```python
from strategy_optimizer.optimizers.signal_optimizer import SignalOptimizer
from strategy_optimizer.extractors.strategy_signal_extractor import StrategySignalExtractor

# 提取信号
extractor = StrategySignalExtractor()
signals_df = extractor.extract_signals_from_strategies(strategies, price_data)

# 创建信号优化器
optimizer = SignalOptimizer(
    model_type="random_forest",
    market_state_aware=True,
    feature_selection=True
)

# 定义目标变量（如明天的价格变动方向）
target = price_data['close'].pct_change().shift(-1) > 0

# 训练优化器
optimizer.fit(
    signals=signals_df,
    target=target,
    market_features=market_features,  # 可选的市场特征数据
    validation_ratio=0.3
)

# 生成优化信号
optimized_signals = optimizer.predict(signals_df, market_features)

# 获取信号重要性
signal_importance = optimizer.get_feature_importance()

# 可视化不同市场状态下的信号权重
optimizer.plot_signal_weights_by_market_state()
```

## 优化方法

### 网格搜索 (Grid Search)

系统地评估参数空间中所有可能的参数组合。适用于参数空间较小的情况。

```python
optimizer = ParameterOptimizer(optimization_method="grid")
```

### 随机搜索 (Random Search)

随机采样参数空间中的参数组合进行评估。适用于参数空间较大的情况。

```python
optimizer = ParameterOptimizer(optimization_method="random", n_iterations=200)
```

### 贝叶斯优化 (Bayesian Optimization)

使用贝叶斯统计方法构建参数性能的概率模型，有效地指导搜索方向。适用于计算资源有限且参数空间复杂的情况。

```python
optimizer = ParameterOptimizer(
    optimization_method="bayesian",
    n_iterations=50,
    acquisition_function="ei"  # 期望改进
)
```

## 评估指标

优化器支持多种评估指标作为优化目标：

- **夏普比率 (Sharpe Ratio)**：风险调整后收益指标
- **索提诺比率 (Sortino Ratio)**：只考虑下行风险的收益指标
- **最大回撤 (Maximum Drawdown)**：最大亏损百分比
- **胜率 (Win Rate)**：盈利交易的比例
- **复合年增长率 (CAGR)**：年化收益率
- **卡尔马比率 (Calmar Ratio)**：年化收益率与最大回撤的比值
- **自定义指标**：支持用户定义的评估函数

## 交叉验证

时间序列交叉验证用于防止过拟合，确保参数在不同时间段上表现稳定：

```python
# 使用时间序列折叠交叉验证
optimizer.optimize(
    # ... 其他参数 ...
    cv_folds=5,
    cv_gap=20  # 训练集和测试集之间的间隔天数
)
```

## 注意事项

1. 参数优化可能导致过拟合，建议使用交叉验证并留出独立的测试集
2. 贝叶斯优化在大多数情况下是效率最高的选择，但可能需要更多的初始样本点
3. 优化过程可能耗时较长，考虑使用并行计算和结果缓存
4. 不同市场状态下的最佳参数可能有显著差异，考虑使用市场状态感知的优化 