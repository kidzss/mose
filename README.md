# 信号组合模型系统

## 项目介绍

信号组合模型系统是一个用于优化多个交易信号组合的Python工具包。该系统提供多种组合算法，可以自动确定最优权重以产生更好的交易策略。

主要功能特点：

- 支持多种信号组合模型（线性模型、神经网络模型、XGBoost模型等）
- 提供信号提取工具，支持各类技术指标和交易信号
- 提供信号标准化工具
- 提供交易策略评估工具
- 提供时间序列交叉验证工具
- 支持模型保存和加载
- 可视化工具

## 最新进展

### 2025年3月更新

- 完成了`SignalExtractor`核心模块，支持快速提取32+种技术分析信号
- 实现了`XGBoostSignalCombiner`模型，支持信号组合和强大的回测功能
- 优化了信号处理流程，增强了对缺失值和极端值的处理
- 改进了模型评估和回测功能，添加了更多金融指标
- 增加了时间序列交叉验证和模型解释功能
- 完善了数据生成工具，支持合成测试数据创建
- 实现了模型序列化和反序列化功能，支持模型保存和加载
- 改进了可视化组件，提供更丰富的性能分析图表

## 安装方法

### 依赖安装

```bash
pip install -r requirements.txt
```

### 从源码安装

```bash
pip install -e .
```

## 快速开始

下面是一个基本使用示例：

```python
import pandas as pd
import numpy as np
from strategy_optimizer.models import LinearCombinationModel
from strategy_optimizer.utils import evaluate_strategy, plot_strategy_performance

# 加载数据
signals = pd.read_csv("signals.csv", index_col=0, parse_dates=True)
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True).iloc[:, 0]

# 创建线性模型
model = LinearCombinationModel(
    model_name="线性组合模型",
    normalize_signals=True,
    normalize_method="zscore",
    allow_short=True,
    weights_constraint="unit_sum",
    optimization_method="sharpe"
)

# 训练模型
model.fit(signals, returns, verbose=True)

# 获取权重
weights = model.get_weights()
print("信号权重:")
print(weights)

# 预测组合信号
combined_signal = model.predict(signals)

# 评估策略表现
performance = model.evaluate(signals, returns)
print("\n策略表现:")
for metric, value in performance.items():
    print(f"{metric}: {value:.4f}")

# 可视化权重
fig_weights = model.plot_weights()
fig_weights.savefig("weights.png")

# 可视化策略表现
fig_perf = plot_strategy_performance(returns, np.sign(combined_signal))
fig_perf.savefig("performance.png")

# 保存模型
model.save("linear_model.pkl")

# 加载模型
loaded_model = LinearCombinationModel.load("linear_model.pkl")
```

## 核心模块

### 模型模块 (`strategy_optimizer.models`)

- `BaseSignalModel`: 信号组合模型的基类
- `LinearCombinationModel`: 线性信号组合模型
- `NeuralCombinationModel`: 神经网络信号组合模型
- `XGBoostSignalCombiner`: XGBoost信号组合模型
- `SignalCombinerModel`: 高级信号组合模型（支持市场状态感知）
- `TransformerModel`: Transformer架构的信号模型

### 工具模块 (`strategy_optimizer.utils`)

- `SignalExtractor`: 交易信号提取器
- `normalize_signals`: 信号标准化函数
- `evaluate_strategy`: 策略评估函数
- `plot_strategy_performance`: 策略表现可视化
- `calculate_ic`: 信息系数计算
- `plot_ic_heatmap`: 信息系数热图可视化
- `SignalOptimizer`: 信号优化器类
- `TimeSeriesSplit`: 时间序列数据分割器
- `walk_forward_validation`: 滚动窗口验证函数
- `DataGenerator`: 数据生成器
- `EnhancedEvaluation`: 增强型策略评估
- `PositionSizing`: 仓位管理工具

## 使用示例

查看 `strategy_optimizer/examples/` 目录下的示例文件获取完整的使用示例。

### 信号提取示例

```python
from strategy_optimizer.utils.signal_extractor import SignalExtractor

# 加载价格数据 (包含 open, high, low, close, volume)
price_data = pd.read_csv("price_data.csv", index_col=0, parse_dates=True)

# 创建信号提取器
extractor = SignalExtractor(price_data)

# 提取各类交易信号
extractor.extract_trend_signals()      # 趋势信号 (移动平均、MACD等)
extractor.extract_momentum_signals()   # 动量信号 (RSI、CCI等)
extractor.extract_volatility_signals() # 波动率信号 (波动率、ATR等)
extractor.extract_volume_signals()     # 成交量信号 (OBV、VWAP等)
extractor.extract_support_resistance_signals() # 支撑阻力信号 (布林带等)

# 获取所有信号
signals_df = extractor.get_signals()

# 获取信号元数据
metadata = extractor.metadata

# 标准化信号
normalized_signals = extractor.normalize_signals(method="zscore")

# 按相关性对信号排序
correlation_df = extractor.rank_signals_by_correlation(target_returns)
```

### 线性模型示例

```python
from strategy_optimizer.models import LinearCombinationModel

# 创建线性模型
model = LinearCombinationModel(
    normalize_signals=True,
    normalize_method="zscore",
    allow_short=True,
    weights_constraint="unit_sum",
    optimization_method="sharpe"
)

# 训练模型
model.fit(signals, returns)

# 预测信号
combined_signal = model.predict(new_signals)
```

### XGBoost模型示例

```python
from strategy_optimizer.models.xgboost_model import XGBoostSignalCombiner

# 创建XGBoost模型
model = XGBoostSignalCombiner(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8
)

# 训练模型
model.fit(
    train_signals, 
    train_returns, 
    eval_set=[(val_signals, val_returns)],
    early_stopping_rounds=30
)

# 预测
predictions = model.predict(test_signals)

# 获取性能评估
performance = model.test_performance

# 获取特征重要性
importance = model.get_feature_importance(plot=True, top_n=15)

# 绘制性能图表
model.plot_performance(test_signals, test_returns, plot_type='cumulative_returns')

# 进行回测
backtest_result = model.backtest(test_signals, test_returns, transaction_cost=0.001)

# 进行时间序列交叉验证
cv_results = model.time_series_cv(signals, returns, n_splits=5, test_size=0.2)
```

### 神经网络模型示例

```python
from strategy_optimizer.models import NeuralCombinationModel

# 创建神经网络模型
model = NeuralCombinationModel(
    normalize_signals=True,
    normalize_method="zscore",
    allow_short=True,
    hidden_dims=[64, 32],
    dropout_rate=0.2,
    learning_rate=0.001,
    batch_size=64,
    epochs=100,
    early_stopping=10
)

# 训练模型（包含验证集）
model.fit(
    train_signals, 
    train_returns, 
    val_signals=val_signals, 
    val_targets=val_returns,
    verbose=True
)

# 绘制训练历史
model.plot_training_history()
```

### 滚动窗口验证示例

```python
from strategy_optimizer.utils import walk_forward_validation
from strategy_optimizer.models import LinearCombinationModel

# 创建模型
model = LinearCombinationModel()

# 运行验证
fold_metrics, avg_metrics, final_model = walk_forward_validation(
    model,
    signals,
    returns,
    n_splits=5,
    test_size=100,
    verbose=True
)
```

## 模型参数

### SignalExtractor 参数

- `price_data`: 价格数据 DataFrame，应包含 open, high, low, close, volume 列
- `use_talib`: 是否使用TA-Lib库，默认为True
- `signal_naming`: 信号命名风格，可选 "short", "long"

### XGBoostSignalCombiner 参数

- `n_estimators`: 决策树数量（默认：100）
- `learning_rate`: 学习率（默认：0.1）
- `max_depth`: 树的最大深度（默认：3）
- `min_child_weight`: 最小子节点权重（默认：1）
- `gamma`: 节点分裂所需的最小损失减少（默认：0）
- `subsample`: 训练实例的抽样比例（默认：1.0）
- `colsample_bytree`: 特征抽样比例（默认：1.0）
- `reg_alpha`: L1正则化项（默认：0）
- `reg_lambda`: L2正则化项（默认：1）
- `random_state`: 随机种子（默认：None）
- `objective`: 目标函数（默认："reg:squarederror"）
- `n_jobs`: 并行工作线程数（默认：-1，使用所有线程）

### LinearCombinationModel 参数

- `model_name`: 模型名称
- `normalize_signals`: 是否标准化信号
- `normalize_method`: 标准化方法，可选 "zscore", "minmax", "maxabs", "robust", "rank", "quantile"
- `normalize_window`: 标准化窗口大小，None表示全局标准化
- `allow_short`: 是否允许做空
- `weights_constraint`: 权重约束方式，可选 "unit_sum", "unit_norm", "simplex", None
- `optimization_method`: 优化方法，可选 "sharpe", "sortino", "returns", "min_variance", "regression"
- `regularization`: 正则化参数

### NeuralCombinationModel 参数

- `model_name`: 模型名称
- `normalize_signals`: 是否标准化信号
- `normalize_method`: 标准化方法
- `normalize_window`: 标准化窗口大小
- `allow_short`: 是否允许做空
- `hidden_dims`: 隐藏层维度列表
- `dropout_rate`: Dropout比率
- `use_batch_norm`: 是否使用批归一化
- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `early_stopping`: 早停轮数
- `loss_fn`: 损失函数，可选 "mse", "mae", "sharpe", "sortino"
- `device`: 设备，None表示自动选择

## 未来计划

### 短期计划 (1-2个月)

1. **增强XGBoostSignalCombiner**
   - 添加更多评估指标和回测功能
   - 优化特征工程流程
   - 增加自动特征选择机制

2. **完善SignalExtractor**
   - 增加更多信号类型
   - 优化信号计算性能
   - 添加自定义信号支持

3. **改进模型评估**
   - 设计更全面的策略评估框架
   - 增加统计显著性测试
   - 优化回测可视化

### 中期计划 (3-6个月)

1. **开发新型模型**
   - 实现强化学习信号组合器
   - 开发基于图神经网络的市场模型
   - 集成多个模型的集成学习框架

2. **优化系统架构**
   - 重构核心模块，提高扩展性
   - 设计流水线处理架构
   - 优化计算性能和内存使用

3. **增强用户界面**
   - 开发Web界面
   - 增加交互式数据可视化
   - 提供用户友好的报告生成

### 长期计划 (6-12个月)

1. **实现实时信号生成系统**
   - 对接交易API
   - 开发实时数据处理流程
   - 构建实时信号监控系统

2. **扩展到更多资产类别**
   - 增加对期货、外汇、加密货币的支持
   - 实现跨资产类别的信号组合
   - 开发多资产动态配置模型

3. **构建完整交易系统**
   - 整合仓位管理模块
   - 开发风险控制系统
   - 实现自动交易执行

## 贡献指南

欢迎提交问题和功能请求！如果您想贡献代码，请遵循以下步骤：

1. Fork 仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。 