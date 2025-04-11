# 模型 (Models)

## 概述

Models 子模块提供了一系列用于信号组合和策略优化的机器学习模型。这些模型可以从多个交易策略的信号中学习，创建更强大的组合信号，适应不同的市场状态，并为交易决策提供支持。

## 主要组件

### 基础模型

- `BaseSignalModel`: 所有信号模型的基类，定义了通用接口
- `LinearCombinationModel`: 线性组合模型，使用线性回归组合信号
- `NeuralCombinationModel`: 神经网络组合模型，使用深度学习组合信号
- `XGBoostModel`: 基于XGBoost的组合模型，利用梯度提升树算法

### 高级模型

- `SignalCombiner`: 自适应信号组合器，根据市场状态动态调整信号权重
- `ConditionalXGBoost`: 条件XGBoost模型，根据市场条件调整预测
- `TransformerModel`: 基于Transformer架构的序列模型，处理时间序列数据

### 工具组件

- `MarketStateClassifier`: 市场状态分类器，识别不同的市场环境
- `AdaptiveWeightModel`: 自适应权重模型，根据市场状态调整权重
- `EarlyStopping`: 早停机制，防止模型过拟合
- `WeightedMSELoss`: 加权均方误差损失函数
- `SimpleMSELoss`: 简单均方误差损失函数

## 使用示例

### 基本信号组合模型

```python
from strategy_optimizer.models import LinearCombinationModel
from strategy_optimizer.extractors.strategy_signal_extractor import StrategySignalExtractor

# 提取策略信号
extractor = StrategySignalExtractor()
signals_df = extractor.extract_signals_from_strategies(strategies, price_data)

# 创建线性组合模型
model = LinearCombinationModel(
    lookback_period=10,  # 使用10天历史数据
    target_type="return"  # 预测收益率
)

# 定义目标变量（明天的收益率）
target = price_data['close'].pct_change().shift(-1)

# 训练模型
model.train(signals_df, target)

# 预测
predictions = model.predict(signals_df)
```

### 神经网络组合模型

```python
from strategy_optimizer.models import NeuralCombinationModel

# 创建神经网络组合模型
model = NeuralCombinationModel(
    hidden_layers=[64, 32],
    dropout=0.2,
    lookback_period=20,
    target_type="direction"  # 预测价格方向
)

# 训练模型
model.train(
    signals_df, 
    target,
    batch_size=32,
    epochs=100,
    validation_split=0.2
)

# 生成预测
predictions = model.predict(signals_df)
```

### 高级信号组合器

```python
from strategy_optimizer.models import SignalCombiner, CombinerConfig

# 创建配置
config = CombinerConfig(
    hidden_dim=64,
    n_layers=2,
    dropout=0.2,
    sequence_length=60,
    batch_size=32,
    epochs=50,
    learning_rate=0.001,
    use_market_state=True,
    time_varying_weights=True
)

# 创建信号组合器
combiner = SignalCombiner(
    n_strategies=len(strategies),
    input_dim=signals_df.shape[1],
    config=config
)

# 准备市场特征数据（用于识别市场状态）
market_features = prepare_market_features(price_data)

# 训练模型
combiner.train(
    signals=signals_df,
    market_features=market_features,
    target=target,
    validation_split=0.2
)

# 生成组合信号
combined_signals, weights, market_states = combiner.predict(
    signals=new_signals,
    market_features=new_market_features
)
```

## 模型选择指南

根据不同的应用场景选择合适的模型：

1. **简单场景，数据量小**:
   - `LinearCombinationModel`: 简单、高效、易于解释
   - `XGBoostModel` (深度限制): 稳定性好，不容易过拟合

2. **复杂场景，数据量大**:
   - `NeuralCombinationModel`: 可以捕捉复杂的非线性关系
   - `TransformerModel`: 擅长处理长序列和依赖关系

3. **需要考虑市场状态**:
   - `SignalCombiner`: 可以根据市场状态动态调整
   - `ConditionalXGBoost`: 在不同市场条件下使用不同的预测模型

4. **实时交易环境**:
   - `LinearCombinationModel`: 预测速度最快
   - 简化版的 `XGBoostModel`: 平衡速度和性能

## 模型评估

### 性能指标

评估模型的常用指标：

- **分类任务**: 准确率、精确率、召回率、F1分数、AUC
- **回归任务**: MSE、MAE、R²
- **交易性能**: 夏普比率、最大回撤、胜率、收益率

### 交叉验证

推荐使用时间序列交叉验证评估模型：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(signals_df):
    X_train, X_test = signals_df.iloc[train_idx], signals_df.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    
    # 计算性能指标...
```

## 模型保存与加载

大多数模型支持保存和加载功能：

```python
# 保存模型
model.save("path/to/model.pkl")

# 加载模型
model.load("path/to/model.pkl")
```

## 注意事项

1. 金融时间序列的预测极具挑战性，模型性能可能有限
2. 避免数据泄露，确保训练时不使用未来信息
3. 定期重新训练模型以适应市场变化
4. 考虑模型的可解释性，特别是在实际交易中使用时
5. 保持简单性，复杂模型不一定比简单模型表现更好 