# 工具函数 (Utilities)

## 概述

Utils 子模块提供了一系列辅助工具和函数，用于支持策略优化过程中的各种任务，包括数据处理、评估、归一化、交叉验证和技术指标计算等。这些工具函数使得策略的开发、测试和优化过程更加高效和标准化。

## 主要组件

### 数据处理工具

- `data_generator.py`: 生成模拟数据用于测试和开发
- `normalization.py`: 信号和特征的标准化/归一化工具
- `signal_extractor.py`: 从策略中提取信号的工具函数
- `config_loader.py`: 配置文件加载和解析工具

### 评估工具

- `evaluation.py`: 策略性能评估函数集合
- `enhanced_evaluation.py`: 高级策略评估指标和工具
- `time_series_cv.py`: 时间序列交叉验证工具

### 交易工具

- `position_sizing.py`: 头寸规模计算工具
- `signal_optimizer.py`: 信号优化工具函数
- `technical_indicators.py`: 技术指标计算函数集合

## 详细功能说明

### 数据生成器 (data_generator.py)

生成用于测试和开发的模拟市场数据。

```python
from strategy_optimizer.utils.data_generator import generate_ohlcv_data, generate_synthetic_signals

# 生成模拟OHLCV数据
ohlcv_data = generate_ohlcv_data(
    n_samples=1000,
    start_price=100,
    volatility=0.02,
    drift=0.0001,
    seed=42
)

# 生成合成信号
synthetic_signals = generate_synthetic_signals(
    ohlcv_data,
    n_signals=5,
    signal_quality=[0.6, 0.4, 0.3, 0.2, 0.1],  # 信号质量
    noise_level=0.2,
    seed=42
)
```

### 归一化工具 (normalization.py)

提供各种信号和特征的标准化方法。

```python
from strategy_optimizer.utils.normalization import normalize_signals, standardize_features, min_max_scale

# 归一化信号
normalized_signals = normalize_signals(signals_df, methods={
    'rsi': 'min_max',  # 最小-最大缩放
    'macd': 'standard',  # Z-分数标准化
    'volatility': 'log',  # 对数变换
    'volume': 'rank'  # 排名变换
})

# Z-分数标准化
standardized_features = standardize_features(features_df)

# 最小-最大缩放
scaled_features = min_max_scale(features_df, feature_range=(0, 1))
```

### 评估工具 (evaluation.py, enhanced_evaluation.py)

计算和评估策略性能的各种指标。

```python
from strategy_optimizer.utils.evaluation import calculate_returns, calculate_sharpe, calculate_drawdown
from strategy_optimizer.utils.enhanced_evaluation import calculate_metrics, plot_equity_curve

# 计算策略收益
returns = calculate_returns(positions, price_data)

# 计算夏普比率
sharpe = calculate_sharpe(returns, risk_free_rate=0.02, annualization=252)

# 计算最大回撤
max_drawdown, drawdown_periods = calculate_drawdown(returns)

# 计算所有性能指标
metrics = calculate_metrics(returns, positions, price_data)

# 绘制权益曲线
plot_equity_curve(returns, benchmark_returns=None, title="策略性能")
```

### 时间序列交叉验证 (time_series_cv.py)

适用于金融时间序列的交叉验证工具。

```python
from strategy_optimizer.utils.time_series_cv import TimeSeriesSplit, walk_forward_validation

# 创建时间序列分割器
tscv = TimeSeriesSplit(n_splits=5, test_size=60, gap=10)

# 执行时间序列交叉验证
for train_idx, test_idx in tscv.split(signals_df):
    train_data = signals_df.iloc[train_idx]
    test_data = signals_df.iloc[test_idx]
    
    # 训练和评估...

# 执行前向验证
performance_metrics = walk_forward_validation(
    model,
    signals_df,
    target,
    initial_train_size=500,
    test_window=60,
    step_size=20,
    metric="sharpe_ratio"
)
```

### 技术指标 (technical_indicators.py)

计算常用的技术分析指标。

```python
from strategy_optimizer.utils.technical_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands, 
    calculate_atr, calculate_momentum
)

# 计算相对强弱指数
rsi = calculate_rsi(price_data['close'], period=14)

# 计算MACD
macd, signal, histogram = calculate_macd(
    price_data['close'], 
    fast_period=12, 
    slow_period=26, 
    signal_period=9
)

# 计算布林带
upper, middle, lower = calculate_bollinger_bands(
    price_data['close'], 
    window=20, 
    num_std=2
)

# 计算平均真实范围
atr = calculate_atr(
    price_data['high'], 
    price_data['low'], 
    price_data['close'], 
    period=14
)

# 计算动量
momentum = calculate_momentum(price_data['close'], period=10)
```

### 头寸规模计算 (position_sizing.py)

根据不同的风险管理方法计算交易头寸大小。

```python
from strategy_optimizer.utils.position_sizing import (
    fixed_size, fixed_value, fixed_risk, 
    kelly_criterion, optimal_f
)

# 固定大小头寸
positions = fixed_size(signals, size=1.0)

# 固定金额头寸
positions = fixed_value(signals, price_data, value=10000)

# 固定风险头寸 (风险1%的总资本)
positions = fixed_risk(
    signals, 
    price_data, 
    risk_per_trade=0.01, 
    stop_loss_pct=0.02, 
    account_balance=100000
)

# 凯利准则头寸
positions = kelly_criterion(
    signals,
    price_data,
    win_rate=0.55,
    win_loss_ratio=1.5,
    fraction=0.5  # 半凯利
)

# 最优f值头寸
positions = optimal_f(
    signals,
    historical_returns,
    risk_tolerance=0.1
)
```

## 使用建议

1. **数据准备流程**:
   ```python
   # 1. 加载原始数据
   raw_data = load_data(...)
   
   # 2. 生成技术指标
   tech_indicators = calculate_indicators(raw_data)
   
   # 3. 归一化特征
   normalized_features = normalize_signals(tech_indicators)
   
   # 4. 划分训练集和测试集
   train_data, test_data = time_series_split(normalized_features)
   ```

2. **评估流程**:
   ```python
   # 1. 生成交易信号
   signals = strategy.generate_signals(data)
   
   # 2. 计算头寸大小
   positions = fixed_risk(signals, data, risk_per_trade=0.01)
   
   # 3. 计算性能指标
   metrics = calculate_metrics(positions, data)
   
   # 4. 可视化结果
   plot_equity_curve(metrics['cumulative_returns'])
   ```

## 注意事项

1. 技术指标计算可能会有前N个数据点为NaN，确保在使用前正确处理
2. 归一化方法应根据指标类型合理选择，不同类型的指标可能需要不同的归一化方法
3. 时间序列交叉验证会随着窗口移动，确保不会引入未来数据
4. 头寸大小计算应配合实际风险管理策略使用，并考虑交易成本
5. 评估指标应综合考虑，单一指标可能无法全面反映策略性能 