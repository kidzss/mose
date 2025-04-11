# 数据处理器 (Data Processors)

## 概述

数据处理器子模块提供了一套工具，用于处理和准备交易数据，生成特征，提取信号，以及分析市场状态。这些工具是策略优化过程中的关键环节，确保模型训练和评估使用的数据质量高且格式一致。

## 主要组件

### 核心处理器

- `data_processor.py`: 主数据处理器，提供数据加载、清洗和特征生成功能
- `feature_importance.py`: 特征重要性分析工具，评估不同特征的预测价值
- `market_state_analyzer.py`: 市场状态分析工具，识别不同市场环境的特征

### 辅助工具

- `signal_extractor.py`: 从原始策略中提取交易信号的工具
- `training_data_generator.py`: 为模型训练生成数据集的工具
- `data_enhancer.py`: 通过特征工程增强原始数据的工具

## 功能详解

### 数据处理器 (data_processor.py)

主数据处理器是整个子模块的核心，提供了完整的数据处理流程。

```python
from strategy_optimizer.data_processors.data_processor import DataProcessor

# 创建数据处理器
processor = DataProcessor(
    db_engine=engine,  # 数据库引擎
    cache_dir="cache"  # 缓存目录
)

# 加载原始数据
raw_data = processor.load_data(
    symbols=["AAPL", "MSFT", "GOOG"],
    start_date="2018-01-01",
    end_date="2023-01-01",
    freq="1d"
)

# 清洗数据
clean_data = processor.clean_data(raw_data)

# 生成特征
features = processor.generate_features(clean_data)

# 准备训练数据
X_train, X_test, y_train, y_test = processor.prepare_training_data(
    features,
    target_col="return_1d",
    test_size=0.2,
    sequence_length=60
)
```

### 特征重要性分析 (feature_importance.py)

评估不同特征对预测的贡献，帮助识别最有价值的信号。

```python
from strategy_optimizer.data_processors.feature_importance import FeatureImportanceAnalyzer

# 创建分析器
analyzer = FeatureImportanceAnalyzer()

# 计算特征重要性
importance = analyzer.calculate_importance(
    features=features_df,
    target=target_series,
    method="permutation"  # 可选: permutation, shap, model_specific
)

# 获取Top-N重要特征
top_features = analyzer.get_top_features(importance, n=10)

# 可视化特征重要性
analyzer.plot_feature_importance(importance, output_file="feature_importance.png")
```

### 市场状态分析器 (market_state_analyzer.py)

分析和识别不同的市场状态，为策略优化提供市场环境上下文。

```python
from strategy_optimizer.data_processors.market_state_analyzer import MarketStateAnalyzer

# 创建市场状态分析器
analyzer = MarketStateAnalyzer(n_states=3)  # 识别3种市场状态

# 拟合市场数据
analyzer.fit(market_data)

# 预测市场状态
market_states = analyzer.predict(new_data)

# 获取市场状态特征
state_features = analyzer.get_state_features()

# 分析不同市场状态下的策略表现
performance_by_state = analyzer.analyze_performance_by_state(
    returns=strategy_returns,
    market_states=market_states
)
```

### 信号提取器 (signal_extractor.py)

从各种交易策略中提取标准化的信号。

```python
from strategy_optimizer.data_processors.signal_extractor import SignalExtractor

# 创建信号提取器
extractor = SignalExtractor()

# 从策略中提取信号
signals = extractor.extract_signals(
    strategies=strategy_list,
    price_data=ohlcv_data,
    normalize=True
)

# 过滤低质量信号
filtered_signals = extractor.filter_signals(
    signals,
    correlation_threshold=0.7,
    variance_threshold=0.01
)
```

### 训练数据生成器 (training_data_generator.py)

为模型训练准备结构化数据集。

```python
from strategy_optimizer.data_processors.training_data_generator import TrainingDataGenerator

# 创建数据生成器
generator = TrainingDataGenerator(
    sequence_length=60,  # 使用60天历史数据作为序列输入
    forecast_horizon=5,  # 预测未来5天
    batch_size=32
)

# 生成训练数据
train_dataset = generator.generate_dataset(
    signals=signal_data,
    market_features=market_data,
    target=target_data,
    is_training=True
)

# 创建数据加载器
train_loader = generator.create_data_loader(train_dataset)
```

## 数据流程示例

典型的数据处理流程如下：

1. **数据加载与清洗**:
   ```python
   processor = DataProcessor(engine)
   raw_data = processor.load_data(symbols, start_date, end_date)
   clean_data = processor.clean_data(raw_data)
   ```

2. **特征生成**:
   ```python
   features = processor.generate_features(clean_data)
   enhanced_features = data_enhancer.enhance_features(features)
   ```

3. **信号提取**:
   ```python
   extractor = SignalExtractor()
   strategy_signals = extractor.extract_signals(strategies, clean_data)
   ```

4. **市场状态分析**:
   ```python
   analyzer = MarketStateAnalyzer()
   market_states = analyzer.analyze(clean_data)
   ```

5. **训练数据准备**:
   ```python
   generator = TrainingDataGenerator()
   train_data = generator.generate_dataset(
       signals=strategy_signals,
       features=enhanced_features,
       market_states=market_states,
       target=target_data
   )
   ```

## 注意事项

1. 数据处理是模型性能的关键，确保处理前了解数据的特性和潜在问题
2. 特征生成应考虑避免前视偏差(look-ahead bias)，确保特征只使用当前时点可用的信息
3. 市场状态分析需要足够长的历史数据才能识别不同的市场环境
4. 信号提取时注意标准化方法的选择，不同类型的信号可能需要不同的标准化方法
5. 在准备训练数据时，确保正确处理时间序列的分割，避免数据泄露 