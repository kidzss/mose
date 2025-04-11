# 信号提取器 (Signal Extractors)

## 概述

信号提取器子模块负责从各种交易策略中提取、标准化和处理信号，使其适合后续的优化和机器学习处理。它是策略优化器模块的基础组件，用于将不同类型的交易策略信号转换为统一格式的特征数据。

## 主要组件

### StrategySignalExtractor

`StrategySignalExtractor` 类是此子模块的核心，它提供了从交易策略中提取信号的完整功能集。

#### 主要功能

1. **信号提取**：从单个或多个策略中提取原始信号
2. **信号标准化**：对不同类型的信号进行适当的归一化处理
   - RSI等振荡器类指标归一化到0-1区间
   - MACD等指标相对于价格进行标准化
   - 其他指标根据需要进行适当处理
3. **元数据管理**：记录和管理信号的元数据，包括信号来源、类别、描述、重要性等
4. **信号重要性分析**：基于元数据对信号进行重要性排序

#### 使用示例

```python
from strategy_optimizer.extractors.strategy_signal_extractor import StrategySignalExtractor
from strategy.strategies import RSIStrategy, MACDStrategy, BollingerBandsStrategy

# 创建多个策略实例
strategies = [
    RSIStrategy(period=14), 
    MACDStrategy(fast=12, slow=26, signal=9),
    BollingerBandsStrategy(window=20, num_std=2)
]

# 创建信号提取器
extractor = StrategySignalExtractor()

# 从单个策略提取信号
rsi_signals = extractor.extract_signals_from_strategy(strategies[0], price_data)

# 从多个策略提取信号
all_signals = extractor.extract_signals_from_strategies(strategies, price_data)

# 查看信号元数据
signal_metadata = extractor.get_metadata()

# 获取特定信号的元数据
rsi_metadata = extractor.get_metadata("RSIStrategy_rsi")

# 按重要性排序信号
importance_ranking = extractor.rank_signals_by_importance()
```

## 信号标准化流程

信号提取过程遵循以下标准化流程：

1. **调用策略的生成信号方法**：调用策略实例的 `generate_signals()` 方法获取原始信号
2. **提取核心信号组件**：通过策略的 `extract_signal_components()` 方法获取信号的各个组成部分
3. **标准化处理**：
   - 针对 RSI 类指标：归一化到 0-1 区间
   - 针对 MACD 类指标：相对于价格均值进行标准化
   - 其他指标：保持原始值或根据具体类型进行特殊处理
4. **添加信号元数据**：记录信号的来源、类别、重要性等信息
5. **返回标准化信号**：返回包含所有标准化信号的 DataFrame

## 扩展信号提取器

如需支持新类型的信号标准化，可按以下步骤扩展 `StrategySignalExtractor` 类：

1. 在 `extract_signals_from_strategy()` 方法中添加针对新信号类型的标准化处理逻辑
2. 更新相应的元数据处理逻辑
3. 为新类型的信号定义适当的重要性级别

示例：添加对 ATR (平均真实范围) 指标的支持

```python
# 在 extract_signals_from_strategy 方法中添加：
elif comp_name in ["atr"]:
    # ATR需要相对于价格进行标准化
    normalized_signals[f"{prefix}{comp_name}"] = comp_data / data["close"].mean()
    
    # 更新元数据
    self.signal_metadata[f"{prefix}{comp_name}"] = {
        "source": strategy.name,
        "category": "volatility",
        "description": "平均真实范围指标，衡量市场波动性",
        "importance": "medium",
        "params": getattr(strategy, "parameters", {})
    }
```

## 注意事项

1. 提取信号前确保策略实例正确实现了所需的接口方法
2. 对于大型数据集，信号提取可能需要较长时间，可考虑实现缓存机制
3. 注意处理缺失值，确保提取的信号数据不包含 NaN 或无穷大值 