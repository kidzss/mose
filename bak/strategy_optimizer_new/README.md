# 策略优化模型

策略优化模型是一个基于深度学习的股票交易策略权重优化系统，用于自动分配不同交易策略的权重以优化整体投资组合表现。

## 项目结构

```
strategy_optimizer/
├── configs/             # 配置文件
├── data_processors/     # 数据处理相关模块
├── models/              # 模型定义
├── outputs/             # 训练输出和模型存储
├── utils/               # 工具函数
├── train.py             # 训练脚本
├── evaluate_model.py    # 模型评估脚本
├── visualize_weights.py # 权重可视化脚本
├── backtest_weights.py  # 权重回测脚本
├── run_strategy_evaluation.py # 一体化评估与回测脚本
└── README.md            # 本文档
```

## 功能

1. **训练模型**：训练一个能够基于股票价格和技术指标数据预测最优策略权重的深度学习模型
2. **评估模型**：使用训练好的模型对新的股票数据进行预测，生成推荐的策略权重
3. **可视化结果**：以图表形式展示模型预测的策略权重分布
4. **回测权重**：对生成的策略权重进行回测，评估其实际表现

## 使用方法

### 1. 训练模型

训练新模型：

```bash
PYTHONPATH=/path/to/project python strategy_optimizer/train.py --config strategy_optimizer/configs/optimizer_config.json --output strategy_optimizer/outputs
```

参数说明：
- `--config`：配置文件路径，包含模型参数、训练参数、数据参数等
- `--output`：输出目录，用于保存训练好的模型和日志

### 2. 评估模型

使用训练好的模型进行评估：

```bash
python strategy_optimizer/evaluate_model.py --model_path strategy_optimizer/outputs/[训练ID]/[模型文件名] --config strategy_optimizer/configs/optimizer_config.json --output strategy_weights.csv
```

参数说明：
- `--model_path`：训练好的模型路径
- `--config`：配置文件路径，与训练时使用的配置相同
- `--output`：输出CSV文件路径，包含预测的策略权重

### 3. 可视化结果

可视化预测的策略权重：

```bash
python strategy_optimizer/visualize_weights.py --input strategy_weights.csv --output strategy_weights_viz.png
```

参数说明：
- `--input`：评估脚本生成的CSV文件路径
- `--output`：输出图表文件路径

### 4. 回测权重

对策略权重进行回测：

```bash
python strategy_optimizer/backtest_weights.py --weights strategy_weights.csv --symbol AAPL --days 365 --output aapl_backtest.png
```

参数说明：
- `--weights`：包含策略权重的CSV文件路径
- `--symbol`：要回测的股票代码
- `--days`：回测天数
- `--output`：输出图表文件路径

### 5. 一体化评估与回测

使用一体化脚本完成从评估到回测的全流程：

```bash
python strategy_optimizer/run_strategy_evaluation.py --model_path strategy_optimizer/outputs/[训练ID]/[模型文件名] --config strategy_optimizer/configs/optimizer_config.json --symbols AAPL MSFT TSLA --days 365
```

参数说明：
- `--model_path`：训练好的模型路径
- `--config`：配置文件路径
- `--output_dir`：输出目录的基础名称（会自动添加时间戳）
- `--symbols`：要回测的股票代码列表
- `--days`：回测天数
- `--diversity`：策略多样性因子(0-1)，控制权重分布的多样性

## 中文字体支持

本项目所有图表均支持中文字体显示。系统会自动尝试使用以下字体（按优先级）：
- Arial Unicode MS
- SimHei（黑体）
- Microsoft YaHei（微软雅黑）
- WenQuanYi Micro Hei（文泉驿微米黑）
- DejaVu Sans

如果图表中出现中文显示为方框，请确保系统中安装了上述至少一种中文字体。

## 模型架构

该模型使用Transformer架构处理时间序列数据，包含以下组件：

1. **编码器**：处理股票时间序列数据，捕捉价格和指标的时间特征
2. **LSTM层**：进一步处理序列数据，捕获长期依赖关系
3. **解码器**：将特征映射到策略权重，输出每个策略的权重预测

## 输入特征

模型使用以下股票数据作为输入：
- 价格数据：开盘价、最高价、最低价、收盘价、交易量
- 技术指标：移动平均线(SMA)、相对强弱指标(RSI)、MACD指标、布林带等

## 输出结果

模型输出各个交易策略的权重分配，包括：
- GoldTriangle策略
- Momentum策略
- TDI策略
- MarketForecast策略
- CPGW策略
- Niuniu策略

## 性能评估

模型使用均方误差(MSE)作为损失函数，通过验证集上的损失来评估模型性能。在训练过程中使用早停机制防止过拟合。

回测性能指标包括：
- 总回报率（Total Return）
- 年化回报率（Annual Return）
- 波动率（Volatility）
- 夏普比率（Sharpe Ratio）
- 最大回撤（Max Drawdown）
- 胜率（Win Rate）

## 依赖库

- PyTorch：深度学习框架
- Pandas / NumPy：数据处理
- Matplotlib / Seaborn：数据可视化
- TA-Lib：技术分析指标计算
- yfinance：获取股票数据

## 维护与更新

要改进模型性能，可尝试以下方法：

1. 添加更多技术指标作为特征
2. 调整模型超参数（学习率、批量大小等）
3. 扩展训练数据集（更多股票、更长时间范围）
4. 尝试不同的网络架构或优化方法

# 策略优化器 (Strategy Optimizer)

## 概述

策略优化器模块是一个用于优化交易策略参数和组合多策略信号的工具集。它通过机器学习和统计方法，从各种交易策略中提取信号并找到最佳组合方式，以提高交易决策的准确性和稳定性。

## 主要功能

1. **信号提取**：从各种交易策略中提取和标准化信号
2. **特征重要性分析**：评估不同信号对交易决策的影响程度
3. **参数优化**：寻找策略的最佳参数设置
4. **多策略融合**：将多个策略的信号组合成更强大的综合信号
5. **回测评估**：评估优化后策略的性能和稳定性
6. **市场状态分析**：识别不同的市场环境，为策略优化提供上下文

## 目录结构

```
strategy_optimizer/
├── __init__.py              # 模块初始化文件
├── extractors/              # 信号提取相关组件
│   ├── __init__.py
│   └── strategy_signal_extractor.py  # 策略信号提取器
├── optimizers/              # 优化器相关组件
│   ├── __init__.py
│   ├── parameter_optimizer.py        # 参数优化器
│   └── signal_optimizer.py           # 信号优化器
├── models/                  # 机器学习模型组件
│   ├── __init__.py
│   ├── feature_importance.py         # 特征重要性分析
│   └── signal_ensemble.py            # 信号集成模型
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── data_preparation.py           # 数据准备工具
│   └── visualization.py              # 可视化工具
├── data_processors/         # 数据处理组件
│   ├── __init__.py
│   ├── data_processor.py             # 主数据处理器
│   └── feature_importance.py         # 特征重要性分析
├── evaluation/              # 评估组件
│   ├── __init__.py
│   └── report_generator.py           # 报告生成器
├── market_analysis/         # 市场分析组件
│   ├── __init__.py
│   └── market_state.py               # 市场状态分析器
└── train.py                 # 模型训练入口脚本
```

## 使用示例

### 策略信号提取

```python
from strategy_optimizer.extractors.strategy_signal_extractor import StrategySignalExtractor
from strategy.strategies import RSIStrategy, MACDStrategy

# 创建策略实例
strategies = [RSIStrategy(period=14), MACDStrategy(fast=12, slow=26, signal=9)]

# 创建信号提取器
extractor = StrategySignalExtractor()

# 提取信号
signals_df = extractor.extract_signals_from_strategies(strategies, price_data)

# 获取信号重要性排名
importance_scores = extractor.rank_signals_by_importance()
```

### 参数优化

```python
from strategy_optimizer.optimizers.parameter_optimizer import ParameterOptimizer
from strategy.strategies import RSIStrategy

# 创建参数优化器
optimizer = ParameterOptimizer()

# 定义参数搜索空间
param_space = {
    "period": [7, 14, 21, 28],
    "overbought": [70, 75, 80],
    "oversold": [20, 25, 30]
}

# 执行优化
best_params = optimizer.optimize(
    strategy_class=RSIStrategy,
    param_space=param_space,
    data=price_data,
    target_metric="sharpe_ratio",
    iterations=100
)

# 使用最佳参数创建策略
optimized_strategy = RSIStrategy(**best_params)
```

### 多策略信号融合

```python
from strategy_optimizer.models.signal_ensemble import SignalEnsemble
from strategy_optimizer.extractors.strategy_signal_extractor import StrategySignalExtractor

# 提取信号
extractor = StrategySignalExtractor()
signals_df = extractor.extract_signals_from_strategies(strategies, price_data)

# 创建信号集成模型
ensemble = SignalEnsemble(model_type="random_forest")

# 训练模型
X = signals_df.iloc[:-1]  # 特征
y = price_data['close'].pct_change().shift(-1).iloc[:-1] > 0  # 明天价格上涨为目标
ensemble.train(X, y)

# 生成集成信号
ensemble_signals = ensemble.predict(signals_df)
```

### 市场状态分析

```python
from strategy_optimizer.market_analysis.market_state import MarketState
from sqlalchemy import create_engine

# 创建数据库连接
engine = create_engine('mysql://user:password@localhost/stockdb')

# 初始化市场状态分析器
market_analyzer = MarketState(engine)

# 获取市场数据
market_data = market_analyzer.get_market_data(
    symbols=['SPY'],
    start_date='2020-01-01',
    end_date='2023-01-01'
)

# 识别市场状态
market_states = market_analyzer.identify_market_state(market_data['SPY'])

# 分析不同市场状态下的策略表现
performance_by_state = market_analyzer.analyze_strategy_by_state(
    strategy_returns=strategy_returns,
    market_states=market_states
)
```

## 注意事项

1. 在使用前，确保已安装所需的依赖包 (`scikit-learn`, `pandas`, `numpy` 等)
2. 信号提取和优化可能需要大量计算资源，对于大型数据集可能耗时较长
3. 优化后的策略参数不一定能在未来数据上表现良好，始终保持谨慎并定期重新优化
4. 对于生产环境，建议添加适当的异常处理和日志记录机制
5. 不同市场状态下的策略性能可能有显著差异，建议使用市场状态分析辅助决策 