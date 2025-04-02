# 多策略交易系统

这个项目实现了一个完整的多策略交易系统，包括策略标准化接口、策略评分系统、多策略回测框架和策略工厂。

## 系统架构

系统由以下几个主要组件组成：

1. **策略基类** (`Strategy`): 所有交易策略的基类，定义了标准化的接口。
2. **策略工厂** (`StrategyFactory`): 负责管理和创建策略实例。
3. **策略评分系统** (`StrategyScorer`): 评估策略表现，为每个策略打分。
4. **多策略回测系统** (`MultiStrategyBacktest`): 运行多策略回测，找出最佳策略组合。

## 策略标准化接口

所有策略都继承自 `Strategy` 基类，实现以下接口：

- `generate_signals`: 生成交易信号
- `calculate_indicators`: 计算技术指标
- `get_market_regime`: 判断市场环境
- `get_position_size`: 计算仓位大小
- `get_stop_loss`: 计算止损价格
- `get_take_profit`: 计算止盈价格
- `should_adjust_stop_loss`: 实现追踪止损
- `optimize_parameters`: 优化策略参数

## 策略评分系统

策略评分系统从三个维度评估策略：

1. **盈利能力** (2分): 评估策略的盈利能力
   - 收益率
   - 胜率
   - 盈亏比
   - 夏普比率

2. **适应性** (2分): 评估策略在不同市场环境下的表现
   - 市场适应性
   - 恢复能力
   - 交易时机把握

3. **稳健性** (1分): 评估策略的风险控制能力
   - 风险控制
   - 波动控制
   - 持仓周期
   - 下行风险

评分系统会根据市场环境动态调整权重，例如在高波动市场中增加稳健性的权重。

## 多策略回测系统

多策略回测系统可以同时回测多个策略，并为每个股票找出最佳策略组合。主要功能包括：

- 并行回测多个策略
- 为每个股票计算策略得分
- 生成详细的回测报告
- 为每个股票找出最佳策略
- 计算策略分配权重

## 策略工厂

策略工厂负责管理和创建策略实例，主要功能包括：

- 自动发现和注册策略
- 创建策略实例
- 提供策略信息

## 已实现的策略

目前已实现的策略包括：

1. **黄金三角策略** (`GoldTriangleStrategy`): 基于三均线交叉的趋势跟踪策略
2. **动量策略** (`MomentumStrategy`): 基于价格突破和技术指标的动量策略
3. **CPGW策略** (`CPGWStrategy`): 基于长庄线、游资线和主力线的交叉关系生成买卖信号
4. **Market Forecast策略** (`MarketForecastStrategy`): 基于三条曲线（Momentum、NearTerm、Intermediate）的反转信号生成买卖信号
5. **牛牛策略** (`NiuniuStrategy`): 基于牛线（主力成本线）和交易线的交叉关系生成买卖信号
6. **TDI策略** (`TDIStrategy`): 基于RSI和其移动平均线的交叉关系生成买卖信号

### 策略详情

#### CPGW策略 (长庄股王)

该策略基于长庄线、游资线和主力线的交叉关系生成买卖信号。

**买入条件**:
1. 长庄线 < 12 且 主力线 < 8 且 (游资线 < 7.2 或 前一天主力线 < 5) 且 (主力线 > 前一天主力线 或 游资线 > 前一天游资线)
2. 或 长庄线 < 8 且 主力线 < 7 且 游资线 < 15 且 游资线 > 前一天游资线
3. 或 长庄线 < 10 且 主力线 < 7 且 游资线 < 1

**卖出条件**:
主力线 < 前一天主力线 且 前一天主力线 > 80 且 (前一天游资线 > 95 或 前两天游资线 > 95) 且 长庄线 > 60 且
游资线 < 83.5 且 游资线 < 主力线 且 游资线 < 主力线 + 4

#### Market Forecast策略

该策略基于三条曲线的反转信号生成买卖信号：
1. Momentum (短期)
2. NearTerm (中期)
3. Intermediate (长期)

**买入条件**: 三条曲线几乎同时在底部区域反转上升
**卖出条件**: 三条曲线几乎同时在顶部区域反转下降

#### 牛牛策略 (Niuniu)

该策略基于牛线（主力成本线）和交易线的交叉关系生成买卖信号。

**买入条件**: 牛线上穿交易线
**卖出条件**: 牛线下穿交易线

#### TDI策略 (Traders Dynamic Index)

该策略基于RSI和其移动平均线的交叉关系生成买卖信号。

**买入条件**: 短期MA下穿长期MA，且前一天短期MA上穿长期MA
**卖出条件**: 短期MA上穿长期MA，且前一天短期MA下穿长期MA

## 使用方法

### 运行多策略回测

```bash
python backtest/run_multi_strategy_backtest.py --start_date 2023-01-01 --end_date 2023-12-31 --strategies GoldTriangleStrategy,MomentumStrategy,CPGWStrategy,MarketForecastStrategy,NiuniuStrategy,TDIStrategy --market_regime normal --parallel
```

### 列出所有可用策略

```bash
python backtest/run_multi_strategy_backtest.py --list_strategies
```

### 查看策略详细信息

```bash
python backtest/run_multi_strategy_backtest.py --strategy_info CPGWStrategy
```

## 后续计划

1. **第二阶段**: 增强评分系统
   - 完善 `StrategyScorer` 类
   - 实现动态权重调整

2. **第三阶段**: 策略组合与轮动
   - 实现策略轮动机制
   - 开发策略组合优化算法

3. **第四阶段**: 风险管理与市场环境分类
   - 实现波动率调整仓位
   - 开发市场环境分类器

4. **第五阶段**: 实时监控与报告
   - 实现实时性能监控
   - 增强回测报告功能 