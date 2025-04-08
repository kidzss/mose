# 评估 (Evaluation)

## 概述

评估子模块提供了用于评估交易策略和策略组合模型性能的工具。它能够生成全面的性能报告，包括关键绩效指标、回测结果和可视化图表，帮助分析策略的优势和劣势。

## 主要组件

### 报告生成器 (report_generator.py)

`report_generator.py` 是评估子模块的核心组件，提供了创建各种评估报告的功能。

#### 主要功能

1. **性能指标计算**：计算各种交易性能指标，如夏普比率、索提诺比率、最大回撤等
2. **策略比较报告**：对比多个策略的性能指标和回测结果
3. **市场状态分析**：分析策略在不同市场状态下的表现
4. **特征重要性报告**：评估不同信号和特征对策略性能的贡献
5. **可视化**：生成各种图表，如权益曲线、回撤曲线、信号权重等

## 使用示例

### 生成综合评估报告

```python
from strategy_optimizer.evaluation.report_generator import (
    generate_performance_report,
    generate_strategy_comparison_report,
    generate_feature_importance_report
)

# 生成单个策略的性能报告
performance_report = generate_performance_report(
    strategy_returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    positions=positions,
    output_dir="reports",
    strategy_name="OptimizedStrategy"
)

# 生成多策略比较报告
comparison_report = generate_strategy_comparison_report(
    strategy_returns={
        "Strategy1": strategy1_returns,
        "Strategy2": strategy2_returns,
        "Combined": combined_returns
    },
    benchmark_returns=benchmark_returns,
    market_state=market_state_data,
    output_dir="reports/comparison"
)

# 生成特征重要性报告
importance_report = generate_feature_importance_report(
    model=signal_combiner_model,
    feature_names=feature_names,
    output_dir="reports/features"
)
```

### 生成自定义指标报告

```python
from strategy_optimizer.evaluation.report_generator import ReportGenerator

# 创建报告生成器实例
report_gen = ReportGenerator(output_dir="reports/custom")

# 添加自定义指标
report_gen.add_custom_metric("profit_factor", profit_factor_values)
report_gen.add_custom_metric("recovery_factor", recovery_factor_values)

# 生成报告
report = report_gen.generate_report(
    returns=strategy_returns,
    benchmark=benchmark_returns,
    include_plots=True,
    include_monthly_breakdown=True
)
```

## 注意事项

1. 评估前确保数据已经正确对齐，特别是在比较多个策略时
2. 为了获得统计显著的结果，确保测试期足够长且包含不同的市场环境
3. 在计算性能指标时，考虑滑点和交易成本的影响
4. 生成报告时包含适当的基准，以便进行相对性能比较 