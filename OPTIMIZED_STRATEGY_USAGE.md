# 优化策略系统使用指南

## 🎯 概述

经过优化后，策略系统已精简为**3个核心策略**，并提供了**增强的持股分析**和**股票筛选**功能。

## 🏗️ 核心架构

### 1. 核心策略（strategy/）
- **strategy_base.py** - 策略基类
- **niuniu_strategy_v3.py** - 牛牛策略V3（主策略，权重50%）
- **tdi_strategy.py** - TDI策略（短期策略，权重30%）
- **cpgw_strategy.py** - CPGW策略（补充策略，权重20%）
- **combined_strategy.py** - 组合策略（集成3个核心策略）
- **strategy_factory.py** - 策略工厂

### 2. 增强应用（monitor/）
- **enhanced_portfolio_advisor.py** - 增强持股分析系统
- **enhanced_stock_screener.py** - 增强股票筛选系统

## 📚 使用方法

### 1. 基础策略使用

```python
from strategy.strategy_factory import StrategyFactory

# 创建策略工厂
factory = StrategyFactory()

# 创建单个策略
niuniu_strategy = factory.create_strategy('NiuniuV3')
tdi_strategy = factory.create_strategy('TDI')
cpgw_strategy = factory.create_strategy('CPGW')

# 创建组合策略（推荐）
combined_strategy = factory.create_combined_strategy()

# 获取策略信息
print(combined_strategy.get_strategy_summary())
```

### 2. 组合策略分析

```python
from strategy.combined_strategy import CombinedStrategy
import yfinance as yf

# 创建组合策略
strategy = CombinedStrategy()

# 获取股票数据
ticker = yf.Ticker('AAPL')
data = ticker.history(period='3mo')
data.columns = [col.lower() for col in data.columns]

# 生成交易信号
signals = strategy.generate_signals(data)
current_signal = signals['signal'].iloc[-1]
signal_strength = signals['signal_strength'].iloc[-1]

print(f"当前信号: {current_signal}")
print(f"信号强度: {signal_strength:.3f}")

# 计算仓位和风险管理
current_price = data['close'].iloc[-1]
position_size = strategy.get_position_size(data, int(current_signal))
stop_loss = strategy.get_stop_loss(data, current_price, 1)
take_profit = strategy.get_take_profit(data, current_price, 1)

print(f"建议仓位: {position_size:.1%}")
print(f"止损价格: ${stop_loss:.2f}")
print(f"止盈价格: ${take_profit:.2f}")
```

### 3. 增强持股分析

```python
import asyncio
from monitor.enhanced_portfolio_advisor import EnhancedPortfolioAdvisor

async def analyze_portfolio():
    advisor = EnhancedPortfolioAdvisor()
    
    # 持仓信息
    positions = {
        'AAPL': {'shares': 100, 'avg_price': 150.0, 'weight': 0.5},
        'GOOGL': {'shares': 50, 'avg_price': 2500.0, 'weight': 0.5}
    }
    
    # 观察列表
    watchlist = ['NVDA', 'TSLA', 'MSFT']
    
    # 执行分析
    analysis = await advisor.analyze_portfolio_with_strategies(positions, watchlist)
    
    # 查看结果
    print("=== 持仓分析 ===")
    for symbol, data in analysis['position_analysis'].items():
        if 'error' not in data:
            print(f"{symbol}: {data['recommendation']['action']} "
                  f"({data['unrealized_pnl_percent']:.1%})")
    
    print("\n=== 投资组合建议 ===")
    for rec in analysis['portfolio_recommendations']:
        print(f"• {rec['title']}: {rec['reason']}")

# 运行分析
asyncio.run(analyze_portfolio())
```

### 4. 智能股票筛选

```python
import asyncio
from monitor.enhanced_stock_screener import EnhancedStockScreener

async def screen_stocks():
    # 使用默认股票池或自定义
    screener = EnhancedStockScreener()  # 默认股票池
    # 或者: screener = EnhancedStockScreener(['AAPL', 'GOOGL', 'TSLA'])
    
    # 自定义筛选条件
    custom_criteria = {
        'min_signal_strength': 0.7,
        'min_strategy_score': 0.8,
        'consensus_required': 2
    }
    
    # 执行筛选
    results = await screener.screen_stocks(
        max_results=10,
        filter_criteria=custom_criteria
    )
    
    # 查看结果
    print(f"找到 {len(results)} 只符合条件的股票:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.symbol}: {result.recommendation}")
        print(f"   价格: ${result.current_price:.2f}")
        print(f"   策略评分: {result.strategy_score:.3f}")
        print(f"   建议仓位: {result.position_size:.1%}")
        print(f"   原因: {', '.join(result.reasons[:2])}")
        print()
    
    # 生成报告
    report = screener.get_screening_report(results)
    print(f"筛选报告: 分析了 {report['total_stocks_analyzed']} 只股票")

# 运行筛选
asyncio.run(screen_stocks())
```

## ⚙️ 配置选项

### 组合策略权重调整

```python
from strategy.strategy_factory import StrategyFactory

factory = StrategyFactory()

# 自定义权重
custom_weights = {
    'niuniu': 0.6,  # 增加主策略权重
    'tdi': 0.25,    # 减少短期策略权重
    'cpgw': 0.15    # 减少补充策略权重
}

combined_strategy = factory.create_combined_strategy(custom_weights)
```

### 筛选条件调整

```python
screener = EnhancedStockScreener()

# 更新筛选条件
new_criteria = {
    'min_signal_strength': 0.8,    # 提高信号强度要求
    'min_strategy_score': 0.75,    # 提高策略评分要求
    'max_volatility': 0.3,         # 降低波动率限制
    'min_price': 10.0,             # 提高最低价格
    'consensus_required': 3         # 要求所有策略一致
}

screener.update_screening_criteria(new_criteria)
```

## 🔍 与现有系统集成

### 集成到现有监控系统

```python
# 在现有的 monitor/stock_monitor.py 中
from monitor.enhanced_portfolio_advisor import EnhancedPortfolioAdvisor

class StockMonitor:
    def __init__(self):
        # 添加增强分析器
        self.enhanced_advisor = EnhancedPortfolioAdvisor()
    
    async def enhanced_analysis(self, positions, watchlist):
        # 使用策略驱动的分析
        return await self.enhanced_advisor.analyze_portfolio_with_strategies(
            positions, watchlist
        )
```

### 集成到现有筛选系统

```python
# 在现有的筛选逻辑中
from monitor.enhanced_stock_screener import EnhancedStockScreener

# 替换或补充现有筛选器
screener = EnhancedStockScreener()
strategy_results = await screener.screen_stocks(max_results=20)
```

## 📊 性能优化建议

1. **并发处理**: 使用 `asyncio` 进行并发股票分析
2. **数据缓存**: 缓存股票数据避免重复获取
3. **批量处理**: 一次性分析多只股票
4. **结果筛选**: 使用合适的筛选条件减少无效分析

## 🚨 注意事项

1. **数据依赖**: 确保网络连接正常，能够获取股票数据
2. **API限制**: 注意yfinance等数据源的访问限制
3. **策略适应**: 根据市场环境调整策略权重
4. **风险控制**: 始终遵循止损止盈建议

## 📈 后续扩展

1. **新增策略**: 通过 `strategy_factory.py` 注册新策略
2. **自定义指标**: 在策略中添加新的技术指标
3. **机器学习**: 集成ML模型优化信号质量
4. **实时监控**: 添加实时数据流处理

---

**优化完成！** 🎉 现在你拥有了一个精简、高效、易于维护的策略系统。 