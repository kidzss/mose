# JSON数据功能使用说明

## 概述

本系统已增强智能股票分析日报功能，在生成HTML邮件报告的同时，自动生成结构化的JSON数据文件。这些JSON数据以股票代码为键，包含完整的分析结果，便于AI输入和系统集成。

## 功能特性

### 🎯 核心功能
- **自动生成JSON数据**: 每次运行日报生成器时自动创建JSON文件
- **结构化数据格式**: 以股票代码为键的字典结构，便于直接访问
- **完整分析数据**: 包含技术分析、基本面分析、流动性分析等所有结果
- **AI友好格式**: 提供专门的格式化方法，生成适合AI输入的文本格式
- **数据加载器**: 提供便捷的数据访问和加载工具

### 📊 数据内容
- **基本信息**: 当前价格、涨跌幅、成交量、RSI等
- **市场环境**: 趋势分析、置信度、分析理由
- **策略建议**: 推荐策略、信号质量、信号强度
- **持仓分析**: 成本价格、持股数量、盈亏情况
- **财务分析**: 估值指标、盈利能力、成长性、财务健康
- **流动性分析**: 流动性评分、风险等级、买卖价差
- **增强分析**: 综合评分、股票类型、右侧交易分析
- **投资组合汇总**: 总价值、配置比例、资产分布
- **宏观分析**: 宏观得分、环境建议、关键指标

## 使用方法

### 1. 生成JSON数据

JSON数据会在运行智能日报生成器时自动生成：

```python
from monitor.smart_daily_report import SmartDailyReportGenerator

# 创建生成器
generator = SmartDailyReportGenerator()

# 生成报告（同时生成HTML和JSON）
html_report = generator.generate_report()
```

### 2. 加载和访问数据

使用`StockDataLoader`类加载和访问JSON数据：

```python
from utils.stock_data_loader import StockDataLoader

# 创建数据加载器
loader = StockDataLoader()

# 加载最新数据
data = loader.load_data()

# 获取所有股票列表
symbols = loader.get_all_stocks()

# 获取单个股票数据
stock_data = loader.get_stock_data('AMD')
```

### 3. AI集成使用

#### 方法1: 获取完整AI输入格式
```python
# 获取所有股票的AI输入格式
ai_input = loader.format_for_ai_input()

# 获取特定股票的AI输入格式
ai_input = loader.format_for_ai_input(['AMD', 'NVDA', 'TSLA'])
```

#### 方法2: 获取单个股票摘要
```python
# 获取股票摘要信息
summary = loader.get_stock_summary('AMD')
print(summary)
```

#### 方法3: 获取投资组合和宏观数据
```python
# 获取投资组合汇总
portfolio = loader.get_portfolio_summary()

# 获取宏观分析
macro = loader.get_macro_analysis()

# 获取数据统计
stats = loader.get_data_statistics()
```

## 数据文件结构

### 文件命名
- 格式: `stock_analysis_data_YYYYMMDD_HHMMSS.json`
- 示例: `stock_analysis_data_20250101_143000.json`

### JSON结构
```json
{
  "timestamp": "2025-01-01 14:30:00",
  "data_version": "1.0",
  "stocks": {
    "AMD": {
      "basic_info": {
        "symbol": "AMD",
        "current_price": 142.35,
        "price_change_pct": 2.85,
        "volume": 12345678,
        "rsi": 65.2
      },
      "market_environment": {
        "trend": "uptrend",
        "confidence": 0.85,
        "reasons": ["RSI上升", "价格突破阻力位"]
      },
      "strategy": {
        "recommended_strategy": "hold",
        "market_environment": "bullish",
        "signal_quality": 0.78,
        "signal_strength": "strong"
      },
      "position_analysis": {
        "cost_price": 126.214,
        "shares": 48,
        "weight": 21.93,
        "investment_amount": 4788.89,
        "current_value": 6832.80,
        "pnl_amount": 2043.91,
        "pnl_percent": 42.7,
        "is_profit": true
      },
      "financial_analysis": {
        "total_score": 85.2,
        "overall_rating": "excellent",
        "valuation_metrics": {
          "pe_ratio": 25.3,
          "peg_ratio": 1.2,
          "pb_ratio": 4.5
        }
      },
      "liquidity_analysis": {
        "liquidity_score": 92.5,
        "risk_level": "low",
        "bid_ask_spread_pct": 0.05,
        "market_cap_tier": "large_cap"
      },
      "enhanced_analysis": {
        "total_score": 0.875,
        "overall_rating": "excellent",
        "stock_type_analysis": {
          "type": "growth_stock",
          "risk_level": "moderate",
          "comprehensive_score": 8.5
        }
      }
    }
  },
  "portfolio_summary": {
    "total_value": 27722.97,
    "stock_allocation": 76.18,
    "cash_allocation": 2.20,
    "money_fund_allocation": 21.59,
    "total_stock_investment": 21121.29,
    "money_fund_value": 5983.65
  },
  "macro_analysis": {
    "macro_score": 0.72,
    "recommendation": "宏观环境偏向积极，可适当增加风险资产配置",
    "components": {
      "interest_rate": {
        "curve_score": 0.75,
        "description": "收益率曲线正常，利率环境稳定有利"
      },
      "market_sentiment": {
        "vix_level": 16.8,
        "sentiment_desc": "市场情绪乐观，投资者信心较强"
      }
    }
  }
}
```

## 实际应用示例

### 示例1: 投资组合分析
```python
from utils.stock_data_loader import StockDataLoader

loader = StockDataLoader()
data = loader.load_data()

# 分析投资组合表现
portfolio = loader.get_portfolio_summary()
print(f"投资组合总价值: ${portfolio['total_value']:,.2f}")
print(f"股票配置比例: {portfolio['stock_allocation']:.2f}%")

# 分析每只股票
symbols = loader.get_all_stocks()
for symbol in symbols:
    stock_data = loader.get_stock_data(symbol)
    if stock_data and 'position_analysis' in stock_data:
        pos = stock_data['position_analysis']
        print(f"{symbol}: 盈亏 {pos['pnl_percent']:+.2f}%")
```

### 示例2: AI投资建议
```python
# 为AI提供完整的投资组合数据
ai_input = loader.format_for_ai_input()

# AI提示词示例
prompt = f"""
请基于以下股票分析数据，提供专业的投资建议：

{ai_input}

请提供：
1. 投资组合整体评估
2. 个股操作建议（买入/卖出/持有/加仓/减仓）
3. 风险提示
4. 短期和中期投资策略
"""
```

### 示例3: 风险监控
```python
# 监控流动性风险
symbols = loader.get_all_stocks()
high_risk_stocks = []

for symbol in symbols:
    stock_data = loader.get_stock_data(symbol)
    if stock_data and 'liquidity_analysis' in stock_data:
        liquidity = stock_data['liquidity_analysis']
        if liquidity['risk_level'] == 'high':
            high_risk_stocks.append({
                'symbol': symbol,
                'liquidity_score': liquidity['liquidity_score'],
                'risk_warning': liquidity['risk_warning']
            })

if high_risk_stocks:
    print("⚠️ 高风险股票提醒:")
    for stock in high_risk_stocks:
        print(f"  {stock['symbol']}: 流动性评分 {stock['liquidity_score']:.1f}")
```

### 示例4: 技术分析筛选
```python
# 筛选技术面强势股票
strong_stocks = []

for symbol in symbols:
    stock_data = loader.get_stock_data(symbol)
    if stock_data:
        basic = stock_data['basic_info']
        strategy = stock_data['strategy']
        
        # 筛选条件：RSI > 50 且 信号质量 > 0.7
        if basic['rsi'] > 50 and strategy['signal_quality'] > 0.7:
            strong_stocks.append({
                'symbol': symbol,
                'rsi': basic['rsi'],
                'signal_quality': strategy['signal_quality'],
                'strategy': strategy['recommended_strategy']
            })

print("📈 技术面强势股票:")
for stock in strong_stocks:
    print(f"  {stock['symbol']}: RSI {stock['rsi']:.1f}, 信号质量 {stock['signal_quality']:.2f}")
```

## 测试和验证

### 运行测试脚本
```bash
python test_json_data_feature.py
```

### 运行使用示例
```bash
python examples/json_data_usage_example.py
```

### 测试数据加载器
```bash
python utils/stock_data_loader.py
```

## 优势特点

### 🔄 数据复用
- **一次生成，多次使用**: 生成一次JSON数据，可多次用于不同分析
- **标准化格式**: 统一的数据结构，便于系统集成
- **版本控制**: 包含数据版本信息，便于追踪

### 🤖 AI友好
- **结构化数据**: JSON格式便于AI理解和处理
- **格式化方法**: 提供专门的AI输入格式化功能
- **完整信息**: 包含所有分析维度的数据

### ⚡ 高效访问
- **快速加载**: 缓存机制，避免重复文件读取
- **灵活查询**: 支持单个股票、批量股票、汇总数据查询
- **内存优化**: 按需加载，减少内存占用

### 🔧 易于集成
- **简单API**: 清晰的接口设计，易于使用
- **错误处理**: 完善的异常处理机制
- **日志记录**: 详细的操作日志，便于调试

## 注意事项

### 数据更新
- JSON数据文件会在每次运行日报生成器时更新
- 建议在美股收盘后运行，确保数据准确性
- 旧数据文件会保留，可通过文件名区分

### 文件管理
- JSON文件会保存在项目根目录
- 建议定期清理旧文件，避免占用过多磁盘空间
- 文件名包含时间戳，便于识别最新数据

### 数据质量
- 数据质量取决于市场数据源的可用性
- 建议检查数据质量评分，确保分析可靠性
- 如遇数据问题，系统会记录错误日志

## 故障排除

### 常见问题

1. **找不到JSON文件**
   - 检查是否已运行日报生成器
   - 确认文件路径和权限
   - 查看错误日志

2. **数据加载失败**
   - 检查JSON文件格式是否正确
   - 确认文件编码为UTF-8
   - 验证文件完整性

3. **股票数据缺失**
   - 检查股票代码是否正确
   - 确认该股票在分析列表中
   - 查看数据生成日志

### 调试方法

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查数据文件**
   ```python
   import json
   with open('stock_analysis_data_*.json', 'r') as f:
       data = json.load(f)
       print(json.dumps(data, indent=2)[:1000])
   ```

3. **验证数据结构**
   ```python
   loader = StockDataLoader()
   stats = loader.get_data_statistics()
   print(stats)
   ```

## 更新日志

### v1.0 (2025-01-01)
- ✅ 实现JSON数据自动生成功能
- ✅ 创建StockDataLoader数据加载器
- ✅ 提供AI输入格式化方法
- ✅ 支持投资组合和宏观数据访问
- ✅ 完善错误处理和日志记录
- ✅ 创建测试脚本和使用示例
- ✅ 编写详细使用文档

---

**💡 提示**: 本功能已完全集成到现有系统中，无需额外配置。每次运行智能日报生成器时，都会自动生成最新的JSON数据文件，为AI分析和系统集成提供强大的数据支持。 