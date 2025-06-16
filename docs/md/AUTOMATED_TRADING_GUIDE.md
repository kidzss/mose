# 自动化交易系统使用指南

## 📋 系统概述

本系统实现了"边用边优化"策略，整合了投资组合分析、股票筛选和自动化调度功能。

### 🎯 核心功能

1. **每日投资组合分析** - 实时监控持仓盈亏和技术信号
2. **股票筛选系统** - 从573只股票中筛选投资机会  
3. **自动化调度** - 定时执行分析任务
4. **数据健康检查** - 确保数据完整性

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 手动运行

#### 1. 每日投资组合分析
```bash
python daily_trading_assistant.py
```
功能：
- 分析当前持仓6只股票的盈亏情况
- 提供技术信号分析（TDI策略）
- 生成投资建议（止盈/止损）
- 自动保存每日记录到 `daily_trading_log.json`

#### 2. 股票筛选
```bash
python enhanced_stock_screener.py
```
功能：
- 从数据库573只股票中筛选
- 综合评分：技术面50% + 基本面30% + 动量20%
- 筛选标准：评分>=65的优质股票
- 生成投资报告

#### 3. 自动化调度器
```bash
# 启动自动化调度（后台运行）
python automated_trading_scheduler.py

# 手动测试所有功能
python automated_trading_scheduler.py --test

# 仅运行每日分析
python automated_trading_scheduler.py --daily

# 仅运行股票筛选  
python automated_trading_scheduler.py --screen

# 仅运行数据检查
python automated_trading_scheduler.py --check
```

## ⏰ 自动化调度安排

### 每日任务
- **09:00** - 数据健康检查（周一到周五）
- **16:30** - 投资组合分析（美股收盘后）

### 每周任务
- **周日 20:00** - 股票筛选分析

## 📊 当前投资组合

| 股票 | 股数 | 成本价 | 状态 |
|------|------|--------|------|
| AMD  | 48   | $126.21| 持有 |
| GOOGL| 34   | $170.54| 持有 |
| PFE  | 80   | $25.90 | 持有 |
| NVDA | 40   | $138.84| 持有 |
| TSLA | 8    | $254.10| 持有 |
| ADBE | 5    | $346.90| 持有 |

*注：AIG已清仓*

## 📈 系统输出示例

### 每日分析输出
```
📊 投资组合分析
========================================
AMD  : $ 121.73 | 盈亏 -3.55% | ⚪ 观望 | 🟡 待评估
GOOGL: $ 176.09 | 盈亏 +3.25% | ⚪ 观望 | 🟡 待评估
TSLA : $ 308.58 | 盈亏+21.44% | ⚪ 观望 | 🟡 待评估
ADBE : $ 416.26 | 盈亏+20.00% | ⚪ 观望 | 🟡 待评估

总盈亏: $+753 (+3.24%)

💡 投资建议:
   🎯 TSLA: 收益良好，考虑部分止盈
```

### 数据检查输出
```
📊 可用股票数量: 573
🎯 数据质量 - 有效: 573
✅ AMD: $121.73
✅ GOOGL: $176.09
✅ NVDA: $142.63
```

## 📂 文件结构

```
mose/
├── daily_trading_assistant.py      # 每日交易助手
├── enhanced_stock_screener.py      # 增强版股票筛选器
├── automated_trading_scheduler.py  # 自动化调度器
├── daily_trading_log.json         # 每日交易记录
├── automated_trading.log          # 调度器日志
└── data/                           # 数据模块
    └── data_interface.py          # 统一数据接口
```

## 🔧 技术架构

### 数据流
```
MySQL Database ← → DataInterface ← → Trading Scripts
                      ↓
                 Real-time APIs
```

### 策略集成
- **TDI策略**: 技术信号分析
- **多因子评分**: 技术面+基本面+动量
- **风险管理**: 止盈止损建议

## 📋 日志记录

### 每日记录格式
```json
{
  "date": "2025-06-11",
  "portfolio_pnl_pct": 3.24,
  "opportunities": 0,
  "signals": {...}
}
```

### 调度器日志
- 位置: `automated_trading.log`
- 包含: 执行时间、结果状态、关键数据

## 🛠️ 故障排除

### 常见问题

1. **数据获取失败**
   ```bash
   python automated_trading_scheduler.py --check
   ```

2. **策略信号异常**
   - 检查TDI策略参数
   - 验证历史数据完整性

3. **调度器无响应**
   - 检查时区设置
   - 确认schedule模块版本

### 数据源配置
- MySQL数据库连接配置在 `config/data_config.py`
- 支持fallback到Yahoo Finance API

## 🔄 系统优化

### 已完成
- ✅ 统一数据接口
- ✅ 移除CSV依赖
- ✅ 集成TDI策略
- ✅ 自动化调度

### 待优化
- 🔄 基本面数据接入
- 🔄 实时数据流集成
- 🔄 风险管理增强
- 🔄 机器学习信号

## 💡 使用建议

1. **首次使用**: 先运行 `--test` 验证功能
2. **日常使用**: 启动自动化调度器后台运行
3. **手动分析**: 可随时运行单独的分析脚本
4. **风险控制**: 关注每日建议，特别是止盈止损信号

## 📞 技术支持

- 查看日志: `automated_trading.log`
- 数据检查: `python automated_trading_scheduler.py --check`
- 手动测试: `python automated_trading_scheduler.py --test` 