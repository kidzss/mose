# 策略优化计划

## 📊 当前系统分析

### 主要使用的策略
1. **NiuniuV3Strategy** - 核心中期策略，权重最高
2. **TDIStrategy** - 短期技术分析策略  
3. **CombinedStrategy** - 组合策略框架

### 当前问题
1. `strategy/` 目录有31个文件，包含大量未使用或重复的策略
2. `strategies/` 目录完全冗余（仅2个简化版本）
3. 策略组合过于复杂（8个子策略），信号冲突严重
4. 缺乏有效的市场环境适应机制

## 🎯 优化目标

### 主要目标
- **提高持股分析准确性**：专注于实用的核心策略
- **优化本地股票扫描**：集成有效的筛选策略
- **简化策略架构**：保留3-5个核心策略

### 保留策略清单
1. **strategy_base.py** - 策略基类（必须）
2. **tdi_strategy.py** - TDI策略（已验证有效）
3. **niuniu_strategy_v3.py** - 牛牛策略V3（核心策略）
4. **cpgw_strategy.py** - CPGW策略（补充策略）
5. **combined_strategy.py** - 简化的组合策略
6. **strategy_factory.py** - 策略工厂
7. **strategy_manager.py** - 策略管理器
8. **market_environment_classifier.py** - 市场环境分析
9. **dynamic_strategy_selector.py** - 动态策略选择

### 移至备份目录的策略
1. **重复策略**：
   - `momentum_strategy.py`
   - `mean_reversion_strategy.py` 
   - `bollinger_bands_strategy.py`
   - `breakout_strategy.py`
   - `intraday_momentum_strategy.py`

2. **示例/测试策略**：
   - `example_strategy.py`
   - `custom_strategy.py`
   - `amd_ma_strategy.py`
   - `test_*.py` 文件

3. **过时策略**：
   - `uss_*.py` 系列
   - `trend_following_strategy.py`
   - `market_sentiment_strategy.py`

4. **完全删除**：
   - `strategies/` 目录

## 🔄 整合计划

### 持股分析整合
- **核心策略权重**：NiuniuV3(50%) + TDI(30%) + CPGW(20%)
- **市场环境适应**：根据波动率和趋势动态调整权重
- **风险管理**：集成止损止盈计算

### 本地股票扫描整合
- **筛选策略**：基于技术指标和基本面的多因子模型
- **评分系统**：综合技术分析、动量、价值评分
- **市场时机**：结合市场环境分析选择最佳入场时机

## 📝 实施步骤

### Phase 1: 清理 ✅ 已完成
1. ✅ 创建新分支 `strategy-optimization`
2. ✅ 删除 `strategies/` 目录（完全冗余）
3. ✅ 移动14个冗余策略到 `strategy/bak/redundant_strategies/`
4. ✅ 移动测试文件到 `tests/`

### Phase 2: 优化核心策略 ✅ 已完成
1. ✅ 重写 `combined_strategy.py`（从8策略简化为3核心策略）
2. ✅ 优化策略权重分配（NiuniuV3:50% + TDI:30% + CPGW:20%）
3. ✅ 集成市场环境感知和信号过滤机制

### Phase 3: 整合应用 ✅ 已完成
1. ✅ 更新 `strategy_factory.py`（移除无效引用）
2. ✅ 创建 `enhanced_portfolio_advisor.py`（策略驱动的持股分析）
3. ✅ 创建 `enhanced_stock_screener.py`（智能股票筛选系统）

### Phase 4: 测试验证 ✅ 已完成
1. ✅ 验证策略工厂和组合策略正常工作
2. ✅ 测试增强持股分析功能
3. ✅ 验证股票筛选系统

## 📈 预期效果

1. **代码简化**：从31个文件减少到9个核心文件
2. **性能提升**：专注于验证有效的策略
3. **维护性**：清晰的策略层次结构
4. **实用性**：针对持股分析和股票筛选优化 