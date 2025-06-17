# 技术指标模块

本模块提供了各种常用的技术分析指标，可用于构建和优化交易策略。

## 设计理念

1. **模块化**: 每种类型的指标都单独实现在各自的模块中，便于维护和扩展
2. **统一接口**: 通过`indicators.py`模块提供统一接口，便于策略中使用
3. **高效计算**: 基于pandas实现，支持批量计算和优化
4. **类型提示**: 使用类型注解提高代码可读性和IDE支持
5. **测试覆盖**: 每个指标模块都有对应的测试文件，确保正确性

## 目录结构

```
indicators/
├── __init__.py          # 包初始化文件
├── indicators.py        # 统一接口模块
├── moving_averages.py   # 移动平均线模块
├── bollinger_bands.py   # 布林带模块
├── rsi.py               # RSI相对强弱指标模块
├── macd.py              # MACD指标模块
├── adx.py               # ADX方向动量指标模块
├── volatility.py        # 波动率指标模块
├── volume.py            # 成交量指标模块
├── oscillators.py       # 震荡指标模块
├── trend_strength.py    # 趋势强度指标模块
├── support_resistance.py # 支撑与阻力指标模块
└── README.md            # 说明文档
```

## 已实现指标

### 移动平均线 (moving_averages.py)
- 简单移动平均线 (SMA)
- 指数移动平均线 (EMA)
- 加权移动平均线 (WMA)
- 平滑移动平均线 (SMMA)
- 三重指数移动平均线 (TEMA)
- 考夫曼自适应移动平均线 (KAMA)
- 赫尔移动平均线 (Hull MA)

### 布林带 (bollinger_bands.py)
- 布林带 (中轨、上轨、下轨)
- 布林带宽度
- 布林带挤压
- 布林带突破信号
- 布林带反转信号

### RSI相对强弱指标 (rsi.py)
- RSI
- 随机RSI
- RSI背离
- RSI超买超卖信号
- RSI反转信号

### MACD指标 (macd.py)
- MACD线、信号线和柱状图
- MACD金叉死叉信号
- MACD零线交叉信号
- MACD背离
- MACD柱状图反转信号
- PPO百分比价格震荡指标

### ADX方向动量指标 (adx.py)
- ADX
- +DI和-DI
- ADX趋势强度判断
- ADX趋势方向判断
- ADX交叉信号
- ADX反转信号
- DMI振荡器

### 波动率指标 (volatility.py)
- ATR平均真实范围
- 历史波动率
- 布林带宽度
- Keltner通道宽度
- 波动率比率
- Chaikin波动率
- Garman-Klass波动率
- Ulcer指数

### 成交量指标 (volume.py)
- 成交量移动平均
- 成交量比率
- OBV能量潮
- CMF蔡金资金流量
- MFI资金流量指标
- VPT成交量价格趋势
- NVI/PVI负/正成交量指标
- EMV简易波动指标
- 成交量振荡器
- A/D积累分配线

### 震荡指标 (oscillators.py)
- 随机指标(%K和%D)
- 威廉指标(%R)
- ROC变动率
- CCI顺势指标
- AO动量震荡指标
- 终极震荡器
- 随机RSI
- TRIX三重指数平滑平均
- TSI真实强度指数
- PPO百分比价格震荡指标

### 趋势强度指标 (trend_strength.py)
- ADX平均方向指标
- Aroon指标
- Vortex指标
- DMI振荡器
- 方向运动指数
- 去趋势价格振荡器
- 抛物线转向指标
- 趋势强度指数
- Supertrend指标

### 支撑与阻力指标 (support_resistance.py)
- 传统枢轴点
- 斐波那契枢轴点
- Woodie枢轴点
- 价格通道
- 唐奇安通道
- 斐波那契回调位
- 支撑阻力水平识别
- Keltner通道
- 一目均衡表(云图)

## 使用方法

### 直接使用各指标模块

```python
import pandas as pd
from strategy.indicators.moving_averages import sma, ema
from strategy.indicators.bollinger_bands import bollinger_bands
from strategy.indicators.rsi import rsi

# 加载数据
data = pd.read_csv('price_data.csv')
close = data['close']

# 计算SMA
sma_20 = sma(close, window=20)

# 计算EMA
ema_20 = ema(close, window=20)

# 计算布林带
bb = bollinger_bands(close, window=20, num_std=2.0)
upper_band = bb['upper']
middle_band = bb['middle']
lower_band = bb['lower']

# 计算RSI
rsi_14 = rsi(close, window=14)
```

### 使用统一接口

```python
import pandas as pd
from strategy.indicators import TechnicalIndicators, calculate_indicators

# 加载数据
data = pd.read_csv('price_data.csv')

# 方法1: 使用TechnicalIndicators类
ti = TechnicalIndicators()
sma_20 = ti.calculate_ma(data['close'], ma_type='sma', window=20)
bb = ti.calculate_bb(data['close'])
rsi_14 = ti.calculate_rsi(data['close'])

# 方法2: 批量计算多个指标
result = calculate_indicators(data, selected_indicators=['sma', 'ema', 'bb', 'rsi', 'macd'])
```

## 扩展指标

要添加新的指标，只需按照以下步骤:

1. 在合适的模块中实现指标函数，或创建新的模块
2. 如果创建了新模块，在`__init__.py`中导入
3. 在`indicators.py`的`TechnicalIndicators`类中添加相应的计算方法
4. 在`calculate_all_indicators`方法中注册新指标
5. 创建测试文件，确保指标计算正确
6. 更新本README文档

## 注意事项

- 所有指标函数都应该接受pandas.Series作为输入
- 所有函数都应该提供清晰的文档字符串，说明参数和返回值
- 确保处理好NaN值和边界情况
- 所有指标计算应该是无状态的，不依赖外部变量 