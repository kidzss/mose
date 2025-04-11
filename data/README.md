# MOSE Data Module - 统一数据访问接口

这是MOSE项目的统一数据访问模块。所有对数据的访问都应该通过这个模块进行，以确保数据的一致性和可维护性。

## 为什么使用这个模块？

- **统一的数据访问点**：所有数据访问都通过 `DataInterface` 进行，避免散乱的数据访问代码
- **标准化的数据格式**：所有数据都经过标准化处理，确保格式一致
- **自动的数据验证**：内置数据验证机制，确保数据质量
- **性能优化**：使用缓存机制提高访问速度
- **多数据源支持**：支持MySQL、Yahoo Finance等多个数据源
- **数据更新和监控**：内置数据更新机制和状态监控功能

## 快速开始

```python
from data import DataInterface

# 创建数据接口实例
data = DataInterface()

# 获取股票数据
aapl_data = data.get_historical_data('AAPL', '2023-01-01', '2023-12-31')

# 获取带技术指标的策略数据
strategy_data = data.get_data_for_strategy('AAPL', lookback_days=120)

# 更新市场数据
data.update_market_data()

# 检查数据状态
status = data.check_data_status('AAPL')
```

## 主要功能

### 1. 数据获取
- 历史数据获取
- 实时数据更新
- 多股票数据批量获取
- 技术指标计算

### 2. 数据验证
- 数据完整性检查
- 异常值检测
- 数据连续性验证
- 价格逻辑验证

### 3. 数据处理
- 缺失值处理
- 异常值处理
- 技术指标计算
- 数据标准化

### 4. 数据更新和监控
- 自动数据更新
- 数据质量监控
- 市场状态监控
- 更新状态追踪

## 详细使用说明

### 1. 初始化

```python
from data import DataInterface
from config.data_config import DataConfig

# 使用默认配置
data = DataInterface()

# 或使用自定义配置
config = DataConfig(
    default_source='mysql',
    mysql={
        'host': 'localhost',
        'port': 3306,
        'database': 'mose'
    }
)
data = DataInterface(config=config)
```

### 2. 获取历史数据

```python
# 单个股票
data = data.get_historical_data(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 多个股票
symbols = ['AAPL', 'MSFT', 'GOOGL']
data_dict = data.get_multiple_symbols_data(symbols, start_date, end_date)
```

### 3. 获取策略数据

```python
# 获取带技术指标的数据
strategy_data = data.get_data_for_strategy(
    symbol='AAPL',
    lookback_days=120
)

# 可用的技术指标
print(strategy_data.columns)
```

### 4. 数据验证

```python
from data import DataValidator

# 验证数据
validated_data, report = DataValidator.validate_data(data)

# 检查验证结果
if report['validation_passed']:
    print("数据验证通过")
else:
    print("数据验证失败:", report)
```

### 5. 数据更新和监控

```python
# 更新市场数据
data.update_market_data()  # 使用默认股票列表
data.update_market_data(['AAPL', 'MSFT'])  # 更新指定股票

# 获取最后更新时间
last_update = data.get_last_update_time('AAPL')
all_updates = data.get_last_update_time()  # 所有股票

# 检查数据状态
status = data.check_data_status('AAPL')
all_status = data.check_data_status()  # 所有股票

# 获取股票详细信息
info = data.get_symbol_info('AAPL')

# 获取市场整体状态
market_status = data.get_market_status()
```

## 配置选项

在 `config/data_config.py` 中可以配置：
- 默认数据源
- 数据库连接参数
- 缓存设置
- 数据验证规则
- 更新策略设置
- 监控参数

## 注意事项

1. **统一入口**：
   - 所有数据访问都应该通过 `DataInterface` 进行
   - 不要直接访问底层数据源
   - 不要创建重复的数据访问代码

2. **数据验证**：
   - 总是验证获取的数据
   - 注意处理验证失败的情况
   - 记录数据质量问题

3. **性能考虑**：
   - 使用缓存减少重复请求
   - 批量获取数据而不是频繁单次请求
   - 注意内存使用

4. **数据更新**：
   - 定期更新数据
   - 检查更新状态
   - 处理更新失败情况

## 扩展和自定义

### 添加新的数据源

1. 继承 `DataSource` 基类
2. 实现必要的方法
3. 在 `DataInterface` 中注册新数据源

```python
from data import DataSource

class MyDataSource(DataSource):
    def get_historical_data(self, symbol, start_date, end_date):
        # 实现数据获取逻辑
        pass
    
    # 实现其他必要方法...

# 注册新数据源
data = DataInterface()
data.add_data_source('my_source', MyDataSource())
```

### 自定义数据验证

```python
from data import DataValidator

class MyValidator(DataValidator):
    @staticmethod
    def validate_data(data):
        # 实现自定义验证逻辑
        pass
```

### 自定义更新策略

```python
from data import DataInterface

class MyDataInterface(DataInterface):
    def update_market_data(self, symbols=None):
        # 实现自定义更新逻辑
        pass
```

## 常见问题

1. **数据不一致**：
   - 确保使用统一的 `DataInterface`
   - 检查数据验证报告
   - 确认数据源配置正确

2. **性能问题**：
   - 使用批量获取而不是循环单次获取
   - 确保缓存配置正确
   - 考虑使用数据预加载

3. **数据质量**：
   - 总是进行数据验证
   - 检查异常值和缺失值
   - 记录和报告数据问题

4. **更新失败**：
   - 检查网络连接
   - 验证数据源配置
   - 查看错误日志
   - 使用备用数据源

## 更多信息

- 查看 `examples` 目录获取更多使用示例
- 查看源代码了解详细实现
- 参考配置文件了解所有配置选项 