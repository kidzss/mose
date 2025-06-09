# OpenBB API密钥设置指南

## 概述

OpenBB是一个强大的金融分析平台，可以访问近100个数据源。虽然部分基础功能不需要API密钥，但要充分利用其功能，建议配置API密钥。

## 1. 配置文件位置

OpenBB的API密钥配置文件存储在以下位置：
- Windows: `C:\Users\你的用户名\.openbb\credentials.ini`
- macOS/Linux: `/home/你的用户名/.openbb/credentials.ini`

## 2. 常用免费API密钥获取方法

以下是一些提供免费计划的数据源：

### FRED (Federal Reserve Economic Data)
- 网址：https://fred.stlouisfed.org/docs/api/api_key.html
- 获取方法：
  1. 创建/登录FRED账户
  2. 访问API Key页面申请密钥
  3. 在收到的邮件中获取密钥
- 配置方式：在credentials.ini的`[fred]`部分添加`fred_key = 你的密钥`

### Alpha Vantage
- 网址：https://www.alphavantage.co/support/#api-key
- 获取方法：
  1. 填写表格申请免费API密钥
  2. 立即获得API密钥
- 配置方式：在credentials.ini的`[alpha_vantage]`部分添加`key = 你的密钥`
- 免费限制：每分钟5个请求，每天500个请求

### Finnhub
- 网址：https://finnhub.io/register
- 获取方法：
  1. 注册账户
  2. 在仪表板获取API密钥
- 配置方式：在credentials.ini的`[finnhub]`部分添加`key = 你的密钥`
- 免费限制：每分钟60个API调用

### Financial Modeling Prep (FMP)
- 网址：https://site.financialmodelingprep.com/developer/docs/
- 获取方法：
  1. 注册账户
  2. 获取免费API密钥
- 配置方式：在credentials.ini的`[fmp]`部分添加`api = 你的密钥`
- 免费限制：每天250个请求

### Polygon.io
- 网址：https://polygon.io/dashboard/signup
- 获取方法：
  1. 注册免费账户
  2. 获取API密钥
- 配置方式：在credentials.ini的`[polygon]`部分添加`key = 你的密钥`
- 免费限制：有限的历史数据访问

### EODHD
- 网址：https://eodhistoricaldata.com/register
- 获取方法：
  1. 注册账户
  2. 获取API密钥
- 配置方式：在credentials.ini的`[eodhd]`部分添加`key = 你的密钥`
- 免费限制：每天20个API调用

## 3. 使用方法

配置好API密钥后，OpenBB会自动使用这些密钥访问相应的数据源。例如：

```python
from openbb import obb

# 使用FRED API获取GDP数据
gdp_data = obb.economy.gdp.real().to_df()

# 使用Alpha Vantage获取股票数据
stock_data = obb.equity.price.historical(symbol="AAPL").to_df()

# 使用FMP获取公司财务数据
income_statement = obb.equity.fundamental.income(symbol="MSFT").to_df()
```

## 4. 无需API密钥的功能

OpenBB有许多功能不需要API密钥即可使用，包括：

1. 基本股票价格数据查询
2. 基本市场数据分析
3. 部分技术指标计算
4. 投资组合优化工具

## 5. 故障排除

如果遇到API相关问题：

1. 确认API密钥格式正确且没有多余空格
2. 检查是否超出API调用限制
3. 确认credentials.ini文件编码为UTF-8
4. 重启Python环境以加载新的配置

## 6. 示例credentials.ini文件

```ini
[fred]
fred_key = your_fred_key_here

[alpha_vantage] 
key = your_alphavantage_key_here

[finnhub]
key = your_finnhub_key_here

[fmp]
api = your_fmp_key_here

[polygon]
key = your_polygon_key_here

[eodhd]
key = your_eodhd_key_here
```

完成这些设置后，您就可以充分利用OpenBB的强大功能来进行金融分析了。 