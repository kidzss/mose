# 📧 股票筛选器邮件功能使用指南

## 🎯 功能概述

我们的股票筛选器现在支持自动发送HTML格式的邮件报告！这个功能可以：

- 📊 **自动发送筛选结果**: 将筛选结果以精美的HTML表格形式发送到您的邮箱
- 🎨 **专业HTML格式**: 包含CSS样式的专业报告，易于阅读
- 📎 **附件支持**: 同时发送JSON格式的详细数据
- 📄 **Markdown报告**: 支持将MD报告转换为HTML邮件发送

## 🔧 配置步骤

### 1. 邮件配置设置

运行配置向导：
```bash
python setup_email_config.py
```

### 2. Gmail配置要求

**重要**: 建议使用Gmail账户，需要以下设置：

1. **开启两步验证**
   - 登录 Google 账户
   - 进入"安全性"设置
   - 开启"两步验证"

2. **生成应用专用密码**
   - 访问: https://myaccount.google.com/apppasswords
   - 选择"邮件"应用类型
   - 生成16位应用专用密码

3. **设置环境变量**
   ```bash
   # Windows (PowerShell)
   $env:EMAIL_SENDER="your_email@gmail.com"
   $env:EMAIL_PASSWORD="your_16_digit_app_password"
   $env:EMAIL_RECEIVER="receiver@email.com"
   
   # Linux/Mac
   export EMAIL_SENDER="your_email@gmail.com"
   export EMAIL_PASSWORD="your_16_digit_app_password"
   export EMAIL_RECEIVER="receiver@email.com"
   ```

## 🚀 使用方法

### 方法1: 一键运行 (推荐)

```bash
# 使用默认配置运行
python run_screening_with_email.py

# 自定义配置
python run_screening_with_email.py --min-score 60 --max-results 20 --subject "我的股票筛选报告"

# 同时发送执行报告
python run_screening_with_email.py --send-report
```

### 方法2: 代码调用

```python
from monitor.phase2_professional_screener import Phase2ProfessionalScreener

# 初始化筛选器
screener = Phase2ProfessionalScreener()

# 筛选并发送邮件
results = screener.screen_and_email(
    min_score=50,
    max_results=25,
    send_email=True,
    email_subject="🚀 我的股票筛选报告"
)

# 发送Markdown报告
screener.send_report_email(
    'PHASE2_EXECUTION_REPORT.md',
    '📊 Phase 2 执行报告'
)
```

### 方法3: 统一邮件API

```python
from utils.unified_email_api import send_html, send_markdown, send_text

# 发送HTML邮件
send_html(subject="报告", html_content="<h1>分析报告</h1>")

# 发送Markdown邮件
send_markdown(subject="分析", md_content="# 分析报告")

# 发送文本邮件
send_text(subject="通知", content="Hello World")
```

### 方法4: 命令行参数

```bash
# 基本用法
python run_screening_with_email.py --min-score 55 --max-results 30

# 禁用邮件发送
python run_screening_with_email.py --no-email

# 自定义邮件主题
python run_screening_with_email.py --subject "今日优质股票推荐"

# 发送完整报告
python run_screening_with_email.py --send-report
```

## 📧 邮件内容

### HTML邮件包含：

1. **📊 筛选摘要**
   - 筛选时间
   - 分析股票数量
   - 发现的优质股票数量
   - 最佳股票信息

2. **🏆 筛选结果表格**
   - 排名
   - 股票代码
   - 多因子评分
   - 质量因子
   - 夏普比率
   - 最大回撤
   - 当前价格

3. **🎯 最佳投资标的分析**
   - 详细的最佳股票信息
   - 投资亮点分析

4. **💡 投资建议**
   - 核心策略建议
   - 风险控制措施
   - 组合管理建议

5. **📎 JSON附件**
   - 完整的筛选数据
   - 可用于进一步分析

## 🎨 HTML邮件样式特色

- **🎯 专业设计**: 现代化的CSS样式
- **📊 表格美化**: 悬停效果、条纹背景
- **🏆 高亮显示**: 高质量股票特殊标记
- **📱 响应式**: 适配不同设备屏幕
- **🎨 颜色编码**: 不同类型信息用不同颜色区分

## 🔧 高级配置

### 自定义SMTP服务器

```python
from utils.unified_email_api import UnifiedEmailAPI

# 使用其他邮件服务
api = UnifiedEmailAPI()
# 配置会自动从环境变量或配置文件加载
```

### 支持的邮件服务

| 服务商 | SMTP服务器 | 端口 | 说明 |
|--------|------------|------|------|
| Gmail | smtp.gmail.com | 587 | 推荐，需要应用专用密码 |
| Outlook | smtp-mail.outlook.com | 587 | 支持，需要应用专用密码 |
| Yahoo | smtp.mail.yahoo.com | 587 | 支持，需要应用专用密码 |
| QQ邮箱 | smtp.qq.com | 587 | 支持，需要授权码 |

## 🚨 故障排除

### 常见问题

1. **认证失败**
   - 确保使用应用专用密码而不是普通密码
   - 检查两步验证是否已开启

2. **连接超时**
   - 检查网络连接
   - 确认SMTP服务器和端口正确

3. **邮件发送失败**
   - 检查邮箱配置
   - 查看日志获取详细错误信息

### 测试邮件配置

```python
from utils.unified_email_api import test_email_config

# 测试邮件配置
if test_email_config():
    print("✅ 邮件配置正常")
else:
    print("❌ 邮件配置有问题")
```

## 📝 更新日志

- **v2.0**: 新增统一邮件API，简化邮件发送
- **v1.5**: 支持Markdown报告邮件
- **v1.0**: 基础HTML邮件功能

---

💡 **提示**: 建议使用统一邮件API (`utils/unified_email_api.py`) 进行所有邮件发送操作，它提供了更简单、更统一的接口。

## 🎯 最佳实践

1. **定期筛选**: 建议每周运行一次筛选
2. **邮件归档**: 在邮箱中创建专门文件夹保存筛选报告
3. **数据备份**: JSON附件可用于历史数据分析
4. **安全性**: 定期更换应用专用密码
5. **监控**: 关注邮件发送状态，确保及时收到报告

## 🚀 未来功能

- 📅 **定时发送**: 支持定时自动筛选和发送
- 📊 **图表支持**: 在邮件中包含股票走势图
- 🔔 **多收件人**: 支持发送给多个邮箱
- 📱 **移动优化**: 更好的移动设备显示效果

---

**💡 提示**: 首次使用建议先运行 `python setup_email_config.py` 进行配置测试，确保邮件功能正常工作后再进行正式筛选。 