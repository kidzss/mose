# 交易监控系统部署指南

## 环境要求
- Python 3.8 或更高版本
- MySQL 数据库（可选，如果使用数据库功能）

## 部署步骤

### 1. 创建虚拟环境
```bash
# 创建新的虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. 安装依赖
```bash
# 安装所有依赖包
pip install -r requirements.txt
```

### 3. 配置系统

#### 3.1 创建配置文件
在项目根目录创建以下配置文件：

1. `monitor/configs/portfolio_config.json` - 持仓配置
2. `config/strategy_config.json` - 策略配置
3. `.env` - 环境变量（可选）

#### 3.2 配置示例

portfolio_config.json 示例：
```json
{
    "positions": {
        "NVDA": {
            "cost_basis": 138.843,
            "weight": 0.2134,
            "shares": 40,
            "stop_loss": 84
        },
        "AMD": {
            "cost_basis": 85.55,
            "weight": 0.15,
            "shares": 30,
            "stop_loss": 72
        }
    },
    "monitor_config": {
        "price_alert_threshold": 0.05,
        "loss_alert_threshold": 0.05,
        "profit_target": 0.25,
        "stop_loss": 0.15,
        "check_interval": 60
    }
}
```

### 4. 运行系统

#### 4.1 测试运行
```bash
python test_monitor.py
```

#### 4.2 正式运行
```bash
python main.py
```

## 注意事项

1. 确保所有配置文件中的股票代码和价格信息正确
2. 如果使用邮件通知功能，需要在 `config/trading_config.py` 中配置正确的邮箱信息
3. 如果使用数据库功能，需要确保 MySQL 服务已启动并正确配置
4. 系统需要网络连接以获取实时股票数据

## 故障排除

1. 如果遇到依赖安装问题，可以尝试：
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --no-cache-dir
   ```

2. 如果遇到数据获取问题，检查网络连接和 Yahoo Finance API 的可用性

3. 如果遇到配置问题，检查所有配置文件是否存在且格式正确

## 系统维护

1. 定期检查日志文件了解系统运行状态
2. 定期更新持仓配置信息
3. 监控系统资源使用情况
4. 定期备份配置文件 