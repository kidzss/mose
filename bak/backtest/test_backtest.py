from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.run_backtest import run_single_backtest

# 设置回测参数
symbol = "AAPL"  # 苹果公司
start_date = datetime(2022, 1, 1)
end_date = datetime.now()

# 运行单个股票回测
print(f"\n运行 {symbol} 的回测...")
run_single_backtest(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    param_ranges={
        'sma_short': [5, 10, 15],  # 短期均线参数范围
        'sma_long': [20, 30, 40]   # 长期均线参数范围
    }
) 