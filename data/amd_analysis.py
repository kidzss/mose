import logging
import pandas as pd
import matplotlib.pyplot as plt
from openbb import openbb
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置股票代码、预测周期（例如，预测未来 30 天）以及历史数据周期（半年）
symbol = "AMD"
forecast_days = 30
# 计算半年（约 180 天）前的日期，并获取历史数据
start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
logger.info(f"获取 {symbol} 自 {start_date} 以来的历史数据...")
df = openbb.stocks.load(symbol, start=start_date, end=datetime.now().strftime("%Y-%m-%d"))
logger.info("历史数据获取完成，数据形状: %s", df.shape)

# 利用 openbb 内置的 ta 模块进行预测（这里以 ta 模块的预测函数为例，具体函数名请参考 openbb 文档）
# 例如，使用 ta 模块的 forecast 函数（假设函数名为 forecast_ta）进行预测
# 注意：请根据 openbb 最新文档调整预测函数名及参数
try:
    logger.info("调用 openbb 内置预测函数（例如 ta 模块的 forecast_ta）...")
    # 这里假设 openbb.ta.forecast_ta 函数存在，并返回预测结果（预测数据框）
    # 如果 openbb 没有内置预测函数，请替换成你自定义的预测逻辑，或使用其他库（如 Prophet、statsmodels 等）
    # 例如，这里用 df 的 Close 列进行预测，预测未来 forecast_days 天
    # 注意：请根据 openbb 文档或自定义预测逻辑调整
    # 示例（假设 openbb.ta.forecast_ta 返回预测数据框，包含日期和预测值）：
    # forecast_df = openbb.ta.forecast_ta(df, forecast_days=forecast_days, target_col="Close")
    # 如果 openbb 没有内置预测函数，可以暂时用简单移动平均线（例如 20 日均线）模拟预测，如下：
    df["forecast"] = df["Close"].rolling(window=20, min_periods=1).mean().shift(-forecast_days)
    forecast_df = df[["forecast"]].dropna()
    logger.info("预测完成，预测数据形状: %s", forecast_df.shape)
except Exception as e:
    logger.error("预测过程中出错: %s", e)
    forecast_df = pd.DataFrame()

# 绘图：绘制 AMD 半年历史走势图，并叠加预测曲线（如果预测数据存在）
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="历史收盘价", color="blue")
if not forecast_df.empty:
    # 注意：预测数据的时间索引（例如，预测的日期）需要与历史数据对齐，这里假设 forecast_df 的索引是预测日期
    # 如果预测数据没有日期索引，请根据实际情况调整绘图代码
    plt.plot(forecast_df.index, forecast_df["forecast"], label="预测（示例：20日均线）", color="red", linestyle="--")
plt.title(f"{symbol} 半年历史走势图及预测（预测周期: {forecast_days} 天）")
plt.xlabel("日期")
plt.ylabel("价格")
plt.legend()
plt.grid(True)
plt.tight_layout()
# 保存图片到当前目录
plt.savefig("amd_forecast.png")
logger.info("绘图完成，图片已保存为 amd_forecast.png")

# 打印数据统计信息，方便查看
logger.info("历史数据统计信息:\n%s", df.describe()) 