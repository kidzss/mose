import yfinance as yf
import numpy as np
from yahoo_fin import options

def get_etf_vix(etf_ticker, option_chain_date, risk_free_rate=0.01):
    # Step 1: 获取ETF价格和期权数据
    etf = yf.Ticker(etf_ticker)
    option_chain = etf.option_chain(option_chain_date)
    calls = option_chain.calls
    puts = option_chain.puts

    # Step 2: 选择接近ETF现价的行权价及到期时间
    K_0 = etf.history(period='1d')['Close'].iloc[-1]  # 获取ETF当前价格
    T = 30 / 365  # 换算成年化到期时间

    # Step 3: 计算ETF的恐慌指数
    def calculate_etf_vix(options, K_0, T, risk_free_rate):
        # 根据VIX公式对行权价和期权价格加权平均
        F = K_0 * np.exp(risk_free_rate * T)
        summation = 0

        # 逐个行权价计算权重
        for i in range(len(options) - 1):
            delta_K = options['strike'][i + 1] - options['strike'][i]
            K = options['strike'][i]
            Q = options['lastPrice'][i]
            summation += (delta_K / K ** 2) * np.exp(risk_free_rate * T) * Q

        vix_value = 100 * np.sqrt((2 / T) * summation - (1 / T) * ((F / K_0) - 1) ** 2)
        return vix_value

    # Step 4: 分别计算看涨和看跌的VIX值
    etf_vix = calculate_etf_vix(calls, K_0, T, risk_free_rate)
    return etf_vix


# 使用函数计算ETF恐慌指数
etf_ticker = "QQQ"  # 替换为所需ETF
# 替换成你要查询的ETF代码
# etf_ticker = "SPY"  # 例如SPY

# option_chain_date = "2024-11-04"  # 替换为期权到期日

# # 获取ETF的期权数据
# etf = yf.Ticker(etf_ticker)
#
# # 检查所有可用的期权到期日
# print("Available Expirations:", etf.options)
#
# # 选择一个有效的日期
# option_chain_date = etf.options[0]  # 选择第一个有效日期
#
# etf_vix_value = get_etf_vix(etf_ticker, option_chain_date)
# print("ETF VIX Value:", etf_vix_value)

# 获取SPY的期权到期日
expirations = options.get_expiration_dates(etf_ticker)
print("Available Expirations:", expirations)