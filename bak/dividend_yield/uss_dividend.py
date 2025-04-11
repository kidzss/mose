import yfinance as yf
import pandas as pd


def filter_by_dividend(stocks_df, dividend_yield_threshold=5.0):
    """
    筛选出股息率超过指定阈值的股票。

    :param stocks_df: 包含股票代码的DataFrame
    :param dividend_yield_threshold: 股息率阈值（百分比）
    :return: 包含高股息股票的DataFrame
    """
    high_dividend_stocks = []

    for ticker in stocks_df['Code']:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            dividend_yield = info.get('dividendYield', 0) * 100  # 获取股息率百分比

            if dividend_yield >= dividend_yield_threshold:
                high_dividend_stocks.append({
                    "Ticker": ticker,
                    "Dividend Yield (%)": dividend_yield
                })

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

    return pd.DataFrame(high_dividend_stocks)


def main():
    # 从文件中读取标普500和纳斯达克100的股票列表
    sp500_stocks = pd.read_csv('../stock_pool/sp500_stocks.csv')  # 文件包含标普500股票的代码
    nasdaq100_stocks = pd.read_csv('../stock_pool/nasdaq100_stocks.csv')  # 文件包含纳斯达克100股票的代码

    # 合并两个股票池
    all_stocks = pd.concat([sp500_stocks, nasdaq100_stocks]).drop_duplicates()

    # 筛选股息率超过5%的股票
    high_dividend_df = filter_by_dividend(all_stocks)

    # 保存结果到CSV文件
    high_dividend_df.to_csv('high_dividend_stocks.csv', index=False)
    print("高股息股票已保存到 'high_dividend_stocks.csv'.")


if __name__ == "__main__":
    main()
