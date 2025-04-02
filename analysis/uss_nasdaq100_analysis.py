import pandas as pd


def read_backtest_results(file_paths):
    """读取回测结果文件并添加策略标识列"""
    dataframes = []
    strategies = ['Momentum', 'Market Forecast', 'Gold Triangle']

    for file_path, strategy in zip(file_paths, strategies):
        df = pd.read_csv(file_path)
        df['Strategy'] = strategy
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def summarize_stats(df):
    """汇总每个股票的各项指标（按股票代码和策略）"""
    return df.groupby(['Stock Code', 'Strategy']).agg({
        'Total Return(%)': 'mean',
        'Gain/loss ratio': 'mean',
        'Batting Avg': 'mean'
    }).reset_index()


def get_top_stocks(summary_stats, metric, top_n=20):
    """根据指定指标获取前n只股票"""
    return summary_stats.sort_values(by=metric, ascending=False).head(top_n)


def save_results_to_csv(results, filename):
    """将结果保存到CSV文件"""
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main():
    top_num = 300

    # 文件路径列表
    file_paths = [
        "../backtest/momentum_nasdaq100_backtest_results.csv",
        "../backtest/forecast_nasdaq100_backtest_results.csv",
        "../backtest/triangle_nasdaq100_backtest_results.csv"
    ]

    # 读取数据
    df_all = read_backtest_results(file_paths)

    # 统计汇总
    summary_stats = summarize_stats(df_all)

    # 获取收益最高的股票
    top_returns = get_top_stocks(summary_stats, 'Total Return(%)', top_n=top_num)
    save_results_to_csv(top_returns, "top_stocks_by_total_return.csv")

    # 获取盈利比率较高的股票
    top_gain_loss_ratio = get_top_stocks(summary_stats, 'Gain/loss ratio', top_n=top_num)
    save_results_to_csv(top_gain_loss_ratio, "top_stocks_by_gain_loss_ratio.csv")

    # 获取盈亏次数较高的股票
    top_batting_avg = get_top_stocks(summary_stats, 'Batting Avg', top_n=top_num)
    save_results_to_csv(top_batting_avg, "top_stocks_by_batting_avg.csv")


if __name__ == "__main__":
    main()
