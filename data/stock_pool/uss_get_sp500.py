import pandas as pd

# 从维基百科获取标普500成分股列表
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_df = pd.read_html(url)[0]

# 提取所需的列
sp500_df = sp500_df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

# 重命名列便于阅读
sp500_df = sp500_df.rename(columns={
    "Symbol": "Code",
    "Security": "Name",
    "GICS Sector": "Sector",
    "GICS Sub-Industry": "Industry"
})

# 将数据保存到CSV文件
sp500_df.to_csv("sp500_stocks.csv", index=False)

print("标普500数据已保存到 sp500_stocks.csv 文件中")
