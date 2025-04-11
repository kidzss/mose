import pandas as pd

# 从维基百科获取纳斯达克100成分股列表
url = "https://en.wikipedia.org/wiki/NASDAQ-100"
nasdaq100_df = pd.read_html(url, header=0)[4]  # 获取表格数据

# 检查抓取到的列名
print("抓取到的列名:", nasdaq100_df.columns)

# 选择需要的列
nasdaq100_df = nasdaq100_df[['Symbol', 'Company', 'GICS Sector', 'GICS Sub-Industry']]

# 重命名列
nasdaq100_df.columns = ['Code', 'Name', 'Sector', 'Industry']

# 保存到 CSV 文件
nasdaq100_df.to_csv("nasdaq100_stocks.csv", index=False)

print("纳斯达克100成分股数据已保存到 nasdaq100_stocks.csv 文件中")
