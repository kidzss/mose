import pandas as pd
from sqlalchemy import create_engine

# 数据库配置信息
db_config = {
    "host": "localhost",  # 替换为你的数据库地址
    "port": 3306,  # 通常是3306
    "user": "root",  # 替换为你的用户名
    "password": "",  # 替换为你的密码
    "database": "mose"
}


def get_stock_info_from_db(stock, start_date, end_date):
    """
    从 MySQL 数据库获取股票数据，并返回与 `pdr.get_data_yahoo` 格式一致的数据。

    :param stock: 股票代码
    :param start_date: 开始日期，格式 'YYYY-MM-DD'
    :param end_date: 结束日期，格式 'YYYY-MM-DD'
    :return: 包含股票数据的 Pandas DataFrame，列为 [High, Low, Open, Close, Volume, Adj Close]
    """
    query = """
    SELECT 
        Date AS `date`, 
        Open AS `Open`, 
        High AS `High`, 
        Low AS `Low`, 
        Close AS `Close`, 
        Volume AS `Volume`, 
        AdjClose AS `Adj Close`
    FROM stock_code_time
    WHERE Code = %s
    AND Date BETWEEN %s AND %s
    ORDER BY Date ASC;
    """
    try:
        # 创建 SQLAlchemy 引擎
        engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # 使用 Pandas 读取查询结果
        data = pd.read_sql_query(query, engine, params=(stock, start_date, end_date))

        # 将日期列设置为索引
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        return data
    except Exception as e:
        print(f"从数据库读取数据失败: {e}")
        return pd.DataFrame()  # 返回空的 DataFrame 以防止程序中断


def get_stock_list_from_db(table='uss_nasdaq_stocks'):
    """
    从数据库的 uss_nasdaq_stocks 表中读取数据
    :return: Pandas DataFrame 包含 Code, Name, Sector, Industry
    """
    try:
        # 创建 SQLAlchemy 引擎
        engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # 查询表数据
        query = "SELECT * FROM "+table
        stock_list = pd.read_sql(query, engine)
        return stock_list
    except Exception as e:
        print("从数据库获取股票列表时出错:", e)
        return None
