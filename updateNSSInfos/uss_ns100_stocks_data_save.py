import pymysql
import pandas as pd
import datetime as dt
import yfinance as yf
from prettytable import PrettyTable

# 数据库配置信息
from utils.get_uss_stocks_datas import get_stock_list_from_db

db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "mose"
}


def get_data(stock, start_date, end_date):
    """获取股票数据"""
    try:
        data = yf.download(stock, start=start_date, end=end_date)
        if data.empty:
            print(f"股票 {stock} 无数据返回")
            return None
        data.reset_index(inplace=True)
        data.rename(columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "AdjClose",
            "Volume": "Volume"
        }, inplace=True)
        data["Code"] = stock
        return data
    except Exception as e:
        print(f"获取 {stock} 数据时出错:", e)
        return None


def save_to_mysql(data, db_config):
    """将数据保存到 MySQL 数据库"""
    if data is None or data.empty:
        print("数据为空，跳过保存")
        return

    try:
        connection = pymysql.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            charset='utf8mb4'
        )
        cursor = connection.cursor()

        for index, row in data.iterrows():
            insert_code_time = """
                INSERT INTO stock_code_time (Code, Date, Open, High, Low, Close, Volume, AdjClose)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                Open=VALUES(Open), High=VALUES(High), Low=VALUES(Low),
                Close=VALUES(Close), Volume=VALUES(Volume), AdjClose=VALUES(AdjClose);
            """
            cursor.execute(insert_code_time, (
                row["Code"], row["Date"], row["Open"], row["High"],
                row["Low"], row["Close"], row["Volume"], row["AdjClose"]
            ))

            insert_time_code = """
                INSERT INTO stock_time_code (Date, Code, Open, High, Low, Close, Volume, AdjClose)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                Open=VALUES(Open), High=VALUES(High), Low=VALUES(Low),
                Close=VALUES(Close), Volume=VALUES(Volume), AdjClose=VALUES(AdjClose);
            """
            cursor.execute(insert_time_code, (
                row["Date"], row["Code"], row["Open"], row["High"],
                row["Low"], row["Close"], row["Volume"], row["AdjClose"]
            ))

        connection.commit()
        print("数据保存成功！")
    except Exception as e:
        print("数据库保存数据时出错:", e)
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()


def validate_data(db_config):
    """
    校验存储的数据，从两个表中读取前 10 条记录。
    """
    try:
        connection = pymysql.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            charset='utf8mb4'
        )
        cursor = connection.cursor()

        # 从 stock_code_time 表中读取前 10 条记录
        print("从 stock_code_time 表中读取前 10 条记录:")
        query_stock_code_time = "SELECT * FROM stock_code_time LIMIT 10"
        cursor.execute(query_stock_code_time)
        results = cursor.fetchall()
        for row in results:
            print(row)

        # 从 stock_time_code 表中读取前 10 条记录
        print("\n从 stock_time_code 表中读取前 10 条记录:")
        query_stock_time_code = "SELECT * FROM stock_time_code LIMIT 10"
        cursor.execute(query_stock_time_code)
        results = cursor.fetchall()
        for row in results:
            print(row)

    except Exception as e:
        print("校验数据时出错:", e)
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()


from prettytable import PrettyTable


def validate_data_and_print(db_config):
    """
    校验存储的数据，从两个表中读取前 10 条记录并以表格形式打印。
    """
    try:
        connection = pymysql.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            charset='utf8mb4'
        )
        cursor = connection.cursor()

        # 从 stock_code_time 表中读取前 10 条记录
        print("从 stock_code_time 表中读取前 10 条记录:")
        query_stock_code_time = "SELECT * FROM stock_code_time LIMIT 10"
        cursor.execute(query_stock_code_time)
        results_code_time = cursor.fetchall()

        # 获取列名
        columns = [desc[0] for desc in cursor.description]

        # 使用 PrettyTable 打印表格
        table_code_time = PrettyTable()
        table_code_time.field_names = columns
        for row in results_code_time:
            table_code_time.add_row(row)
        print(table_code_time)

        # 从 stock_time_code 表中读取前 10 条记录
        print("\n从 stock_time_code 表中读取前 10 条记录:")
        query_stock_time_code = "SELECT * FROM stock_time_code LIMIT 10"
        cursor.execute(query_stock_time_code)
        results_time_code = cursor.fetchall()

        # 使用 PrettyTable 打印表格
        table_time_code = PrettyTable()
        table_time_code.field_names = columns
        for row in results_time_code:
            table_time_code.add_row(row)
        print(table_time_code)

    except Exception as e:
        print("校验数据时出错:", e)
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()


if __name__ == "__main__":
    validate_data_and_print(db_config)

    stock_list = get_stock_list_from_db()
    if stock_list is not None:
        start_date = dt.datetime(2022, 1, 1).strftime('%Y-%m-%d')
        end_date = dt.datetime.now().strftime('%Y-%m-%d')
        for stock in stock_list['Code']:
            stock_data = get_data(stock, start_date, end_date)
            save_to_mysql(stock_data, db_config)
    else:
        print("未能从数据库中读取股票列表，程序终止")
