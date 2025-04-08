import pymysql
import pandas as pd
import datetime as dt
import yfinance as yf
import time
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据库配置信息
from utils.get_uss_stocks_datas import get_stock_list_from_db

db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "mose"
}


def get_stock_data(stock_list, start_date, end_date):
    """
    批量获取股票数据
    :param stock_list: 股票代码列表
    :param start_date: 开始日期 (格式: YYYY-MM-DD)
    :param end_date: 结束日期 (格式: YYYY-MM-DD)
    :return: 包含所有股票数据的 Pandas DataFrame
    """
    try:
        # 检查日期是否有效
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # 确保结束日期不超过当前日期
        current_date = dt.datetime.now().date()
        if end_date_dt.date() > current_date:
            end_date = current_date.strftime('%Y-%m-%d')
            print(f"结束日期调整为当前日期: {end_date}")
        
        # 确保开始日期不超过结束日期
        if start_date_dt.date() > pd.to_datetime(end_date).date():
            print(f"开始日期 {start_date} 晚于结束日期 {end_date}，无需更新数据")
            return None
            
        # 使用单个股票获取方式，而不是批量获取
        all_data = []
        for stock in stock_list:
            try:
                print(f"获取股票 {stock} 的数据...")
                ticker = yf.Ticker(stock)
                stock_data = ticker.history(start=start_date, end=end_date)
                
                if not stock_data.empty:
                    stock_data.reset_index(inplace=True)
                    stock_data["Code"] = stock
                    
                    # 确保所有必要的列都存在
                    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
                    missing_columns = [col for col in required_columns if col not in stock_data.columns]
                    if missing_columns:
                        print(f"警告: 股票 {stock} 数据缺少必要的列: {missing_columns}")
                        print(f"可用的列: {stock_data.columns.tolist()}")
                        continue
                    
                    # 检查是否有Adj Close列，如果没有，使用Close列的值
                    if "Adj Close" not in stock_data.columns:
                        print(f"警告: 股票 {stock} 数据缺少 'Adj Close' 列，使用 'Close' 列的值代替")
                        stock_data["Adj Close"] = stock_data["Close"]
                    
                    # 重命名列
                    stock_data.rename(columns={
                        "Date": "Date",
                        "Open": "Open",
                        "High": "High",
                        "Low": "Low",
                        "Close": "Close",
                        "Adj Close": "AdjClose",
                        "Volume": "Volume"
                    }, inplace=True)
                    
                    # 确保数据类型正确
                    for col in ["Open", "High", "Low", "Close", "AdjClose"]:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                    stock_data["Volume"] = pd.to_numeric(stock_data["Volume"], errors='coerce')
                    
                    all_data.append(stock_data)
                else:
                    print(f"股票 {stock} 在指定时间范围内没有数据")
                
                # 添加延迟，避免API限制
                time.sleep(1)
            except Exception as e:
                print(f"Failed to get ticker '{stock}' reason: {str(e)}")
                # 继续处理下一个股票
                continue

        if not all_data:
            print("未获取到任何股票数据")
            return None
            
        # 合并所有股票数据
        try:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"成功获取 {len(combined_data)} 行数据，涉及 {combined_data['Code'].nunique()} 只股票")
            return combined_data
        except Exception as e:
            print(f"合并数据时出错: {str(e)}")
            return None
    except Exception as e:
        print("获取股票数据时出错:", e)
        return None


def save_to_mysql(data, db_config):
    """
    将数据保存到 MySQL 数据库
    :param data: 包含股票数据的 Pandas DataFrame
    :param db_config: 数据库配置字典
    """
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

        # 确保所有必要的列都存在
        required_columns = ["Code", "Date", "Open", "High", "Low", "Close", "Volume", "AdjClose"]
        for col in required_columns:
            if col not in data.columns:
                print(f"错误: 数据中缺少必要的列 '{col}'")
                print(f"可用的列: {data.columns.tolist()}")
                return
        
        # 确保数据类型正确
        data["Date"] = pd.to_datetime(data["Date"]).dt.date  # 确保日期格式正确
        
        # 将NaN值转换为None，避免数据库插入错误
        data = data.replace({np.nan: None})
        
        success_count = 0
        error_count = 0

        for index, row in data.iterrows():
            try:
                # 检查AdjClose是否为None或NaN
                adj_close = row["AdjClose"]
                if pd.isna(adj_close):
                    adj_close = row["Close"]  # 如果AdjClose为空，使用Close的值
                
                insert_code_time = """
                    INSERT INTO stock_code_time (Code, Date, Open, High, Low, Close, Volume, AdjClose)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    Open=VALUES(Open), High=VALUES(High), Low=VALUES(Low),
                    Close=VALUES(Close), Volume=VALUES(Volume), AdjClose=VALUES(AdjClose);
                """
                cursor.execute(insert_code_time, (
                    row["Code"], row["Date"], row["Open"], row["High"],
                    row["Low"], row["Close"], row["Volume"], adj_close
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
                    row["Low"], row["Close"], row["Volume"], adj_close
                ))
                
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"插入行 {index} 时出错: {e}")
                print(f"问题数据: {row}")
                # 继续处理下一行，不中断整个过程
                continue

        connection.commit()
        print(f"数据保存完成! 成功: {success_count} 行, 失败: {error_count} 行")
    except Exception as e:
        print("数据库保存数据时出错:", e)
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()


def get_latest_date_from_db(table_name, db_config):
    """
    从指定表中获取最新的日期
    :param table_name: 表名
    :param db_config: 数据库配置字典
    :return: 最新日期，格式为字符串 'YYYY-MM-DD'
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
        query = f"SELECT MAX(Date) AS latest_date FROM {table_name}"
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            latest_date = result[0]
            return latest_date
    except Exception as e:
        print(f"从表 {table_name} 获取最新日期时出错:", e)
        return None
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()


if __name__ == "__main__":
    try:
        # 从数据库中读取股票列表
        stock_list = get_stock_list_from_db()

        if stock_list is None or stock_list.empty:
            print("股票列表为空，退出程序")
            exit()

        # 确定增量更新时间范围
        latest_date_code_time = get_latest_date_from_db("stock_code_time", db_config)
        latest_date_time_code = get_latest_date_from_db("stock_time_code", db_config)

        # 确保两个表的日期一致，选择最新的日期
        latest_date = max(latest_date_code_time, latest_date_time_code) if latest_date_code_time and latest_date_time_code else None

        # 获取当前日期
        current_date = dt.datetime.now().date()

        if latest_date:
            # 将日期转换为datetime对象
            latest_date_dt = pd.to_datetime(latest_date)
            # 添加一天得到开始日期
            start_date = (latest_date_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # 如果数据库中没有数据，默认从2022年开始获取
            start_date = '2022-01-01'

        # 使用当前日期作为结束日期
        end_date = current_date.strftime('%Y-%m-%d')

        print(f"开始更新数据，时间范围: {start_date} 到 {end_date}")
        
        # 获取股票数据
        stock_data = get_stock_data(stock_list['Code'].tolist(), start_date, end_date)
        
        # 保存数据到数据库
        if stock_data is not None and not stock_data.empty:
            save_to_mysql(stock_data, db_config)
        else:
            print("没有新数据需要更新")
            
    except Exception as e:
        print("程序执行出错:", e)
