import pymysql
import logging
from tabulate import tabulate
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_import():
    try:
        # 连接到 MySQL
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='mose'
        )
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print("\nAvailable tables:")
        print(tabulate(tables, headers=['Table Name']))
        
        # 检查每个表的记录数
        print("\nRecord counts:")
        counts = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            counts.append([table_name, count])
        print(tabulate(counts, headers=['Table', 'Record Count']))
        
        # 检查股票数据
        print("\nSample stock data (with date format check):")
        cursor.execute("""
            SELECT Code, 
                   COUNT(*) as record_count,
                   MIN(STR_TO_DATE(Date, '%Y-%m-%d')) as earliest_date,
                   MAX(STR_TO_DATE(Date, '%Y-%m-%d')) as latest_date,
                   MIN(Date) as raw_earliest_date,
                   MAX(Date) as raw_latest_date
            FROM stock_time_code
            GROUP BY Code
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        print(tabulate(sample_data, headers=['Stock Code', 'Records', 'Earliest Date', 'Latest Date', 'Raw Earliest', 'Raw Latest']))
        
        # 检查日期格式
        print("\nDate format check:")
        cursor.execute("""
            SELECT Date, COUNT(*) as count
            FROM stock_time_code
            WHERE Date = '0000-00-00'
            GROUP BY Date
            LIMIT 1
        """)
        invalid_dates = cursor.fetchall()
        if invalid_dates:
            print(f"Found {invalid_dates[0][1]} records with invalid date format")
            
        # 检查有效的日期范围
        cursor.execute("""
            SELECT MIN(STR_TO_DATE(Date, '%Y-%m-%d')) as min_date,
                   MAX(STR_TO_DATE(Date, '%Y-%m-%d')) as max_date
            FROM stock_time_code
            WHERE Date != '0000-00-00'
        """)
        date_range = cursor.fetchone()
        print(f"\nValid date range: {date_range[0]} to {date_range[1]}")
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_import() 