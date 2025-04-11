from data.data_updater import DB_CONFIG
from sqlalchemy import create_engine, text
import pandas as pd

def check_data_info():
    """检查数据库中的数据情况"""
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    # 1. 检查AAPL的数据范围
    query = text("""
        SELECT 
            MIN(Date) as first_date,
            MAX(Date) as last_date,
            COUNT(DISTINCT Date) as total_days,
            COUNT(CASE WHEN Volume > 0 THEN 1 END) as trading_days
        FROM stock_time_code
        WHERE Code = 'AAPL'
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
        print("\nAAPL数据情况:")
        print(f"第一条数据日期: {result[0]}")
        print(f"最后数据日期: {result[1]}")
        print(f"总天数: {result[2]}")
        print(f"有交易的天数: {result[3]}")
    
    # 2. 检查数据库中的日期范围分布
    query = text("""
        SELECT 
            YEAR(Date) as year,
            COUNT(DISTINCT Date) as days_count,
            COUNT(DISTINCT Code) as stocks_count
        FROM stock_time_code
        GROUP BY YEAR(Date)
        ORDER BY year
    """)
    
    with engine.connect() as conn:
        results = conn.execute(query).fetchall()
        print("\n按年份统计:")
        print("年份\t天数\t股票数")
        for row in results:
            print(f"{row[0]}\t{row[1]}\t{row[2]}")
    
    # 3. 检查最近30天的数据情况
    query = text("""
        SELECT 
            Date,
            COUNT(DISTINCT Code) as stocks_count,
            COUNT(CASE WHEN Volume > 0 THEN 1 END) as trading_records
        FROM stock_time_code
        WHERE Date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        GROUP BY Date
        ORDER BY Date DESC
    """)
    
    with engine.connect() as conn:
        results = conn.execute(query).fetchall()
        print("\n最近30天数据情况:")
        print("日期\t\t股票数\t交易记录数")
        for row in results:
            print(f"{row[0]}\t{row[1]}\t{row[2]}")

if __name__ == '__main__':
    check_data_info() 