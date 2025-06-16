from data.data_updater import DatabaseManager, DB_CONFIG
import pandas as pd
from sqlalchemy import text

def check_data():
    # 创建数据库管理器
    db = DatabaseManager(DB_CONFIG)
    
    # 检查数据量
    with db.engine.connect() as conn:
        # 检查总记录数
        result = conn.execute(text('SELECT COUNT(*) FROM stock_code_time WHERE Code = "SPY" AND Date BETWEEN "2024-01-01" AND "2025-04-11"'))
        total_records = result.scalar()
        print(f"总记录数: {total_records}")
        
        # 检查数据范围
        result = conn.execute(text('SELECT MIN(Date), MAX(Date) FROM stock_code_time WHERE Code = "SPY"'))
        min_date, max_date = result.fetchone()
        print(f"数据范围: {min_date} 到 {max_date}")
        
        # 检查是否有空值
        result = conn.execute(text('''
            SELECT COUNT(*) 
            FROM stock_code_time 
            WHERE Code = "SPY" 
            AND (Open IS NULL OR High IS NULL OR Low IS NULL OR Close IS NULL OR Volume IS NULL)
        '''))
        null_records = result.scalar()
        print(f"空值记录数: {null_records}")
        
        # 检查价格异常
        result = conn.execute(text('''
            SELECT COUNT(*) 
            FROM stock_code_time 
            WHERE Code = "SPY" 
            AND (High < Low OR High < Open OR High < Close OR Low > Open OR Low > Close)
        '''))
        invalid_prices = result.scalar()
        print(f"价格异常记录数: {invalid_prices}")
        
        # 获取最新的10条记录
        result = conn.execute(text('''
            SELECT Date, Open, High, Low, Close, Volume 
            FROM stock_code_time 
            WHERE Code = "SPY" 
            ORDER BY Date DESC 
            LIMIT 10
        '''))
        latest_records = result.fetchall()
        print("\n最新的10条记录:")
        for record in latest_records:
            print(record)

if __name__ == "__main__":
    check_data() 