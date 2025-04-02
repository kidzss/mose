from sqlalchemy import create_engine, text
import pandas as pd

def check_database():
    engine = create_engine('mysql+pymysql://root@localhost/mose')
    
    try:
        with engine.connect() as conn:
            # 检查 stock_code_time 表
            print("检查 stock_code_time 表...")
            result = conn.execute(text('SELECT COUNT(*) as count, MIN(Date) as min_date, MAX(Date) as max_date FROM stock_code_time'))
            print('stock_code_time 表统计:')
            for row in result:
                print(f'总记录数: {row.count}, 最早日期: {row.min_date}, 最新日期: {row.max_date}')
            
            # 检查数据分布
            result = conn.execute(text('SELECT Code, COUNT(*) as count FROM stock_code_time GROUP BY Code LIMIT 5'))
            print('\n每个股票的记录数示例:')
            for row in result:
                print(f'股票代码: {row.Code}, 记录数: {row.count}')
            
            # 检查 stock_time_code 表
            print("\n检查 stock_time_code 表...")
            result = conn.execute(text('SELECT COUNT(*) as count, MIN(Date) as min_date, MAX(Date) as max_date FROM stock_time_code'))
            print('stock_time_code 表统计:')
            for row in result:
                print(f'总记录数: {row.count}, 最早日期: {row.min_date}, 最新日期: {row.max_date}')
            
            # 检查表结构
            print("\n表结构:")
            for table in ['stock_code_time', 'stock_time_code']:
                result = conn.execute(text(f'DESCRIBE {table}'))
                print(f'\n{table} 表结构:')
                for row in result:
                    print(f'字段: {row.Field}, 类型: {row.Type}, 可空: {row.Null}, 键: {row.Key}')
    
    except Exception as e:
        print(f"检查数据库时出错: {e}")

if __name__ == '__main__':
    check_database() 