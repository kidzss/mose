import pymysql
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database():
    try:
        # 连接到 MySQL
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='123456'
        )
        cursor = conn.cursor()
        
        # 创建数据库
        logger.info("Creating database 'mose'...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS mose")
        
        # 使用 mose 数据库
        cursor.execute("USE mose")
        
        # 创建必要的表
        logger.info("Creating tables...")
        
        # stock_time_code 表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_time_code (
            Code VARCHAR(20),
            Date DATE,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume BIGINT,
            AdjClose FLOAT,
            Dividends FLOAT,
            StockSplits FLOAT,
            Capital_Gains FLOAT,
            PRIMARY KEY (Date, Code)
        )
        """)
        
        # stock_code_time 表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_code_time (
            Code VARCHAR(20),
            Date DATE,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume BIGINT,
            AdjClose FLOAT,
            Dividends FLOAT,
            StockSplits FLOAT,
            Capital_Gains FLOAT,
            PRIMARY KEY (Code, Date)
        )
        """)
        
        conn.commit()
        logger.info("Database and tables created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    create_database() 