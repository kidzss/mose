<<<<<<< HEAD
import pandas as pd
from sqlalchemy import create_engine, text
import logging
from config.trading_config import default_config
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('check_invalid_dates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    "host": default_config.database.host,
    "port": default_config.database.port,
    "user": default_config.database.user,
    "password": default_config.database.password,
    "database": default_config.database.database
}

def check_invalid_dates():
    """检查数据库中的无效日期数据"""
    try:
        # 创建数据库连接
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        
        with engine.connect() as conn:
            # 临时修改 SQL 模式以允许无效日期
            conn.execute(text("SET SESSION sql_mode=''"))
            
            # 获取表结构
            table_info_query = text("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME IN ('stock_time_code', 'stock_code_time')
                ORDER BY TABLE_NAME, ORDINAL_POSITION
            """)
            
            table_info = pd.read_sql(table_info_query, conn)
            logger.info("\n表结构信息：")
            logger.info("\n" + str(table_info))
            
            # 获取无效日期记录的详细信息
            invalid_records_query = text("""
                SELECT 
                    t1.Code,
                    t1.Date as invalid_date,
                    t1.Open,
                    t1.High,
                    t1.Low,
                    t1.Close,
                    t1.Volume,
                    t2.Date as code_time_date,
                    t2.Open as code_time_open,
                    t2.High as code_time_high,
                    t2.Low as code_time_low,
                    t2.Close as code_time_close,
                    t2.Volume as code_time_volume
                FROM stock_time_code t1
                LEFT JOIN stock_code_time t2 
                    ON t1.Code = t2.Code 
                    AND t1.Date = t2.Date
                WHERE t1.Date = '0000-00-00'
                ORDER BY t1.Code
            """)
            
            invalid_records = pd.read_sql(invalid_records_query, conn)
            if not invalid_records.empty:
                logger.info("\n无效日期记录的详细信息：")
                logger.info("\n" + str(invalid_records))
                
                # 分析数据差异
                price_diff = invalid_records[
                    (invalid_records['Open'] != invalid_records['code_time_open']) |
                    (invalid_records['High'] != invalid_records['code_time_high']) |
                    (invalid_records['Low'] != invalid_records['code_time_low']) |
                    (invalid_records['Close'] != invalid_records['code_time_close']) |
                    (invalid_records['Volume'] != invalid_records['code_time_volume'])
                ]
                
                if not price_diff.empty:
                    logger.warning("\n发现与 stock_code_time 表数据不一致的记录：")
                    logger.warning("\n" + str(price_diff))
                else:
                    logger.info("\n所有无效日期记录在 stock_code_time 表中都有对应的数据")
                
                # 检查每个股票的前后日期
                date_context_query = text("""
                    WITH invalid_records AS (
                        SELECT Code, Date
                        FROM stock_time_code
                        WHERE Date = '0000-00-00'
                    )
                    SELECT 
                        i.Code,
                        i.Date as invalid_date,
                        MAX(CASE WHEN t.Date < '0000-00-00' THEN t.Date END) as prev_date,
                        MIN(CASE WHEN t.Date > '0000-00-00' THEN t.Date END) as next_date,
                        COUNT(*) as total_records
                    FROM invalid_records i
                    LEFT JOIN stock_time_code t ON i.Code = t.Code
                    GROUP BY i.Code, i.Date
                    ORDER BY i.Code
                """)
                
                date_context = pd.read_sql(date_context_query, conn)
                logger.info("\n无效日期记录的前后日期上下文：")
                logger.info("\n" + str(date_context))
            
            # 检查 stock_time_code 表
            time_code_query = text("""
                SELECT Code, Date, COUNT(*) as count
                FROM stock_time_code
                WHERE Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' 
                   OR Date = '0000-00-00'
                   OR Date IS NULL
                   OR LENGTH(Date) != 10
                GROUP BY Code, Date
                ORDER BY Code, Date
            """)
            
            # 检查 stock_code_time 表
            code_time_query = text("""
                SELECT Code, Date, COUNT(*) as count
                FROM stock_code_time
                WHERE Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                   OR Date = '0000-00-00'
                   OR Date IS NULL
                   OR LENGTH(Date) != 10
                GROUP BY Code, Date
                ORDER BY Code, Date
            """)
            
            # 检查 stock_time_code 表
            time_code_results = pd.read_sql(time_code_query, conn)
            if not time_code_results.empty:
                logger.warning("在 stock_time_code 表中发现无效日期数据：")
                logger.warning("\n" + str(time_code_results))
                logger.warning(f"总计发现 {len(time_code_results)} 条无效日期记录")
            else:
                logger.info("stock_time_code 表中没有无效日期数据")
            
            # 检查 stock_code_time 表
            code_time_results = pd.read_sql(code_time_query, conn)
            if not code_time_results.empty:
                logger.warning("在 stock_code_time 表中发现无效日期数据：")
                logger.warning("\n" + str(code_time_results))
                logger.warning(f"总计发现 {len(code_time_results)} 条无效日期记录")
            else:
                logger.info("stock_code_time 表中没有无效日期数据")
            
            # 检查日期范围（只检查有效日期）
            date_range_query = text("""
                SELECT 
                    'stock_time_code' as table_name,
                    MIN(STR_TO_DATE(Date, '%Y-%m-%d')) as min_date,
                    MAX(STR_TO_DATE(Date, '%Y-%m-%d')) as max_date,
                    COUNT(DISTINCT Date) as unique_dates,
                    COUNT(*) as total_records
                FROM stock_time_code
                WHERE Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                  AND LENGTH(Date) = 10
                UNION ALL
                SELECT 
                    'stock_code_time' as table_name,
                    MIN(STR_TO_DATE(Date, '%Y-%m-%d')) as min_date,
                    MAX(STR_TO_DATE(Date, '%Y-%m-%d')) as max_date,
                    COUNT(DISTINCT Date) as unique_dates,
                    COUNT(*) as total_records
                FROM stock_code_time
                WHERE Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                  AND LENGTH(Date) = 10
            """)
            
            date_range_results = pd.read_sql(date_range_query, conn)
            logger.info("\n日期范围统计：")
            logger.info("\n" + str(date_range_results))
            
            # 检查每个表的记录总数
            total_records_query = text("""
                SELECT 
                    'stock_time_code' as table_name,
                    COUNT(*) as total_records
                FROM stock_time_code
                UNION ALL
                SELECT 
                    'stock_code_time' as table_name,
                    COUNT(*) as total_records
                FROM stock_code_time
            """)
            
            total_records_results = pd.read_sql(total_records_query, conn)
            logger.info("\n表记录总数统计：")
            logger.info("\n" + str(total_records_results))
            
    except Exception as e:
        logger.error(f"检查无效日期时出错: {str(e)}")
        raise

if __name__ == "__main__":
=======
import pandas as pd
from sqlalchemy import create_engine, text
import logging
from config.trading_config import default_config
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('check_invalid_dates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    "host": default_config.database.host,
    "port": default_config.database.port,
    "user": default_config.database.user,
    "password": default_config.database.password,
    "database": default_config.database.database
}

def check_invalid_dates():
    """检查数据库中的无效日期数据"""
    try:
        # 创建数据库连接
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        
        with engine.connect() as conn:
            # 临时修改 SQL 模式以允许无效日期
            conn.execute(text("SET SESSION sql_mode=''"))
            
            # 获取表结构
            table_info_query = text("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME IN ('stock_time_code', 'stock_code_time')
                ORDER BY TABLE_NAME, ORDINAL_POSITION
            """)
            
            table_info = pd.read_sql(table_info_query, conn)
            logger.info("\n表结构信息：")
            logger.info("\n" + str(table_info))
            
            # 获取无效日期记录的详细信息
            invalid_records_query = text("""
                SELECT 
                    t1.Code,
                    t1.Date as invalid_date,
                    t1.Open,
                    t1.High,
                    t1.Low,
                    t1.Close,
                    t1.Volume,
                    t2.Date as code_time_date,
                    t2.Open as code_time_open,
                    t2.High as code_time_high,
                    t2.Low as code_time_low,
                    t2.Close as code_time_close,
                    t2.Volume as code_time_volume
                FROM stock_time_code t1
                LEFT JOIN stock_code_time t2 
                    ON t1.Code = t2.Code 
                    AND t1.Date = t2.Date
                WHERE t1.Date = '0000-00-00'
                ORDER BY t1.Code
            """)
            
            invalid_records = pd.read_sql(invalid_records_query, conn)
            if not invalid_records.empty:
                logger.info("\n无效日期记录的详细信息：")
                logger.info("\n" + str(invalid_records))
                
                # 分析数据差异
                price_diff = invalid_records[
                    (invalid_records['Open'] != invalid_records['code_time_open']) |
                    (invalid_records['High'] != invalid_records['code_time_high']) |
                    (invalid_records['Low'] != invalid_records['code_time_low']) |
                    (invalid_records['Close'] != invalid_records['code_time_close']) |
                    (invalid_records['Volume'] != invalid_records['code_time_volume'])
                ]
                
                if not price_diff.empty:
                    logger.warning("\n发现与 stock_code_time 表数据不一致的记录：")
                    logger.warning("\n" + str(price_diff))
                else:
                    logger.info("\n所有无效日期记录在 stock_code_time 表中都有对应的数据")
                
                # 检查每个股票的前后日期
                date_context_query = text("""
                    WITH invalid_records AS (
                        SELECT Code, Date
                        FROM stock_time_code
                        WHERE Date = '0000-00-00'
                    )
                    SELECT 
                        i.Code,
                        i.Date as invalid_date,
                        MAX(CASE WHEN t.Date < '0000-00-00' THEN t.Date END) as prev_date,
                        MIN(CASE WHEN t.Date > '0000-00-00' THEN t.Date END) as next_date,
                        COUNT(*) as total_records
                    FROM invalid_records i
                    LEFT JOIN stock_time_code t ON i.Code = t.Code
                    GROUP BY i.Code, i.Date
                    ORDER BY i.Code
                """)
                
                date_context = pd.read_sql(date_context_query, conn)
                logger.info("\n无效日期记录的前后日期上下文：")
                logger.info("\n" + str(date_context))
            
            # 检查 stock_time_code 表
            time_code_query = text("""
                SELECT Code, Date, COUNT(*) as count
                FROM stock_time_code
                WHERE Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' 
                   OR Date = '0000-00-00'
                   OR Date IS NULL
                   OR LENGTH(Date) != 10
                GROUP BY Code, Date
                ORDER BY Code, Date
            """)
            
            # 检查 stock_code_time 表
            code_time_query = text("""
                SELECT Code, Date, COUNT(*) as count
                FROM stock_code_time
                WHERE Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                   OR Date = '0000-00-00'
                   OR Date IS NULL
                   OR LENGTH(Date) != 10
                GROUP BY Code, Date
                ORDER BY Code, Date
            """)
            
            # 检查 stock_time_code 表
            time_code_results = pd.read_sql(time_code_query, conn)
            if not time_code_results.empty:
                logger.warning("在 stock_time_code 表中发现无效日期数据：")
                logger.warning("\n" + str(time_code_results))
                logger.warning(f"总计发现 {len(time_code_results)} 条无效日期记录")
            else:
                logger.info("stock_time_code 表中没有无效日期数据")
            
            # 检查 stock_code_time 表
            code_time_results = pd.read_sql(code_time_query, conn)
            if not code_time_results.empty:
                logger.warning("在 stock_code_time 表中发现无效日期数据：")
                logger.warning("\n" + str(code_time_results))
                logger.warning(f"总计发现 {len(code_time_results)} 条无效日期记录")
            else:
                logger.info("stock_code_time 表中没有无效日期数据")
            
            # 检查日期范围（只检查有效日期）
            date_range_query = text("""
                SELECT 
                    'stock_time_code' as table_name,
                    MIN(STR_TO_DATE(Date, '%Y-%m-%d')) as min_date,
                    MAX(STR_TO_DATE(Date, '%Y-%m-%d')) as max_date,
                    COUNT(DISTINCT Date) as unique_dates,
                    COUNT(*) as total_records
                FROM stock_time_code
                WHERE Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                  AND LENGTH(Date) = 10
                UNION ALL
                SELECT 
                    'stock_code_time' as table_name,
                    MIN(STR_TO_DATE(Date, '%Y-%m-%d')) as min_date,
                    MAX(STR_TO_DATE(Date, '%Y-%m-%d')) as max_date,
                    COUNT(DISTINCT Date) as unique_dates,
                    COUNT(*) as total_records
                FROM stock_code_time
                WHERE Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                  AND LENGTH(Date) = 10
            """)
            
            date_range_results = pd.read_sql(date_range_query, conn)
            logger.info("\n日期范围统计：")
            logger.info("\n" + str(date_range_results))
            
            # 检查每个表的记录总数
            total_records_query = text("""
                SELECT 
                    'stock_time_code' as table_name,
                    COUNT(*) as total_records
                FROM stock_time_code
                UNION ALL
                SELECT 
                    'stock_code_time' as table_name,
                    COUNT(*) as total_records
                FROM stock_code_time
            """)
            
            total_records_results = pd.read_sql(total_records_query, conn)
            logger.info("\n表记录总数统计：")
            logger.info("\n" + str(total_records_results))
            
    except Exception as e:
        logger.error(f"检查无效日期时出错: {str(e)}")
        raise

if __name__ == "__main__":
>>>>>>> 3d7330be7ea0ecb409ac485e1c8391bc6d56a2de
    check_invalid_dates() 