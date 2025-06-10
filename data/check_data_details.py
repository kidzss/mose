from data.data_updater import DB_CONFIG
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_details_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_data_details():
    """检查数据库中的具体数据情况"""
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    try:
        with engine.connect() as conn:
            # 1. 检查最近30天的数据情况
            query = text("""
                SELECT 
                    Date,
                    COUNT(DISTINCT Code) as stocks_count,
                    COUNT(CASE WHEN Volume > 0 THEN 1 END) as trading_records,
                    COUNT(CASE WHEN Volume = 0 THEN 1 END) as non_trading_records
                FROM stock_time_code
                WHERE Date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                GROUP BY Date
                ORDER BY Date DESC
            """)
            
            recent_data = pd.read_sql(query, conn)
            logger.info("\n最近30天数据情况:")
            logger.info("\n" + str(recent_data))
            
            # 2. 检查数据完整性
            query = text("""
                SELECT 
                    Code,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN Volume > 0 THEN 1 END) as trading_days,
                    MIN(Date) as first_date,
                    MAX(Date) as last_date,
                    COUNT(DISTINCT Date) as unique_dates
                FROM stock_time_code
                GROUP BY Code
                ORDER BY trading_days DESC
                LIMIT 10
            """)
            
            completeness = pd.read_sql(query, conn)
            logger.info("\n数据完整性检查 (前10只股票):")
            logger.info("\n" + str(completeness))
            
            # 3. 检查数据异常
            query = text("""
                SELECT 
                    Code,
                    Date,
                    Open,
                    High,
                    Low,
                    Close,
                    Volume
                FROM stock_time_code
                WHERE 
                    (High < Low) OR
                    (Open > High) OR
                    (Open < Low) OR
                    (Close > High) OR
                    (Close < Low) OR
                    (Volume < 0)
                ORDER BY Code, Date
                LIMIT 10
            """)
            
            anomalies = pd.read_sql(query, conn)
            logger.info("\n数据异常检查 (前10条记录):")
            logger.info("\n" + str(anomalies))
            
            # 4. 检查表一致性
            query = text("""
                SELECT 
                    t1.Code,
                    COUNT(*) as time_code_count,
                    (SELECT COUNT(*) FROM stock_code_time t2 WHERE t2.Code = t1.Code) as code_time_count
                FROM stock_time_code t1
                GROUP BY t1.Code
                HAVING time_code_count != code_time_count
                LIMIT 10
            """)
            
            consistency = pd.read_sql(query, conn)
            logger.info("\n表一致性检查 (前10条不一致记录):")
            logger.info("\n" + str(consistency))
            
            # 5. 检查数据分布
            query = text("""
                SELECT 
                    YEAR(Date) as year,
                    COUNT(DISTINCT Code) as stocks_count,
                    COUNT(DISTINCT Date) as trading_days,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN Volume > 0 THEN 1 END) as valid_records
                FROM stock_time_code
                GROUP BY YEAR(Date)
                ORDER BY year
            """)
            
            distribution = pd.read_sql(query, conn)
            logger.info("\n数据分布情况 (按年份):")
            logger.info("\n" + str(distribution))
            
    except Exception as e:
        logger.error(f"检查数据时出错: {str(e)}")

if __name__ == "__main__":
    check_data_details() 