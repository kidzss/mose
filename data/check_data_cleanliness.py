from data.data_updater import DB_CONFIG
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime, timedelta
import logging
import pandas_market_calendars as mcal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_cleanliness_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_data_cleanliness():
    """详细检查数据清洗状态"""
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    # 获取NYSE日历
    nyse = mcal.get_calendar('NYSE')
    
    try:
        with engine.connect() as conn:
            # 1. 检查数据覆盖情况
            query = text("""
                SELECT 
                    COUNT(DISTINCT Code) as total_stocks,
                    COUNT(DISTINCT Date) as total_dates,
                    MIN(Date) as earliest_date,
                    MAX(Date) as latest_date,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN Volume > 0 THEN 1 END) as valid_records,
                    COUNT(CASE WHEN Volume = 0 THEN 1 END) as zero_volume_records
                FROM stock_time_code
            """)
            
            coverage = pd.read_sql(query, conn).iloc[0]
            logger.info("\n=== 数据覆盖情况 ===")
            logger.info(f"总股票数: {coverage['total_stocks']}")
            logger.info(f"总交易日数: {coverage['total_dates']}")
            logger.info(f"数据时间范围: {coverage['earliest_date']} 至 {coverage['latest_date']}")
            logger.info(f"总记录数: {coverage['total_records']}")
            logger.info(f"有效交易记录数: {coverage['valid_records']}")
            logger.info(f"零成交量记录数: {coverage['zero_volume_records']}")
            
            # 2. 检查数据质量问题
            query = text("""
                SELECT 
                    COUNT(*) as total_issues,
                    SUM(CASE WHEN High < Low THEN 1 ELSE 0 END) as high_low_issues,
                    SUM(CASE WHEN Open > High OR Open < Low THEN 1 ELSE 0 END) as open_range_issues,
                    SUM(CASE WHEN Close > High OR Close < Low THEN 1 ELSE 0 END) as close_range_issues,
                    SUM(CASE WHEN Volume < 0 THEN 1 ELSE 0 END) as negative_volume_issues
                FROM stock_time_code
            """)
            
            quality = pd.read_sql(query, conn).iloc[0]
            logger.info("\n=== 数据质量问题 ===")
            logger.info(f"总问题数: {quality['total_issues']}")
            logger.info(f"最高价低于最低价: {quality['high_low_issues']}")
            logger.info(f"开盘价超出范围: {quality['open_range_issues']}")
            logger.info(f"收盘价超出范围: {quality['close_range_issues']}")
            logger.info(f"负成交量: {quality['negative_volume_issues']}")
            
            # 3. 检查表一致性
            query = text("""
                SELECT 
                    COUNT(*) as inconsistent_stocks
                FROM (
                    SELECT t1.Code
                    FROM stock_time_code t1
                    LEFT JOIN stock_code_time t2 
                    ON t1.Code = t2.Code AND t1.Date = t2.Date
                    WHERE t2.Code IS NULL
                    UNION
                    SELECT t2.Code
                    FROM stock_code_time t2
                    LEFT JOIN stock_time_code t1 
                    ON t1.Code = t2.Code AND t1.Date = t2.Date
                    WHERE t1.Code IS NULL
                ) as diff
            """)
            
            consistency = pd.read_sql(query, conn).iloc[0]
            logger.info("\n=== 表一致性检查 ===")
            logger.info(f"不一致的股票数: {consistency['inconsistent_stocks']}")
            
            # 4. 检查最近30天的数据质量
            query = text("""
                SELECT 
                    Date,
                    COUNT(DISTINCT Code) as stocks_count,
                    COUNT(CASE WHEN Volume > 0 THEN 1 END) as trading_records,
                    COUNT(CASE WHEN Volume = 0 THEN 1 END) as non_trading_records,
                    COUNT(CASE WHEN High < Low OR Open > High OR Open < Low OR Close > High OR Close < Low THEN 1 END) as price_issues
                FROM stock_time_code
                WHERE Date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                GROUP BY Date
                ORDER BY Date DESC
            """)
            
            recent_quality = pd.read_sql(query, conn)
            logger.info("\n=== 最近30天数据质量 ===")
            logger.info("\n" + str(recent_quality))
            
            # 5. 检查缺失日期
            query = text("""
                SELECT Code, COUNT(*) as missing_dates
                FROM (
                    SELECT t1.Code, t1.Date
                    FROM stock_time_code t1
                    WHERE t1.Date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                    AND NOT EXISTS (
                        SELECT 1 
                        FROM stock_time_code t2 
                        WHERE t2.Code = t1.Code 
                        AND t2.Date = DATE_ADD(t1.Date, INTERVAL 1 DAY)
                    )
                ) as gaps
                GROUP BY Code
                HAVING missing_dates > 0
                ORDER BY missing_dates DESC
                LIMIT 10
            """)
            
            missing_dates = pd.read_sql(query, conn)
            logger.info("\n=== 最近30天缺失日期最多的股票 ===")
            logger.info("\n" + str(missing_dates))
            
            # 6. 检查数据连续性
            query = text("""
                SELECT 
                    Code,
                    COUNT(DISTINCT Date) as actual_dates,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN Volume > 0 THEN 1 END) as trading_days,
                    MIN(Date) as first_date,
                    MAX(Date) as last_date
                FROM stock_time_code
                GROUP BY Code
                HAVING actual_dates != DATEDIFF(MAX(Date), MIN(Date)) + 1
                ORDER BY (DATEDIFF(MAX(Date), MIN(Date)) + 1 - COUNT(DISTINCT Date)) DESC
                LIMIT 10
            """)
            
            continuity = pd.read_sql(query, conn)
            logger.info("\n=== 数据连续性检查（前10只不连续的股票）===")
            logger.info("\n" + str(continuity))
            
    except Exception as e:
        logger.error(f"检查数据清洗状态时出错: {str(e)}")

if __name__ == "__main__":
    check_data_cleanliness() 