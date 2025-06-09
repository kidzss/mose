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
        logging.FileHandler(f'stock_time_code_cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_stock_time_code():
    """清理和优化stock_time_code表"""
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    try:
        with engine.connect() as conn:
            # 保存当前的SQL模式
            result = conn.execute(text("SELECT @@SESSION.sql_mode"))
            original_mode = result.scalar()
            
            try:
                # 临时关闭严格模式
                logger.info("临时关闭严格模式...")
                conn.execute(text("SET SESSION sql_mode=''"))
                
                # 1. 删除无效日期数据
                logger.info("开始清理无效日期数据...")
                conn.execute(text("""
                    DELETE FROM stock_time_code 
                    WHERE Date = '0000-00-00' 
                    OR Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                """))
                conn.commit()
                logger.info("无效日期数据清理完成")
                
                # 2. 修正价格异常数据
                logger.info("开始修正价格异常数据...")
                conn.execute(text("""
                    UPDATE stock_time_code
                    SET Open = CASE 
                        WHEN Open > High THEN High
                        WHEN Open < Low THEN Low
                        ELSE Open
                    END,
                    Close = CASE 
                        WHEN Close > High THEN High
                        WHEN Close < Low THEN Low
                        ELSE Close
                    END
                    WHERE Open > High OR Open < Low OR Close > High OR Close < Low
                """))
                conn.commit()
                logger.info("价格异常数据修正完成")
                
                # 3. 删除重复记录
                logger.info("开始删除重复记录...")
                # 只保留每组(Code, Date)的第一条
                conn.execute(text("""
                    DELETE FROM stock_time_code
                    WHERE (Code, Date) NOT IN (
                        SELECT Code, Date FROM (
                            SELECT Code, Date, ROW_NUMBER() OVER (PARTITION BY Code, Date ORDER BY Code) as rn
                            FROM stock_time_code
                        ) t WHERE t.rn = 1
                    )
                """))
                conn.commit()
                logger.info("重复记录删除完成")
                
                # 4. 创建优化索引
                logger.info("开始创建优化索引...")
                try:
                    conn.execute(text("""
                        CREATE INDEX idx_date_code ON stock_time_code (Date, Code)
                    """))
                    conn.execute(text("""
                        CREATE INDEX idx_code_date ON stock_time_code (Code, Date)
                    """))
                    logger.info("索引创建完成")
                except Exception as e:
                    logger.warning(f"创建索引时出错（可能已存在）: {str(e)}")
                
            finally:
                # 恢复原来的SQL模式
                logger.info("恢复SQL模式...")
                conn.execute(text(f"SET SESSION sql_mode='{original_mode}'"))
            
            # 5. 验证数据质量
            logger.info("\n=== 数据质量验证 ===")
            
            # 5.1 检查数据覆盖情况
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
            logger.info("\n数据覆盖情况:")
            logger.info(f"总股票数: {coverage['total_stocks']}")
            logger.info(f"总交易日数: {coverage['total_dates']}")
            logger.info(f"数据时间范围: {coverage['earliest_date']} 至 {coverage['latest_date']}")
            logger.info(f"总记录数: {coverage['total_records']}")
            logger.info(f"有效交易记录数: {coverage['valid_records']}")
            logger.info(f"零成交量记录数: {coverage['zero_volume_records']}")
            
            # 5.2 检查数据质量问题
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
            logger.info("\n数据质量问题:")
            logger.info(f"总问题数: {quality['total_issues']}")
            logger.info(f"最高价低于最低价: {quality['high_low_issues']}")
            logger.info(f"开盘价超出范围: {quality['open_range_issues']}")
            logger.info(f"收盘价超出范围: {quality['close_range_issues']}")
            logger.info(f"负成交量: {quality['negative_volume_issues']}")
            
            # 5.3 检查最近30天的数据质量
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
            logger.info("\n最近30天数据质量:")
            logger.info("\n" + str(recent_quality))
            
            logger.info("\n=== 数据清理和优化完成 ===")
            
    except Exception as e:
        logger.error(f"清理数据时出错: {str(e)}")
        raise

if __name__ == "__main__":
    clean_stock_time_code() 