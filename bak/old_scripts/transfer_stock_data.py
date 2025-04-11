import logging
from sqlalchemy import create_engine, text
from config.trading_config import default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transfer_stock.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def transfer_stock_data():
    """临时脚本：将stock_time_code的数据迁移到stock_code_time"""
    try:
        # 创建数据库连接
        db_config = default_config.database
        engine = create_engine(
            f"mysql+pymysql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
        )
        
        with engine.connect() as conn:
            # 1. 首先检查源表中的数据
            check_source_query = """
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(DISTINCT Code) as stock_count
                FROM stock_time_code
            """
            result = conn.execute(text(check_source_query))
            stats = result.fetchone()
            logger.info(f"源表统计:")
            logger.info(f"总记录数: {stats[0]}")
            logger.info(f"股票数量: {stats[1]}")
            
            # 检查无效日期数量
            invalid_date_query = """
                SELECT COUNT(*) as invalid_count
                FROM stock_time_code 
                WHERE CHAR_LENGTH(Date) != 10 
                   OR Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
            """
            result = conn.execute(text(invalid_date_query))
            invalid_count = result.fetchone()[0]
            logger.info(f"无效日期数: {invalid_count}")
            
            # 2. 检查日期格式问题
            check_date_query = """
                SELECT DISTINCT Date 
                FROM stock_time_code 
                WHERE CHAR_LENGTH(Date) != 10 
                   OR Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                LIMIT 5
            """
            result = conn.execute(text(check_date_query))
            invalid_dates = result.fetchall()
            if invalid_dates:
                logger.warning("发现异常日期格式示例:")
                for date in invalid_dates:
                    logger.warning(f"异常日期: {date[0]}")
            
            # 3. 清空目标表
            logger.info("清空 stock_code_time 表...")
            conn.execute(text("TRUNCATE TABLE stock_code_time"))
            
            # 4. 转移数据
            logger.info("开始转移数据...")
            transfer_query = """
                INSERT INTO stock_code_time (Code, Date, Open, High, Low, Close, Volume)
                SELECT 
                    Code,
                    Date,
                    Open,
                    High,
                    Low,
                    Close,
                    Volume
                FROM stock_time_code 
                WHERE CHAR_LENGTH(Date) = 10 
                  AND Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                  AND Date != '0000-00-00'
                ORDER BY Code, Date
            """
            conn.execute(text(transfer_query))
            conn.commit()
            
            # 5. 验证转移结果
            verify_query = """
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(DISTINCT Code) as stock_count,
                    MIN(Date) as earliest_date,
                    MAX(Date) as latest_date
                FROM stock_code_time
            """
            result = conn.execute(text(verify_query))
            verify = result.fetchone()
            logger.info(f"\n转移结果统计:")
            logger.info(f"成功转移记录数: {verify[0]}")
            logger.info(f"股票数量: {verify[1]}")
            logger.info(f"数据日期范围: {verify[2]} 到 {verify[3]}")
            
    except Exception as e:
        logger.error(f"数据转移过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        transfer_stock_data()
        logger.info("数据转移完成")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}") 