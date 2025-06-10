import logging
from sqlalchemy import create_engine, text
from config.trading_config import default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('delete_invalid_dates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": default_config.database.host,
    "port": default_config.database.port,
    "user": default_config.database.user,
    "password": default_config.database.password,
    "database": default_config.database.database
}

def delete_invalid_dates():
    try:
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        with engine.connect() as conn:
            conn.execute(text("SET SESSION sql_mode=''"))
            # 先统计要删除的数量
            count_query = text("SELECT COUNT(*) FROM stock_time_code WHERE Date = '0000-00-00'")
            result = conn.execute(count_query)
            count = result.scalar()
            if count == 0:
                logger.info("没有需要删除的无效日期记录。")
                return
            logger.info(f"即将删除 {count} 条无效日期记录...")
            # 执行删除
            delete_query = text("DELETE FROM stock_time_code WHERE Date = '0000-00-00'")
            conn.execute(delete_query)
            logger.info(f"已成功删除 {count} 条无效日期记录。")
    except Exception as e:
        logger.error(f"删除无效日期时出错: {str(e)}")
        raise

if __name__ == "__main__":
    delete_invalid_dates() 