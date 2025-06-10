import pymysql
import logging
import os
import time
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('import_backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def import_backup():
    try:
        # 连接到 MySQL
        logger.info("Connecting to MySQL...")
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='mose'
        )
        cursor = conn.cursor()
        
        # 获取备份文件大小
        backup_file = 'mose_backup.sql'
        file_size = os.path.getsize(backup_file)
        logger.info(f"Backup file size: {file_size / (1024*1024):.2f} MB")
        
        # 读取并执行 SQL 文件
        logger.info("Starting import...")
        start_time = time.time()
        
        with open(backup_file, 'r', encoding='utf-8') as f:
            # 读取所有 SQL 语句
            sql_commands = f.read().split(';')
            
            # 使用 tqdm 显示进度
            for cmd in tqdm(sql_commands, desc="Importing SQL"):
                if cmd.strip():
                    try:
                        cursor.execute(cmd)
                        conn.commit()
                    except pymysql.Error as e:
                        logger.error(f"Error executing command: {e}")
                        logger.error(f"Problematic command: {cmd[:200]}...")  # 只显示前200个字符
                        continue
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Import completed in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during import: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    logger.info("Starting backup import process...")
    import_backup()
    logger.info("Backup import process completed!") 