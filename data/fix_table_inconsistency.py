<<<<<<< HEAD
import logging
from datetime import datetime
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.data_updater import DatabaseManager, DB_CONFIG
from sqlalchemy import text

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_invalid_data(db_manager):
    """清理无效的日期数据"""
    try:
        with db_manager.engine.connect() as conn:
            # 保存当前的SQL模式
            result = conn.execute(text("SELECT @@SESSION.sql_mode"))
            original_mode = result.scalar()
            
            try:
                # 临时关闭严格模式，允许处理无效日期
                conn.execute(text("SET SESSION sql_mode=''"))
                
                # 只删除日期格式不正确的记录，避免使用'0000-00-00'字面值
                clean_query = """
                    DELETE FROM stock_time_code
                    WHERE Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' 
                    OR Date < '1900-01-01'
                """
                conn.execute(text(clean_query))
                conn.commit()
                logger.info("已清理无效日期数据")
            finally:
                # 恢复原来的SQL模式
                conn.execute(text(f"SET SESSION sql_mode='{original_mode}'"))
            
            return True
    except Exception as e:
        logger.error(f"清理无效数据时出错: {str(e)}")
        return False

def get_stock_codes(db_manager):
    """获取所有股票代码"""
    try:
        with db_manager.engine.connect() as conn:
            query = text("SELECT DISTINCT Code FROM stock_time_code")
            result = conn.execute(query)
            stock_codes = [row[0] for row in result]
            logger.info(f"共获取到 {len(stock_codes)} 只股票")
            return stock_codes
    except Exception as e:
        logger.error(f"获取股票代码时出错: {str(e)}")
        return []

def fix_inconsistency_for_stock(db_manager, stock_code):
    """修复单个股票的数据不一致"""
    try:
        # 1. 将stock_time_code中有而stock_code_time中没有的记录插入stock_code_time
        sync_to_code_time_query = """
            INSERT IGNORE INTO stock_code_time 
            (Code, Date, Open, High, Low, Close, Volume, Amount, AdjClose, Dividends, StockSplits, Capital_Gains)
            SELECT 
                t.Code, t.Date, t.Open, t.High, t.Low, t.Close, t.Volume, 
                t.Amount, t.AdjClose, t.Dividends, t.StockSplits, t.Capital_Gains
            FROM stock_time_code t
            LEFT JOIN stock_code_time c ON t.Code = c.Code AND t.Date = c.Date
            WHERE c.Code IS NULL AND t.Code = :stock_code
        """
        
        # 2. 将stock_code_time中有而stock_time_code中没有的记录插入stock_time_code
        sync_to_time_code_query = """
            INSERT IGNORE INTO stock_time_code
            (Date, Code, Open, High, Low, Close, Volume, Amount, AdjClose, Dividends, StockSplits, Capital_Gains)
            SELECT 
                c.Date, c.Code, c.Open, c.High, c.Low, c.Close, c.Volume, 
                c.Amount, c.AdjClose, c.Dividends, c.StockSplits, c.Capital_Gains
            FROM stock_code_time c
            LEFT JOIN stock_time_code t ON c.Code = t.Code AND c.Date = c.Date
            WHERE t.Code IS NULL AND c.Code = :stock_code
        """
        
        # 3. 更新两表之间存在但数据不一致的记录
        update_inconsistent_records_query = """
            UPDATE stock_code_time c
            JOIN stock_time_code t ON c.Code = t.Code AND c.Date = t.Date
            SET 
                c.Open = t.Open,
                c.High = t.High,
                c.Low = t.Low,
                c.Close = t.Close,
                c.Volume = t.Volume,
                c.Amount = t.Amount,
                c.AdjClose = t.AdjClose,
                c.Dividends = t.Dividends,
                c.StockSplits = t.StockSplits,
                c.Capital_Gains = t.Capital_Gains
            WHERE c.Code = :stock_code AND (
                c.Open != t.Open OR
                c.High != t.High OR
                c.Low != t.Low OR
                c.Close != t.Close OR
                c.Volume != t.Volume OR
                c.Amount != t.Amount OR
                (c.AdjClose IS NULL AND t.AdjClose IS NOT NULL) OR
                (c.AdjClose IS NOT NULL AND t.AdjClose IS NULL) OR
                (c.AdjClose != t.AdjClose AND c.AdjClose IS NOT NULL AND t.AdjClose IS NOT NULL) OR
                (c.Dividends != t.Dividends) OR
                (c.StockSplits != t.StockSplits) OR
                (c.Capital_Gains != t.Capital_Gains AND c.Capital_Gains IS NOT NULL AND t.Capital_Gains IS NOT NULL)
            )
        """
        
        with db_manager.engine.connect() as conn:
            # 执行同步
            result = conn.execute(text(sync_to_code_time_query), {"stock_code": stock_code})
            to_code_time_inserted = result.rowcount
            conn.commit()
            
            result = conn.execute(text(sync_to_time_code_query), {"stock_code": stock_code})
            to_time_code_inserted = result.rowcount
            conn.commit()
            
            result = conn.execute(text(update_inconsistent_records_query), {"stock_code": stock_code})
            rows_updated = result.rowcount
            conn.commit()
            
            return to_code_time_inserted, to_time_code_inserted, rows_updated
    except Exception as e:
        logger.error(f"修复股票 {stock_code} 的数据不一致时出错: {str(e)}")
        return 0, 0, 0

def check_inconsistency_for_stock(db_manager, stock_code):
    """检查单个股票的数据不一致数量"""
    try:
        check_query = """
            SELECT COUNT(*) AS inconsistent_count
            FROM stock_time_code t
            JOIN stock_code_time c ON t.Code = c.Code AND t.Date = c.Date
            WHERE t.Code = :stock_code AND (
                t.Open != c.Open OR
                t.High != c.High OR
                t.Low != c.Low OR
                t.Close != c.Close OR
                t.Volume != c.Volume OR
                t.Amount != c.Amount OR
                (t.AdjClose IS NULL AND c.AdjClose IS NOT NULL) OR
                (t.AdjClose IS NOT NULL AND c.AdjClose IS NULL) OR
                (t.AdjClose != c.AdjClose AND t.AdjClose IS NOT NULL AND c.AdjClose IS NOT NULL) OR
                (t.Dividends != c.Dividends) OR
                (t.StockSplits != c.StockSplits) OR
                (t.Capital_Gains != c.Capital_Gains AND t.Capital_Gains IS NOT NULL AND c.Capital_Gains IS NOT NULL)
            )
        """
        
        with db_manager.engine.connect() as conn:
            result = conn.execute(text(check_query), {"stock_code": stock_code})
            inconsistent_count = result.scalar()
            return inconsistent_count
    except Exception as e:
        logger.error(f"检查股票 {stock_code} 的数据不一致时出错: {str(e)}")
        return None

def main():
    """
    主函数：修复stock_time_code和stock_code_time表之间的数据不一致问题
    """
    logger.info("开始修复表之间的数据不一致...")
    
    try:
        # 创建数据库管理器
        db_manager = DatabaseManager(DB_CONFIG)
        
        # 1. 清理无效数据
        if not clean_invalid_data(db_manager):
            logger.error("清理无效数据失败")
            return
        
        # 2. 获取所有股票代码
        stock_codes = get_stock_codes(db_manager)
        if not stock_codes:
            logger.error("无法获取股票代码")
            return
        
        # 3. 按股票代码分批修复
        total_to_code_time = 0
        total_to_time_code = 0
        total_updated = 0
        stocks_with_issues = 0
        
        for i, stock_code in enumerate(stock_codes):
            # 检查不一致数量
            before_count = check_inconsistency_for_stock(db_manager, stock_code)
            
            if before_count is None:
                logger.error(f"无法检查股票 {stock_code} 的数据不一致")
                continue
                
            if before_count > 0:
                logger.info(f"处理股票 [{i+1}/{len(stock_codes)}] {stock_code} - 不一致记录数: {before_count}")
                
                # 修复不一致
                to_code_time, to_time_code, updated = fix_inconsistency_for_stock(db_manager, stock_code)
                total_to_code_time += to_code_time
                total_to_time_code += to_time_code
                total_updated += updated
                
                # 再次检查不一致
                after_count = check_inconsistency_for_stock(db_manager, stock_code)
                
                if after_count is None:
                    logger.error(f"无法检查股票 {stock_code} 修复后的数据不一致")
                    continue
                    
                if after_count > 0:
                    logger.warning(f"股票 {stock_code} 修复后仍有 {after_count} 条不一致记录")
                    stocks_with_issues += 1
                else:
                    logger.info(f"股票 {stock_code} 数据已修复一致")
            
            # 每处理10只股票输出一次统计信息
            if (i + 1) % 10 == 0:
                logger.info(f"已处理 {i+1}/{len(stock_codes)} 只股票，"
                           f"新增到code_time: {total_to_code_time}，"
                           f"新增到time_code: {total_to_time_code}，"
                           f"更新: {total_updated}，"
                           f"仍有问题的股票: {stocks_with_issues}")
        
        # 4. 输出最终统计信息
        logger.info(f"修复完成！共处理 {len(stock_codes)} 只股票")
        logger.info(f"新增到stock_code_time: {total_to_code_time} 条记录")
        logger.info(f"新增到stock_time_code: {total_to_time_code} 条记录")
        logger.info(f"更新: {total_updated} 条记录")
        logger.info(f"仍有问题的股票: {stocks_with_issues} 只")
        
        if stocks_with_issues == 0:
            logger.info("修复成功！两个表数据现在完全一致")
        else:
            logger.warning(f"修复后仍有 {stocks_with_issues} 只股票数据不一致，可能需要进一步排查")
        
    except Exception as e:
        logger.error(f"修复过程中发生错误: {str(e)}")

if __name__ == "__main__":
=======
import logging
from datetime import datetime
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.data_updater import DatabaseManager, DB_CONFIG
from sqlalchemy import text

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_invalid_data(db_manager):
    """清理无效的日期数据"""
    try:
        with db_manager.engine.connect() as conn:
            # 保存当前的SQL模式
            result = conn.execute(text("SELECT @@SESSION.sql_mode"))
            original_mode = result.scalar()
            
            try:
                # 临时关闭严格模式，允许处理无效日期
                conn.execute(text("SET SESSION sql_mode=''"))
                
                # 只删除日期格式不正确的记录，避免使用'0000-00-00'字面值
                clean_query = """
                    DELETE FROM stock_time_code
                    WHERE Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' 
                    OR Date < '1900-01-01'
                """
                conn.execute(text(clean_query))
                conn.commit()
                logger.info("已清理无效日期数据")
            finally:
                # 恢复原来的SQL模式
                conn.execute(text(f"SET SESSION sql_mode='{original_mode}'"))
            
            return True
    except Exception as e:
        logger.error(f"清理无效数据时出错: {str(e)}")
        return False

def get_stock_codes(db_manager):
    """获取所有股票代码"""
    try:
        with db_manager.engine.connect() as conn:
            query = text("SELECT DISTINCT Code FROM stock_time_code")
            result = conn.execute(query)
            stock_codes = [row[0] for row in result]
            logger.info(f"共获取到 {len(stock_codes)} 只股票")
            return stock_codes
    except Exception as e:
        logger.error(f"获取股票代码时出错: {str(e)}")
        return []

def fix_inconsistency_for_stock(db_manager, stock_code):
    """修复单个股票的数据不一致"""
    try:
        # 1. 将stock_time_code中有而stock_code_time中没有的记录插入stock_code_time
        sync_to_code_time_query = """
            INSERT IGNORE INTO stock_code_time 
            (Code, Date, Open, High, Low, Close, Volume, Amount, AdjClose, Dividends, StockSplits, Capital_Gains)
            SELECT 
                t.Code, t.Date, t.Open, t.High, t.Low, t.Close, t.Volume, 
                t.Amount, t.AdjClose, t.Dividends, t.StockSplits, t.Capital_Gains
            FROM stock_time_code t
            LEFT JOIN stock_code_time c ON t.Code = c.Code AND t.Date = c.Date
            WHERE c.Code IS NULL AND t.Code = :stock_code
        """
        
        # 2. 将stock_code_time中有而stock_time_code中没有的记录插入stock_time_code
        sync_to_time_code_query = """
            INSERT IGNORE INTO stock_time_code
            (Date, Code, Open, High, Low, Close, Volume, Amount, AdjClose, Dividends, StockSplits, Capital_Gains)
            SELECT 
                c.Date, c.Code, c.Open, c.High, c.Low, c.Close, c.Volume, 
                c.Amount, c.AdjClose, c.Dividends, c.StockSplits, c.Capital_Gains
            FROM stock_code_time c
            LEFT JOIN stock_time_code t ON c.Code = t.Code AND c.Date = c.Date
            WHERE t.Code IS NULL AND c.Code = :stock_code
        """
        
        # 3. 更新两表之间存在但数据不一致的记录
        update_inconsistent_records_query = """
            UPDATE stock_code_time c
            JOIN stock_time_code t ON c.Code = t.Code AND c.Date = t.Date
            SET 
                c.Open = t.Open,
                c.High = t.High,
                c.Low = t.Low,
                c.Close = t.Close,
                c.Volume = t.Volume,
                c.Amount = t.Amount,
                c.AdjClose = t.AdjClose,
                c.Dividends = t.Dividends,
                c.StockSplits = t.StockSplits,
                c.Capital_Gains = t.Capital_Gains
            WHERE c.Code = :stock_code AND (
                c.Open != t.Open OR
                c.High != t.High OR
                c.Low != t.Low OR
                c.Close != t.Close OR
                c.Volume != t.Volume OR
                c.Amount != t.Amount OR
                (c.AdjClose IS NULL AND t.AdjClose IS NOT NULL) OR
                (c.AdjClose IS NOT NULL AND t.AdjClose IS NULL) OR
                (c.AdjClose != t.AdjClose AND c.AdjClose IS NOT NULL AND t.AdjClose IS NOT NULL) OR
                (c.Dividends != t.Dividends) OR
                (c.StockSplits != t.StockSplits) OR
                (c.Capital_Gains != t.Capital_Gains AND c.Capital_Gains IS NOT NULL AND t.Capital_Gains IS NOT NULL)
            )
        """
        
        with db_manager.engine.connect() as conn:
            # 执行同步
            result = conn.execute(text(sync_to_code_time_query), {"stock_code": stock_code})
            to_code_time_inserted = result.rowcount
            conn.commit()
            
            result = conn.execute(text(sync_to_time_code_query), {"stock_code": stock_code})
            to_time_code_inserted = result.rowcount
            conn.commit()
            
            result = conn.execute(text(update_inconsistent_records_query), {"stock_code": stock_code})
            rows_updated = result.rowcount
            conn.commit()
            
            return to_code_time_inserted, to_time_code_inserted, rows_updated
    except Exception as e:
        logger.error(f"修复股票 {stock_code} 的数据不一致时出错: {str(e)}")
        return 0, 0, 0

def check_inconsistency_for_stock(db_manager, stock_code):
    """检查单个股票的数据不一致数量"""
    try:
        check_query = """
            SELECT COUNT(*) AS inconsistent_count
            FROM stock_time_code t
            JOIN stock_code_time c ON t.Code = c.Code AND t.Date = c.Date
            WHERE t.Code = :stock_code AND (
                t.Open != c.Open OR
                t.High != c.High OR
                t.Low != c.Low OR
                t.Close != c.Close OR
                t.Volume != c.Volume OR
                t.Amount != c.Amount OR
                (t.AdjClose IS NULL AND c.AdjClose IS NOT NULL) OR
                (t.AdjClose IS NOT NULL AND c.AdjClose IS NULL) OR
                (t.AdjClose != c.AdjClose AND t.AdjClose IS NOT NULL AND c.AdjClose IS NOT NULL) OR
                (t.Dividends != c.Dividends) OR
                (t.StockSplits != c.StockSplits) OR
                (t.Capital_Gains != c.Capital_Gains AND t.Capital_Gains IS NOT NULL AND c.Capital_Gains IS NOT NULL)
            )
        """
        
        with db_manager.engine.connect() as conn:
            result = conn.execute(text(check_query), {"stock_code": stock_code})
            inconsistent_count = result.scalar()
            return inconsistent_count
    except Exception as e:
        logger.error(f"检查股票 {stock_code} 的数据不一致时出错: {str(e)}")
        return None

def main():
    """
    主函数：修复stock_time_code和stock_code_time表之间的数据不一致问题
    """
    logger.info("开始修复表之间的数据不一致...")
    
    try:
        # 创建数据库管理器
        db_manager = DatabaseManager(DB_CONFIG)
        
        # 1. 清理无效数据
        if not clean_invalid_data(db_manager):
            logger.error("清理无效数据失败")
            return
        
        # 2. 获取所有股票代码
        stock_codes = get_stock_codes(db_manager)
        if not stock_codes:
            logger.error("无法获取股票代码")
            return
        
        # 3. 按股票代码分批修复
        total_to_code_time = 0
        total_to_time_code = 0
        total_updated = 0
        stocks_with_issues = 0
        
        for i, stock_code in enumerate(stock_codes):
            # 检查不一致数量
            before_count = check_inconsistency_for_stock(db_manager, stock_code)
            
            if before_count is None:
                logger.error(f"无法检查股票 {stock_code} 的数据不一致")
                continue
                
            if before_count > 0:
                logger.info(f"处理股票 [{i+1}/{len(stock_codes)}] {stock_code} - 不一致记录数: {before_count}")
                
                # 修复不一致
                to_code_time, to_time_code, updated = fix_inconsistency_for_stock(db_manager, stock_code)
                total_to_code_time += to_code_time
                total_to_time_code += to_time_code
                total_updated += updated
                
                # 再次检查不一致
                after_count = check_inconsistency_for_stock(db_manager, stock_code)
                
                if after_count is None:
                    logger.error(f"无法检查股票 {stock_code} 修复后的数据不一致")
                    continue
                    
                if after_count > 0:
                    logger.warning(f"股票 {stock_code} 修复后仍有 {after_count} 条不一致记录")
                    stocks_with_issues += 1
                else:
                    logger.info(f"股票 {stock_code} 数据已修复一致")
            
            # 每处理10只股票输出一次统计信息
            if (i + 1) % 10 == 0:
                logger.info(f"已处理 {i+1}/{len(stock_codes)} 只股票，"
                           f"新增到code_time: {total_to_code_time}，"
                           f"新增到time_code: {total_to_time_code}，"
                           f"更新: {total_updated}，"
                           f"仍有问题的股票: {stocks_with_issues}")
        
        # 4. 输出最终统计信息
        logger.info(f"修复完成！共处理 {len(stock_codes)} 只股票")
        logger.info(f"新增到stock_code_time: {total_to_code_time} 条记录")
        logger.info(f"新增到stock_time_code: {total_to_time_code} 条记录")
        logger.info(f"更新: {total_updated} 条记录")
        logger.info(f"仍有问题的股票: {stocks_with_issues} 只")
        
        if stocks_with_issues == 0:
            logger.info("修复成功！两个表数据现在完全一致")
        else:
            logger.warning(f"修复后仍有 {stocks_with_issues} 只股票数据不一致，可能需要进一步排查")
        
    except Exception as e:
        logger.error(f"修复过程中发生错误: {str(e)}")

if __name__ == "__main__":
>>>>>>> 3d7330be7ea0ecb409ac485e1c8391bc6d56a2de
    main() 