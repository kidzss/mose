import sys
import logging
import pandas as pd
from datetime import datetime
import sqlite3
import os
import pymysql
from sqlalchemy import create_engine, text

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从data_updater导入数据库配置
try:
    from data_updater import DB_CONFIG
    # 构建MySQL连接字符串
    DB_URI = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
except ImportError:
    logger.error("无法导入配置，使用默认配置")
    # 默认MySQL配置
    DB_URI = "mysql+pymysql://root:password@localhost:3306/stock_data"

def check_stock_completeness(symbols):
    """检查特定股票的数据完整性"""
    try:
        # 创建数据库连接
        engine = create_engine(DB_URI)
        
        for symbol in symbols:
            try:
                # 查询股票数据 - 使用stock_time_code表
                query = text(f"SELECT Date FROM stock_time_code WHERE Code = :symbol ORDER BY Date")
                with engine.connect() as conn:
                    data_df = pd.read_sql_query(query, conn, params={"symbol": symbol})
                
                # 如果没有数据，继续下一个
                if len(data_df) == 0:
                    logger.error(f"股票 {symbol} 没有找到任何数据")
                    continue
                
                # 获取数据日期范围
                min_date = data_df['Date'].min()
                max_date = data_df['Date'].max()
                
                # 查询交易日历 - 从交易数据中获取所有日期作为基准
                calendar_query = text("""
                    SELECT DISTINCT Date 
                    FROM stock_time_code 
                    WHERE Date BETWEEN :min_date AND :max_date
                    ORDER BY Date
                """)
                
                with engine.connect() as conn:
                    calendar_df = pd.read_sql_query(
                        calendar_query, 
                        conn, 
                        params={"min_date": min_date, "max_date": max_date}
                    )
                
                # 转换日期格式为字符串进行比较
                data_df['Date'] = pd.to_datetime(data_df['Date']).dt.strftime('%Y-%m-%d')
                calendar_df['Date'] = pd.to_datetime(calendar_df['Date']).dt.strftime('%Y-%m-%d')
                
                # 转换为集合进行比较
                data_dates = set(data_df['Date'].tolist())
                calendar_dates = set(calendar_df['Date'].tolist())
                
                # 计算缺失的日期
                missing_dates = calendar_dates - data_dates
                
                # 输出结果
                completion_rate = (len(data_dates) / len(calendar_dates)) * 100 if len(calendar_dates) > 0 else 0
                
                logger.info(f"股票 {symbol} 数据范围: {min_date} 到 {max_date}")
                logger.info(f"股票 {symbol} 交易日历天数: {len(calendar_dates)}")
                logger.info(f"股票 {symbol} 实际数据天数: {len(data_dates)}")
                
                if missing_dates:
                    logger.warning(f"股票 {symbol} 缺失 {len(missing_dates)} 个交易日")
                    logger.warning(f"完整率: {completion_rate:.2f}%")
                    
                    # 输出部分缺失日期作为示例
                    missing_list = sorted(list(missing_dates))
                    if len(missing_list) > 10:
                        logger.warning(f"缺失日期示例: {missing_list[:10]}...")
                    else:
                        logger.warning(f"缺失日期: {missing_list}")
                else:
                    logger.info(f"股票 {symbol} 数据完整，无缺失日期")
                    logger.info(f"完整率: 100.00%")
            
            except Exception as e:
                logger.error(f"检查股票 {symbol} 时出错: {str(e)}")
    
    except Exception as e:
        logger.error(f"连接数据库时出错: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_specific_stocks.py SYMBOL1 SYMBOL2 ...")
        sys.exit(1)
    
    symbols = sys.argv[1:]
    logger.info(f"正在检查 {len(symbols)} 只股票的数据完整性...")
    check_stock_completeness(symbols)
    logger.info("检查完成") 