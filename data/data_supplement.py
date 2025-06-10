import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text
import yfinance as yf
import time
from tqdm import tqdm
import pandas_market_calendars as mcal
from typing import List, Dict, Set, Optional
import sys
import os
import pymysql

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_supplement.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "mose"
}

class DataSupplementer:
    """数据补充器，用于补充缺失的交易日数据"""
    
    def __init__(self, db_config: Dict):
        """初始化数据补充器"""
        self.db_config = db_config
        self.engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        # 获取NYSE日历
        self.nyse = mcal.get_calendar('NYSE')
        
    def get_missing_dates(self, symbol: str) -> Set[str]:
        """获取股票缺失的交易日数据"""
        # 获取股票的数据范围
        query = text("""
            SELECT 
                MIN(Date) as first_date,
                MAX(Date) as last_date
            FROM stock_time_code 
            WHERE Code = :symbol
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol}).fetchone()
            if not result or not result[0]:
                logger.error(f"股票 {symbol} 没有数据记录")
                return set()
            
            first_date = result[0].strftime('%Y-%m-%d') if isinstance(result[0], datetime) else result[0]
            last_date = result[1].strftime('%Y-%m-%d') if isinstance(result[1], datetime) else result[1]
        
        # 获取该时间范围内的所有交易日
        trading_days = set(self.nyse.schedule(start_date=first_date, end_date=last_date).index.strftime('%Y-%m-%d'))
        
        # 获取实际的数据日期
        query = text("""
            SELECT DISTINCT Date 
            FROM stock_time_code 
            WHERE Code = :symbol 
            AND Date BETWEEN :start_date AND :end_date
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                "symbol": symbol,
                "start_date": first_date,
                "end_date": last_date
            })
            dates = {row[0].strftime('%Y-%m-%d') if isinstance(row[0], (datetime, pd.Timestamp)) else row[0] 
                    for row in result}
        
        # 计算缺失的日期
        missing_dates = trading_days - dates
        
        if missing_dates:
            logger.info(f"股票 {symbol} 缺失 {len(missing_dates)} 个交易日数据")
            logger.info(f"缺失日期范围: {min(missing_dates)} 到 {max(missing_dates)}")
        else:
            logger.info(f"股票 {symbol} 数据完整，无缺失日期")
            
        return missing_dates
    
    def fetch_missing_data(self, symbol: str, missing_dates: Set[str]) -> Optional[pd.DataFrame]:
        """获取缺失的交易日数据"""
        if not missing_dates:
            return None
            
        try:
            # 获取缺失日期范围
            start_date = min(missing_dates)
            end_date = max(missing_dates)
            
            # 从Yahoo Finance获取数据
            stock = yf.Ticker(symbol)
            df = stock.history(
                start=start_date,
                end=end_date,
                interval="1d",
                prepost=False,
                actions=True
            )
            
            if df.empty:
                logger.warning(f"股票 {symbol} 在 {start_date} 到 {end_date} 期间没有数据")
                return None
                
            # 处理数据
            df = df.reset_index()
            df = df.rename(columns={
                'Stock Splits': 'StockSplits',
                'Capital Gains': 'Capital_Gains'
            })
            df['Code'] = symbol
            df['Amount'] = df['Close'] * df['Volume']
            df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)
            
            # 只保留缺失日期的数据
            df = df[df['Date'].isin(missing_dates)]
            
            if df.empty:
                logger.warning(f"股票 {symbol} 在缺失日期范围内没有有效数据")
                return None
                
            logger.info(f"成功获取股票 {symbol} 的 {len(df)} 条缺失数据")
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的缺失数据时出错: {str(e)}")
            return None
    
    def save_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """保存数据到数据库"""
        if df is None or df.empty:
            return False
            
        try:
            # 准备数据
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)
            
            # 获取连接
            conn = pymysql.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            
            try:
                with conn.cursor() as cursor:
                    # 临时关闭严格模式
                    cursor.execute("SET SESSION sql_mode=''")
                    
                    # 获取列名
                    columns = df.columns.tolist()
                    column_names = ', '.join(['`' + col + '`' for col in columns])
                    placeholders = ', '.join(['%s'] * len(columns))
                    
                    # 构建SQL - 使用REPLACE INTO
                    sql = f"""
                    REPLACE INTO stock_time_code ({column_names})
                    VALUES ({placeholders})
                    """
                    
                    # 逐行插入
                    for _, row in df.iterrows():
                        values = [row[col] for col in columns]
                        cursor.execute(sql, values)
                        
                    # 提交事务
                    conn.commit()
                    
                    # 对stock_code_time表执行相同的操作
                    sql = f"""
                    REPLACE INTO stock_code_time ({column_names})
                    VALUES ({placeholders})
                    """
                    
                    for _, row in df.iterrows():
                        values = [row[col] for col in columns]
                        cursor.execute(sql, values)
                        
                    # 提交事务
                    conn.commit()
                    
                logger.info(f"成功保存股票 {symbol} 的 {len(df)} 条数据到数据库")
                return True
                
            except Exception as e:
                conn.rollback()
                logger.error(f"保存股票 {symbol} 的数据时出错: {str(e)}")
                return False
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"保存股票 {symbol} 的数据时出错: {str(e)}")
            return False
    
    def supplement_stock_data(self, symbol: str) -> bool:
        """补充单个股票的数据"""
        try:
            # 获取缺失的日期
            missing_dates = self.get_missing_dates(symbol)
            if not missing_dates:
                return True
                
            # 获取缺失的数据
            df = self.fetch_missing_data(symbol, missing_dates)
            if df is None:
                return False
                
            # 保存数据
            return self.save_data(df, symbol)
            
        except Exception as e:
            logger.error(f"补充股票 {symbol} 的数据时出错: {str(e)}")
            return False
    
    def supplement_multiple_stocks(self, symbols: List[str]) -> Dict[str, bool]:
        """补充多个股票的数据"""
        results = {}
        for symbol in tqdm(symbols, desc="补充股票数据"):
            success = self.supplement_stock_data(symbol)
            results[symbol] = success
            # 添加延迟以避免API限制
            time.sleep(1)
        return results

def main():
    """主函数"""
    # 需要补充数据的股票列表
    target_symbols = [
        'DDOG', 'ASML', 'AZN', 'CCEP', 'MELI', 'MRVL', 'WDAY',
        'TEAM', 'TTD', 'HWM', 'MDB', 'ZS', 'PDD', 'TMDX'
    ]
    
    # 创建数据补充器
    supplementer = DataSupplementer(DB_CONFIG)
    
    # 补充数据
    results = supplementer.supplement_multiple_stocks(target_symbols)
    
    # 打印结果
    print("\n=== 数据补充结果 ===")
    success_count = sum(1 for success in results.values() if success)
    print(f"成功补充: {success_count}/{len(target_symbols)} 只股票")
    
    # 打印详细信息
    print("\n详细结果:")
    for symbol, success in results.items():
        status = "成功" if success else "失败"
        print(f"{symbol}: {status}")

if __name__ == '__main__':
    main() 