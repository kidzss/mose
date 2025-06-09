import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, date
import logging
import os
import time
import requests
from requests.exceptions import RequestException
import concurrent.futures
from tqdm import tqdm
import sys
from config.trading_config import default_config
import pymysql
from typing import List, Dict, Any

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
        logging.FileHandler('data_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置 - 使用统一配置
DB_CONFIG = {
    "host": default_config.database.host,
    "port": default_config.database.port,
    "user": default_config.database.user,
    "password": default_config.database.password,
    "database": default_config.database.database
}

# 代理配置（如果需要）
PROXIES = {
    'http': '',  # 如果需要代理，在这里填写
    'https': ''  # 如果需要代理，在这里填写
}

class DatabaseManager:
    """数据库管理类，处理数据库连接和查询操作"""
    
    def __init__(self, config):
        """
        初始化数据库管理器
        
        Args:
            config: 数据库配置
        """
        self.config = config
        self.engine = self._create_engine_with_retry()
    
    def _create_engine_with_retry(self, max_retries=3):
        """创建数据库连接引擎，带重试机制"""
        for i in range(max_retries):
            try:
                engine = create_engine(
                    f"mysql+pymysql://{self.config['user']}:{self.config['password']}@"
                    f"{self.config['host']}:{self.config['port']}/{self.config['database']}",
                    pool_recycle=3600,
                    pool_timeout=60,
                    connect_args={
                        'connect_timeout': 60,
                        'read_timeout': 60,
                        'write_timeout': 60
                    }
                )
                # 测试连接
                with engine.connect() as conn:
                    pass
                return engine
            except Exception as e:
                logger.error(f"数据库连接失败，尝试次数 {i + 1}/{max_retries}: {str(e)}")
                if i == max_retries - 1:
                    raise
                time.sleep(2 ** i)  # 指数退避
    
    def get_existing_stocks(self):
        """获取数据库中已存在的股票列表"""
        try:
            # 修改查询，只获取有效的股票代码
            query = text("""
                SELECT DISTINCT Code 
                FROM stock_time_code 
                WHERE Code NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                AND Code IS NOT NULL
                AND Code != ''
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                existing_stocks = [row[0] for row in result]
                
            # 过滤掉任何看起来像日期的代码
            existing_stocks = [
                code for code in existing_stocks 
                if not (isinstance(code, str) and len(code) == 10 and code.count('-') == 2)
            ]
            
            logger.info(f"从数据库加载了 {len(existing_stocks)} 只股票")
            return existing_stocks
        except Exception as e:
            logger.error(f"获取已存在股票列表时出错: {str(e)}")
            return []
    
    def get_last_update_date(self, symbol):
        """获取股票最后更新日期，同时检查两个表中的最新日期"""
        try:
            # 检查stock_time_code表中的最后更新日期
            time_code_query = text("SELECT MAX(Date) FROM stock_time_code WHERE Code = :symbol")
            with self.engine.connect() as conn:
                result = conn.execute(time_code_query, {"symbol": symbol})
                time_code_last_date = result.scalar()
            
            # 检查stock_code_time表中的最后更新日期
            code_time_query = text("SELECT MAX(Date) FROM stock_code_time WHERE Code = :symbol")
            with self.engine.connect() as conn:
                result = conn.execute(code_time_query, {"symbol": symbol})
                code_time_last_date = result.scalar()
                
            # 获取两个日期中的最新日期
            if time_code_last_date and code_time_last_date:
                last_date = max(time_code_last_date, code_time_last_date)
            elif time_code_last_date:
                last_date = time_code_last_date
            elif code_time_last_date:
                last_date = code_time_last_date
            else:
                last_date = None
                
            # 如果结果是datetime.date对象，转换为datetime对象以便于后续计算
            if last_date:
                # 使用直接的类型检查
                if isinstance(last_date, date) and not isinstance(last_date, datetime):
                    last_date = datetime.combine(last_date, datetime.min.time())
                
            return last_date
        except Exception as e:
            logger.error(f"获取股票 {symbol} 最后更新日期时出错: {str(e)}")
            return None
    
    def get_table_columns(self, table_name):
        """获取数据库表的列名"""
        try:
            query = text(f"SHOW COLUMNS FROM {table_name}")
            with self.engine.connect() as conn:
                result = conn.execute(query)
                columns = [row[0] for row in result]
            return columns
        except Exception as e:
            logger.error(f"获取表 {table_name} 列名时出错: {str(e)}")
            return []
    
    def sync_tables_differences(self):
        """检查并同步stock_time_code和stock_code_time表之间的差异"""
        try:
            # 清理无效数据 - 临时关闭严格模式处理无效日期
            with self.engine.connect() as conn:
                # 保存当前的SQL模式
                result = conn.execute(text("SELECT @@SESSION.sql_mode"))
                original_mode = result.scalar()
                
                try:
                    # 临时关闭严格模式
                    conn.execute(text("SET SESSION sql_mode=''"))
                    
                    # 执行清理查询
                    clean_query = """
                        DELETE FROM stock_time_code
                        WHERE Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                        OR Date = '0000-00-00'
                    """
                    conn.execute(text(clean_query))
                    conn.commit()
                    logger.info("已清理stock_time_code表中的无效日期数据")
                finally:
                    # 恢复原来的SQL模式
                    conn.execute(text(f"SET SESSION sql_mode='{original_mode}'"))
            
            # 1. 将stock_time_code中有而stock_code_time中没有的记录插入stock_code_time
            insert_code_time_query = """
                INSERT IGNORE INTO stock_code_time
                SELECT
                    t.Code,
                    t.Date,
                    t.Open,
                    t.High,
                    t.Low,
                    t.Close,
                    t.Volume,
                    t.Amount,
                    t.AdjClose,
                    t.Dividends,
                    t.StockSplits,
                    t.Capital_Gains
                FROM stock_time_code t
                LEFT JOIN stock_code_time c ON t.Code = c.Code AND t.Date = c.Date
                WHERE c.Code IS NULL
            """
            with self.engine.connect() as conn:
                conn.execute(text(insert_code_time_query))
                conn.commit()
            
            # 2. 将stock_code_time中有而stock_time_code中没有的记录插入stock_time_code
            insert_time_code_query = """
                INSERT IGNORE INTO stock_time_code
                SELECT
                    c.Date,
                    c.Code,
                    c.Open,
                    c.High,
                    c.Low,
                    c.Close,
                    c.Volume,
                    c.Amount,
                    c.AdjClose,
                    c.Dividends,
                    c.StockSplits,
                    c.Capital_Gains
                FROM stock_code_time c
                LEFT JOIN stock_time_code t ON c.Code = t.Code AND c.Date = t.Date
                WHERE t.Code IS NULL
            """
            with self.engine.connect() as conn:
                conn.execute(text(insert_time_code_query))
                conn.commit()
            
            # 查询不一致的记录数量
            inconsistent_query = """
                SELECT COUNT(*) AS inconsistent_count
                FROM stock_time_code t
                JOIN stock_code_time c ON t.Code = c.Code AND t.Date = c.Date
                WHERE 
                    t.Open != c.Open OR 
                    t.High != c.High OR 
                    t.Low != c.Low OR 
                    t.Close != c.Close OR 
                    t.Volume != c.Volume OR
                    t.Amount != c.Amount OR
                    t.AdjClose != c.AdjClose OR
                    t.Dividends != c.Dividends OR
                    t.StockSplits != c.StockSplits OR
                    t.Capital_Gains != c.Capital_Gains
            """
            with self.engine.connect() as conn:
                result = conn.execute(text(inconsistent_query))
                inconsistent_count = result.scalar()
            
            # 如果存在不一致的记录，尝试同步它们
            if inconsistent_count > 0:
                logger.info(f"发现 {inconsistent_count} 条不一致的记录，正在同步...")
                
                # 更新stock_code_time表使用stock_time_code的数据
                update_code_time_query = """
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
                    WHERE 
                        t.Open != c.Open OR 
                        t.High != c.High OR 
                        t.Low != c.Low OR 
                        t.Close != c.Close OR 
                        t.Volume != c.Volume OR
                        t.Amount != c.Amount OR
                        t.AdjClose != c.AdjClose OR
                        t.Dividends != c.Dividends OR
                        t.StockSplits != c.StockSplits OR
                        t.Capital_Gains != c.Capital_Gains
                """
                with self.engine.connect() as conn:
                    conn.execute(text(update_code_time_query))
                    conn.commit()
                
                logger.info("同步完成")
            else:
                logger.info("两个表的数据已完全一致")
            
        except Exception as e:
            logger.error(f"同步表差异时出错: {e}")
            
    def save_stock_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        保存股票数据到数据库
        
        Args:
            symbol: 股票代码
            df: 包含股票数据的DataFrame
            
        Returns:
            bool: 是否成功保存
        """
        try:
            if df.empty:
                logger.warning(f"股票 {symbol} 没有数据需要保存")
                return False
                
            # 确保日期格式正确
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # 添加股票代码列
            df['Code'] = symbol
            
            # 移除Amount列（如果存在）
            if 'Amount' in df.columns:
                df = df.drop('Amount', axis=1)
            
            # 查询已有记录，用于后续确定是插入新记录而不是替换旧记录
            try:
                conn = pymysql.connect(
                    host=self.config['host'],
                    port=self.config['port'],
                    user=self.config['user'],
                    password=self.config['password'],
                    database=self.config['database']
                )
                
                try:
                    with conn.cursor() as cursor:
                        # 获取现有数据的日期
                        cursor.execute(
                            "SELECT Date FROM stock_time_code WHERE Code = %s",
                            (symbol,)
                        )
                        existing_dates = set([row[0] for row in cursor.fetchall()])
                        
                        # 检查有多少新记录
                        df_dates = set(df['Date'].tolist())
                        new_dates = df_dates - existing_dates
                        
                        logger.info(f"股票 {symbol} 数据分析: 总获取记录数 {len(df)}, 需要插入的新记录数 {len(new_dates)}")
                        
                        if len(new_dates) == 0:
                            logger.info(f"股票 {symbol} 没有新数据需要添加，跳过保存")
                            return True
                            
                        # 只保留新记录
                        df_new = df[df['Date'].isin(new_dates)]
                        if df_new.empty:
                            logger.info(f"过滤后股票 {symbol} 没有新数据需要添加，跳过保存")
                            return True
                            
                        logger.info(f"将为股票 {symbol} 插入 {len(df_new)} 条新记录")
                        
                        # 获取列名
                        columns = df_new.columns.tolist()
                        column_names = ', '.join(['`' + col + '`' for col in columns])
                        
                        # 构建占位符
                        placeholders = ', '.join(['%s'] * len(columns))
                        
                        # 准备数据，确保两个表插入完全相同的数据
                        insert_data = []
                        for _, row in df_new.iterrows():
                            values = [row[col] for col in columns]
                            insert_data.append(values)
                        
                        # 构建SQL - 使用INSERT IGNORE而不是REPLACE INTO
                        sql_time_code = f"""
                        INSERT IGNORE INTO stock_time_code ({column_names})
                        VALUES ({placeholders})
                        """
                        
                        sql_code_time = f"""
                        INSERT IGNORE INTO stock_code_time ({column_names})
                        VALUES ({placeholders})
                        """
                        
                        # 批量插入新记录到两个表
                        # 1. 先插入stock_time_code表
                        inserted_time_code = 0
                        for values in insert_data:
                            cursor.execute(sql_time_code, values)
                            inserted_time_code += cursor.rowcount
                        
                        # 2. 再插入stock_code_time表 - 使用完全相同的数据
                        inserted_code_time = 0
                        for values in insert_data:
                            cursor.execute(sql_code_time, values)
                            inserted_code_time += cursor.rowcount
                        
                        # 提交事务
                        conn.commit()
                        
                        logger.info(f"成功插入股票 {symbol} 数据: stock_time_code表 {inserted_time_code} 条, stock_code_time表 {inserted_code_time} 条")
                        
                        # 3. 检查两个表的记录数是否一致
                        if inserted_time_code != inserted_code_time:
                            logger.warning(f"股票 {symbol} 插入记录数不一致: stock_time_code {inserted_time_code}, stock_code_time {inserted_code_time}")
                            
                            # 尝试同步两个表的差异
                            self.sync_tables_differences()
                        
                    logger.info(f"成功保存股票 {symbol} 的数据，总计 {len(df_new)} 条新记录")
                    return True
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"保存股票 {symbol} 数据时出错: {str(e)}")
                    return False
                finally:
                    conn.close()
                    
            except Exception as e:
                logger.error(f"保存股票 {symbol} 数据时出错: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"保存股票 {symbol} 数据初始化阶段出错: {str(e)}")
            return False


class StockDataFetcher:
    """负责从API获取股票数据的类"""
    
    def __init__(self, proxies=None):
        """
        初始化数据获取器
        
        Args:
            proxies: 代理设置
        """
        self.proxies = proxies
    
    def get_stock_data(self, symbol, start_date, end_date):
        """获取单个股票的历史数据"""
        max_retries = 3
        base_delay = 1  # 将基础延迟从2秒减少到1秒

        for attempt in range(max_retries):
            try:
                # 创建Ticker对象
                stock = yf.Ticker(symbol)

                # 获取数据
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    prepost=False,  # 不包括盘前盘后数据
                    actions=True  # 包括分红和拆分信息
                )

                if df.empty:
                    logger.warning(f"股票 {symbol} 返回空数据")
                    return None

                # 重置索引，将日期变为列
                df = df.reset_index()

                # 重命名列
                df = df.rename(columns={
                    'Stock Splits': 'StockSplits',
                    'Capital Gains': 'Capital_Gains'
                })

                # 添加股票代码列
                df['Code'] = symbol
                
                # 添加Amount列（计算为价格乘以成交量）
                df['Amount'] = df['Close'] * df['Volume']

                # 确保日期格式正确 - 将日期转换为字符串格式，与process_stock_data一致
                df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)

                # 按日期排序
                df = df.sort_values('Date')

                # 减少延时以加快处理速度
                time.sleep(0.3)  # 进一步减少延时

                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"获取股票 {symbol} 数据时出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    delay = base_delay * (2 ** attempt)  # 指数退避
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"获取股票 {symbol} 数据失败: {str(e)}")
                    return None

        return None
    
    def get_stock_data_batch(self, symbols, start_dates, end_date, max_workers=5):
        """
        并行获取多只股票的数据
        
        Args:
            symbols: 股票代码列表
            start_dates: 每只股票对应的开始日期
            end_date: 结束日期
            max_workers: 并行数量
            
        Returns:
            字典 {股票代码: 数据DataFrame}
        """
        results = {}
        
        # 确保start_dates是字典形式
        if not isinstance(start_dates, dict):
            start_dates = {symbol: start_dates for symbol in symbols}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建future到symbol的映射
            future_to_symbol = {
                executor.submit(self.get_stock_data, symbol, start_dates.get(symbol), end_date): symbol
                for symbol in symbols
            }
            
            # 收集结果，使用tqdm显示进度
            for future in tqdm(concurrent.futures.as_completed(future_to_symbol), total=len(symbols), desc="获取股票数据"):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[symbol] = data
                except Exception as e:
                    logger.error(f"获取股票 {symbol} 数据时出错: {str(e)}")
        
        return results


def get_trading_days(end_date, days=30):
    """获取最近的交易日列表

    使用简单的方法：排除周末（未考虑节假日）
    可以根据需要扩展为使用专业数据源获取准确的交易日历
    """
    trading_days = []
    current_date = end_date - timedelta(days=days)

    while current_date <= end_date:
        # 排除周末 (5=Saturday, 6=Sunday)
        if current_date.weekday() < 5:
            trading_days.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return trading_days


class MarketDataUpdater:
    """
    市场数据更新器类，整合了数据库操作和数据获取的功能
    """
    
    def __init__(self, db_config, proxies=None):
        """
        初始化更新器
        
        Args:
            db_config: 数据库配置
            proxies: 代理设置
        """
        self.db_manager = DatabaseManager(db_config)
        self.data_fetcher = StockDataFetcher(proxies)
        self._last_update_times = {}
    
    def get_last_update_time(self, symbol: str) -> datetime:
        """
        获取股票最后更新时间
        
        Args:
            symbol: 股票代码
            
        Returns:
            最后更新时间
        """
        if symbol in self._last_update_times:
            return self._last_update_times[symbol]
            
        last_update = self.db_manager.get_last_update_date(symbol)
        self._last_update_times[symbol] = last_update
        return last_update
        
    def get_last_update_times(self) -> Dict[str, datetime]:
        """
        获取所有股票的最后更新时间
        
        Returns:
            股票代码到更新时间的映射字典
        """
        symbols = self.db_manager.get_existing_stocks()
        return {symbol: self.get_last_update_time(symbol) for symbol in symbols}
    
    def load_stock_lists(self):
        """加载SP500和Nasdaq100的股票列表"""
        # 读取SP500股票列表
        sp500_df = pd.read_csv('stock_pool/sp500_stocks.csv')
        sp500_symbols = sp500_df['Code'].tolist()

        nasdaq100_df = pd.read_csv('stock_pool/nasdaq100_stocks.csv')
        nasdaq100_symbols = nasdaq100_df['Code'].tolist()

        etf_stocks_df = pd.read_csv('stock_pool/uss_etf_stocks.csv')
        etf_symbols = etf_stocks_df['Code'].tolist()

        # 添加Nasdaq100股票
        for symbol in nasdaq100_symbols:
            if symbol not in sp500_symbols:
                sp500_symbols.append(symbol)

        # 添加ETF股票
        for symbol in etf_symbols:
            if symbol not in sp500_symbols:
                sp500_symbols.append(symbol)

        # 添加BABA, BRK.B, LVMUY
        for symbol in ['BABA', 'BRK.B', 'LVMUY']:
            if symbol not in sp500_symbols:
                sp500_symbols.append(symbol)

        # 添加SPY ETF
        if 'SPY' not in sp500_symbols:
            sp500_symbols.append('SPY')

        logger.info(f"从SP500列表中加载了 {len(sp500_symbols)} 只股票")
        return sp500_symbols

    def get_next_trading_day(self, date, trading_days):
        """获取给定日期之后的下一个交易日"""
        # 确保日期是字符串格式
        if isinstance(date, datetime) or isinstance(date, date):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
            
        for trading_day in trading_days:
            if trading_day > date_str:
                return datetime.strptime(trading_day, '%Y-%m-%d').date()
        return None
    
    def process_stock_data(self, symbol, df, time_code_columns, code_time_columns):
        """
        处理单只股票的数据，插入到数据库
        
        Args:
            symbol: 股票代码
            df: 股票数据DataFrame
            time_code_columns: stock_time_code表的列
            code_time_columns: stock_code_time表的列
            
        Returns:
            是否成功处理
        """
        try:
            if df.empty:
                logger.info(f"股票 {symbol} 没有数据需要更新")
                return False
            
            # 详细记录数据类型，便于调试
            logger.info(f"股票 {symbol} 原始数据类型: {df.dtypes.to_dict()}")
            
            # 过滤掉无效日期
            df = df[df['Date'].notna()]  # 移除空日期
            df = df[~df['Date'].str.contains('0000-00-00', na=False)]  # 移除无效日期
            
            if df.empty:
                logger.warning(f"股票 {symbol} 过滤后没有有效数据")
                return False
            
            # 明确转换所有列的数据类型
            # 1. 先转换日期列
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # 使用coerce处理无效日期
            df = df[df['Date'].notna()]  # 再次移除转换后仍为无效的日期
            
            if df.empty:
                logger.warning(f"股票 {symbol} 日期转换后没有有效数据")
                return False
                
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # 转换为字符串格式
            
            # 2. 转换数值列
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose', 'Dividends', 'StockSplits', 'Capital_Gains', 'Amount']
            for col in df.columns:
                if col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
            
            # 3. 确保Code列为字符串
            df['Code'] = df['Code'].astype(str)
            
            # 4. 验证价格数据的合理性
            # 检查最高价是否低于最低价
            invalid_hl = df[df['High'] < df['Low']]
            if not invalid_hl.empty:
                logger.warning(f"股票 {symbol} 存在最高价低于最低价的情况，日期: {invalid_hl['Date'].tolist()}")
                # 修正数据：将最高价和最低价交换
                df.loc[invalid_hl.index, ['High', 'Low']] = df.loc[invalid_hl.index, ['Low', 'High']].values
            
            # 检查最高价是否低于开盘价或收盘价
            invalid_ho = df[df['High'] < df['Open']]
            invalid_hc = df[df['High'] < df['Close']]
            if not invalid_ho.empty:
                logger.warning(f"股票 {symbol} 存在最高价低于开盘价的情况，日期: {invalid_ho['Date'].tolist()}")
                # 修正数据：将最高价设置为开盘价和收盘价中的较大值
                df.loc[invalid_ho.index, 'High'] = df.loc[invalid_ho.index, ['Open', 'Close']].max(axis=1)
            if not invalid_hc.empty:
                logger.warning(f"股票 {symbol} 存在最高价低于收盘价的情况，日期: {invalid_hc['Date'].tolist()}")
                # 修正数据：将最高价设置为开盘价和收盘价中的较大值
                df.loc[invalid_hc.index, 'High'] = df.loc[invalid_hc.index, ['Open', 'Close']].max(axis=1)
            
            # 检查最低价是否高于开盘价或收盘价
            invalid_lo = df[df['Low'] > df['Open']]
            invalid_lc = df[df['Low'] > df['Close']]
            if not invalid_lo.empty:
                logger.warning(f"股票 {symbol} 存在最低价高于开盘价的情况，日期: {invalid_lo['Date'].tolist()}")
                # 修正数据：将最低价设置为开盘价和收盘价中的较小值
                df.loc[invalid_lo.index, 'Low'] = df.loc[invalid_lo.index, ['Open', 'Close']].min(axis=1)
            if not invalid_lc.empty:
                logger.warning(f"股票 {symbol} 存在最低价高于收盘价的情况，日期: {invalid_lc['Date'].tolist()}")
                # 修正数据：将最低价设置为开盘价和收盘价中的较小值
                df.loc[invalid_lc.index, 'Low'] = df.loc[invalid_lc.index, ['Open', 'Close']].min(axis=1)
            
            # 记录处理后的数据类型，便于调试
            logger.info(f"股票 {symbol} 处理后数据类型: {df.dtypes.to_dict()}")
            
            # 分别处理两个表
            try:
                # 1. 准备并更新stock_time_code表
                df_time_code = df.copy()
                valid_time_code_columns = [col for col in df_time_code.columns if col in time_code_columns]
                
                if valid_time_code_columns:
                    df_time_code = df_time_code[valid_time_code_columns]
                    total_rows = len(df_time_code)
                    
                    # 直接使用原始MySQL连接和直接SQL字符串拼接，跳过SQLAlchemy的类型处理
                    conn = pymysql.connect(
                        host=self.db_manager.config['host'],
                        port=self.db_manager.config['port'],
                        user=self.db_manager.config['user'],
                        password=self.db_manager.config['password'],
                        database=self.db_manager.config['database']
                    )
                    
                    try:
                        with conn.cursor() as cursor:
                            # 临时关闭严格模式
                            cursor.execute("SET SESSION sql_mode=''")
                            
                            # 获取列名
                            columns = df_time_code.columns.tolist()
                            column_names = ', '.join(['`' + col + '`' for col in columns])
                            
                            # 构建占位符
                            placeholders = ', '.join(['%s'] * len(columns))
                            
                            # 构建SQL - 使用REPLACE INTO
                            sql = f"""
                            REPLACE INTO stock_time_code ({column_names})
                            VALUES ({placeholders})
                            """
                            
                            # 逐行插入
                            for _, row in df_time_code.iterrows():
                                values = [row[col] for col in columns]
                                cursor.execute(sql, values)
                                
                            # 提交事务
                            conn.commit()
                            
                            # 对stock_code_time表执行相同的操作
                            sql = f"""
                            REPLACE INTO stock_code_time ({column_names})
                            VALUES ({placeholders})
                            """
                            
                            for _, row in df_time_code.iterrows():
                                values = [row[col] for col in columns]
                                cursor.execute(sql, values)
                                
                            # 提交事务
                            conn.commit()
                            
                        logger.info(f"成功保存股票 {symbol} 的数据，总计 {total_rows} 条记录")
                        return True
                        
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"保存股票 {symbol} 数据时出错: {str(e)}")
                        return False
                    finally:
                        conn.close()
                    
                else:
                    logger.error(f"股票 {symbol} 的数据与stock_time_code表结构不兼容，无法插入")
                    return False
                    
            except Exception as e:
                logger.error(f"处理股票 {symbol} 数据时出错: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"处理股票 {symbol} 数据初始化阶段出错: {str(e)}")
            return False
    
    def is_market_closed(self):
        """检查当前是否为收盘后时间"""
        now = datetime.now()
        market_close_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        return now > market_close_time

    def is_data_complete(self, symbol, date_str):
        """检查指定日期的数据是否完整"""
        try:
            query = text("""
                SELECT Volume, Close 
                FROM stock_time_code 
                WHERE Code = :symbol 
                AND Date = :date
            """)
            with self.db_manager.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol, "date": date_str}).fetchone()
                
                if result is None:
                    return False
                    
                volume, close = result
                # 如果成交量为0或收盘价为None，认为数据不完整
                return volume > 0 and close is not None
        except Exception as e:
            logger.error(f"检查数据完整性时出错: {str(e)}")
            return False

    def update_stock_data(self, symbols: List[str] = None, force_update: bool = False) -> Dict[str, Any]:
        """
        更新股票数据
        
        Args:
            symbols: 要更新的股票列表，如果为None则使用默认列表
            force_update: 是否强制更新（忽略最后更新时间）
            
        Returns:
            更新报告，包含更新状态和统计信息
        """
        if symbols is None:
            symbols = self.db_manager.get_existing_stocks()
            
        # 记录更新模式    
        if force_update:
            logger.info(f"开始强制更新 {len(symbols)} 只股票的数据")
        else:
            logger.info(f"开始增量更新 {len(symbols)} 只股票的数据")
            
        report = {
            'total': len(symbols),
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'details': {},
            'new_records_count': 0  # 新增一个统计，记录实际插入的新记录数
        }
        
        # 获取当前日期，用于跳过当天已更新的股票
        today = datetime.now().date()
        
        for symbol in symbols:
            try:
                # 检查是否需要更新
                last_update = self.get_last_update_time(symbol)
                
                # 记录当前处理的股票和最后更新时间
                logger.info(f"处理股票 {symbol}，最后更新时间: {last_update}")
                
                # 判断是否需要跳过
                if not force_update and last_update:
                    # 确保last_update是datetime对象
                    if isinstance(last_update, str):
                        last_update = pd.to_datetime(last_update)
                    elif isinstance(last_update, date) and not isinstance(last_update, datetime):
                        last_update = datetime.combine(last_update, datetime.min.time())
                    
                    # 计算时间差
                    time_diff = datetime.now() - last_update
                    
                    # 详细记录时间差
                    logger.info(f"股票 {symbol} 的数据已更新 {time_diff.days} 天 {time_diff.seconds//3600} 小时前")
                    
                    # 如果最后更新是今天，并且已经过了交易时间，则跳过
                    if last_update.date() == today and self.is_market_closed():
                        logger.info(f"跳过股票 {symbol}，因为它在今天已更新且已过交易时间")
                        report['skipped'] += 1
                        report['details'][symbol] = 'skipped (updated today after market close)'
                        continue
                    
                    # 如果最后更新在24小时内，则跳过
                    if time_diff.days < 1:
                        logger.info(f"跳过股票 {symbol}，因为它在过去24小时内已更新")
                        report['skipped'] += 1
                        report['details'][symbol] = 'skipped (up to date)'
                        continue
                        
                # 更新数据
                before_update = datetime.now()
                success = self._update_single_stock(symbol)
                after_update = datetime.now()
                update_time = (after_update - before_update).total_seconds()
                
                if success:
                    report['updated'] += 1
                    # 查询实际插入了多少条新记录
                    conn = pymysql.connect(
                        host=self.db_manager.config['host'],
                        port=self.db_manager.config['port'],
                        user=self.db_manager.config['user'],
                        password=self.db_manager.config['password'],
                        database=self.db_manager.config['database']
                    )
                    
                    try:
                        with conn.cursor() as cursor:
                            # 查询在更新期间插入的记录数
                            latest_update = self.get_last_update_time(symbol)
                            if latest_update and last_update:
                                if isinstance(latest_update, date) and not isinstance(latest_update, datetime):
                                    latest_update = datetime.combine(latest_update, datetime.min.time())
                                if isinstance(last_update, date) and not isinstance(last_update, datetime):
                                    last_update = datetime.combine(last_update, datetime.min.time())
                                    
                                # 将日期转换为字符串格式
                                if isinstance(latest_update, datetime):
                                    latest_update_str = latest_update.strftime('%Y-%m-%d')
                                else:
                                    latest_update_str = str(latest_update)
                                    
                                if isinstance(last_update, datetime):
                                    last_update_str = last_update.strftime('%Y-%m-%d')
                                else:
                                    last_update_str = str(last_update)
                                
                                # 计算新记录数
                                if latest_update_str > last_update_str:
                                    cursor.execute(
                                        "SELECT COUNT(*) FROM stock_time_code WHERE Code = %s AND Date > %s AND Date <= %s",
                                        (symbol, last_update_str, latest_update_str)
                                    )
                                    new_records = cursor.fetchone()[0]
                                    report['new_records_count'] += new_records
                                    logger.info(f"股票 {symbol} 成功插入 {new_records} 条新记录，更新耗时 {update_time:.2f} 秒")
                                    report['details'][symbol] = f'updated with {new_records} new records'
                                else:
                                    logger.info(f"股票 {symbol} 没有新记录插入，最后更新日期未变")
                                    report['details'][symbol] = 'updated (no new records)'
                            else:
                                logger.info(f"股票 {symbol} 更新成功，但无法确定新增记录数")
                                report['details'][symbol] = 'updated'
                    finally:
                        conn.close()
                        
                    # 更新内存中的最后更新时间
                    self._last_update_times[symbol] = datetime.now()
                else:
                    report['failed'] += 1
                    report['details'][symbol] = 'failed'
                    
            except Exception as e:
                logger.error(f"更新股票 {symbol} 数据时出错: {str(e)}")
                report['failed'] += 1
                report['details'][symbol] = f'error: {str(e)}'
                
        # 记录最终结果
        logger.info(f"更新完成: 总计 {report['total']} 只股票, 更新 {report['updated']} 只, 跳过 {report['skipped']} 只, 失败 {report['failed']} 只, 共新增 {report['new_records_count']} 条记录")
        
        return report
        
    def _update_single_stock(self, symbol: str) -> bool:
        """更新单个股票的数据"""
        try:
            # 获取最新数据
            end_date = datetime.now()
            
            # 获取最后更新日期
            last_update = self.get_last_update_time(symbol)
            
            # 详细记录最后更新日期信息
            logger.info(f"股票 {symbol} 的最后更新日期为: {last_update}")
            
            # 根据最后更新日期决定数据获取范围
            if last_update:
                # 确保last_update是datetime对象
                if isinstance(last_update, str):
                    last_update = pd.to_datetime(last_update)
                elif isinstance(last_update, date) and not isinstance(last_update, datetime):
                    last_update = datetime.combine(last_update, datetime.min.time())
                
                # 从最后更新日期的下一天开始获取数据
                start_date = last_update + timedelta(days=1)
                logger.info(f"将获取股票 {symbol} 从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据")
                
                # 检查是否需要更新
                if start_date.date() >= end_date.date():
                    logger.info(f"股票 {symbol} 数据已是最新，无需更新")
                    return True
            else:
                # 新股票则获取历史数据
                start_date = end_date - timedelta(days=120)
                logger.info(f"股票 {symbol} 没有历史数据，将获取从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据")
            
            # 从Yahoo Finance获取数据
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                prepost=False,  # 不包括盘前盘后数据
                actions=True  # 包括分红和拆分信息
            )
            
            if df.empty:
                logger.warning(f"未获取到股票 {symbol} 的新数据")
                return False
                
            # 记录获取到的数据行数
            logger.info(f"从Yahoo获取到股票 {symbol} 的数据共 {len(df)} 条记录")
            
            # 处理数据
            df = self._process_data(df, symbol)
            
            # 如果有最后更新日期，验证是否真的获取了新数据
            if last_update and not df.empty:
                # 确保last_update是字符串格式
                if isinstance(last_update, datetime) or isinstance(last_update, date):
                    last_update_str = last_update.strftime('%Y-%m-%d')
                else:
                    last_update_str = str(last_update)
                
                # 记录新数据的日期范围
                if not df.empty:
                    min_date = df['Date'].min()
                    max_date = df['Date'].max()
                    logger.info(f"获取到的新数据日期范围: 从 {min_date} 到 {max_date}")
            
            # 保存到数据库
            result = self.db_manager.save_stock_data(symbol, df)
            
            # 记录保存结果
            if result:
                logger.info(f"成功保存股票 {symbol} 的新数据，共 {len(df)} 条记录")
            
            return result
            
        except Exception as e:
            logger.error(f"更新股票 {symbol} 数据时出错: {str(e)}")
            return False
            
    def _process_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """处理股票数据"""
        try:
            if df.empty:
                logger.warning(f"股票 {symbol} 数据为空，无法处理")
                return pd.DataFrame()
                
            # 记录原始数据行数
            original_rows = len(df)
                
            # 重置索引，将日期变为列
            df = df.reset_index()
            
            # 重命名列
            df = df.rename(columns={
                'Stock Splits': 'StockSplits',
                'Capital Gains': 'Capital_Gains'
            })
            
            # 添加股票代码列
            df['Code'] = symbol
            
            # 添加Amount列（计算为价格乘以成交量）
            df['Amount'] = df['Close'] * df['Volume']
            
            # 确保日期格式正确
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # 按日期排序
            df = df.sort_values('Date')
            
            # 检查数据质量
            # 1. 删除任何具有NaN价格的行
            nan_prices = df[df['Close'].isna()]
            if not nan_prices.empty:
                logger.warning(f"股票 {symbol} 有 {len(nan_prices)} 条记录价格为NaN，将被删除")
                df = df.dropna(subset=['Close'])
                
            # 2. 验证最高价和最低价的一致性
            invalid_prices = df[(df['High'] < df['Low']) | (df['High'] < df['Close']) | (df['Low'] > df['Close'])]
            if not invalid_prices.empty:
                logger.warning(f"股票 {symbol} 有 {len(invalid_prices)} 条记录价格不一致，将进行修正")
                # 修正价格
                for idx in invalid_prices.index:
                    high = max(df.loc[idx, 'High'], df.loc[idx, 'Close'], df.loc[idx, 'Low'])
                    low = min(df.loc[idx, 'High'], df.loc[idx, 'Close'], df.loc[idx, 'Low'])
                    df.loc[idx, 'High'] = high
                    df.loc[idx, 'Low'] = low
            
            # 记录处理后的数据行数
            processed_rows = len(df)
            if processed_rows != original_rows:
                logger.warning(f"股票 {symbol} 数据处理过程中移除了 {original_rows - processed_rows} 条无效记录")
            
            logger.info(f"股票 {symbol} 数据处理完成，共 {processed_rows} 条有效记录")
            return df
            
        except Exception as e:
            logger.error(f"处理股票 {symbol} 数据时出错: {str(e)}")
            return pd.DataFrame()


def test_update_logic():
    """测试数据更新逻辑，用于诊断固定返回83条记录的问题"""
    logger.info("开始测试数据更新逻辑...")
    
    try:
        # 创建市场数据更新器
        updater = MarketDataUpdater(DB_CONFIG, PROXIES)
        
        # 选择一只典型的股票进行测试 - SPY是常用的ETF，更新频率高
        test_symbol = "SPY"
        
        # 查询该股票的历史记录数量
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        
        try:
            with conn.cursor() as cursor:
                # 查询该股票的总记录数
                cursor.execute(
                    "SELECT COUNT(*) FROM stock_time_code WHERE Code = %s",
                    (test_symbol,)
                )
                record_count = cursor.fetchone()[0]
                logger.info(f"测试前，股票 {test_symbol} 在数据库中有 {record_count} 条历史记录")
                
                # 查询最早和最晚的日期
                cursor.execute(
                    "SELECT MIN(Date), MAX(Date) FROM stock_time_code WHERE Code = %s",
                    (test_symbol,)
                )
                min_date, max_date = cursor.fetchone()
                logger.info(f"测试前，股票 {test_symbol} 的数据范围: 从 {min_date} 到 {max_date}")
                
                # 先检查是否有最新交易日的数据（当天或前一个交易日）
                # 这样我们可以验证当已有最新数据时是否会正确跳过
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                today_str = today.strftime('%Y-%m-%d')
                yesterday_str = yesterday.strftime('%Y-%m-%d')
                
                cursor.execute(
                    "SELECT COUNT(*) FROM stock_time_code WHERE Code = %s AND (Date = %s OR Date = %s)",
                    (test_symbol, today_str, yesterday_str)
                )
                recent_count = cursor.fetchone()[0]
                logger.info(f"测试前，股票 {test_symbol} 在最近两天有 {recent_count} 条记录")
                
        finally:
            conn.close()
        
        # 1. 先进行一次常规更新，看是否会跳过或只更新新数据
        logger.info(f"对股票 {test_symbol} 执行常规更新，预期：如果已有最新数据则跳过，否则只更新新数据...")
        update_result = updater.update_stock_data([test_symbol], force_update=False)
        logger.info(f"常规更新结果: {update_result}")
        
        # 2. 然后进行强制更新测试
        logger.info(f"对股票 {test_symbol} 执行强制更新，预期：应始终更新数据...")
        force_update_result = updater.update_stock_data([test_symbol], force_update=True)
        logger.info(f"强制更新结果: {force_update_result}")
        
        # 3. 再次查询记录数量，看是否有变化
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        
        try:
            with conn.cursor() as cursor:
                # 查询更新后的总记录数
                cursor.execute(
                    "SELECT COUNT(*) FROM stock_time_code WHERE Code = %s",
                    (test_symbol,)
                )
                new_record_count = cursor.fetchone()[0]
                logger.info(f"测试后，股票 {test_symbol} 在数据库中有 {new_record_count} 条记录")
                
                # 如果记录数增加，检查新增的是哪些日期
                if new_record_count > record_count:
                    added_records = new_record_count - record_count
                    logger.info(f"共新增了 {added_records} 条记录")
                    
                    # 查询最新的日期
                    cursor.execute(
                        "SELECT MAX(Date) FROM stock_time_code WHERE Code = %s",
                        (test_symbol,)
                    )
                    new_max_date = cursor.fetchone()[0]
                    logger.info(f"更新后，股票 {test_symbol} 的最新数据日期为 {new_max_date}")
                    
                    # 查询最近10天的记录，看看具体更新了哪些日期
                    ten_days_ago = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
                    cursor.execute(
                        "SELECT Date, Close FROM stock_time_code WHERE Code = %s AND Date >= %s ORDER BY Date",
                        (test_symbol, ten_days_ago)
                    )
                    recent_data = cursor.fetchall()
                    logger.info(f"最近10天的记录: {recent_data}")
                else:
                    logger.info("没有新增记录，验证跳过逻辑正常工作")
                
        finally:
            conn.close()
        
        # 4. 等待3秒后再次尝试常规更新，应该会跳过
        logger.info("等待3秒后再次对股票 SPY 执行常规更新，预期：应该会跳过...")
        time.sleep(3)
        skip_update_result = updater.update_stock_data([test_symbol], force_update=False)
        logger.info(f"再次常规更新结果: {skip_update_result}")
        
        logger.info("测试数据更新逻辑完成")
        
    except Exception as e:
        logger.error(f"测试数据更新逻辑时出错: {str(e)}")
        raise


def main():
    """主函数"""
    try:
        # 测试更新逻辑 - 添加这一行
        test_update_logic()
        
        # 创建市场数据更新器
        updater = MarketDataUpdater(DB_CONFIG, PROXIES)
        
        # 加载股票列表并更新数据
        updater.update_stock_data()
        
        logger.info("股票数据处理完成 - 两个表已同步更新")

    except Exception as e:
        logger.error(f"运行过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()
