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

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
            query = text("SELECT DISTINCT Code FROM stock_time_code")
            with self.engine.connect() as conn:
                result = conn.execute(query)
                existing_stocks = [row[0] for row in result]
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
            sync_to_code_time_query = """
                INSERT IGNORE INTO stock_code_time 
                SELECT 
                    t.Code,
                    t.Date,
                    t.Open,
                    t.High,
                    t.Low,
                    t.Close,
                    t.Volume,
                    t.AdjClose,
                    t.Dividends,
                    t.StockSplits,
                    t.Capital_Gains
                FROM stock_time_code t
                LEFT JOIN stock_code_time c ON t.Code = c.Code AND t.Date = c.Date
                WHERE c.Code IS NULL
                    AND t.Date IS NOT NULL
                    AND t.Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                    AND t.Date != '0000-00-00'
            """
            
            # 临时关闭严格模式执行同步操作
            with self.engine.connect() as conn:
                # 保存当前的SQL模式
                result = conn.execute(text("SELECT @@SESSION.sql_mode"))
                original_mode = result.scalar()
                
                try:
                    # 临时关闭严格模式
                    conn.execute(text("SET SESSION sql_mode=''"))
                    
                    # 执行同步
                    result = conn.execute(text(sync_to_code_time_query))
                    affected_rows = result.rowcount
                    conn.commit()
                    
                    logger.info(f"从stock_time_code同步到stock_code_time: {affected_rows}条记录")
                finally:
                    # 恢复原来的SQL模式
                    conn.execute(text(f"SET SESSION sql_mode='{original_mode}'"))
            
            # 2. 将stock_code_time中有而stock_time_code中没有的记录插入stock_time_code
            sync_to_time_code_query = """
                INSERT IGNORE INTO stock_time_code 
                SELECT 
                    c.Code,
                    c.Date,
                    c.Open,
                    c.High,
                    c.Low,
                    c.Close,
                    c.Volume,
                    c.AdjClose,
                    c.Dividends,
                    c.StockSplits,
                    c.Capital_Gains
                FROM stock_code_time c
                LEFT JOIN stock_time_code t ON c.Code = t.Code AND c.Date = t.Date
                WHERE t.Code IS NULL
                    AND c.Date IS NOT NULL
                    AND c.Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                    AND c.Date != '0000-00-00'
            """
            
            # 临时关闭严格模式执行同步操作
            with self.engine.connect() as conn:
                # 保存当前的SQL模式
                result = conn.execute(text("SELECT @@SESSION.sql_mode"))
                original_mode = result.scalar()
                
                try:
                    # 临时关闭严格模式
                    conn.execute(text("SET SESSION sql_mode=''"))
                    
                    # 执行同步
                    result = conn.execute(text(sync_to_time_code_query))
                    affected_rows = result.rowcount
                    conn.commit()
                    
                    logger.info(f"从stock_code_time同步到stock_time_code: {affected_rows}条记录")
                finally:
                    # 恢复原来的SQL模式
                    conn.execute(text(f"SET SESSION sql_mode='{original_mode}'"))
            
        except Exception as e:
            logger.error(f"同步表之间差异时出错: {str(e)}")
            raise


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
            
            # 明确转换所有列的数据类型
            # 1. 先转换日期列
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # 2. 转换数值列
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose', 'Dividends', 'StockSplits', 'Capital_Gains', 'Amount']
            for col in df.columns:
                if col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
            
            # 3. 确保Code列为字符串
            df['Code'] = df['Code'].astype(str)
            
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
                            
                            # 构建ON DUPLICATE KEY UPDATE部分
                            update_clause = ', '.join([f"`{col}` = VALUES(`{col}`)" for col in columns if col not in ['Date', 'Code']])
                            
                            # 构建占位符
                            placeholders = ', '.join(['%s'] * len(columns))
                            
                            # 构建SQL
                            sql = f"""
                            INSERT INTO stock_time_code ({column_names})
                            VALUES ({placeholders})
                            ON DUPLICATE KEY UPDATE {update_clause}
                            """
                            
                            # 逐行插入 - 完全避免类型比较问题
                            for _, row in df_time_code.iterrows():
                                values = []
                                for col in columns:
                                    value = row[col]
                                    # 确保正确的数据类型
                                    if pd.isna(value):
                                        if col in numeric_columns:
                                            values.append(0.0)
                                        else:
                                            values.append("")
                                    else:
                                        values.append(value)
                                
                                # 执行SQL
                                cursor.execute(sql, values)
                            
                            # 提交事务
                            conn.commit()
                            
                        logger.info(f"股票 {symbol}: 插入/更新 {total_rows} 条记录到stock_time_code表")
                    except Exception as batch_error:
                        conn.rollback()
                        logger.error(f"插入 {symbol} 数据到stock_time_code表时出错: {str(batch_error)}")
                        raise
                    finally:
                        conn.close()
                    
                    # 2. 准备并更新stock_code_time表
                    df_code_time = df.copy()
                    valid_code_time_columns = [col for col in df_code_time.columns if col in code_time_columns]
                    
                    if not valid_code_time_columns:
                        logger.error(f"股票 {symbol} 的数据与stock_code_time表结构不兼容，无法插入")
                        return False
                    
                    df_code_time = df_code_time[valid_code_time_columns]
                    total_rows = len(df_code_time)
                    
                    # 使用相同的方法处理stock_code_time表
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
                            columns = df_code_time.columns.tolist()
                            column_names = ', '.join(['`' + col + '`' for col in columns])
                            
                            # 构建ON DUPLICATE KEY UPDATE部分
                            update_clause = ', '.join([f"`{col}` = VALUES(`{col}`)" for col in columns if col not in ['Date', 'Code']])
                            
                            # 构建占位符
                            placeholders = ', '.join(['%s'] * len(columns))
                            
                            # 构建SQL
                            sql = f"""
                            INSERT INTO stock_code_time ({column_names})
                            VALUES ({placeholders})
                            ON DUPLICATE KEY UPDATE {update_clause}
                            """
                            
                            # 逐行插入
                            for _, row in df_code_time.iterrows():
                                values = []
                                for col in columns:
                                    value = row[col]
                                    # 确保正确的数据类型
                                    if pd.isna(value):
                                        if col in numeric_columns:
                                            values.append(0.0)
                                        else:
                                            values.append("")
                                    else:
                                        values.append(value)
                                
                                # 执行SQL
                                cursor.execute(sql, values)
                            
                            # 提交事务
                            conn.commit()
                            
                        logger.info(f"股票 {symbol}: 插入/更新 {total_rows} 条记录到stock_code_time表")
                    except Exception as batch_error:
                        conn.rollback()
                        logger.error(f"插入 {symbol} 数据到stock_code_time表时出错: {str(batch_error)}")
                        raise
                    finally:
                        conn.close()
                    
                    logger.info(f"成功更新股票 {symbol} 的数据，总计 {total_rows} 条记录")
                    return True
                else:
                    logger.error(f"股票 {symbol} 的数据与stock_time_code表结构不兼容，无法插入")
                    return False
                
            except Exception as e:
                logger.error(f"处理股票 {symbol} 数据时出错: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"处理股票 {symbol} 数据初始化阶段出错: {str(e)}")
            return False
    
    def update_stock_data(self, symbols=None):
        """更新股票数据 - 增量更新两张表
        
        Args:
            symbols: 要更新的股票列表，如果为None则使用默认列表
        """
        end_date = datetime.now()
        default_start_date = datetime(2015, 1, 1)  # 默认从2015年开始

        logger.info(f"开始增量更新股票数据，截止日期：{end_date.date()}")
        
        # 加载股票列表
        if symbols is None:
            symbols = self.load_stock_lists()
        
        # 获取已存在的股票列表
        existing_stocks = self.db_manager.get_existing_stocks()
        logger.info(f"数据库中已有 {len(existing_stocks)} 只股票的数据")
        
        # 获取表列名
        time_code_columns = self.db_manager.get_table_columns('stock_time_code')
        code_time_columns = self.db_manager.get_table_columns('stock_code_time')
        
        # 获取市场开放的交易日，避免更新非交易日数据
        trading_days = get_trading_days(end_date)

        # 记录更新的股票
        updated_symbols = []
        skipped_symbols = []
        
        # 为每只股票确定开始日期
        start_dates = {}
        max_batch_size = 20  # 每批处理的股票数
        
        # 1. 确定每只股票的开始日期
        for symbol in symbols:
            last_update_date = None
            if symbol in existing_stocks:
                last_update_date = self.db_manager.get_last_update_date(symbol)
                if last_update_date:
                    # 将日期向后推一天，避免重复数据
                    start_date = last_update_date + timedelta(days=1)
                    
                    # 确保start_date是datetime对象
                    if not isinstance(start_date, datetime):
                        if isinstance(start_date, date):
                            start_date = datetime.combine(start_date, datetime.min.time())
                        else:
                            try:
                                start_date = datetime.strptime(str(start_date), '%Y-%m-%d')
                            except ValueError:
                                logger.error(f"无法解析日期: {start_date}，使用默认开始日期")
                                start_date = default_start_date
                    
                    # 检查是否为交易日
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    current_date_str = end_date.strftime('%Y-%m-%d')
                    
                    if start_date_str not in trading_days and start_date_str < current_date_str:
                        # 如果开始日期不是交易日，找到下一个交易日
                        next_trading_day = self.get_next_trading_day(start_date, trading_days)
                        if next_trading_day:
                            start_date = datetime.combine(next_trading_day, datetime.min.time())
                    
                    # 如果最后更新日期是今天或未来，跳过这只股票
                    if start_date.strftime('%Y-%m-%d') >= end_date.strftime('%Y-%m-%d'):
                        logger.info(f"股票 {symbol} 已是最新数据，跳过更新")
                        skipped_symbols.append(symbol)
                        continue
                        
                    logger.info(f"股票 {symbol} 最后更新日期: {last_update_date.date() if hasattr(last_update_date, 'date') else last_update_date}, 将获取从 {start_date.date() if hasattr(start_date, 'date') else start_date} 到 {end_date.date()} 的数据")
                    start_dates[symbol] = start_date
                else:
                    logger.info(f"股票 {symbol} 在数据库中存在但无法确定最后更新日期，将获取从 {default_start_date.date()} 的完整历史数据")
                    start_dates[symbol] = default_start_date
            else:
                logger.info(f"股票 {symbol} 在数据库中不存在，将获取从 {default_start_date.date()} 的完整历史数据")
                start_dates[symbol] = default_start_date
        
        # 2. 按批次处理股票
        symbols_to_process = [s for s in symbols if s not in skipped_symbols]
        for i in range(0, len(symbols_to_process), max_batch_size):
            batch_symbols = symbols_to_process[i:i+max_batch_size]
            batch_start_dates = {symbol: start_dates.get(symbol, default_start_date) for symbol in batch_symbols}
            
            logger.info(f"处理第 {i//max_batch_size + 1} 批股票，共 {len(batch_symbols)} 只")
            
            # 获取数据
            batch_data = self.data_fetcher.get_stock_data_batch(batch_symbols, batch_start_dates, end_date)
            
            # 处理数据
            for symbol, df in batch_data.items():
                if df is not None and not df.empty:
                    success = self.process_stock_data(symbol, df, time_code_columns, code_time_columns)
                    if success:
                        updated_symbols.append(symbol)
                    else:
                        skipped_symbols.append(symbol)
                else:
                    logger.warning(f"股票 {symbol} 没有获取到数据")
                    skipped_symbols.append(symbol)
        
        logger.info(f"股票数据增量更新完成，共更新了 {len(updated_symbols)} 只股票，跳过了 {len(skipped_symbols)} 只股票")
        
        # 检查并同步两个表之间的差异
        try:
            logger.info("开始检查并同步两表之间的差异")
            self.db_manager.sync_tables_differences()
            logger.info("表同步完成")
        except Exception as e:
            logger.error(f"同步表之间差异时出错: {str(e)}")


def main():
    """主函数"""
    try:
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
