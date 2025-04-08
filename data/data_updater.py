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
            
            # 分别保存到两个表
            try:
                # 获取连接
                conn = pymysql.connect(
                    host=self.config['host'],
                    port=self.config['port'],
                    user=self.config['user'],
                    password=self.config['password'],
                    database=self.config['database']
                )
                
                try:
                    with conn.cursor() as cursor:
                        # 临时关闭严格模式
                        cursor.execute("SET SESSION sql_mode=''")
                        
                        # 获取列名
                        columns = df.columns.tolist()
                        column_names = ', '.join(['`' + col + '`' for col in columns])
                        
                        # 构建占位符
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
                        
                    logger.info(f"成功保存股票 {symbol} 的数据，总计 {len(df)} 条记录")
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
            
        report = {
            'total': len(symbols),
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'details': {}
        }
        
        for symbol in symbols:
            try:
                # 检查是否需要更新
                last_update = self.get_last_update_time(symbol)
                if not force_update and last_update and (datetime.now() - last_update).days < 1:
                    report['skipped'] += 1
                    report['details'][symbol] = 'skipped (up to date)'
                    continue
                    
                # 更新数据
                success = self._update_single_stock(symbol)
                if success:
                    report['updated'] += 1
                    report['details'][symbol] = 'updated'
                    self._last_update_times[symbol] = datetime.now()
                else:
                    report['failed'] += 1
                    report['details'][symbol] = 'failed'
                    
            except Exception as e:
                logger.error(f"更新股票 {symbol} 数据时出错: {str(e)}")
                report['failed'] += 1
                report['details'][symbol] = f'error: {str(e)}'
                
        return report
        
    def _update_single_stock(self, symbol: str) -> bool:
        """更新单个股票的数据"""
        try:
            # 获取最新数据
            end_date = datetime.now()
            # 增加获取数据的时间范围，确保有足够的数据
            start_date = self.get_last_update_time(symbol) or (end_date - timedelta(days=60))
            
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
                
            # 处理数据
            df = self._process_data(df, symbol)
            
            # 验证数据
            if len(df) < 20:  # 如果数据点不足20个
                logger.warning(f"股票 {symbol} 数据点不足: {len(df)} < 20")
                # 尝试获取更多历史数据
                extended_start_date = end_date - timedelta(days=120)  # 扩展到120天
                extended_df = ticker.history(
                    start=extended_start_date,
                    end=end_date,
                    interval="1d",
                    prepost=False,
                    actions=True
                )
                if not extended_df.empty:
                    extended_df = self._process_data(extended_df, symbol)
                    if len(extended_df) > len(df):
                        df = extended_df
                        logger.info(f"成功获取更多历史数据，现在有 {len(df)} 个数据点")
            
            # 保存到数据库
            self.db_manager.save_stock_data(symbol, df)
            
            return True
            
        except Exception as e:
            logger.error(f"更新股票 {symbol} 数据时出错: {str(e)}")
            return False
            
    def _process_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """处理股票数据"""
        try:
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
            
            return df
            
        except Exception as e:
            logger.error(f"处理股票 {symbol} 数据时出错: {str(e)}")
            return pd.DataFrame()


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
