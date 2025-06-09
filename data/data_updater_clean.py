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
from data.data_validator import DataValidator  # 集成数据验证器

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
        logging.FileHandler('data_updater_clean.log'),
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
        self.config = config
        self.engine = self._create_engine_with_retry()

    def _create_engine_with_retry(self, max_retries=3):
        for i in range(max_retries):
            try:
                engine = create_engine(
                    f"mysql+pymysql://{self.config['user']}:{self.config['password']}@"
                    f"{self.config['host']}:{self.config['port']}/{self.config['database']}",
                    pool_recycle=3600, pool_timeout=60, connect_args={'connect_timeout': 60, 'read_timeout': 60, 'write_timeout': 60}
                )
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
            query = text("""
                SELECT DISTINCT Code FROM stock_time_code
                WHERE Code NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' AND Code IS NOT NULL AND Code != ''
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query)
                existing_stocks = [row[0] for row in result]
            # 过滤掉任何看起来像日期的代码
            existing_stocks = [code for code in existing_stocks if not (isinstance(code, str) and len(code) == 10 and code.count('-') == 2)]
            logger.info(f"从数据库加载了 {len(existing_stocks)} 只股票")
            return existing_stocks
        except Exception as e:
            logger.error(f"获取已存在股票列表时出错: {str(e)}")
            return []

    def get_last_update_date(self, symbol):
        """获取股票最后更新日期，同时检查两个表中的最新日期"""
        try:
            time_code_query = text("SELECT MAX(Date) FROM stock_time_code WHERE Code = :symbol")
            with self.engine.connect() as conn:
                result = conn.execute(time_code_query, {"symbol": symbol})
                time_code_last_date = result.scalar()
            code_time_query = text("SELECT MAX(Date) FROM stock_code_time WHERE Code = :symbol")
            with self.engine.connect() as conn:
                result = conn.execute(code_time_query, {"symbol": symbol})
                code_time_last_date = result.scalar()
            if time_code_last_date and code_time_last_date:
                last_date = max(time_code_last_date, code_time_last_date)
            elif time_code_last_date:
                last_date = time_code_last_date
            elif code_time_last_date:
                last_date = code_time_last_date
            else:
                last_date = None
            if last_date and isinstance(last_date, date) and not isinstance(last_date, datetime):
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
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT @@SESSION.sql_mode"))
                original_mode = result.scalar()
                try:
                    conn.execute(text("SET SESSION sql_mode=''"))
                    clean_query = """
                        DELETE FROM stock_time_code
                        WHERE Date NOT REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' OR Date = '0000-00-00'
                    """
                    conn.execute(text(clean_query))
                    conn.commit()
                    logger.info("已清理stock_time_code表中的无效日期数据")
                finally:
                    conn.execute(text(f"SET SESSION sql_mode='{original_mode}'"))
        except Exception as e:
            logger.error(f"同步表之间差异时出错: {str(e)}")
            raise

    def save_stock_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """保存股票数据到数据库"""
        try:
            if df.empty:
                logger.warning(f"股票 {symbol} 没有数据需要保存")
                return False
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            df['Code'] = symbol
            if 'Amount' in df.columns:
                df = df.drop('Amount', axis=1)
            conn = pymysql.connect(**self.config)
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SET SESSION sql_mode=''")
                    columns = df.columns.tolist()
                    column_names = ', '.join(['`' + col + '`' for col in columns])
                    placeholders = ', '.join(['%s'] * len(columns))
                    sql = f"REPLACE INTO stock_time_code ({column_names}) VALUES ({placeholders})"
                    for _, row in df.iterrows():
                        values = [row[col] for col in columns]
                        cursor.execute(sql, values)
                    conn.commit()
                    sql = f"REPLACE INTO stock_code_time ({column_names}) VALUES ({placeholders})"
                    for _, row in df.iterrows():
                        values = [row[col] for col in columns]
                        cursor.execute(sql, values)
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
            logger.error(f"保存股票 {symbol} 数据初始化阶段出错: {str(e)}")
            return False

class StockDataFetcher:
    """负责从API获取股票数据的类"""
    def __init__(self, proxies=None):
        self.proxies = proxies

    def get_stock_data(self, symbol, start_date, end_date):
        """获取单个股票的历史数据"""
        max_retries = 3
        base_delay = 1
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=start_date, end=end_date, interval="1d", prepost=False, actions=True)
                if df.empty:
                    logger.warning(f"股票 {symbol} 返回空数据")
                    return None
                df = df.reset_index()
                df = df.rename(columns={'Stock Splits': 'StockSplits', 'Capital Gains': 'Capital_Gains'})
                df['Code'] = symbol
                df['Amount'] = df['Close'] * df['Volume']
                df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)
                df = df.sort_values('Date')
                time.sleep(0.3)
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"获取股票 {symbol} 数据时出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"获取股票 {symbol} 数据失败: {str(e)}")
                    return None
        return None

    def get_stock_data_batch(self, symbols, start_dates, end_date, max_workers=5):
        """并行获取多只股票的数据"""
        results = {}
        if not isinstance(start_dates, dict):
            start_dates = {symbol: start_dates for symbol in symbols}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.get_stock_data, symbol, start_dates.get(symbol), end_date): symbol for symbol in symbols}
            for future in tqdm(concurrent.futures.as_completed(future_to_symbol), total=len(symbols), desc="获取股票数据"):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[symbol] = data
                except Exception as e:
                    logger.error(f"获取股票 {symbol} 数据时出错: {str(e)}")
        return results

class MarketDataUpdater:
    def __init__(self, db_config, proxies=None):
        self.db_manager = DatabaseManager(db_config)
        self.data_fetcher = StockDataFetcher(proxies)
        self._last_update_times = {}

    def process_stock_data(self, symbol, df, time_code_columns, code_time_columns):
        """
        处理单只股票的数据，先用 DataValidator 清洗和修复，不能修复的直接丢弃
        """
        try:
            if df.empty:
                logger.info(f"股票 {symbol} 没有数据需要更新")
                return False

            # 统一列名为小写，便于 DataValidator 处理
            df.columns = [col.lower() for col in df.columns]
            # 用 DataValidator 清洗和修复
            df, report = DataValidator.validate_data(df)
            if not report.get('validation_passed', True):
                logger.warning(f"股票 {symbol} 数据清洗后仍有严重问题，已丢弃")
                return False
            if df.empty:
                logger.warning(f"股票 {symbol} 清洗后无有效数据")
                return False

            # 恢复原有的列名格式（首字母大写）以兼容数据库
            df.columns = [col.capitalize() for col in df.columns]
            if 'Code' not in df.columns:
                df['Code'] = symbol
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

            # 只保留数据库表需要的列
            valid_time_code_columns = [col for col in df.columns if col in time_code_columns]
            if not valid_time_code_columns:
                logger.error(f"股票 {symbol} 的数据与stock_time_code表结构不兼容，无法插入")
                return False
            df = df[valid_time_code_columns]

            # 保存到数据库
            return self.db_manager.save_stock_data(symbol, df)
        except Exception as e:
            logger.error(f"处理股票 {symbol} 数据时出错: {str(e)}")
            return False

    def get_last_update_time(self, symbol: str) -> datetime:
        if symbol in self._last_update_times:
            return self._last_update_times[symbol]
        last_update = self.db_manager.get_last_update_date(symbol)
        self._last_update_times[symbol] = last_update
        return last_update

    def get_last_update_times(self) -> Dict[str, datetime]:
        symbols = self.db_manager.get_existing_stocks()
        return {symbol: self.get_last_update_time(symbol) for symbol in symbols}

    def load_stock_lists(self):
        """加载SP500和Nasdaq100的股票列表"""
        sp500_df = pd.read_csv('stock_pool/sp500_stocks.csv')
        sp500_symbols = sp500_df['Code'].tolist()
        nasdaq100_df = pd.read_csv('stock_pool/nasdaq100_stocks.csv')
        nasdaq100_symbols = nasdaq100_df['Code'].tolist()
        etf_stocks_df = pd.read_csv('stock_pool/uss_etf_stocks.csv')
        etf_symbols = etf_stocks_df['Code'].tolist()
        for symbol in nasdaq100_symbols + etf_symbols + ['BABA', 'BRK.B', 'LVMUY', 'SPY']:
            if symbol not in sp500_symbols:
                sp500_symbols.append(symbol)
        logger.info(f"从SP500列表中加载了 {len(sp500_symbols)} 只股票")
        return sp500_symbols

    def get_next_trading_day(self, date, trading_days):
        if isinstance(date, (datetime, date)):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        for trading_day in trading_days:
            if trading_day > date_str:
                return datetime.strptime(trading_day, '%Y-%m-%d').date()
        return None

    def is_market_closed(self):
        now = datetime.now()
        market_close_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        return now > market_close_time

    def is_data_complete(self, symbol, date_str):
        try:
            query = text("""
                SELECT Volume, Close FROM stock_time_code WHERE Code = :symbol AND Date = :date
            """)
            with self.db_manager.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol, "date": date_str}).fetchone()
                if result is None:
                    return False
                volume, close = result
                return volume > 0 and close is not None
        except Exception as e:
            logger.error(f"检查数据完整性时出错: {str(e)}")
            return False

    def update_stock_data(self, symbols: List[str] = None, force_update: bool = False) -> Dict[str, Any]:
        if symbols is None:
            symbols = self.db_manager.get_existing_stocks()
        report = {'total': len(symbols), 'updated': 0, 'skipped': 0, 'failed': 0, 'details': {}}
        for symbol in symbols:
            try:
                last_update = self.get_last_update_time(symbol)
                if not force_update and last_update:
                    if isinstance(last_update, str):
                        last_update = pd.to_datetime(last_update)
                    elif isinstance(last_update, date) and not isinstance(last_update, datetime):
                        last_update = datetime.combine(last_update, datetime.min.time())
                    time_diff = datetime.now() - last_update
                    if time_diff.days < 1:
                        report['skipped'] += 1
                        report['details'][symbol] = 'skipped (up to date)'
                        continue
                end_date = datetime.now()
                start_date = end_date - timedelta(days=120)
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval="1d", prepost=False, actions=True)
                if df.empty:
                    logger.warning(f"未获取到股票 {symbol} 的新数据")
                    report['failed'] += 1
                    report['details'][symbol] = 'failed (empty data)'
                    continue
                df = self._process_data(df, symbol)
                if self.db_manager.save_stock_data(symbol, df):
                    report['updated'] += 1
                    report['details'][symbol] = 'updated'
                    self._last_update_times[symbol] = datetime.now()
                else:
                    report['failed'] += 1
                    report['details'][symbol] = 'failed (save error)'
            except Exception as e:
                logger.error(f"更新股票 {symbol} 数据时出错: {str(e)}")
                report['failed'] += 1
                report['details'][symbol] = f'error: {str(e)}'
        return report

    def _process_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        try:
            df = df.reset_index()
            df = df.rename(columns={'Stock Splits': 'StockSplits', 'Capital Gains': 'Capital_Gains'})
            df['Code'] = symbol
            df['Amount'] = df['Close'] * df['Volume']
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            df = df.sort_values('Date')
            return df
        except Exception as e:
            logger.error(f"处理股票 {symbol} 数据时出错: {str(e)}")
            return pd.DataFrame()

    def find_missing_dates(self, symbol, start_date=None, end_date=None):
        """
        查找某只股票在数据库中缺失的日期
        :param symbol: 股票代码
        :param start_date: 检查的起始日期（默认数据库最小日期）
        :param end_date: 检查的结束日期（默认数据库最大日期）
        :return: 缺失日期列表（字符串格式 'YYYY-MM-DD'）
        """
        try:
            with self.db_manager.engine.connect() as conn:
                # 获取该股票所有已存在的日期
                query = text("SELECT DISTINCT Date FROM stock_time_code WHERE Code = :symbol AND Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'")
                result = conn.execute(query, {"symbol": symbol})
                existing_dates = set([str(row[0]) for row in result if row[0] is not None])
                # 获取全市场日期范围
                min_date_query = text("SELECT MIN(Date) FROM stock_time_code WHERE Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'")
                max_date_query = text("SELECT MAX(Date) FROM stock_time_code WHERE Date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'")
                min_date = conn.execute(min_date_query).scalar()
                max_date = conn.execute(max_date_query).scalar()
                if not min_date or not max_date:
                    return []
                if start_date is None:
                    start_date = min_date
                if end_date is None:
                    end_date = max_date
                # 生成完整日期序列
                all_dates = pd.date_range(start=start_date, end=end_date, freq='B').strftime('%Y-%m-%d').tolist()
                missing_dates = [d for d in all_dates if d not in existing_dates]
                return missing_dates
        except Exception as e:
            logger.error(f"查找股票 {symbol} 缺失日期时出错: {str(e)}")
            return []

    def fill_missing_dates(self, symbol):
        """
        自动补齐某只股票在数据库中缺失的日期数据
        """
        missing_dates = self.find_missing_dates(symbol)
        if not missing_dates:
            logger.info(f"股票 {symbol} 没有缺失日期，无需补齐")
            return 0
        logger.info(f"股票 {symbol} 缺失 {len(missing_dates)} 个交易日，正在补齐...")
        # 按年份分批拉取，避免yfinance接口超时
        filled_count = 0
        for year, group in pd.DataFrame({'date': missing_dates}).groupby(lambda x: pd.to_datetime(missing_dates[x]).year):
            year_dates = group['date'].tolist()
            if not year_dates:
                continue
            start = year_dates[0]
            end = year_dates[-1]
            df = self.data_fetcher.get_stock_data(symbol, start, end)
            if df is not None and not df.empty:
                # 只保留缺失的日期
                df = df[df['Date'].isin(year_dates)]
                # 清洗和保存
                time_code_columns = self.db_manager.get_table_columns('stock_time_code')
                code_time_columns = self.db_manager.get_table_columns('stock_code_time')
                if self.process_stock_data(symbol, df, time_code_columns, code_time_columns):
                    filled_count += len(df)
        logger.info(f"股票 {symbol} 已补齐 {filled_count} 条缺失数据")
        return filled_count

    def fill_all_missing_dates(self):
        """
        对所有股票进行全量缺失日期补齐
        """
        symbols = self.db_manager.get_existing_stocks()
        total_filled = 0
        for symbol in tqdm(symbols, desc="补齐所有股票缺失日期"):
            total_filled += self.fill_missing_dates(symbol)
        logger.info(f"所有股票已补齐缺失数据，共补齐 {total_filled} 条记录")
        return total_filled

def main():
    try:
        updater = MarketDataUpdater(DB_CONFIG, PROXIES)
        updater.update_stock_data()
        # 新增：补齐所有股票缺失日期
        updater.fill_all_missing_dates()
        logger.info("股票数据处理完成 - 数据已清洗、同步并补齐缺失日期")
    except Exception as e:
        logger.error(f"运行过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 