import pandas as pd
import numpy as np
import yfinance as yf
import time
import datetime as dt
from typing import List, Dict, Optional, Union, Tuple
from sqlalchemy import create_engine, text
import pymysql
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataFetcher")

# 数据库配置信息
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "mose"
}


class DataFetcher:
    """数据获取器，负责从各种数据源获取市场数据"""
    
    def __init__(self, config=None):
        """初始化数据获取器"""
        self.config = config or {}
        self.api_delay = self.config.get('api_delay', 1.0)  # 默认延迟1秒
        self.logger = logging.getLogger(__name__)
        self.max_workers = 5  # 默认最大工作线程数
        self.engine = None
        self.cache_data = True
        self.data_cache = {}
        self.cache_timestamps = {}
        self.cache_expiry = 3600  # 缓存过期时间（秒）
        self._executor = None  # 线程池执行器
        
    def __del__(self):
        """析构函数，确保线程池被正确关闭"""
        self._cleanup()
        
    def _cleanup(self):
        """清理资源"""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
                self._executor = None
            except Exception as e:
                self.logger.error(f"关闭线程池时出错: {e}")
                
    def _get_executor(self):
        """获取或创建线程池执行器"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor
        
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        获取历史数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            interval: 数据间隔 ('1d', '1h', '1m'等)
            
        返回:
            历史数据DataFrame
        """
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if data.empty:
                self.logger.warning(f"无法获取 {symbol} 的历史数据")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            self.logger.error(f"获取 {symbol} 的历史数据时发生错误: {str(e)}")
            return pd.DataFrame()
            
    def get_latest_data(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        获取多个股票的最新数据
        
        参数:
            symbols: 股票代码列表
            days: 获取最近多少天的数据
            
        返回:
            字典，键为股票代码，值为对应的DataFrame
        """
        try:
            end_date = dt.datetime.now().strftime('%Y-%m-%d')
            start_date = (dt.datetime.now() - dt.timedelta(days=days)).strftime('%Y-%m-%d')
            
            result = {}
            executor = self._get_executor()
            
            # 创建任务
            future_to_symbol = {
                executor.submit(self.get_historical_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # 获取结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        result[symbol] = data
                    else:
                        self.logger.warning(f"未获取到 {symbol} 的数据")
                except Exception as e:
                    self.logger.error(f"获取 {symbol} 的数据时出错: {e}")
                    
                # API调用间隔
                time.sleep(self.api_delay)
                    
            return result
        except Exception as e:
            self.logger.error(f"获取最新数据时出错: {e}")
            return {}
            
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        获取实时数据
        
        参数:
            symbols: 股票代码列表
            
        返回:
            实时数据字典
        """
        try:
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info:
                    data[symbol] = pd.DataFrame([{
                        'open': info.get('regularMarketOpen'),
                        'high': info.get('regularMarketDayHigh'),
                        'low': info.get('regularMarketDayLow'),
                        'close': info.get('regularMarketPrice'),
                        'volume': info.get('regularMarketVolume'),
                        'timestamp': dt.datetime.now()
                    }])
                    
            return data
            
        except Exception as e:
            self.logger.error(f"获取实时数据时发生错误: {str(e)}")
            return {}

    def get_stock_list(self, table: str = 'uss_nasdaq_stocks') -> pd.DataFrame:
        """
        从数据库获取股票列表
        
        参数:
            table: 股票列表表名
            
        返回:
            股票列表DataFrame
        """
        try:
            # 检查缓存
            cache_key = f"stock_list_{table}"
            if self.cache_data and cache_key in self.data_cache:
                cache_time = self.cache_timestamps.get(cache_key, 0)
                if time.time() - cache_time < self.cache_expiry:
                    self.logger.debug(f"从缓存获取股票列表 {table}")
                    return self.data_cache[cache_key]
            
            # 查询表数据
            query = f"SELECT * FROM {table}"
            stock_list = pd.read_sql(query, self.engine)
            
            # 更新缓存
            if self.cache_data:
                self.data_cache[cache_key] = stock_list
                self.cache_timestamps[cache_key] = time.time()
                
            self.logger.info(f"从数据库获取股票列表成功，共 {len(stock_list)} 只股票")
            return stock_list
        except Exception as e:
            self.logger.error(f"从数据库获取股票列表时出错: {e}")
            return pd.DataFrame()
            
    def _get_data_from_db(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从数据库获取数据"""
        query = """
        SELECT 
            Date as date,
            Open as open,
            High as high,
            Low as low,
            Close as close,
            Volume as volume,
            AdjClose as adj_close
        FROM stock_code_time
        WHERE Code = %s
        AND Date BETWEEN %s AND %s
        ORDER BY Date ASC
        """
        try:
            # 使用 Pandas 读取查询结果
            data = pd.read_sql_query(query, self.engine, params=(symbol, start_date, end_date))
            
            if data.empty:
                self.logger.warning(f"数据库中没有找到 {symbol} 的数据")
                return pd.DataFrame()
                
            # 将日期列设置为索引
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            self.logger.info(f"从数据库获取 {symbol} 的历史数据成功，共 {len(data)} 条记录")
            return data
        except Exception as e:
            self.logger.error(f"从数据库读取 {symbol} 的数据失败: {e}")
            return pd.DataFrame()
            
    def _get_data_from_api(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从API获取股票数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            股票数据DataFrame
        """
        try:
            self.logger.info(f"从API获取 {symbol} 的数据，时间范围：{start_date} 至 {end_date}")
            
            # 获取数据
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"未获取到 {symbol} 的数据")
                return pd.DataFrame()
                
            # 确保所有必要的列都存在
            if 'Adj Close' not in data.columns and 'Close' in data.columns:
                data['Adj Close'] = data['Close']
                
            # 重命名列
            if 'Adj Close' in data.columns:
                data = data.rename(columns={'Adj Close': 'AdjClose'})
                
            # 确保所有必要的列都存在
            for col in ['Dividends', 'Stock Splits', 'Capital Gains']:
                if col not in data.columns:
                    data[col] = 0
                    
            # 重命名列
            data = data.rename(columns={
                'Stock Splits': 'StockSplits',
                'Capital Gains': 'Capital_Gains'
            })
            
            # 添加股票代码列
            data['Code'] = symbol
            
            # 重置索引，将日期变为列
            data = data.reset_index()
            data = data.rename(columns={'Date': 'date'})
            
            # 确保日期列是datetime类型
            data['date'] = pd.to_datetime(data['date'])
            
            self.logger.info(f"成功获取 {symbol} 的数据，共 {len(data)} 条记录")
            
            # 保存到数据库
            self._save_data_to_db(data, symbol)
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取 {symbol} 的数据时出错: {e}")
            return pd.DataFrame()
            
    def _save_data_to_db(self, df: pd.DataFrame, symbol: str) -> None:
        """
        将数据保存到数据库
        :param df: 要保存的数据
        :param symbol: 股票代码
        :return: None
        """
        if df.empty:
            self.logger.warning("没有数据需要保存到数据库")
            return

        try:
            # 重置索引，确保date列存在
            df = df.reset_index()
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)
            elif 'Date' in df.columns:
                df['date'] = df['Date'].astype(str)
                df = df.drop('Date', axis=1)

            # 添加股票代码列
            df['Code'] = symbol

            # 确保所有数值列都是float类型
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Dividends', 'StockSplits', 'Capital_Gains']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            # 准备SQL语句
            sql = """
                INSERT INTO stock_code_time 
                (date, Code, Open, High, Low, Close, Volume, AdjClose, Dividends, StockSplits, Capital_Gains)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                Open = VALUES(Open),
                High = VALUES(High),
                Low = VALUES(Low),
                Close = VALUES(Close),
                Volume = VALUES(Volume),
                AdjClose = VALUES(AdjClose),
                Dividends = VALUES(Dividends),
                StockSplits = VALUES(StockSplits),
                Capital_Gains = VALUES(Capital_Gains)
            """

            # 使用事务批量插入数据
            with self.engine.connect() as conn:
                with conn.begin():
                    for _, row in df.iterrows():
                        values = (
                            row['date'], row['Code'],
                            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],
                            row['AdjClose'], row['Dividends'], row['StockSplits'], row['Capital_Gains']
                        )
                        conn.execute(sql, values)

            self.logger.info(f"成功保存 {symbol} 的数据到数据库，共 {len(df)} 条记录")

        except Exception as e:
            self.logger.error(f"保存 {symbol} 的数据到数据库时出错: {e}")
            raise
            
    def update_all_stocks(
        self,
        stock_list: Optional[pd.DataFrame] = None,
        batch_size: int = 20,
        start_date: Optional[str] = None
    ) -> None:
        """
        更新所有股票的数据
        
        参数:
            stock_list: 股票列表DataFrame，如果为None则从数据库获取
            batch_size: 每批处理的股票数量
            start_date: 开始日期，如果为None则使用数据库中最新日期的下一天
        """
        try:
            # 获取股票列表
            if stock_list is None:
                stock_list = self.get_stock_list()
                
            if stock_list.empty:
                self.logger.error("股票列表为空，无法更新数据")
                return
                
            # 确定增量更新时间范围
            if start_date is None:
                # 获取数据库中最新的日期
                latest_date = self._get_latest_date_from_db()
                
                if latest_date:
                    # 将日期转换为datetime对象
                    latest_date_dt = pd.to_datetime(latest_date)
                    # 添加一天得到开始日期
                    start_date = (latest_date_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    # 如果数据库中没有数据，默认从2022年开始获取
                    start_date = "2022-01-01"
            
            # 使用当前日期作为结束日期
            end_date = dt.datetime.now().strftime('%Y-%m-%d')
            
            # 检查日期是否有效
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            
            # 如果开始日期晚于结束日期，则无需更新
            if start_date_dt > end_date_dt:
                self.logger.info(f"开始日期 {start_date} 晚于结束日期 {end_date}，无需更新数据")
                return
                
            self.logger.info(f"开始增量更新数据，时间范围: {start_date} 至 {end_date}")
            
            # 获取股票代码列表
            stock_codes = stock_list["Code"].tolist()
            
            # 分批处理
            for i in range(0, len(stock_codes), batch_size):
                batch_codes = stock_codes[i:i+batch_size]
                self.logger.info(f"处理第 {i//batch_size + 1} 批股票，共 {len(batch_codes)} 只")
                
                # 使用线程池并发获取数据
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_symbol = {
                        executor.submit(self.get_historical_data, symbol, start_date, end_date, False): symbol
                        for symbol in batch_codes
                    }
                    
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            future.result()  # 数据已经在函数内部保存到数据库
                        except Exception as e:
                            self.logger.error(f"更新 {symbol} 的数据时出错: {e}")
                
                # 批次之间添加延迟
                if i + batch_size < len(stock_codes):
                    self.logger.info("等待5秒后处理下一批...")
                    time.sleep(5)
                    
            self.logger.info("所有数据更新完成")
        except Exception as e:
            self.logger.error(f"更新所有股票数据时出错: {e}")
            
    def _get_latest_date_from_db(self) -> Optional[str]:
        """从数据库获取最新日期"""
        try:
            # 查询stock_code_time表的最新日期
            query1 = "SELECT MAX(Date) AS latest_date FROM stock_code_time"
            # 查询stock_time_code表的最新日期
            query2 = "SELECT MAX(Date) AS latest_date FROM stock_time_code"
            
            # 执行查询
            date1 = pd.read_sql(query1, self.engine).iloc[0, 0]
            date2 = pd.read_sql(query2, self.engine).iloc[0, 0]
            
            # 取两个日期中的最大值
            if date1 and date2:
                return max(date1, date2)
            elif date1:
                return date1
            elif date2:
                return date2
            else:
                return None
        except Exception as e:
            self.logger.error(f"从数据库获取最新日期时出错: {e}")
            return None
            
    def check_connection(self) -> bool:
        """检查数据库连接是否正常"""
        try:
            if self.engine is None:
                return False
            # 尝试执行一个简单的查询
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(f"数据库连接检查失败: {e}")
            return False

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        清除数据缓存
        
        参数:
            symbol: 指定要清除缓存的股票代码，如果为None则清除所有缓存
        """
        try:
            if symbol:
                # 清除指定股票的缓存
                keys_to_remove = [k for k in self.data_cache.keys() if symbol in k]
                for key in keys_to_remove:
                    self.data_cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
                self.logger.info(f"清除 {symbol} 的缓存数据")
            else:
                # 清除所有缓存
                self.data_cache.clear()
                self.cache_timestamps.clear()
                self.logger.info("清除所有缓存数据")
        except Exception as e:
            self.logger.error(f"清除缓存时出错: {e}")
            
    def get_cache_status(self) -> Dict:
        """
        获取缓存状态
        
        返回:
            包含缓存信息的字典
        """
        try:
            return {
                'cache_size': len(self.data_cache),
                'cached_symbols': list(set(k.split('_')[1] for k in self.data_cache.keys() if k.startswith('hist_'))),
                'last_update': {k: dt.datetime.fromtimestamp(v).strftime('%Y-%m-%d %H:%M:%S')
                              for k, v in self.cache_timestamps.items()}
            }
        except Exception as e:
            self.logger.error(f"获取缓存状态时出错: {e}")
            return {}

    def update_config(self, new_config: Dict):
        """更新配置"""
        try:
            if 'cache_data' in new_config:
                self.cache_data = new_config['cache_data']
            if 'cache_expiry' in new_config:
                self.cache_expiry = new_config['cache_expiry']
            if 'max_workers' in new_config:
                self.max_workers = new_config['max_workers']
            if 'api_delay' in new_config:
                self.api_delay = new_config['api_delay']
            
            # 如果数据库配置改变，重新创建连接
            if 'db_config' in new_config:
                self.db_config = new_config['db_config']
                self.engine = self._create_db_engine()
                
            self.logger.info("DataFetcher配置已更新")
        except Exception as e:
            self.logger.error(f"更新配置失败: {e}")

    def get_status(self) -> Dict:
        """获取数据获取器状态"""
        return {
            "db_connected": self.check_connection(),
            "cache_status": self.get_cache_status(),
            "max_workers": self.max_workers,
            "api_delay": self.api_delay
        }

    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        检查数据质量
        
        参数:
            data: 待检查的数据
        
        返回:
            数据是否有效
        """
        try:
            # 检查必要的列是否存在
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.warning("数据缺少必要的列")
                return False
            
            # 检查是否有空值
            if data[required_columns].isnull().any().any():
                self.logger.warning("数据存在空值")
                return False
            
            # 检查数据是否合理
            if (data['High'] < data['Low']).any():
                self.logger.warning("数据存在不合理的高低价")
                return False
            
            # 检查成交量是否为负
            if (data['Volume'] < 0).any():
                self.logger.warning("数据存在负的成交量")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"数据质量检查出错: {e}")
            return False

    def detect_anomalies(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        检测数据异常
        
        参数:
            data: 原始数据
            window: 移动窗口大小
            
        返回:
            带有异常标记的数据
        """
        try:
            # 计算移动平均和标准差
            rolling_mean = data['Close'].rolling(window=window).mean()
            rolling_std = data['Close'].rolling(window=window).std()
            
            # 计算z-score
            z_scores = (data['Close'] - rolling_mean) / rolling_std
            
            # 标记异常（z-score超过3个标准差）
            data['is_anomaly'] = abs(z_scores) > 3
            
            # 记录异常
            anomalies = data[data['is_anomaly']]
            if not anomalies.empty:
                self.logger.warning(f"检测到 {len(anomalies)} 个数据异常")
            
            return data
        except Exception as e:
            self.logger.error(f"检测数据异常时出错: {e}")
            return data

    def backup_database(self, backup_path: str = "backups") -> bool:
        """
        备份数据库
        
        参数:
            backup_path: 备份文件保存路径
            
        返回:
            备份是否成功
        """
        try:
            # 创建备份目录
            os.makedirs(backup_path, exist_ok=True)
            
            # 生成备份文件名
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_path, f"db_backup_{timestamp}.sql")
            
            # 执行备份命令
            command = f"mysqldump -h {self.db_config['host']} -P {self.db_config['port']} " \
                     f"-u {self.db_config['user']} -p{self.db_config['password']} " \
                     f"{self.db_config['database']} > {backup_file}"
            
            os.system(command)
            self.logger.info(f"数据库备份成功: {backup_file}")
            return True
        except Exception as e:
            self.logger.error(f"数据库备份失败: {e}")
            return False

    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据格式"""
        if df.empty:
            return df
            
        # 统一列名为小写
        df.columns = df.columns.str.lower()
        return df 