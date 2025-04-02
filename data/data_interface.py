# data/data_interface.py
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple
import datetime as dt
import pymysql
from pymysql.cursors import DictCursor
import os
import sys

# 添加项目根目录到sys.path以便导入config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.data_config import default_data_config

class DataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: dt.datetime, 
                           end_date: dt.datetime, timeframe: str = 'daily') -> pd.DataFrame:
        """获取历史OHLCV数据"""
        pass
    
    @abstractmethod
    def get_multiple_symbols(self, symbols: List[str], start_date: dt.datetime,
                            end_date: dt.datetime, timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
        """获取多个股票的历史数据"""
        pass
    
    @abstractmethod
    def get_latest_data(self, symbol: str, n_bars: int = 1, 
                       timeframe: str = 'daily') -> pd.DataFrame:
        """获取最新的n条数据"""
        pass
    
    @abstractmethod
    def search_symbols(self, query: str) -> List[Dict]:
        """搜索股票代码"""
        pass


class MySQLDataSource(DataSource):
    """MySQL数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化MySQL连接
        
        Args:
            config: 包含数据库连接信息的字典，如果为None则使用默认配置
                - host: 数据库主机
                - user: 用户名
                - password: 密码
                - database: 数据库名
                - port: 端口号
        """
        # 如果没有提供配置，使用默认配置
        if config is None:
            config = default_data_config.get_mysql_dict()
            
        self.conn_params = {
            'host': config.get('host', 'localhost'),
            'user': config.get('user', 'root'),
            'password': config.get('password', ''),
            'database': config.get('database', 'mose'),
            'port': config.get('port', 3306),
            'charset': 'utf8mb4',
            'cursorclass': DictCursor
        }
        # 测试连接
        self._test_connection()
        
    def _test_connection(self):
        """测试数据库连接"""
        try:
            conn = pymysql.connect(**self.conn_params)
            conn.close()
        except Exception as e:
            raise ConnectionError(f"无法连接到MySQL数据库: {e}")
    
    def get_connection(self):
        """获取数据库连接"""
        return pymysql.connect(**self.conn_params)
    
    def get_historical_data(self, symbol: str, start_date: dt.datetime, 
                           end_date: dt.datetime, timeframe: str = 'daily') -> pd.DataFrame:
        """从MySQL获取历史OHLCV数据"""
        
        # 根据timeframe选择表名，目前仅支持日线数据
        if timeframe.lower() != 'daily':
            print(f"警告: 目前仅支持日线数据，已忽略timeframe: {timeframe}")
        
        # 使用stock_code_time表 (按股票代码和时间索引)
        table_name = 'stock_code_time'
        
        # 构建SQL查询
        query = f"""
        SELECT Date, Code, Open, High, Low, Close, Volume, AdjClose
        FROM {table_name}
        WHERE Code = %s AND Date BETWEEN %s AND %s
        ORDER BY Date
        """
        
        with self.get_connection() as conn:
            # 查询数据
            cursor = conn.cursor()
            cursor.execute(query, (
                symbol, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            ))
            
            # 获取列名
            columns = [column[0] for column in cursor.description]
            
            # 获取所有数据
            rows = cursor.fetchall()
            
            # 如果没有数据，返回空DataFrame
            if not rows:
                return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'])
            
            # 创建DataFrame
            df = pd.DataFrame(list(rows), columns=columns)
            
            # 标准化列名和格式
            df = self._standardize_dataframe(df)
            
        return df
    
    def get_multiple_symbols(self, symbols: List[str], start_date: dt.datetime,
                            end_date: dt.datetime, timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
        """获取多个股票的历史数据"""
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_historical_data(symbol, start_date, end_date, timeframe)
        return result
    
    def get_latest_data(self, symbol: str, n_bars: int = 1, 
                       timeframe: str = 'daily') -> pd.DataFrame:
        """获取最新的n条数据"""
        # 目前仅支持日线数据
        if timeframe.lower() != 'daily':
            print(f"警告: 目前仅支持日线数据，已忽略timeframe: {timeframe}")
        
        # 使用stock_code_time表
        table_name = 'stock_code_time'
        
        # 构建SQL查询
        query = f"""
        SELECT Date, Code, Open, High, Low, Close, Volume, AdjClose
        FROM {table_name}
        WHERE Code = %s
        ORDER BY Date DESC
        LIMIT %s
        """
        
        with self.get_connection() as conn:
            # 查询数据
            cursor = conn.cursor()
            cursor.execute(query, (symbol, n_bars))
            
            # 获取列名
            columns = [column[0] for column in cursor.description]
            
            # 获取所有数据
            rows = cursor.fetchall()
            
            # 如果没有数据，返回空DataFrame
            if not rows:
                return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'])
            
            # 创建DataFrame
            df = pd.DataFrame(list(rows), columns=columns)
            
            # 标准化列名和格式
            df = self._standardize_dataframe(df)
            
        return df.sort_values('date')  # 确保时间升序排列
    
    def search_symbols(self, query: str) -> List[Dict]:
        """搜索股票代码"""
        # 从stock_code_time表获取唯一的Code值
        search_query = f"""
        SELECT DISTINCT Code as symbol
        FROM stock_code_time
        WHERE Code LIKE %s
        LIMIT 20
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(search_query, (f'%{query}%',))
                results = cursor.fetchall()
                
        return list(results)
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame格式"""
        # 重命名列
        renaming = {
            'Date': 'date',
            'Code': 'symbol',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'AdjClose': 'adj_close'
        }
        df = df.rename(columns={k: v for k, v in renaming.items() if k in df.columns})
        
        # 确保date是datetime类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 设置date为索引
        df.set_index('date', inplace=True)
        
        # 确保所有必要的列存在
        for col in ['open', 'high', 'low', 'close', 'volume', 'adj_close']:
            if col not in df.columns:
                df[col] = 0.0
        
        return df


# class FutuDataSource(DataSource):
#     """富途API数据源实现"""
#     
#     def __init__(self, config: Dict = None):
#         """
#         初始化富途API连接
#         
#         Args:
#             config: 包含API连接信息的字典
#                 - ip: API主机IP
#                 - port: API端口
#                 - password: 连接密码
#         """
#         # 设置默认配置
#         if config is None:
#             config = default_data_config.get_futu_dict()
#             
#         try:
#             # 动态导入，以便不依赖此库的用户也能使用其他数据源
#             from futu import OpenQuoteContext
#             
#             self.ip = config.get('ip', '127.0.0.1')
#             self.port = config.get('port', 11111)
#             self.password = config.get('password', '')
#             
#             # 创建API连接上下文
#             self.quote_ctx = OpenQuoteContext(host=self.ip, port=self.port)
#             
#             # 尝试连接
#             if not config.get('skip_auth', False):
#                 self.quote_ctx.unlock_trade(password=self.password)
#                 
#         except ImportError:
#             raise ImportError("缺少futu依赖，请先安装: pip install futu-api")
#     
#     def __del__(self):
#         """析构函数，确保关闭API连接"""
#         if hasattr(self, 'quote_ctx'):
#             self.quote_ctx.close()
#     
#     def get_historical_data(self, symbol: str, start_date: dt.datetime, 
#                            end_date: dt.datetime, timeframe: str = 'daily') -> pd.DataFrame:
#         """从富途API获取历史OHLCV数据"""
#         
#         # 处理股票代码格式（富途要求特定格式）
#         if '.' not in symbol:
#             # 默认假设美股
#             symbol = f"US.{symbol}"
#         
#         # 映射时间粒度
#         ktype_map = {
#             'daily': 'K_DAY',
#             'weekly': 'K_WEEK',
#             'monthly': 'K_MON'
#         }
#         ktype = ktype_map.get(timeframe.lower(), 'K_DAY')
#         
#         # 调用API
#         ret, data, page_req_key = self.quote_ctx.request_history_kline(
#             symbol, 
#             start=start_date.strftime('%Y-%m-%d'),
#             end=end_date.strftime('%Y-%m-%d'),
#             ktype=ktype
#         )
#         
#         if ret != 0:
#             # API调用失败
#             return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
#         
#         # 处理数据
#         df = data.copy()
#         df = self._standardize_futu_dataframe(df)
#         
#         return df
#     
#     def get_multiple_symbols(self, symbols: List[str], start_date: dt.datetime,
#                             end_date: dt.datetime, timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
#         """获取多个股票的历史数据"""
#         result = {}
#         for symbol in symbols:
#             result[symbol] = self.get_historical_data(symbol, start_date, end_date, timeframe)
#         return result
#     
#     def get_latest_data(self, symbol: str, n_bars: int = 1, 
#                        timeframe: str = 'daily') -> pd.DataFrame:
#         """获取最新的n条数据"""
#         # 处理股票代码格式
#         if '.' not in symbol:
#             symbol = f"US.{symbol}"
#         
#         # 映射时间粒度
#         ktype_map = {
#             'daily': 'K_DAY',
#             'weekly': 'K_WEEK',
#             'monthly': 'K_MON'
#         }
#         ktype = ktype_map.get(timeframe.lower(), 'K_DAY')
#         
#         # 获取今天日期
#         end_date = dt.datetime.now()
#         # 获取n天前的日期（保险起见多获取一些）
#         start_date = end_date - dt.timedelta(days=n_bars * 3)
#         
#         # 调用API
#         ret, data, page_req_key = self.quote_ctx.request_history_kline(
#             symbol, 
#             start=start_date.strftime('%Y-%m-%d'),
#             end=end_date.strftime('%Y-%m-%d'),
#             ktype=ktype,
#             max_count=n_bars
#         )
#         
#         if ret != 0:
#             return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
#         
#         # 处理数据
#         df = data.copy().tail(n_bars)
#         df = self._standardize_futu_dataframe(df)
#         
#         return df
#     
#     def search_symbols(self, query: str) -> List[Dict]:
#         """搜索股票代码"""
#         # 调用富途API搜索股票
#         ret, data = self.quote_ctx.search_stocks(query)
#         
#         if ret != 0:
#             return []
#         
#         # 转换为标准格式
#         result = []
#         for _, row in data.iterrows():
#             result.append({
#                 'symbol': row['code'],
#                 'name': row['name'],
#                 'exchange': row['market']
#             })
#             
#         return result
#     
#     def _standardize_futu_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
#         """标准化富途API返回的数据格式"""
#         # 重命名列
#         renaming = {
#             'time_key': 'date',
#             'open': 'open',
#             'high': 'high',
#             'low': 'low',
#             'close': 'close',
#             'volume': 'volume',
#             'turnover': 'turnover',
#             'code': 'symbol'
#         }
#         df = df.rename(columns={k: v for k, v in renaming.items() if k in df.columns})
#         
#         # 确保date是datetime类型
#         df['date'] = pd.to_datetime(df['date'])
#         
#         # 设置date为索引
#         df.set_index('date', inplace=True)
#         
#         # 确保所有必要的列存在
#         for col in ['open', 'high', 'low', 'close', 'volume']:
#             if col not in df.columns:
#                 df[col] = 0.0
#         
#         return df


class YahooFinanceDataSource(DataSource):
    """Yahoo Finance API数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化Yahoo Finance数据源
        
        Args:
            config: 配置参数
                - proxy: 代理服务器地址
                - timeout: 请求超时时间(秒)
                - max_retries: 最大重试次数
        """
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("缺少yfinance依赖，请先安装: pip install yfinance")
            
        # 如果没有提供配置，使用默认配置
        if config is None:
            config = default_data_config.get_yahoo_dict()
            
        self.config = config
        self.proxy = config.get('proxy')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
    
    def get_historical_data(self, symbol: str, start_date: dt.datetime, 
                           end_date: dt.datetime, timeframe: str = 'daily') -> pd.DataFrame:
        """从Yahoo Finance获取历史OHLCV数据"""
        
        # 映射时间周期格式
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
        interval = interval_map.get(timeframe.lower(), '1d')
        
        # 尝试获取数据，支持重试
        for attempt in range(self.max_retries):
            try:
                # 使用yfinance获取数据
                ticker = self.yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=interval,
                    proxy=self.proxy,
                    timeout=self.timeout
                )
                
                # 如果成功获取数据，退出重试循环
                if not df.empty:
                    break
                    
            except Exception as e:
                # 最后一次尝试仍然失败，则抛出异常
                if attempt == self.max_retries - 1:
                    print(f"获取Yahoo数据失败 ({symbol}): {e}")
                    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                # 否则等待一段时间后重试
                else:
                    import time
                    time.sleep(1)  # 等待1秒后重试
        
        # 标准化Yahoo Finance数据格式
        if not df.empty:
            # 重命名列
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            # 删除不需要的列
            for col in df.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume', 'adj_close']:
                    df = df.drop(col, axis=1)
            
            # 确保所有必要的列存在
            for col in ['open', 'high', 'low', 'close', 'volume', 'adj_close']:
                if col not in df.columns:
                    df[col] = 0.0
        
        return df
    
    def get_multiple_symbols(self, symbols: List[str], start_date: dt.datetime,
                            end_date: dt.datetime, timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
        """获取多个股票的历史数据"""
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_historical_data(symbol, start_date, end_date, timeframe)
        return result
    
    def get_latest_data(self, symbol: str, n_bars: int = 1, 
                       timeframe: str = 'daily') -> pd.DataFrame:
        """获取最新的n条数据"""
        # 计算开始日期，保守估计 (因为可能有非交易日)
        end_date = dt.datetime.now()
        
        # 根据时间周期和所需条数计算所需天数
        days_map = {
            'daily': 1.5,   # 每条数据约1.5天 (考虑周末和假日)
            'weekly': 7,     # 每条数据7天
            'monthly': 31    # 每条数据31天
        }
        
        # 计算需要获取的日期范围，多获取一些以确保足够
        days_factor = days_map.get(timeframe.lower(), 1.5)
        days_needed = int(n_bars * days_factor * 2)  # 多取2倍时间以确保数据足够
        
        start_date = end_date - dt.timedelta(days=days_needed)
        
        # 获取历史数据
        df = self.get_historical_data(symbol, start_date, end_date, timeframe)
        
        # 截取所需的最新n条数据
        if len(df) > n_bars:
            return df.iloc[-n_bars:]
        return df
    
    def search_symbols(self, query: str) -> List[Dict]:
        """搜索股票代码"""
        try:
            # 使用yfinance的搜索功能
            from yfinance.utils import get_market_symbols
            # 注意：yfinance没有直接的搜索API，这里是模拟实现
            # 获取所有股票列表然后过滤
            all_symbols = get_market_symbols()
            
            # 过滤包含查询字符串的股票
            results = []
            for symbol in all_symbols:
                if query.lower() in symbol.lower():
                    results.append({
                        'symbol': symbol,
                        'name': '', # yfinance API不直接提供名称信息
                        'exchange': '' # yfinance API不直接提供交易所信息
                    })
                    if len(results) >= 20:
                        break
                        
            return results
            
        except Exception as e:
            print(f"搜索股票出错: {e}")
            
            # 返回空列表
            return []


class DataInterface:
    """统一数据接口，管理多个数据源"""
    
    def __init__(self, default_source: str = None, config: Dict = None):
        """
        初始化数据接口
        
        Args:
            default_source: 默认数据源名称，如果为None则使用data_config中的配置
            config: 配置字典，包含各数据源的配置。如果为None，将使用默认配置
        """
        # 使用配置文件中的默认数据源
        if default_source is None:
            default_source = default_data_config.default_source
            
        # 如果没有提供配置，使用配置文件中的配置
        if config is None:
            config = default_data_config.get_all_configs()
            
        self.config = config
        self.default_source = default_source
        self.data_sources = {}
        
        # 初始化所有配置的数据源
        self._init_data_sources()
    
    def _init_data_sources(self):
        """初始化所有配置的数据源"""
        # MySQL数据源
        if 'mysql' in self.config:
            self.data_sources['mysql'] = MySQLDataSource(self.config['mysql'])
            
        # # 富途数据源
        # if 'futu' in self.config:
        #     try:
        #         self.data_sources['futu'] = FutuDataSource(self.config['futu'])
        #     except (ImportError, ConnectionError) as e:
        #         print(f"富途数据源初始化失败: {e}")
        
        # Yahoo Finance数据源
        if 'yahoo' in self.config:
            try:
                self.data_sources['yahoo'] = YahooFinanceDataSource(self.config['yahoo'])
            except (ImportError, ConnectionError) as e:
                print(f"Yahoo Finance数据源初始化失败: {e}")
    
    def add_data_source(self, name: str, source: DataSource):
        """添加自定义数据源"""
        self.data_sources[name] = source
    
    def get_data_source(self, name: str = None) -> DataSource:
        """获取指定数据源"""
        source_name = name or self.default_source
        if source_name not in self.data_sources:
            raise ValueError(f"数据源 '{source_name}' 不存在")
        return self.data_sources[source_name]
    
    def get_historical_data(self, symbol: str, start_date: Union[str, dt.datetime], 
                          end_date: Union[str, dt.datetime], timeframe: str = 'daily',
                          source: str = None) -> pd.DataFrame:
        """获取历史数据"""
        # 处理日期格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 获取数据源
        data_source = self.get_data_source(source)
        
        # 获取数据
        return data_source.get_historical_data(symbol, start_date, end_date, timeframe)
    
    def get_data_for_strategy(self, symbol: str, lookback_days: int = None, 
                            timeframe: str = 'daily', source: str = None) -> pd.DataFrame:
        """获取适合策略使用的数据"""
        # 使用配置文件中的默认回溯天数
        if lookback_days is None:
            lookback_days = default_data_config.default_lookback_days
            
        # 计算日期范围
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_days * 2)  # 多获取一些数据以备不足
        
        # 获取原始数据
        df = self.get_historical_data(symbol, start_date, end_date, timeframe, source)
        
        # 进行必要的预处理
        # 1. 确保没有缺失值
        df = df.ffill()  # 使用前向填充替代method='ffill'
        
        # 2. 计算一些基本的衍生变量
        if not df.empty:
            # 计算收益率
            df['returns'] = df['close'].pct_change()
            
            # 计算波动率
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # 计算移动平均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            
            # 计算布林带
            df['upper_band'] = df['ma20'] + (df['close'].rolling(window=20).std() * 2)
            df['lower_band'] = df['ma20'] - (df['close'].rolling(window=20).std() * 2)
        
        # 3. 截取所需的最近数据
        if len(df) > lookback_days:
            df = df.iloc[-lookback_days:]
            
        return df
    
    def get_multiple_symbols_data(self, symbols: List[str], start_date: Union[str, dt.datetime], 
                                end_date: Union[str, dt.datetime], timeframe: str = 'daily',
                                source: str = None) -> Dict[str, pd.DataFrame]:
        """获取多个股票的历史数据"""
        # 处理日期格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 获取数据源
        data_source = self.get_data_source(source)
        
        # 获取数据
        return data_source.get_multiple_symbols(symbols, start_date, end_date, timeframe)
    
    def search_symbols(self, query: str, source: str = None) -> List[Dict]:
        """搜索股票代码"""
        data_source = self.get_data_source(source)
        return data_source.search_symbols(query)