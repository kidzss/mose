# data/data_interface.py
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import datetime as dt
import pymysql
from pymysql.cursors import DictCursor
import os
import sys
from functools import lru_cache
import hashlib
import json
import random
import asyncio

# 添加项目根目录到sys.path以便导入config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.data_config import default_data_config
from .data_validator import DataValidator

def cache_key(*args, **kwargs):
    """生成缓存键"""
    # 将参数转换为字符串
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    # 使用MD5生成唯一键
    return hashlib.md5(json.dumps(key_parts).encode()).hexdigest()

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


class RealTimeDataSource(DataSource):
    """实时数据源基类"""
    
    @abstractmethod
    async def get_realtime_data(self, symbols: List[str], 
                              timeframe: str = '1m') -> Dict[str, pd.DataFrame]:
        """获取实时数据
        
        参数:
            symbols: 股票代码列表
            timeframe: 时间周期，如'1m', '5m', '15m'等
            
        返回:
            字典，key为股票代码，value为DataFrame
        """
        pass
    
    @abstractmethod
    async def subscribe_updates(self, symbols: List[str], 
                              callback: Callable[[str, pd.DataFrame], None],
                              timeframe: str = '1m') -> None:
        """订阅数据更新
        
        参数:
            symbols: 股票代码列表
            callback: 回调函数，接收股票代码和DataFrame
            timeframe: 时间周期
        """
        pass
    
    @abstractmethod
    async def unsubscribe_updates(self, symbols: List[str]) -> None:
        """取消订阅数据更新
        
        参数:
            symbols: 股票代码列表
        """
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


class YahooFinanceDataSource(RealTimeDataSource):
    """Yahoo Finance数据源实现"""
    
    def __init__(self):
        import yfinance as yf
        self.yf = yf
        
    def get_historical_data(self, symbol: str, start_date: dt.datetime, 
                           end_date: dt.datetime, timeframe: str = 'daily') -> pd.DataFrame:
        """获取历史数据"""
        try:
            # 获取股票数据
            stock = self.yf.Ticker(symbol)
            # 获取历史数据
            df = stock.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=timeframe
            )
            if not df.empty:
                # 标准化列名
                df = self._standardize_dataframe(df)
            return df
        except Exception as e:
            print(f"获取{symbol}历史数据失败: {e}")
            return pd.DataFrame()
            
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
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=n_bars * 2)  # 多取一些数据确保足够
        df = self.get_historical_data(symbol, start_date, end_date, timeframe)
        if not df.empty and len(df) > n_bars:
            return df.iloc[-n_bars:]
        return df
        
    def search_symbols(self, query: str) -> List[Dict]:
        """搜索股票代码"""
        try:
            # 使用yfinance的搜索功能
            from yfinance.utils import get_market_symbols
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
            return []
            
    async def get_realtime_data(self, symbols: List[str], 
                              timeframe: str = '1m') -> Dict[str, pd.DataFrame]:
        """获取实时数据"""
        result = {}
        for symbol in symbols:
            try:
                # 获取股票数据
                stock = self.yf.Ticker(symbol)
                # 获取实时数据
                df = stock.history(period='1d', interval='1m')
                if not df.empty:
                    # 标准化列名
                    df = self._standardize_dataframe(df)
                    result[symbol] = df
            except Exception as e:
                print(f"获取{symbol}实时数据失败: {e}")
        return result
        
    async def subscribe_updates(self, symbols: List[str], 
                              callback: Callable[[str, pd.DataFrame], None],
                              timeframe: str = '1m') -> None:
        """订阅数据更新"""
        # Yahoo Finance不支持实时订阅，这里使用轮询方式
        while True:
            data = await self.get_realtime_data(symbols, timeframe)
            for symbol, df in data.items():
                if not df.empty:
                    callback(symbol, df)
            await asyncio.sleep(60)  # 每分钟更新一次
            
    async def unsubscribe_updates(self, symbols: List[str]) -> None:
        """取消订阅数据更新"""
        # Yahoo Finance不支持取消订阅，这里只是占位
        pass
        
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame格式"""
        # 重命名列
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        })
        # 确保所有必需的列都存在
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        return df


class YahooFinanceRealTimeSource(RealTimeDataSource):
    """基于YahooFinance的实时数据源实现"""
    
    def __init__(self, config: Dict = None):
        """初始化Yahoo Finance实时数据源
        
        Args:
            config: 配置参数
                - proxy: 代理服务器地址
                - timeout: 请求超时时间(秒)
                - max_retries: 最大重试次数
                - update_interval: 更新间隔(秒)
        """
        super().__init__()
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
        self.update_interval = config.get('update_interval', 60)  # 默认60秒更新一次
        
        # 用于存储订阅信息
        self.subscriptions = {}
        self.running = False
        self.update_task = None
        
    async def get_historical_data(self, symbol: str, start_date: dt.datetime, 
                                end_date: dt.datetime, timeframe: str = 'daily') -> pd.DataFrame:
        """获取历史OHLCV数据"""
        try:
            ticker = self.yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=timeframe
            )
            return self._standardize_dataframe(df)
        except Exception as e:
            print(f"获取历史数据失败 ({symbol}): {e}")
            return pd.DataFrame()
    
    async def get_multiple_symbols(self, symbols: List[str], start_date: dt.datetime,
                                 end_date: dt.datetime, timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
        """获取多个股票的历史数据"""
        result = {}
        for symbol in symbols:
            result[symbol] = await self.get_historical_data(symbol, start_date, end_date, timeframe)
        return result
    
    async def get_latest_data(self, symbol: str, n_bars: int = 1, 
                            timeframe: str = 'daily') -> pd.DataFrame:
        """获取最新的n条数据"""
        try:
            ticker = self.yf.Ticker(symbol)
            df = ticker.history(period=f"{n_bars}d", interval=timeframe)
            return self._standardize_dataframe(df)
        except Exception as e:
            print(f"获取最新数据失败 ({symbol}): {e}")
            return pd.DataFrame()
    
    async def search_symbols(self, query: str) -> List[Dict]:
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
            return []
        
    async def get_realtime_data(self, symbols: List[str], 
                              timeframe: str = '1m') -> Dict[str, pd.DataFrame]:
        """获取实时数据
        
        Args:
            symbols: 股票代码列表
            timeframe: 时间周期，如'1m', '5m', '15m'等
            
        Returns:
            字典，key为股票代码，value为DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                # 使用yfinance获取最新数据
                ticker = self.yf.Ticker(symbol)
                # 获取最近的分钟数据
                df = ticker.history(period='1d', interval=timeframe)
                
                if not df.empty:
                    # 标准化数据格式
                    df = self._standardize_dataframe(df)
                    result[symbol] = df
                else:
                    result[symbol] = pd.DataFrame()
                    
            except Exception as e:
                print(f"获取{symbol}实时数据失败: {e}")
                result[symbol] = pd.DataFrame()
                
        return result
        
    async def subscribe_updates(self, symbols: List[str], 
                              callback: Callable[[str, pd.DataFrame], None],
                              timeframe: str = '1m') -> None:
        """订阅数据更新
        
        Args:
            symbols: 股票代码列表
            callback: 回调函数，接收股票代码和DataFrame
            timeframe: 时间周期
        """
        import asyncio
        
        # 存储订阅信息
        for symbol in symbols:
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = set()
            self.subscriptions[symbol].add(callback)
        
        # 如果更新任务未运行，启动它
        if not self.running:
            self.running = True
            self.update_task = asyncio.create_task(self._update_loop(timeframe))
            
    async def unsubscribe_updates(self, symbols: List[str]) -> None:
        """取消订阅数据更新
        
        Args:
            symbols: 股票代码列表
        """
        # 移除订阅
        for symbol in symbols:
            if symbol in self.subscriptions:
                del self.subscriptions[symbol]
                
        # 如果没有任何订阅，停止更新任务
        if not self.subscriptions and self.update_task:
            self.running = False
            self.update_task.cancel()
            self.update_task = None
            
    async def _update_loop(self, timeframe: str) -> None:
        """更新循环"""
        import asyncio
        
        while self.running:
            try:
                # 获取所有订阅的股票的最新数据
                symbols = list(self.subscriptions.keys())
                data = await self.get_realtime_data(symbols, timeframe)
                
                # 调用回调函数
                for symbol, df in data.items():
                    if symbol in self.subscriptions and not df.empty:
                        for callback in self.subscriptions[symbol]:
                            try:
                                callback(symbol, df)
                            except Exception as e:
                                print(f"调用回调函数失败 ({symbol}): {e}")
                                
            except Exception as e:
                print(f"更新循环出错: {e}")
                
            # 等待下一次更新
            await asyncio.sleep(self.update_interval)
            
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame格式"""
        if df.empty:
            return df
            
        # 重命名列
        renaming = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        df = df.rename(columns={k: v for k, v in renaming.items() if k in df.columns})
        
        # 确保所有必要的列存在
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0.0
                
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']
            
        return df
            
    def __del__(self):
        """清理资源"""
        if self.update_task:
            self.running = False
            self.update_task.cancel()


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
        self._cache = {}
        
        # 初始化所有配置的数据源
        self._init_data_sources()
        
        # 初始化数据更新器
        self._init_data_updater()
    
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
                self.data_sources['yahoo'] = YahooFinanceDataSource()
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
    
    @lru_cache(maxsize=100)
    def get_historical_data(self, symbol: str, start_date: Union[str, dt.datetime], 
                          end_date: Union[str, dt.datetime], timeframe: str = 'daily',
                          source: str = None) -> pd.DataFrame:
        """获取历史数据（带缓存）"""
        # 处理日期格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 获取数据源
        data_source = self.get_data_source(source)
        
        # 获取数据
        df = data_source.get_historical_data(symbol, start_date, end_date, timeframe)
        
        return df
    
    @lru_cache(maxsize=50)
    def get_data_for_strategy(self, symbol: str, lookback_days: int = None, 
                            timeframe: str = 'daily', source: str = None) -> pd.DataFrame:
        """获取适合策略使用的数据（带缓存）"""
        # 使用配置文件中的默认回溯天数
        if lookback_days is None:
            lookback_days = default_data_config.default_lookback_days
            
        # 计算日期范围
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_days * 2)  # 多获取一些数据以备不足
        
        # 获取原始数据
        df = self.get_historical_data(symbol, start_date, end_date, timeframe, source)
        
        # 进行必要的预处理
        df = self._preprocess_strategy_data(df)
        
        return df
    
    def _preprocess_strategy_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理策略数据"""
        if df.empty:
            return df
            
        # 1. 确保没有缺失值
        df = df.ffill()
        
        # 2. 计算基本的衍生变量
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
    
    def _init_data_updater(self):
        """初始化数据更新器"""
        from .data_updater import MarketDataUpdater
        self.updater = MarketDataUpdater(self.config.get('mysql', {}))
    
    def update_market_data(self, symbols: List[str] = None, force_update: bool = False) -> Dict[str, Any]:
        """
        更新市场数据
        
        Args:
            symbols: 要更新的股票列表，如果为None则使用默认列表
            force_update: 是否强制更新（忽略最后更新时间）
            
        Returns:
            更新报告，包含更新状态和统计信息
        """
        return self.updater.update_stock_data(symbols, force_update=force_update)
    
    def get_last_update_time(self, symbol: str = None) -> Union[dt.datetime, Dict[str, dt.datetime]]:
        """
        获取数据最后更新时间
        
        Args:
            symbol: 股票代码，如果为None则返回所有股票的最后更新时间
            
        Returns:
            最后更新时间或更新时间字典
        """
        if symbol is None:
            # 获取所有股票的最后更新时间
            return self.updater.get_last_update_times()
        return self.updater.get_last_update_time(symbol)
    
    def check_data_status(self, symbol: str = None) -> Dict[str, Any]:
        """
        检查数据状态
        
        Args:
            symbol: 股票代码，如果为None则检查所有股票
            
        Returns:
            数据状态报告，包含：
            - 最后更新时间
            - 数据完整性
            - 缺失区间
            - 异常值统计
        """
        if symbol is None:
            symbols = self.get_available_symbols()
        else:
            symbols = [symbol]
            
        report = {}
        for sym in symbols:
            # 获取最新数据
            data = self.get_historical_data(sym, 
                                         start_date=dt.datetime.now() - dt.timedelta(days=30),
                                         end_date=dt.datetime.now())
            
            # 验证数据
            _, validation_report = DataValidator.validate_data(data)
            
            # 获取最后更新时间
            last_update = self.get_last_update_time(sym)
            
            report[sym] = {
                'last_update': last_update,
                'validation': validation_report,
                'data_points': len(data) if not data.empty else 0,
                'status': 'ok' if validation_report['validation_passed'] else 'warning'
            }
            
        return report
    
    def get_available_symbols(self) -> List[str]:
        """获取可用的股票代码列表"""
        return self.updater.db_manager.get_existing_stocks()
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取股票详细信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票信息字典，包含：
            - 基本信息（代码、名称等）
            - 数据统计（数据点数量、时间范围等）
            - 最后更新时间
            - 数据质量报告
        """
        # 获取基本数据
        data = self.get_historical_data(symbol,
                                     start_date=dt.datetime.now() - dt.timedelta(days=365),
                                     end_date=dt.datetime.now())
        
        # 获取数据验证报告
        _, validation_report = DataValidator.validate_data(data)
        
        # 获取最后更新时间
        last_update = self.get_last_update_time(symbol)
        
        # 计算基本统计信息
        if not data.empty:
            stats = {
                'data_points': len(data),
                'date_range': {
                    'start': data.index[0].strftime('%Y-%m-%d'),
                    'end': data.index[-1].strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min': data['low'].min(),
                    'max': data['high'].max(),
                    'current': data['close'].iloc[-1]
                },
                'volume_stats': {
                    'avg': data['volume'].mean(),
                    'max': data['volume'].max()
                }
            }
        else:
            stats = {
                'data_points': 0,
                'date_range': None,
                'price_range': None,
                'volume_stats': None
            }
            
        return {
            'symbol': symbol,
            'last_update': last_update,
            'validation': validation_report,
            'statistics': stats
        }
    
    @lru_cache(maxsize=100)
    def get_market_status(self) -> Dict[str, Any]:
        """
        获取市场整体状态
        
        Returns:
            市场状态报告，包含：
            - 可用股票数量
            - 数据更新统计
            - 数据质量统计
            - 系统状态
        """
        symbols = self.get_available_symbols()
        
        status = {
            'total_symbols': len(symbols),
            'last_update': dt.datetime.now(),
            'data_quality': {
                'valid': 0,
                'warning': 0,
                'error': 0
            },
            'system_status': 'operational'
        }
        
        # 检查随机样本
        sample_size = min(10, len(symbols))
        sample_symbols = random.sample(symbols, sample_size)
        
        for symbol in sample_symbols:
            symbol_status = self.check_data_status(symbol)
            if symbol_status[symbol]['status'] == 'ok':
                status['data_quality']['valid'] += 1
            elif symbol_status[symbol]['status'] == 'warning':
                status['data_quality']['warning'] += 1
            else:
                status['data_quality']['error'] += 1
        
        # 扩展到总体
        factor = len(symbols) / sample_size
        status['data_quality'] = {
            k: int(v * factor) for k, v in status['data_quality'].items()
        }
        
        return status