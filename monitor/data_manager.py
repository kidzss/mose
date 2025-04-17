import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import logging
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataManager")

class DataManager:
    """Data manager for US stock market data with layered storage and intelligent updates"""
    
    def __init__(self, db_config: Dict, historical_days: int = 30):
        """
        Initialize data manager
        
        Args:
            db_config: Database configuration
            historical_days: Number of historical days to load
        """
        self.db_config = db_config
        self.historical_days = historical_days
        
        # Data storage
        self.memory_data = {}      # Real-time data cache
        self.historical_data = {}  # Historical data cache
        
        # Update time records
        self.last_historical_update = None
        self.last_realtime_update = None
        
        # Create database connection
        self.engine = self._create_db_engine()
        
        # Cache directory
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache data
        self.market_data_cache = {}
        self.stock_data_cache = {}
        
        # Cache expiry time (minutes)
        self.cache_expiry = {
            'market': 5,    # Market data updates every 5 minutes
            'stock': 1      # Stock data updates every minute during market hours
        }
        
        # Market indices mapping
        self.market_indices = {
            'NDX': '^NDX',   # NASDAQ 100
            'SPX': '^GSPC'   # S&P 500
        }
        
        logger.info("DataManager initialized")
        
    def _create_db_engine(self):
        """Create database engine"""
        try:
            engine = create_engine(
                f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
                f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            return engine
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            return None
            
    def _is_market_open_time(self) -> bool:
        """Check if US market is open"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # Check if it's weekend
        if now.weekday() > 4:  # 5 is Saturday, 6 is Sunday
            return False
            
        # Check if within trading hours (9:30 - 16:00)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
        
    def _is_same_trading_day(self, last_update: datetime) -> bool:
        """Check if it's the same trading day"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        last = last_update.astimezone(pytz.timezone('US/Eastern'))
        
        # If same day, return True
        if now.date() == last.date():
            return True
            
        # If one day apart and it's early morning, consider it same trading day
        if (now.date() - last.date()).days == 1:
            return now.hour < 9 or (now.hour == 9 and now.minute < 30)
            
        return False
        
    def load_historical_data(self, symbols: List[str]) -> None:
        """Load historical data for given symbols"""
        try:
            start_date = (datetime.now() - timedelta(days=self.historical_days)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            for symbol in symbols:
                try:
                    # Get data from database
                    query = """
                    SELECT * FROM stock_code_time
                    WHERE Code = %s AND Date BETWEEN %s AND %s
                    ORDER BY Date ASC
                    """
                    data = pd.read_sql_query(
                        query, 
                        self.engine, 
                        params=(symbol, start_date, end_date)
                    )
                    
                    if not data.empty:
                        # Set date index
                        data['Date'] = pd.to_datetime(data['Date'])
                        data.set_index('Date', inplace=True)
                        self.historical_data[symbol] = data
                        logger.info(f"Successfully loaded historical data for {symbol}, {len(data)} records")
                    else:
                        logger.warning(f"No historical data found for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error loading historical data for {symbol}: {e}")
                    
            self.last_historical_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            
    def update_realtime_data(self, symbols: List[str]) -> None:
        """Update real-time data for given symbols"""
        try:
            for symbol in symbols:
                try:
                    # Get real-time data using yfinance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d', interval='1m')
                    
                    if not data.empty:
                        self.memory_data[symbol] = data
                        logger.debug(f"Successfully updated real-time data for {symbol}")
                    else:
                        logger.warning(f"Unable to get real-time data for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error updating real-time data for {symbol}: {e}")
                    
            self.last_realtime_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating real-time data: {e}")
            
    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get latest data (merged historical and real-time data)"""
        try:
            # Get historical data
            hist_data = self.historical_data.get(symbol)
            # Get real-time data
            real_data = self.memory_data.get(symbol)
            
            if hist_data is None and real_data is None:
                logger.warning(f"No data found for {symbol}")
                return None
                
            # If only historical data
            if real_data is None:
                return hist_data
                
            # If only real-time data
            if hist_data is None:
                return real_data
                
            # Merge data
            # Remove last record from historical data (might be incomplete)
            if not hist_data.empty:
                hist_data = hist_data.iloc[:-1]
                
            # Merge data
            combined_data = pd.concat([hist_data, real_data])
            return combined_data
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return None
            
    def get_market_data(self, index_code: str = 'SPX',
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get market index data
        
        Args:
            index_code: Index code ('NDX' for NASDAQ 100, 'SPX' for S&P 500)
            start_date: Start date
            end_date: End date
        """
        try:
            # Check cache
            cache_key = f"market_{index_code}"
            if cache_key in self.market_data_cache:
                cached_data, cache_time = self.market_data_cache[cache_key]
                if datetime.now() - cache_time < timedelta(minutes=self.cache_expiry['market']):
                    return self._filter_date_range(cached_data, start_date, end_date)
            
            # Get index symbol
            index_symbol = self.market_indices.get(index_code)
            if not index_symbol:
                raise ValueError(f"Unsupported index code: {index_code}")
            
            # Get data using yfinance
            data = yf.download(
                index_symbol,
                start=start_date or (datetime.now() - timedelta(days=365)),
                end=end_date or datetime.now(),
                interval='1d'
            )
            
            # Standardize data format
            data = self._standardize_market_data(data)
            
            # Update cache
            self.market_data_cache[cache_key] = (data, datetime.now())
            
            return self._filter_date_range(data, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
            
    def get_market_stats(self) -> Dict[str, float]:
        """Get market statistics"""
        try:
            stats = {}
            for index_name, index_symbol in self.market_indices.items():
                # Get today's data
                data = yf.download(index_symbol, period='5d', interval='1d')
                if not data.empty:
                    # Calculate daily return
                    daily_return = (data['Close'][-1] / data['Close'][-2] - 1) * 100
                    # Calculate 5-day volatility
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    
                    stats[index_name] = {
                        'price': data['Close'][-1],
                        'daily_return': daily_return,
                        'volatility': volatility,
                        'volume': data['Volume'][-1]
                    }
                    
            return stats
            
        except Exception as e:
            logger.error(f"Error getting market statistics: {e}")
            return {}
            
    def need_update_historical(self) -> bool:
        """Check if historical data needs update"""
        if self.last_historical_update is None:
            return True
            
        if not self._is_same_trading_day(self.last_historical_update):
            return True
            
        return False
        
    def need_update_realtime(self) -> bool:
        """Check if real-time data needs update"""
        if self.last_realtime_update is None:
            return True
            
        time_diff = (datetime.now() - self.last_realtime_update).total_seconds()
        
        if self._is_market_open_time():
            return time_diff >= 60  # 1 minute during market hours
        return time_diff >= 3600   # 1 hour outside market hours
        
    def clear_cache(self) -> None:
        """Clear cached data"""
        self.memory_data.clear()
        self.historical_data.clear()
        self.last_historical_update = None
        self.last_realtime_update = None
        self.market_data_cache.clear()
        self.stock_data_cache.clear()
        logger.info("Cache cleared")
        
    def get_cache_status(self) -> Dict:
        """Get cache status"""
        return {
            "historical_symbols": list(self.historical_data.keys()),
            "realtime_symbols": list(self.memory_data.keys()),
            "last_historical_update": self.last_historical_update,
            "last_realtime_update": self.last_realtime_update,
            "is_market_open": self._is_market_open_time()
        }
        
    def _standardize_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化市场数据格式
        :param df: 输入的DataFrame
        :return: 标准化后的DataFrame
        """
        try:
            if df.empty:
                self.logger.warning("输入的DataFrame为空")
                return df

            # 复制数据以避免修改原始数据
            df = df.copy()

            # 确保索引是datetime类型
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                elif 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)

            # 列名映射
            column_mapping = {
                'Open': ['open', 'Open', 'OPEN'],
                'High': ['high', 'High', 'HIGH'],
                'Low': ['low', 'Low', 'LOW'],
                'Close': ['close', 'Close', 'CLOSE'],
                'Volume': ['volume', 'Volume', 'VOLUME'],
                'Adj Close': ['adj close', 'Adj Close', 'AdjClose', 'adj_close'],
                'Dividends': ['dividends', 'Dividends'],
                'Stock Splits': ['stock splits', 'Stock Splits', 'StockSplits'],
                'Capital Gains': ['capital gains', 'Capital Gains', 'CapitalGains']
            }

            # 标准化列名
            for standard_name, variations in column_mapping.items():
                for var in variations:
                    if var in df.columns:
                        df.rename(columns={var: standard_name}, inplace=True)
                        break

            # 确保所有必需的列都存在
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Adj Close' and 'Close' in df.columns:
                        df[col] = df['Close']
                    else:
                        df[col] = 0
                        self.logger.warning(f"列 {col} 不存在，已用0填充")

            # 初始化可选列
            optional_columns = ['Dividends', 'Stock Splits', 'Capital Gains']
            for col in optional_columns:
                if col not in df.columns:
                    df[col] = 0

            # 确保数据类型正确
            numeric_columns = required_columns + optional_columns
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 填充缺失值
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"标准化市场数据时出错: {e}")
            return pd.DataFrame()
        
    def _filter_date_range(self, data: pd.DataFrame,
                          start_date: Optional[datetime],
                          end_date: Optional[datetime]) -> pd.DataFrame:
        """Filter data by date range"""
        if start_date:
            data = data[data.index >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data.index <= pd.Timestamp(end_date)]
        return data 

    def get_historical_data(self, symbol: str, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Get historical data for a symbol within a date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date (default: None, will use self.historical_days)
            end_date: End date (default: None, will use current date)
            
        Returns:
            DataFrame with historical data
        """
        try:
            # If no dates provided, use default range
            if start_date is None:
                start_date = datetime.now() - timedelta(days=self.historical_days)
            if end_date is None:
                end_date = datetime.now()
                
            # Try to get data from database
            query = f"""
                SELECT * FROM stock_prices 
                WHERE symbol = '{symbol}' 
                AND date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
                ORDER BY date ASC
            """
            
            with self.engine.connect() as conn:
                data = pd.read_sql(query, conn)
                
            if not data.empty:
                # Convert date column to datetime
                data['date'] = pd.to_datetime(data['date'])
                # Set date as index
                data.set_index('date', inplace=True)
                
                # 重命名列以匹配策略期望的格式
                column_mapping = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adj_close': 'Adj Close'
                }
                data.rename(columns=column_mapping, inplace=True)
                return data
                
            # If no data in database, try to fetch from API
            logger.info(f"No historical data for {symbol} in database, fetching from API")
            data = self._get_data_from_api(symbol, start_date, end_date)
            
            if not data.empty:
                # Save to database
                self._save_data_to_db(data, symbol)
                
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame() 