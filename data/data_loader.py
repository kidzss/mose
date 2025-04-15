import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, List
import time
from sqlalchemy import create_engine, text
from config.trading_config import DatabaseConfig

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retry_count = 3
        self.retry_delay = 1
        self.db_config = {
            'host': DatabaseConfig.host,
            'port': DatabaseConfig.port,
            'user': DatabaseConfig.user,
            'password': DatabaseConfig.password,
            'database': DatabaseConfig.database
        }
        self.engine = self._create_engine()
        
    def _create_engine(self):
        """创建SQLAlchemy引擎"""
        try:
            connection_str = (
                f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
                f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            return create_engine(connection_str)
        except Exception as e:
            self.logger.error(f"创建数据库引擎失败: {str(e)}")
            raise
        
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据"""
        try:
            query = text("""
                SELECT 
                    Date as date,
                    Open as open,
                    High as high,
                    Low as low,
                    Close as close,
                    Volume as volume,
                    AdjClose as adj_close
                FROM stock_code_time
                WHERE Code = :symbol
                AND Date BETWEEN :start_date AND :end_date
                ORDER BY Date ASC
            """)
            
            params = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date
            }
            
            df = pd.read_sql_query(query, self.engine, params=params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败 ({symbol}): {str(e)}")
            return pd.DataFrame()
        
    def load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        加载历史数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔，默认为'1d'（日线）
            
        返回:
            包含OHLCV数据的DataFrame，如果获取失败则返回None
        """
        for attempt in range(self.retry_count):
            try:
                # 下载数据
                ticker = yf.Ticker(symbol)
                
                # 根据时间范围选择合适的间隔
                if (end_date - start_date).days <= 5:
                    interval = '1h'  # 5天内使用小时线
                elif (end_date - start_date).days <= 30:
                    interval = '1d'  # 30天内使用日线
                else:
                    interval = '1wk'  # 超过30天使用周线
                
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    self.logger.warning(f"获取 {symbol} 的数据为空")
                    return None
                    
                # 重命名列
                df.columns = [col.lower() for col in df.columns]
                
                # 删除不需要的列
                if 'dividends' in df.columns:
                    df = df.drop('dividends', axis=1)
                if 'stock splits' in df.columns:
                    df = df.drop('stock splits', axis=1)
                    
                # 确保所有必要的列都存在
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"数据中缺少必要的列: {col}")
                        
                return df
                
            except Exception as e:
                self.logger.error(f"获取 {symbol} 数据失败 (尝试 {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    continue
                return None
                
    def load_positions(self) -> Dict[str, Dict]:
        """
        加载持仓数据
        
        返回:
            字典，key为股票代码，value为包含size和avg_price的字典
        """
        try:
            # 从配置文件加载持仓数据
            import json
            import os
            
            # 获取文件路径
            file_path = os.path.join(os.path.dirname(__file__), '..', 'monitor', 'configs', 'portfolio_config.json')
            
            # 读取持仓数据
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # 转换格式
            positions = {}
            for symbol, data in config['positions'].items():
                positions[symbol] = {
                    'size': data['shares'],
                    'avg_price': data['cost_basis']
                }
            
            return positions
            
        except Exception as e:
            self.logger.error(f"加载持仓数据失败: {e}")
            return {}

    def get_all_symbols(self) -> List[str]:
        """获取所有股票代码"""
        try:
            query = text("SELECT DISTINCT Code FROM stock_code_time ORDER BY Code")
            
            df = pd.read_sql_query(query, self.engine)
            
            return df['Code'].tolist()
            
        except Exception as e:
            self.logger.error(f"获取股票代码列表失败: {str(e)}")
            return []
            
    def get_real_time_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取实时数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            pd.DataFrame: 包含实时数据的DataFrame
        """
        try:
            # 这里应该实现实时数据获取逻辑
            # 可以是从实时行情API获取,或者从其他数据源获取
            pass
            
        except Exception as e:
            self.logger.error(f"获取实时数据失败: {str(e)}")
            return pd.DataFrame()
            
    def get_fundamental_data(self, 
                            symbol: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取基本面数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (可选)
            end_date: 结束日期 (可选)
            
        Returns:
            pd.DataFrame: 包含基本面数据的DataFrame
        """
        try:
            query = text("""
                SELECT *
                FROM stock_fundamentals
                WHERE Code = :symbol
            """)
            params = [symbol]
            
            if start_date:
                query += " AND Date >= :start_date"
                params.append(start_date)
            if end_date:
                query += " AND Date <= :end_date"
                params.append(end_date)
                
            query += " ORDER BY Date"
            
            df = pd.read_sql_query(query, self.engine, params=tuple(params))
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取基本面数据失败 ({symbol}): {str(e)}")
            return pd.DataFrame()
            
    def get_market_data(self, 
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        获取市场数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 包含市场数据的DataFrame
        """
        try:
            query = text("""
                SELECT 
                    Date as date,
                    Open as open,
                    High as high,
                    Low as low,
                    Close as close,
                    Volume as volume
                FROM market_data
                WHERE Date BETWEEN :start_date AND :end_date
                ORDER BY Date
            """)
            
            params = {
                'start_date': start_date,
                'end_date': end_date
            }
            
            df = pd.read_sql_query(query, self.engine, params=params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {str(e)}")
            return pd.DataFrame()
            
    def get_sector_data(self,
                       sector: str,
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        获取行业数据
        
        Args:
            sector: 行业名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 包含行业数据的DataFrame
        """
        try:
            query = text("""
                SELECT 
                    Date as date,
                    Open as open,
                    High as high,
                    Low as low,
                    Close as close,
                    Volume as volume
                FROM sector_data
                WHERE Sector = :sector
                AND Date BETWEEN :start_date AND :end_date
                ORDER BY Date
            """)
            
            params = {
                'sector': sector,
                'start_date': start_date,
                'end_date': end_date
            }
            
            df = pd.read_sql_query(query, self.engine, params=params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取行业数据失败 ({sector}): {str(e)}")
            return pd.DataFrame() 