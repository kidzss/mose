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
        
    def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """获取股票数据"""
        try:
            # 首先尝试从数据库获取数据
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
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
            df = pd.read_sql_query(query, self.engine, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
                
            # 如果数据库中没有数据，从Yahoo Finance获取
            self.logger.info(f"从Yahoo Finance获取{symbol}的数据")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                self.logger.warning(f"无法获取{symbol}的数据")
                return pd.DataFrame()
                
            # 重命名列以匹配数据库格式
            df.columns = [col.lower() for col in df.columns]
            df.index.name = 'date'
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败 ({symbol}): {str(e)}")
            return pd.DataFrame()
            
    def get_market_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """获取多个股票的数据"""
        data = {}
        for symbol in symbols:
            df = self.get_stock_data(symbol, start_date, end_date)
            if not df.empty:
                data[symbol] = df
        return data 