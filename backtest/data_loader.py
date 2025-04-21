import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine
import logging

class DataLoader:
    def __init__(self, db_config: Dict):
        """
        初始化数据加载器
        
        参数:
            db_config: 数据库配置，包含host, port, user, password, database
        """
        self.engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        self.logger = logging.getLogger(__name__)
        
    def load_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime = None,
        adjust: bool = False
    ) -> pd.DataFrame:
        """
        加载股票数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期（默认为当前日期）
            adjust: 是否使用复权价格（默认为False，因为AdjClose数据可能不可用）
        """
        try:
            # 构建查询
            query = f"""
            SELECT 
                date, open, high, low, 
                {'adjclose' if adjust else 'close'} as close,
                volume 
            FROM stock_code_time 
            WHERE code = '{symbol}'
                AND date >= '{start_date.strftime('%Y-%m-%d')}'
            """
            
            if end_date:
                query += f" AND date <= '{end_date.strftime('%Y-%m-%d')}'"
                
            query += " ORDER BY date"
            
            # 读取数据
            df = pd.read_sql(query, self.engine, index_col='date', parse_dates=['date'])
            
            if df.empty:
                self.logger.warning(f"未找到股票 {symbol} 的数据")
                return pd.DataFrame()
                
            # 确保数据完整性
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载股票数据时出错: {str(e)}")
            return pd.DataFrame()
            
    def load_benchmark_data(
        self,
        benchmark_symbol: str = 'SPY',  # 默认使用SPY ETF作为基准
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        加载基准数据
        
        参数:
            benchmark_symbol: 基准指数代码
            start_date: 开始日期
            end_date: 结束日期
        """
        try:
            # 构建查询
            query = f"""
            SELECT 
                date, open, high, low, 
                close,  # 使用普通收盘价
                volume 
            FROM stock_code_time 
            WHERE code = '{benchmark_symbol}'
            """
            
            if start_date:
                query += f" AND date >= '{start_date.strftime('%Y-%m-%d')}'"
            if end_date:
                query += f" AND date <= '{end_date.strftime('%Y-%m-%d')}'"
                
            query += " ORDER BY date"
            
            # 读取数据
            df = pd.read_sql(query, self.engine, index_col='date', parse_dates=['date'])
            
            if df.empty:
                self.logger.warning(f"未找到基准 {benchmark_symbol} 的数据")
                return pd.DataFrame()
                
            # 确保数据完整性
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载基准数据时出错: {str(e)}")
            return pd.DataFrame()
            
    def load_multiple_stocks(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime = None,
        adjust: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        加载多个股票的数据
        
        参数:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adjust: 是否使用复权价格
        """
        data_dict = {}
        for symbol in symbols:
            df = self.load_stock_data(symbol, start_date, end_date, adjust)
            if not df.empty:
                data_dict[symbol] = df
                
        return data_dict
        
    def prepare_backtest_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime = None,
        benchmark_symbol: str = '^GSPC',
        adjust: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备回测数据，包括股票数据和基准数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            benchmark_symbol: 基准指数代码
            adjust: 是否使用复权价格
        """
        # 加载股票数据
        stock_data = self.load_stock_data(symbol, start_date, end_date, adjust)
        if stock_data.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        # 加载基准数据
        benchmark_data = self.load_benchmark_data(benchmark_symbol, start_date, end_date)
        
        return stock_data, benchmark_data 