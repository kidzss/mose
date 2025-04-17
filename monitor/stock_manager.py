import pandas as pd
import logging
import mysql.connector
from typing import List, Optional
from .data_fetcher import DataFetcher
import json

logger = logging.getLogger("StockManager")

class StockManager:
    """股票管理类，负责管理监控的股票列表和数据"""
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        初始化股票管理器
        
        参数:
            data_fetcher: 数据获取器实例
        """
        self.data_fetcher = data_fetcher
        self.monitored_stocks = pd.DataFrame()
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'database': 'mose'
        }
        self._load_monitored_stocks()
        
    def _get_db_connection(self):
        """获取数据库连接"""
        return mysql.connector.connect(**self.db_config)
        
    def _load_monitored_stocks(self) -> None:
        """从数据库加载监控的股票列表"""
        try:
            conn = self._get_db_connection()
            query = """
                SELECT symbol as Code, name as Name, sector, industry 
                FROM monitored_stocks 
                WHERE is_active = 1
            """
            self.monitored_stocks = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"从数据库加载监控股票列表成功，共 {len(self.monitored_stocks)} 只股票")
        except Exception as e:
            logger.error(f"从数据库加载监控股票列表失败: {e}")
            self.monitored_stocks = pd.DataFrame()
            
    def get_monitored_stocks(self) -> pd.DataFrame:
        """获取监控的股票列表"""
        if self.monitored_stocks.empty:
            self._load_monitored_stocks()
        return self.monitored_stocks
        
    def add_stock(self, symbol: str, name: str = "", sector: str = "", industry: str = "") -> bool:
        """
        添加股票到监控列表
        
        参数:
            symbol: 股票代码
            name: 股票名称
            sector: 行业分类
            industry: 具体行业
            
        返回:
            是否添加成功
        """
        try:
            if symbol not in self.monitored_stocks['Code'].values:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # 插入新股票
                query = """
                    INSERT INTO monitored_stocks (symbol, name, sector, industry)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(query, (symbol, name, sector, industry))
                conn.commit()
                cursor.close()
                conn.close()
                
                # 更新内存中的股票列表
                new_stock = pd.DataFrame({
                    'Code': [symbol],
                    'Name': [name],
                    'sector': [sector],
                    'industry': [industry]
                })
                self.monitored_stocks = pd.concat([self.monitored_stocks, new_stock], ignore_index=True)
                
                logger.info(f"添加股票 {symbol} 到监控列表")
                return True
            return False
        except Exception as e:
            logger.error(f"添加股票 {symbol} 失败: {e}")
            return False
            
    def remove_stock(self, symbol: str) -> bool:
        """
        从监控列表中移除股票（设置为非活跃）
        
        参数:
            symbol: 股票代码
            
        返回:
            是否移除成功
        """
        try:
            if symbol in self.monitored_stocks['Code'].values:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # 将股票设置为非活跃
                query = "UPDATE monitored_stocks SET is_active = 0 WHERE symbol = %s"
                cursor.execute(query, (symbol,))
                conn.commit()
                cursor.close()
                conn.close()
                
                # 更新内存中的股票列表
                self.monitored_stocks = self.monitored_stocks[self.monitored_stocks['Code'] != symbol]
                
                logger.info(f"从监控列表中移除股票 {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"移除股票 {symbol} 失败: {e}")
            return False
            
    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        获取股票数据
        
        参数:
            symbol: 股票代码
            
        返回:
            股票数据DataFrame
        """
        try:
            return self.data_fetcher.get_latest_data([symbol])[symbol]
        except Exception as e:
            logger.error(f"获取股票 {symbol} 数据失败: {e}")
            return pd.DataFrame()
            
    def sync_with_database(self) -> None:
        """同步数据库中的股票列表"""
        self._load_monitored_stocks()
        
    def get_all_symbols(self) -> List[str]:
        """
        获取所有监控的股票代码列表
        
        返回:
            股票代码列表
        """
        try:
            # 从portfolio_config.json获取当前持仓
            with open('monitor/configs/portfolio_config.json', 'r') as f:
                config = json.load(f)
                symbols = list(config['positions'].keys())
                
            return symbols
        except Exception as e:
            logger.error(f"获取监控股票列表失败: {str(e)}")
            return []