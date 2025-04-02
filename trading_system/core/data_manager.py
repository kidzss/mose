import logging
import datetime as dt
from typing import Dict, List, Optional
import pandas as pd
from monitor.data_fetcher import DataFetcher

logger = logging.getLogger("DataManager")

class DataManager:
    """数据管理器 - 负责数据的获取、存储和管理"""
    
    def __init__(self, db_config: Dict):
        """初始化数据管理器"""
        self.data_fetcher = DataFetcher(
            db_config=db_config,
            cache_data=True,
            cache_expiry=300,  # 5分钟缓存
            max_workers=5,
            api_delay=1.0
        )
        logger.info("数据管理器初始化完成")
        
    def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """获取股票数据"""
        try:
            return self.data_fetcher.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                use_db=True
            )
        except Exception as e:
            logger.error(f"获取股票 {symbol} 数据时出错: {e}")
            return pd.DataFrame()
            
    def get_market_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """获取多个股票的数据"""
        return {
            symbol: self.get_stock_data(symbol, start_date, end_date)
            for symbol in symbols
        }
        
    def get_latest_data(
        self,
        symbols: List[str],
        days: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """获取最新的数据"""
        return self.data_fetcher.get_latest_data(
            symbols=symbols,
            days=days,
            use_realtime=True
        )
        
    def get_realtime_data(
        self,
        symbols: List[str]
    ) -> pd.DataFrame:
        """获取实时数据"""
        return self.data_fetcher.get_realtime_data(symbols)
        
    def clear_cache(self):
        """清除缓存"""
        self.data_fetcher.clear_cache()
        logger.info("数据缓存已清除")
        
    def get_status(self) -> Dict:
        """获取数据管理器状态"""
        return {
            "cache_enabled": self.data_fetcher.cache_data,
            "cache_expiry": self.data_fetcher.cache_expiry,
            "max_workers": self.data_fetcher.max_workers,
            "api_delay": self.data_fetcher.api_delay,
            "last_update": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        } 