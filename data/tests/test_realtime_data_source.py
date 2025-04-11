import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Callable
from data.data_interface import RealTimeDataSource

class MockRealTimeDataSource(RealTimeDataSource):
    """模拟实时数据源实现"""
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, timeframe: str = 'daily') -> pd.DataFrame:
        return pd.DataFrame()
    
    async def get_multiple_symbols(self, symbols: List[str], start_date: datetime,
                                 end_date: datetime, timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
        return {}
    
    async def get_latest_data(self, symbol: str, n_bars: int = 1, 
                            timeframe: str = 'daily') -> pd.DataFrame:
        return pd.DataFrame()
    
    async def search_symbols(self, query: str) -> List[Dict]:
        return []
    
    async def get_realtime_data(self, symbols: List[str], 
                              timeframe: str = '1m') -> Dict[str, pd.DataFrame]:
        return {symbol: pd.DataFrame() for symbol in symbols}
    
    async def subscribe_updates(self, symbols: List[str], 
                              callback: Callable[[str, pd.DataFrame], None],
                              timeframe: str = '1m') -> None:
        pass
    
    async def unsubscribe_updates(self, symbols: List[str]) -> None:
        pass

@pytest.mark.asyncio
async def test_realtime_data_source():
    """测试实时数据源基类"""
    source = MockRealTimeDataSource()
    
    # 测试获取实时数据
    symbols = ['AAPL', 'MSFT']
    data = await source.get_realtime_data(symbols)
    assert isinstance(data, dict)
    assert all(symbol in data for symbol in symbols)
    assert all(isinstance(df, pd.DataFrame) for df in data.values())
    
    # 测试订阅更新
    async def callback(symbol: str, data: pd.DataFrame):
        pass
    
    await source.subscribe_updates(symbols, callback)
    await source.unsubscribe_updates(symbols) 