import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from data.data_interface import YahooFinanceRealTimeSource

@pytest.mark.asyncio
async def test_yahoo_realtime_source():
    """测试Yahoo Finance实时数据源"""
    # 初始化数据源
    source = YahooFinanceRealTimeSource()
    
    # 测试获取实时数据
    symbols = ['AAPL', 'MSFT']
    data = await source.get_realtime_data(symbols)
    
    # 验证返回的数据结构
    assert isinstance(data, dict)
    assert all(symbol in data for symbol in symbols)
    assert all(isinstance(df, pd.DataFrame) for df in data.values())
    
    # 验证数据内容
    for symbol, df in data.items():
        if not df.empty:  # 如果获取到了数据
            assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
            assert len(df) > 0
    
    # 测试订阅功能
    received_data = {}
    
    def callback(symbol: str, data: pd.DataFrame):
        received_data[symbol] = data
    
    # 订阅更新
    await source.subscribe_updates(symbols, callback)
    
    # 等待一次更新
    await asyncio.sleep(source.update_interval + 1)
    
    # 验证是否收到数据
    assert len(received_data) > 0
    
    # 取消订阅
    await source.unsubscribe_updates(symbols)
    
    # 验证更新任务已停止
    assert not source.running
    assert source.update_task is None 