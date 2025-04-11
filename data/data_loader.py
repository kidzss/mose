import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import time

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retry_count = 3
        self.retry_delay = 1
        
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