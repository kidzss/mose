import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from typing import Dict

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        加载历史数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            包含OHLCV数据的DataFrame
        """
        try:
            # 下载数据
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
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
            self.logger.error(f"加载数据时出错: {str(e)}")
            raise 

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