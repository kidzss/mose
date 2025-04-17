import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict, Any
import json
import os
import time
import random
import pickle

logger = logging.getLogger(__name__)

class MarketSentimentData:
    """市场情绪数据类"""
    
    def __init__(self):
        """初始化市场情绪数据类"""
        self.cache_file = 'data/market_sentiment_cache.pkl'
        self._ensure_cache_dir()
        self._load_or_generate_data()
        
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
    def _load_or_generate_data(self):
        """加载或生成数据"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                self.data = self._generate_simulated_data()
                self._save_data()
        except Exception as e:
            logger.error(f"加载/生成数据失败: {str(e)}")
            self.data = self._generate_simulated_data()
            
    def _save_data(self):
        """保存数据到缓存文件"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data, f)
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            
    def _generate_simulated_data(self, start_date='2024-01-01', end_date='2024-12-31'):
        """生成模拟的市场情绪数据"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # 固定随机种子以确保可重复性
        
        # 生成VIX数据（基于日期生成，确保每天的值是固定的）
        vix_values = []
        for date in dates:
            np.random.seed(date.toordinal())  # 使用日期作为随机种子
            vix = np.random.normal(20, 5)  # 均值20，标准差5
            vix = max(10, min(40, vix))  # 限制在10-40之间
            vix_values.append(vix)
            
        # 生成PCR数据（基于日期生成，确保每天的值是固定的）
        pcr_values = []
        for date in dates:
            np.random.seed(date.toordinal() + 1000)  # 使用不同的种子
            pcr = np.random.normal(1.0, 0.2)  # 均值1.0，标准差0.2
            pcr = max(0.5, min(1.5, pcr))  # 限制在0.5-1.5之间
            pcr_values.append(pcr)
            
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'vix': vix_values,
            'pcr': pcr_values
        })
        
        # 计算市场情绪
        df['sentiment'] = 'neutral'
        df.loc[(df['vix'] < 15) & (df['pcr'] < 0.8), 'sentiment'] = 'bullish'
        df.loc[(df['vix'] > 25) & (df['pcr'] > 1.2), 'sentiment'] = 'bearish'
        
        return df
        
    def get_historical_sentiment(self, start_date, end_date):
        """获取历史市场情绪数据"""
        try:
            mask = (self.data['date'] >= pd.Timestamp(start_date)) & \
                   (self.data['date'] <= pd.Timestamp(end_date))
            return self.data[mask].copy()
        except Exception as e:
            logger.error(f"获取历史情绪数据失败: {str(e)}")
            return pd.DataFrame()
            
    def get_current_sentiment(self):
        """获取当前市场情绪数据"""
        try:
            # 获取今天的日期
            today = pd.Timestamp.now().normalize()
            
            # 如果今天的数据不存在，生成新的数据
            if today not in self.data['date'].values:
                new_data = self._generate_simulated_data(
                    start_date=today.strftime('%Y-%m-%d'),
                    end_date=today.strftime('%Y-%m-%d')
                )
                self.data = pd.concat([self.data, new_data], ignore_index=True)
                self._save_data()
            
            # 返回今天的数据
            return self.data[self.data['date'] == today].iloc[0].to_dict()
        except Exception as e:
            logger.error(f"获取当前情绪数据失败: {str(e)}")
            return None

    def get_vix(self, date: str) -> float:
        """获取指定日期的VIX值"""
        # 检查历史数据
        if 'vix' in self.data and date in self.data['date'].dt.strftime('%Y-%m-%d').values:
            return self.data[self.data['date'].dt.strftime('%Y-%m-%d') == date]['vix'].values[0]
            
        # 生成模拟数据
        vix, _ = self._generate_simulated_data(date)
        
        # 更新历史数据
        if 'vix' not in self.data:
            self.data['vix'] = []
        self.data['vix'].append(vix)
        
        return vix
        
    def get_put_call_ratio(self, date: str) -> float:
        """获取指定日期的Put/Call Ratio值"""
        # 检查历史数据
        if 'pcr' in self.data and date in self.data['date'].dt.strftime('%Y-%m-%d').values:
            return self.data[self.data['date'].dt.strftime('%Y-%m-%d') == date]['pcr'].values[0]
            
        # 生成模拟数据
        _, pcr = self._generate_simulated_data(date)
        
        # 更新历史数据
        if 'pcr' not in self.data:
            self.data['pcr'] = []
        self.data['pcr'].append(pcr)
        
        return pcr
        
    def get_sentiment(self, date: str) -> str:
        """获取市场情绪"""
        vix = self.get_vix(date)
        pcr = self.get_put_call_ratio(date)
        
        # 根据VIX和PCR判断市场情绪
        if vix > 30 and pcr > 1.2:
            return 'bearish'
        elif vix < 20 and pcr < 0.8:
            return 'bullish'
        else:
            return 'neutral'
            
    def get_historical_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史市场情绪数据"""
        dates = pd.date_range(start=start_date, end=end_date)
        sentiment_data = []
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            vix = self.get_vix(date_str)
            pcr = self.get_put_call_ratio(date_str)
            sentiment = self.get_sentiment(date_str)
            
            sentiment_data.append({
                'date': date_str,
                'vix': vix,
                'pcr': pcr,
                'sentiment': sentiment
            })
            
        return pd.DataFrame(sentiment_data) 