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

logger = logging.getLogger(__name__)

class MarketSentimentData:
    def __init__(self, cache_dir: str = 'data/cache'):
        self.vix_data = None
        self.put_call_ratio_data = None
        self.last_update = None
        self.cache_dir = cache_dir
        self.historical_data = {}
        self.retry_delay = 5  # seconds
        self.max_retries = 3
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载历史数据
        self._load_historical_data()
        
    def _load_historical_data(self):
        """加载历史数据"""
        try:
            vix_file = os.path.join(self.cache_dir, 'vix_history.json')
            pcr_file = os.path.join(self.cache_dir, 'pcr_history.json')
            
            if os.path.exists(vix_file):
                with open(vix_file, 'r') as f:
                    self.historical_data['vix'] = json.load(f)
                    
            if os.path.exists(pcr_file):
                with open(pcr_file, 'r') as f:
                    self.historical_data['pcr'] = json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            
    def _save_historical_data(self):
        """保存历史数据"""
        try:
            vix_file = os.path.join(self.cache_dir, 'vix_history.json')
            pcr_file = os.path.join(self.cache_dir, 'pcr_history.json')
            
            if 'vix' in self.historical_data:
                with open(vix_file, 'w') as f:
                    json.dump(self.historical_data['vix'], f)
                    
            if 'pcr' in self.historical_data:
                with open(pcr_file, 'w') as f:
                    json.dump(self.historical_data['pcr'], f)
                    
        except Exception as e:
            logger.error(f"Error saving historical data: {str(e)}")
            
    def _fetch_with_retry(self, url: str) -> dict:
        """Fetch data with retry logic and rate limiting"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Too Many Requests
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Failed to fetch data: {response.status_code}")
                    return None
            except Exception as e:
                logger.warning(f"Error fetching data: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
        return None
        
    def update_data(self):
        """Update market sentiment data"""
        try:
            # Fetch VIX data
            vix_url = "https://cdn.cboe.com/api/global/delayed_quotes/indices/VIX.json"
            vix_response = self._fetch_with_retry(vix_url)
            if vix_response:
                self.vix_data = vix_response.get('data', {}).get('last', 0)
                
            # Fetch Put/Call Ratio data
            pcr_url = "https://cdn.cboe.com/api/global/delayed_quotes/indices/PCR.json"
            pcr_response = self._fetch_with_retry(pcr_url)
            if pcr_response:
                self.put_call_ratio_data = pcr_response.get('data', {}).get('last', 0)
                
            self.last_update = datetime.now()
            self._save_historical_data()
            logger.info("Market sentiment data updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating market sentiment data: {str(e)}")
            
    def get_vix(self, date: str) -> float:
        """获取指定日期的VIX值"""
        # 检查历史数据
        if 'vix' in self.historical_data and date in self.historical_data['vix']:
            return self.historical_data['vix'][date]
            
        # 检查是否需要更新数据
        if not self.vix_data or (datetime.now() - self.last_update).days > 0:
            self.update_data()
            
        return self.vix_data if self.vix_data else 20.0  # 默认值
        
    def get_put_call_ratio(self, date: str) -> float:
        """获取指定日期的Put/Call Ratio值"""
        # 检查历史数据
        if 'pcr' in self.historical_data and date in self.historical_data['pcr']:
            return self.historical_data['pcr'][date]
            
        # 检查是否需要更新数据
        if not self.put_call_ratio_data or (datetime.now() - self.last_update).days > 0:
            self.update_data()
            
        return self.put_call_ratio_data if self.put_call_ratio_data else 1.0  # 默认值
        
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