#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import time
import json
import os
import logging
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YFinanceClient:
    """yfinance客户端，用于获取财务基本面数据"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """初始化yfinance客户端
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间（秒）
        """
        self.cache_dir = "data/yfinance_cache"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._ensure_cache_dir()
        self._setup_session()
        
    def _setup_session(self):
        """设置requests session，配置重试策略"""
        self.session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=self.retry_delay
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置超时
        self.session.timeout = 30
        
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{symbol}_info.json")
    
    def _load_from_cache(self, symbol: str) -> Optional[Dict]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(symbol)
        if os.path.exists(cache_path):
            try:
                # 检查缓存是否过期（1天）
                cache_age = time.time() - os.path.getmtime(cache_path)
                if cache_age < 24 * 3600:  # 24小时内的缓存有效
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"读取缓存失败 {symbol}: {e}")
        return None
    
    def _save_to_cache(self, symbol: str, data: Dict):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(symbol)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"保存缓存失败 {symbol}: {e}")
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        error_str = str(error).lower()
        
        # curl错误16: HTTP/2 stream 0 was not closed cleanly
        if "curl" in error_str and "16" in error_str:
            return True
            
        # 网络相关错误
        retryable_errors = [
            "connection", "timeout", "network", "ssl", "tls",
            "http/2", "stream", "reset", "refused", "unreachable"
        ]
        
        return any(err in error_str for err in retryable_errors)
    
    def get_stock_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        获取股票基本信息，带重试机制
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            股票信息字典
        """
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(symbol)
            if cached_data:
                return cached_data
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"📊 获取 {symbol} 的yfinance数据... (尝试 {attempt + 1}/{self.max_retries + 1})")
                
                # 创建Ticker对象
                ticker = yf.Ticker(symbol)
                
                # 获取股票信息
                info = ticker.info
                
                if info and len(info) > 5:  # 确保获取到有效数据
                    # 保存到缓存
                    self._save_to_cache(symbol, info)
                    logger.info(f"✅ {symbol}: 数据获取成功")
                    return info
                else:
                    logger.warning(f"❌ {symbol}: yfinance数据无效")
                    return None
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"❌ 获取 {symbol} yfinance数据失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                
                # 判断是否可重试
                if attempt < self.max_retries and self._is_retryable_error(e):
                    delay = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.info(f"🔄 {symbol}: {delay}秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ {symbol}: 最终获取失败，不再重试")
                    return None
        
        return None
    
    def extract_financial_metrics(self, info: Dict) -> Dict[str, float]:
        """
        从yfinance info中提取关键财务指标
        
        Args:
            info: yfinance ticker.info返回的数据
            
        Returns:
            标准化的财务指标字典
        """
        def safe_float(value, default=0.0):
            """安全转换为浮点数"""
            if value in [None, 'None', '-', '', 'N/A']:
                return default
            try:
                if isinstance(value, str):
                    # 处理百分比字符串
                    if '%' in value:
                        return float(value.replace('%', '')) / 100
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_percentage(value, default=0.0):
            """安全转换百分比"""
            result = safe_float(value, default)
            # 如果值看起来像百分比（0-1之间），转换为百分数
            if 0 <= result <= 1:
                return result * 100
            return result
        
        metrics = {
            # 盈利能力指标
            'roe': safe_percentage(info.get('returnOnEquity'), 12.0),
            'roa': safe_percentage(info.get('returnOnAssets'), 8.0),
            'gross_margin': safe_percentage(info.get('grossMargins'), 25.0),
            'profit_margin': safe_percentage(info.get('profitMargins'), 10.0),
            'operating_margin': safe_percentage(info.get('operatingMargins'), 15.0),
            
            # 财务健康指标
            'debt_to_equity': safe_float(info.get('debtToEquity'), 50.0) / 100,  # 转换为比率
            'current_ratio': safe_float(info.get('currentRatio'), 1.5),
            'quick_ratio': safe_float(info.get('quickRatio'), 1.2),
            'total_debt': safe_float(info.get('totalDebt'), 0),
            'total_cash': safe_float(info.get('totalCash'), 0),
            
            # 估值指标
            'pe_ratio': safe_float(info.get('trailingPE'), 15.0),
            'forward_pe': safe_float(info.get('forwardPE'), 15.0),
            'pb_ratio': safe_float(info.get('priceToBook'), 2.0),
            'ps_ratio': safe_float(info.get('priceToSalesTrailing12Months'), 3.0),
            'peg_ratio': safe_float(info.get('pegRatio'), 1.0),
            
            # 规模指标
            'market_cap': safe_float(info.get('marketCap'), 1000000000),
            'enterprise_value': safe_float(info.get('enterpriseValue'), 1000000000),
            'revenue': safe_float(info.get('totalRevenue'), 1000000000),
            'net_income': safe_float(info.get('netIncomeToCommon'), 100000000),
            
            # 其他指标
            'eps': safe_float(info.get('trailingEps'), 1.0),
            'forward_eps': safe_float(info.get('forwardEps'), 1.0),
            'dividend_yield': safe_percentage(info.get('dividendYield'), 2.0),
            'beta': safe_float(info.get('beta'), 1.0),
            'book_value': safe_float(info.get('bookValue'), 10.0),
            
            # 增长指标
            'revenue_growth': safe_percentage(info.get('revenueGrowth'), 5.0),
            'earnings_growth': safe_percentage(info.get('earningsGrowth'), 5.0),
            
            # 业务指标
            'shares_outstanding': safe_float(info.get('sharesOutstanding'), 1000000000),
            'float_shares': safe_float(info.get('floatShares'), 1000000000),
            'held_percent_institutions': safe_percentage(info.get('heldPercentInstitutions'), 50.0),
        }
        
        # 计算衍生指标
        if metrics['total_debt'] > 0 and metrics['total_cash'] > 0:
            metrics['net_debt'] = metrics['total_debt'] - metrics['total_cash']
        else:
            metrics['net_debt'] = metrics['total_debt']
        
        # 资产负债率
        if metrics['total_debt'] > 0 and metrics['market_cap'] > 0:
            metrics['debt_to_market_cap'] = metrics['total_debt'] / metrics['market_cap']
        else:
            metrics['debt_to_market_cap'] = 0.1
        
        return metrics
    
    def get_batch_financial_data(self, symbols: List[str], max_symbols: int = 50) -> Dict[str, Dict]:
        """
        批量获取多个股票的财务数据，带错误处理和重试机制
        
        Args:
            symbols: 股票代码列表
            max_symbols: 最大处理股票数量
            
        Returns:
            {symbol: financial_metrics} 的字典
        """
        results = {}
        processed = 0
        failed_symbols = []
        
        logger.info(f"🔄 开始批量获取yfinance财务数据，目标股票数: {min(len(symbols), max_symbols)}")
        
        for symbol in symbols[:max_symbols]:
            try:
                info = self.get_stock_info(symbol)
                if info:
                    metrics = self.extract_financial_metrics(info)
                    results[symbol] = metrics
                    processed += 1
                    logger.info(f"✅ {symbol}: 数据获取成功 ({processed}/{min(len(symbols), max_symbols)})")
                else:
                    logger.warning(f"❌ {symbol}: 数据获取失败")
                    failed_symbols.append(symbol)
                
                # 避免请求过快，但减少延迟
                time.sleep(0.05)
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: 处理异常 - {e}")
                failed_symbols.append(symbol)
                continue
        
        success_rate = len(results) / min(len(symbols), max_symbols) * 100
        logger.info(f"📊 批量获取完成，成功: {len(results)}/{min(len(symbols), max_symbols)} ({success_rate:.1f}%)")
        
        if failed_symbols:
            logger.warning(f"❌ 失败的股票: {', '.join(failed_symbols)}")
        
        return results
    
    def save_financial_data(self, financial_data: Dict[str, Dict], filename: str = None):
        """
        保存财务数据到本地文件
        
        Args:
            financial_data: 财务数据字典
            filename: 保存文件名
        """
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"yfinance_financial_data_{timestamp}.json"
        
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(financial_data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"💾 yfinance财务数据已保存到: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ 保存yfinance财务数据失败: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = YFinanceClient()
    
    # 测试单个股票
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # 批量获取数据
    financial_data = client.get_batch_financial_data(test_symbols, max_symbols=3)
    
    # 保存数据
    if financial_data:
        client.save_financial_data(financial_data)
        
        # 显示结果
        for symbol, metrics in financial_data.items():
            print(f"\n📊 {symbol} 财务指标:")
            print(f"  ROE: {metrics['roe']:.2f}%")
            print(f"  市值: ${metrics['market_cap']:,.0f}")
            print(f"  PE比率: {metrics['pe_ratio']:.2f}")
            print(f"  债务权益比: {metrics['debt_to_equity']:.2f}") 