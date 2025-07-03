#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级yfinance客户端
支持配置文件、智能重试、错误分类和性能优化
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import time
import json
import os
import logging
import re
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

class AdvancedYFinanceClient:
    """高级yfinance客户端，支持智能错误处理和重试机制"""
    
    def __init__(self, config_path: str = "config/yfinance_error_config.json"):
        """初始化高级yfinance客户端
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_session()
        self._setup_cache()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_attempts': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "retry_settings": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True,
                "max_delay": 10.0
            },
            "timeout_settings": {
                "request_timeout": 30,
                "connect_timeout": 10,
                "read_timeout": 30
            },
            "error_handling": {
                "retryable_errors": [
                    "curl.*16",
                    "http/2.*stream.*not.*closed",
                    "connection.*timeout",
                    "network.*unreachable"
                ],
                "non_retryable_errors": [
                    "invalid.*symbol",
                    "not.*found",
                    "unauthorized",
                    "forbidden"
                ]
            },
            "cache_settings": {
                "enabled": True,
                "cache_duration_hours": 24,
                "cache_dir": "data/yfinance_cache"
            },
            "rate_limiting": {
                "requests_per_second": 2,
                "delay_between_requests": 0.5,
                "batch_size": 10
            },
            "logging": {
                "level": "INFO",
                "log_errors": True,
                "log_retries": True,
                "log_success": False
            }
        }
    
    def _setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_session(self):
        """设置requests session"""
        self.session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=self.config['retry_settings']['max_retries'],
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=self.config['retry_settings']['retry_delay']
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置超时
        timeout_config = self.config['timeout_settings']
        self.session.timeout = (timeout_config['connect_timeout'], timeout_config['read_timeout'])
        
    def _setup_cache(self):
        """设置缓存"""
        cache_config = self.config['cache_settings']
        if cache_config['enabled']:
            self.cache_dir = Path(cache_config['cache_dir'])
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_duration = timedelta(hours=cache_config['cache_duration_hours'])
        else:
            self.cache_dir = None
            self.cache_duration = None
    
    def _get_cache_path(self, symbol: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{symbol}_info.json"
    
    def _load_from_cache(self, symbol: str) -> Optional[Dict]:
        """从缓存加载数据"""
        if not self.cache_dir or not self.cache_duration:
            return None
            
        cache_path = self._get_cache_path(symbol)
        if cache_path.exists():
            try:
                # 检查缓存是否过期
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < self.cache_duration.total_seconds():
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.stats['cache_hits'] += 1
                        return data
            except Exception as e:
                self.logger.warning(f"读取缓存失败 {symbol}: {e}")
        
        self.stats['cache_misses'] += 1
        return None
    
    def _save_to_cache(self, symbol: str, data: Dict):
        """保存数据到缓存"""
        if not self.cache_dir:
            return
            
        cache_path = self._get_cache_path(symbol)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"保存缓存失败 {symbol}: {e}")
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        error_str = str(error).lower()
        
        # 检查不可重试的错误
        non_retryable_patterns = self.config['error_handling']['non_retryable_errors']
        for pattern in non_retryable_patterns:
            if re.search(pattern, error_str, re.IGNORECASE):
                return False
        
        # 检查可重试的错误
        retryable_patterns = self.config['error_handling']['retryable_errors']
        for pattern in retryable_patterns:
            if re.search(pattern, error_str, re.IGNORECASE):
                return True
        
        # 默认情况下，网络相关错误可重试
        return any(keyword in error_str for keyword in [
            'connection', 'timeout', 'network', 'ssl', 'tls', 'http'
        ])
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        retry_config = self.config['retry_settings']
        base_delay = retry_config['retry_delay']
        max_delay = retry_config['max_delay']
        
        if retry_config['exponential_backoff']:
            delay = base_delay * (2 ** attempt)
        else:
            delay = base_delay * (attempt + 1)
        
        return min(delay, max_delay)
    
    def get_stock_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        获取股票基本信息，带智能重试机制
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            股票信息字典
        """
        self.stats['total_requests'] += 1
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(symbol)
            if cached_data:
                return cached_data
        
        max_retries = self.config['retry_settings']['max_retries']
        
        for attempt in range(max_retries + 1):
            try:
                if self.config['logging']['log_retries'] and attempt > 0:
                    self.logger.info(f"📊 获取 {symbol} 的yfinance数据... (尝试 {attempt + 1}/{max_retries + 1})")
                elif self.config['logging']['log_success']:
                    self.logger.info(f"📊 获取 {symbol} 的yfinance数据...")
                
                # 创建Ticker对象
                ticker = yf.Ticker(symbol)
                
                # 获取股票信息
                info = ticker.info
                
                if info and len(info) > 5:  # 确保获取到有效数据
                    # 保存到缓存
                    self._save_to_cache(symbol, info)
                    self.stats['successful_requests'] += 1
                    
                    if self.config['logging']['log_success']:
                        self.logger.info(f"✅ {symbol}: 数据获取成功")
                    return info
                else:
                    self.logger.warning(f"❌ {symbol}: yfinance数据无效")
                    self.stats['failed_requests'] += 1
                    return None
                    
            except Exception as e:
                error_msg = str(e)
                self.stats['retry_attempts'] += 1
                
                if self.config['logging']['log_errors']:
                    self.logger.warning(f"❌ 获取 {symbol} yfinance数据失败 (尝试 {attempt + 1}/{max_retries + 1}): {error_msg}")
                
                # 判断是否可重试
                if attempt < max_retries and self._is_retryable_error(e):
                    delay = self._calculate_delay(attempt)
                    if self.config['logging']['log_retries']:
                        self.logger.info(f"🔄 {symbol}: {delay}秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    if self.config['logging']['log_errors']:
                        self.logger.error(f"❌ {symbol}: 最终获取失败，不再重试")
                    self.stats['failed_requests'] += 1
                    return None
        
        return None
    
    def get_batch_financial_data(self, symbols: List[str], max_symbols: int = None) -> Dict[str, Dict]:
        """
        批量获取多个股票的财务数据，带智能错误处理和性能优化
        
        Args:
            symbols: 股票代码列表
            max_symbols: 最大处理股票数量
            
        Returns:
            {symbol: financial_metrics} 的字典
        """
        if max_symbols is None:
            max_symbols = self.config['rate_limiting']['batch_size']
        
        symbols = symbols[:max_symbols]
        results = {}
        failed_symbols = []
        
        self.logger.info(f"🔄 开始批量获取yfinance财务数据，目标股票数: {len(symbols)}")
        
        start_time = time.time()
        
        for i, symbol in enumerate(symbols):
            try:
                info = self.get_stock_info(symbol)
                if info:
                    metrics = self.extract_financial_metrics(info)
                    results[symbol] = metrics
                    
                    if self.config['logging']['log_success']:
                        self.logger.info(f"✅ {symbol}: 数据获取成功 ({i+1}/{len(symbols)})")
                else:
                    self.logger.warning(f"❌ {symbol}: 数据获取失败")
                    failed_symbols.append(symbol)
                
                # 速率限制
                delay = self.config['rate_limiting']['delay_between_requests']
                if i < len(symbols) - 1:  # 不是最后一个请求
                    time.sleep(delay)
                    
            except Exception as e:
                self.logger.error(f"❌ {symbol}: 处理异常 - {e}")
                failed_symbols.append(symbol)
                continue
        
        end_time = time.time()
        
        # 统计结果
        success_rate = len(results) / len(symbols) * 100
        self.logger.info(f"📊 批量获取完成，成功: {len(results)}/{len(symbols)} ({success_rate:.1f}%)")
        self.logger.info(f"⏱️ 总耗时: {end_time - start_time:.2f}秒")
        
        if failed_symbols:
            self.logger.warning(f"❌ 失败的股票: {', '.join(failed_symbols)}")
        
        return results
    
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
            'debt_to_equity': safe_float(info.get('debtToEquity'), 50.0) / 100,
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats['total_requests']
        if total > 0:
            success_rate = self.stats['successful_requests'] / total * 100
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100
        else:
            success_rate = 0
            cache_hit_rate = 0
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'cache_hit_rate': cache_hit_rate
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_attempts': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = AdvancedYFinanceClient()
    
    # 测试单个股票
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'INVALID_SYMBOL']
    
    print("测试单个股票获取:")
    for symbol in test_symbols:
        info = client.get_stock_info(symbol)
        if info:
            print(f"✅ {symbol}: 成功")
        else:
            print(f"❌ {symbol}: 失败")
    
    # 测试批量获取
    print("\n测试批量获取:")
    results = client.get_batch_financial_data(test_symbols)
    print(f"成功获取 {len(results)} 只股票的数据")
    
    # 显示统计信息
    print("\n统计信息:")
    stats = client.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}") 