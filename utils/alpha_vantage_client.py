#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import json
import os
from typing import Dict, Optional, Any
import pandas as pd

class AlphaVantageClient:
    """Alpha Vantage API客户端，用于获取财务基本面数据"""
    
    def __init__(self, api_key: str = None):
        """
        初始化Alpha Vantage客户端
        
        Args:
            api_key: Alpha Vantage API密钥
        """
        self.api_key = api_key or self._get_api_key()
        self.base_url = "https://www.alphavantage.co/query"
        self.cache_dir = "data/alpha_vantage_cache"
        self._ensure_cache_dir()
        
        # API限制：免费版每分钟5次调用，每天25次
        self.calls_per_minute = 5
        self.daily_limit = 25
        self.call_interval = 12  # 秒，确保不超过每分钟5次
        
    def _get_api_key(self) -> str:
        """从环境变量或配置文件获取API密钥"""
        # 首先尝试环境变量
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if api_key:
            return api_key
            
        # 尝试从配置文件读取
        config_path = "monitor/configs/alpha_vantage_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    api_key = config.get('api_key', '')
                    
                    # 检查是否为有效的API密钥
                    if api_key and api_key not in ['', 'demo', '请将此处替换为您的真实API密钥']:
                        return api_key
                    
            except Exception as e:
                print(f"读取Alpha Vantage配置失败: {e}")
        
        # 使用演示密钥（有限制）
        print("⚠️ 使用演示API密钥，功能受限。请设置真实API密钥。")
        return "demo"
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol: str, function: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{symbol}_{function}.json")
    
    def _load_from_cache(self, symbol: str, function: str) -> Optional[Dict]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(symbol, function)
        if os.path.exists(cache_path):
            try:
                # 检查缓存是否过期（1天）
                cache_age = time.time() - os.path.getmtime(cache_path)
                if cache_age < 24 * 3600:  # 24小时内的缓存有效
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                print(f"读取缓存失败 {symbol}: {e}")
        return None
    
    def _save_to_cache(self, symbol: str, function: str, data: Dict):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(symbol, function)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败 {symbol}: {e}")
    
    def _make_request(self, params: Dict[str, str]) -> Optional[Dict]:
        """发起API请求"""
        try:
            # 添加API密钥
            params['apikey'] = self.api_key
            
            # 发起请求
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查API错误
            if 'Error Message' in data:
                print(f"❌ Alpha Vantage API错误: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                print(f"⚠️ Alpha Vantage API限制: {data['Note']}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            return None
    
    def get_company_overview(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        获取公司基本面数据概览
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            包含财务指标的字典
        """
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(symbol, 'OVERVIEW')
            if cached_data:
                return cached_data
        
        # 发起API请求
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        print(f"📊 获取 {symbol} 的基本面数据...")
        data = self._make_request(params)
        
        if data:
            # 保存到缓存
            self._save_to_cache(symbol, 'OVERVIEW', data)
            
            # API限制：等待避免超限
            time.sleep(self.call_interval)
            
        return data
    
    def extract_financial_metrics(self, overview_data: Dict) -> Dict[str, float]:
        """
        从概览数据中提取关键财务指标
        
        Args:
            overview_data: Alpha Vantage OVERVIEW API返回的数据
            
        Returns:
            标准化的财务指标字典
        """
        def safe_float(value, default=0.0):
            """安全转换为浮点数"""
            if value in [None, 'None', '-', '']:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        metrics = {
            # 盈利能力指标
            'roe': safe_float(overview_data.get('ReturnOnEquityTTM')),  # 股本回报率
            'roa': safe_float(overview_data.get('ReturnOnAssetsTTM')),  # 资产回报率
            'gross_margin': safe_float(overview_data.get('GrossProfitTTM', 0)) / max(safe_float(overview_data.get('RevenueTTM', 1)), 1),
            
            # 财务健康指标
            'debt_to_equity': safe_float(overview_data.get('DebtEquityRatio')),  # 债务股权比
            'current_ratio': safe_float(overview_data.get('CurrentRatio')),      # 流动比率
            'quick_ratio': safe_float(overview_data.get('QuickRatio')),          # 速动比率
            
            # 估值指标
            'pe_ratio': safe_float(overview_data.get('PERatio')),                # 市盈率
            'pb_ratio': safe_float(overview_data.get('PriceToBookRatio')),       # 市净率
            'ps_ratio': safe_float(overview_data.get('PriceToSalesRatioTTM')),   # 市销率
            
            # 规模指标
            'market_cap': safe_float(overview_data.get('MarketCapitalization')), # 市值
            'revenue': safe_float(overview_data.get('RevenueTTM')),              # 收入
            'net_income': safe_float(overview_data.get('NetIncomeTTM')),         # 净利润
            
            # 其他指标
            'eps': safe_float(overview_data.get('EPS')),                         # 每股收益
            'dividend_yield': safe_float(overview_data.get('DividendYield')),    # 股息率
            'beta': safe_float(overview_data.get('Beta')),                       # Beta系数
        }
        
        return metrics
    
    def get_batch_financial_data(self, symbols: list, max_symbols: int = 20) -> Dict[str, Dict]:
        """
        批量获取多个股票的财务数据
        
        Args:
            symbols: 股票代码列表
            max_symbols: 最大处理股票数量（考虑API限制）
            
        Returns:
            {symbol: financial_metrics} 的字典
        """
        results = {}
        processed = 0
        
        print(f"🔄 开始批量获取财务数据，目标股票数: {min(len(symbols), max_symbols)}")
        
        for symbol in symbols[:max_symbols]:
            try:
                overview_data = self.get_company_overview(symbol)
                if overview_data:
                    metrics = self.extract_financial_metrics(overview_data)
                    results[symbol] = metrics
                    processed += 1
                    print(f"✅ {symbol}: 数据获取成功 ({processed}/{min(len(symbols), max_symbols)})")
                else:
                    print(f"❌ {symbol}: 数据获取失败")
                    
            except Exception as e:
                print(f"❌ {symbol}: 处理异常 - {e}")
                continue
        
        print(f"📊 批量获取完成，成功: {len(results)}/{min(len(symbols), max_symbols)}")
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
            filename = f"financial_data_{timestamp}.json"
        
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(financial_data, f, ensure_ascii=False, indent=2)
            print(f"💾 财务数据已保存到: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ 保存财务数据失败: {e}")
            return None
    
    def load_financial_data(self, filename: str) -> Optional[Dict[str, Dict]]:
        """
        从本地文件加载财务数据
        
        Args:
            filename: 文件名
            
        Returns:
            财务数据字典
        """
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📂 财务数据已加载: {filepath}")
            return data
        except Exception as e:
            print(f"❌ 加载财务数据失败: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = AlphaVantageClient()
    
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
            for key, value in metrics.items():
                print(f"  {key}: {value}") 