#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试改进的yfinance客户端
验证重试机制和错误处理功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.yfinance_client import YFinanceClient
import time

def test_single_stock():
    """测试单个股票数据获取"""
    print("=" * 50)
    print("测试单个股票数据获取")
    print("=" * 50)
    
    client = YFinanceClient(max_retries=3, retry_delay=1.0)
    
    # 测试正常股票
    print("\n1. 测试正常股票 (AAPL):")
    info = client.get_stock_info("AAPL")
    if info:
        print(f"✅ 成功获取AAPL数据，包含 {len(info)} 个字段")
        print(f"   当前价格: {info.get('regularMarketPrice', 'N/A')}")
        print(f"   市值: {info.get('marketCap', 'N/A')}")
    else:
        print("❌ 获取AAPL数据失败")
    
    # 测试无效股票
    print("\n2. 测试无效股票 (INVALID_SYMBOL):")
    info = client.get_stock_info("INVALID_SYMBOL")
    if info:
        print("✅ 意外获取到数据")
    else:
        print("❌ 正确识别无效股票")

def test_batch_stocks():
    """测试批量股票数据获取"""
    print("\n" + "=" * 50)
    print("测试批量股票数据获取")
    print("=" * 50)
    
    client = YFinanceClient(max_retries=2, retry_delay=0.5)
    
    # 测试股票列表（包含一些可能出错的股票）
    test_symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",  # 正常股票
        "INVALID1", "INVALID2",  # 无效股票
        "NVDA", "AMD", "META"  # 更多正常股票
    ]
    
    print(f"测试股票列表: {test_symbols}")
    
    start_time = time.time()
    results = client.get_batch_financial_data(test_symbols, max_symbols=len(test_symbols))
    end_time = time.time()
    
    print(f"\n📊 批量获取结果:")
    print(f"   总耗时: {end_time - start_time:.2f}秒")
    print(f"   成功获取: {len(results)}/{len(test_symbols)} 只股票")
    print(f"   成功率: {len(results)/len(test_symbols)*100:.1f}%")
    
    if results:
        print(f"\n✅ 成功获取的股票:")
        for symbol, metrics in results.items():
            print(f"   {symbol}: PE={metrics.get('pe_ratio', 'N/A'):.2f}, "
                  f"市值={metrics.get('market_cap', 0)/1e9:.1f}B")

def test_cache_functionality():
    """测试缓存功能"""
    print("\n" + "=" * 50)
    print("测试缓存功能")
    print("=" * 50)
    
    client = YFinanceClient(max_retries=1, retry_delay=0.1)
    
    symbol = "MSFT"
    
    print(f"1. 第一次获取 {symbol} 数据（不使用缓存）:")
    start_time = time.time()
    info1 = client.get_stock_info(symbol, use_cache=False)
    time1 = time.time() - start_time
    print(f"   耗时: {time1:.2f}秒")
    
    print(f"\n2. 第二次获取 {symbol} 数据（使用缓存）:")
    start_time = time.time()
    info2 = client.get_stock_info(symbol, use_cache=True)
    time2 = time.time() - start_time
    print(f"   耗时: {time2:.2f}秒")
    
    if info1 and info2:
        print(f"   缓存加速: {time1/time2:.1f}x")
        print(f"   数据一致性: {'✅' if info1 == info2 else '❌'}")

def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 50)
    print("测试错误处理")
    print("=" * 50)
    
    client = YFinanceClient(max_retries=2, retry_delay=0.1)
    
    # 测试各种可能的错误情况
    error_test_cases = [
        "",  # 空字符串
        "A" * 100,  # 超长字符串
        "123",  # 纯数字
        "AAPL.INVALID",  # 包含特殊字符
    ]
    
    for test_case in error_test_cases:
        print(f"\n测试股票代码: '{test_case}'")
        try:
            info = client.get_stock_info(test_case, use_cache=False)
            if info:
                print(f"   ✅ 意外获取到数据")
            else:
                print(f"   ❌ 正确返回None")
        except Exception as e:
            print(f"   ❌ 抛出异常: {type(e).__name__}: {e}")

def main():
    """主测试函数"""
    print("🚀 开始测试改进的yfinance客户端")
    print("=" * 60)
    
    try:
        # 测试单个股票
        test_single_stock()
        
        # 测试批量获取
        test_batch_stocks()
        
        # 测试缓存功能
        test_cache_functionality()
        
        # 测试错误处理
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 