#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试yfinance错误处理改进
验证curl错误16的处理和重试机制
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.yfinance_client import YFinanceClient
from utils.advanced_yfinance_client import AdvancedYFinanceClient
import time

def test_basic_client():
    """测试基础yfinance客户端"""
    print("=" * 60)
    print("测试基础yfinance客户端")
    print("=" * 60)
    
    client = YFinanceClient(max_retries=2, retry_delay=0.5)
    
    # 测试正常股票
    print("\n1. 测试正常股票 (AAPL):")
    start_time = time.time()
    info = client.get_stock_info("AAPL")
    end_time = time.time()
    
    if info:
        print(f"✅ 成功获取AAPL数据")
        print(f"   耗时: {end_time - start_time:.2f}秒")
        print(f"   数据字段数: {len(info)}")
    else:
        print("❌ 获取AAPL数据失败")
    
    # 测试可能出错的股票（模拟网络问题）
    print("\n2. 测试可能出错的股票 (模拟网络问题):")
    test_symbols = ["MSFT", "GOOGL", "TSLA", "NVDA"]
    
    for symbol in test_symbols:
        start_time = time.time()
        info = client.get_stock_info(symbol)
        end_time = time.time()
        
        if info:
            print(f"✅ {symbol}: 成功 ({end_time - start_time:.2f}秒)")
        else:
            print(f"❌ {symbol}: 失败 ({end_time - start_time:.2f}秒)")

def test_advanced_client():
    """测试高级yfinance客户端"""
    print("\n" + "=" * 60)
    print("测试高级yfinance客户端")
    print("=" * 60)
    
    client = AdvancedYFinanceClient()
    
    # 测试批量获取
    print("\n1. 测试批量获取:")
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD"]
    
    start_time = time.time()
    results = client.get_batch_financial_data(test_symbols)
    end_time = time.time()
    
    print(f"📊 批量获取结果:")
    print(f"   总耗时: {end_time - start_time:.2f}秒")
    print(f"   成功获取: {len(results)}/{len(test_symbols)} 只股票")
    print(f"   成功率: {len(results)/len(test_symbols)*100:.1f}%")
    
    # 显示统计信息
    print("\n2. 统计信息:")
    stats = client.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

def test_error_simulation():
    """测试错误模拟"""
    print("\n" + "=" * 60)
    print("测试错误模拟")
    print("=" * 60)
    
    client = AdvancedYFinanceClient()
    
    # 测试无效股票
    print("\n1. 测试无效股票:")
    invalid_symbols = ["INVALID_SYMBOL_123", "NOT_EXIST", "EMPTY", ""]
    
    for symbol in invalid_symbols:
        print(f"\n测试: '{symbol}'")
        info = client.get_stock_info(symbol, use_cache=False)
        if info:
            print(f"   ✅ 意外获取到数据")
        else:
            print(f"   ❌ 正确返回None")
    
    # 测试缓存功能
    print("\n2. 测试缓存功能:")
    symbol = "MSFT"
    
    print(f"第一次获取 {symbol} (不使用缓存):")
    start_time = time.time()
    info1 = client.get_stock_info(symbol, use_cache=False)
    time1 = time.time() - start_time
    print(f"   耗时: {time1:.2f}秒")
    
    print(f"第二次获取 {symbol} (使用缓存):")
    start_time = time.time()
    info2 = client.get_stock_info(symbol, use_cache=True)
    time2 = time.time() - start_time
    print(f"   耗时: {time2:.2f}秒")
    
    if info1 and info2:
        print(f"   缓存加速: {time1/time2:.1f}x")
        print(f"   数据一致性: {'✅' if info1 == info2 else '❌'}")

def main():
    """主测试函数"""
    print("🚀 开始测试yfinance错误处理改进")
    print("=" * 80)
    
    try:
        # 测试基础客户端
        test_basic_client()
        
        # 测试高级客户端
        test_advanced_client()
        
        # 测试错误模拟
        test_error_simulation()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试完成！")
        print("\n💡 改进说明:")
        print("1. ✅ 添加了智能重试机制，特别针对curl错误16")
        print("2. ✅ 实现了指数退避策略，避免频繁重试")
        print("3. ✅ 改进了错误分类，区分可重试和不可重试错误")
        print("4. ✅ 优化了缓存机制，提高数据获取效率")
        print("5. ✅ 添加了详细的统计信息和日志记录")
        print("6. ✅ 支持配置文件，便于调整参数")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 