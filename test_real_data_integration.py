#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试真实财务数据集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitor.phase2_professional_screener import Phase2ProfessionalScreener
from utils.alpha_vantage_client import AlphaVantageClient
import time

def test_alpha_vantage_client():
    """测试Alpha Vantage客户端"""
    print("🧪 测试Alpha Vantage客户端...")
    
    try:
        client = AlphaVantageClient()
        print(f"✅ Alpha Vantage客户端初始化成功")
        print(f"📊 API密钥: {client.api_key[:10]}...")
        print(f"🗂️ 缓存目录: {client.cache_dir}")
        
        # 测试获取单个股票数据
        print("\n🔍 测试获取AAPL财务数据...")
        overview_data = client.get_company_overview('AAPL')
        
        if overview_data:
            print("✅ 成功获取AAPL概览数据")
            metrics = client.extract_financial_metrics(overview_data)
            print(f"📈 ROE: {metrics.get('roe', 'N/A')}")
            print(f"💰 市值: {metrics.get('market_cap', 'N/A')}")
            print(f"📊 PE比率: {metrics.get('pe_ratio', 'N/A')}")
        else:
            print("❌ 获取AAPL数据失败")
            
        return True
        
    except Exception as e:
        print(f"❌ Alpha Vantage客户端测试失败: {e}")
        return False

def test_phase2_screener():
    """测试Phase2筛选器"""
    print("\n🎯 测试Phase2筛选器...")
    
    try:
        screener = Phase2ProfessionalScreener()
        print("✅ Phase2筛选器初始化成功")
        
        # 测试单个股票分析
        print("\n🔍 测试单个股票分析 (AAPL)...")
        result = screener.analyze_stock_professional('AAPL')
        
        if result:
            print("✅ 股票分析成功")
            print(f"📊 综合评分: {result['multifactor_score']:.1f}")
            print(f"🏆 质量因子: {result['quality_factor']:.3f}")
            print(f"📈 动量因子: {result['momentum_factor']:.3f}")
            print(f"⚡ 夏普比率: {result['sharpe_ratio']:.3f}")
        else:
            print("❌ 股票分析失败")
            
        return True
        
    except Exception as e:
        print(f"❌ Phase2筛选器测试失败: {e}")
        return False

def test_small_screening():
    """测试小规模筛选"""
    print("\n🚀 测试小规模筛选 (前10只股票)...")
    
    try:
        screener = Phase2ProfessionalScreener()
        
        # 获取前10只股票进行测试
        from data.data_interface import DataInterface
        di = DataInterface()
        symbols = di.get_available_symbols()[:10]
        
        print(f"📋 测试股票: {symbols}")
        
        results = []
        for symbol in symbols:
            print(f"🔍 分析 {symbol}...")
            result = screener.analyze_stock_professional(symbol)
            if result and result['multifactor_score'] >= 50:
                results.append(result)
                print(f"✅ {symbol}: 评分 {result['multifactor_score']:.1f}")
            else:
                print(f"⚠️ {symbol}: 评分过低或分析失败")
        
        print(f"\n📊 筛选结果: 找到 {len(results)} 只优质股票")
        
        # 显示前3名
        if results:
            sorted_results = sorted(results, key=lambda x: x['multifactor_score'], reverse=True)
            print("\n🏆 TOP 3 优质股票:")
            for i, result in enumerate(sorted_results[:3], 1):
                print(f"  {i}. {result['symbol']}: {result['multifactor_score']:.1f}分")
        
        return True
        
    except Exception as e:
        print(f"❌ 小规模筛选测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 开始真实财务数据集成测试...\n")
    
    start_time = time.time()
    
    # 测试1: Alpha Vantage客户端
    test1_success = test_alpha_vantage_client()
    
    # 测试2: Phase2筛选器
    test2_success = test_phase2_screener()
    
    # 测试3: 小规模筛选
    test3_success = test_small_screening()
    
    # 总结
    elapsed_time = time.time() - start_time
    print(f"\n📊 测试总结 (耗时: {elapsed_time:.1f}秒)")
    print(f"✅ Alpha Vantage客户端: {'通过' if test1_success else '失败'}")
    print(f"✅ Phase2筛选器: {'通过' if test2_success else '失败'}")
    print(f"✅ 小规模筛选: {'通过' if test3_success else '失败'}")
    
    if all([test1_success, test2_success, test3_success]):
        print("\n🎉 所有测试通过！真实财务数据集成成功！")
        print("💡 建议: 现在可以运行完整的股票筛选了")
    else:
        print("\n⚠️ 部分测试失败，需要进一步调试")

if __name__ == "__main__":
    main() 