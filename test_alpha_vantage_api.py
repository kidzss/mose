#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Alpha Vantage API密钥
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.alpha_vantage_client import AlphaVantageClient
import time

def test_api_key():
    """测试API密钥是否有效"""
    print("🔑 测试Alpha Vantage API密钥...")
    
    try:
        client = AlphaVantageClient()
        print(f"✅ Alpha Vantage客户端初始化成功")
        print(f"📊 API密钥: {client.api_key[:10]}...")
        
        # 测试获取AAPL数据
        print("\n🔍 测试获取AAPL财务数据...")
        overview_data = client.get_company_overview('AAPL')
        
        if overview_data and 'Information' not in overview_data:
            print("✅ 成功获取AAPL真实财务数据！")
            
            # 提取财务指标
            metrics = client.extract_financial_metrics(overview_data)
            
            print(f"\n📈 财务指标:")
            print(f"  ROE: {metrics.get('roe', 'N/A'):.2f}%")
            print(f"  ROA: {metrics.get('roa', 'N/A'):.2f}%")
            print(f"  市值: ${metrics.get('market_cap', 'N/A'):,.0f}")
            print(f"  PE比率: {metrics.get('pe_ratio', 'N/A'):.2f}")
            print(f"  PB比率: {metrics.get('pb_ratio', 'N/A'):.2f}")
            print(f"  债务权益比: {metrics.get('debt_to_equity', 'N/A'):.2f}")
            print(f"  流动比率: {metrics.get('current_ratio', 'N/A'):.2f}")
            print(f"  毛利率: {metrics.get('gross_margin', 'N/A'):.2f}%")
            
            return True
        else:
            print("❌ API密钥可能无效或遇到限制")
            if overview_data:
                print(f"API响应: {overview_data}")
            return False
            
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

def test_multiple_stocks():
    """测试多个股票的数据获取"""
    print("\n🚀 测试多个股票数据获取...")
    
    try:
        client = AlphaVantageClient()
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            print(f"\n📊 获取 {symbol} 数据...")
            overview_data = client.get_company_overview(symbol)
            
            if overview_data and 'Information' not in overview_data:
                metrics = client.extract_financial_metrics(overview_data)
                print(f"✅ {symbol}: ROE {metrics.get('roe', 0):.1f}%, 市值 ${metrics.get('market_cap', 0)/1e9:.1f}B")
            else:
                print(f"❌ {symbol}: 数据获取失败")
            
            # API限制：等待避免超限
            time.sleep(15)
        
        return True
        
    except Exception as e:
        print(f"❌ 多股票测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 Alpha Vantage API密钥验证测试\n")
    
    # 测试1: 验证API密钥
    test1_success = test_api_key()
    
    if test1_success:
        # 测试2: 多股票测试
        test2_success = test_multiple_stocks()
        
        print(f"\n📊 测试总结:")
        print(f"✅ API密钥验证: {'通过' if test1_success else '失败'}")
        print(f"✅ 多股票测试: {'通过' if test2_success else '失败'}")
        
        if test1_success and test2_success:
            print("\n🎉 API密钥配置成功！现在可以获取完整的财务数据了！")
            print("💡 建议: 运行完整的股票筛选来测试新的财务数据")
        else:
            print("\n⚠️ 部分测试失败，请检查API密钥配置")
    else:
        print("\n❌ API密钥验证失败，请检查配置")

if __name__ == "__main__":
    main() 