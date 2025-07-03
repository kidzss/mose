#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON数据功能测试脚本
测试JSON数据生成、保存、加载和使用功能
"""

import sys
import os
import json
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitor.smart_daily_report import SmartDailyReportGenerator
from utils.stock_data_loader import StockDataLoader

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_json_data_generation():
    """测试JSON数据生成功能"""
    print("🧪 测试JSON数据生成功能")
    print("=" * 50)
    
    try:
        # 创建智能日报生成器
        generator = SmartDailyReportGenerator(auto_update_data=False)  # 不自动更新数据，使用现有数据
        
        # 生成报告（这会同时生成HTML和JSON）
        print("📊 生成智能日报...")
        html_report = generator.generate_report()
        
        if html_report:
            print("✅ HTML报告生成成功")
        else:
            print("❌ HTML报告生成失败")
            return False
        
        # 检查是否生成了JSON文件
        import glob
        json_files = glob.glob("stock_analysis_data_*.json")
        if json_files:
            latest_json = max(json_files, key=os.path.getmtime)
            print(f"✅ JSON数据文件生成成功: {latest_json}")
            return True
        else:
            print("❌ 未找到JSON数据文件")
            return False
            
    except Exception as e:
        logger.error(f"JSON数据生成测试失败: {e}")
        print(f"❌ 测试失败: {e}")
        return False

def test_json_data_loading():
    """测试JSON数据加载功能"""
    print("\n🧪 测试JSON数据加载功能")
    print("=" * 50)
    
    try:
        # 创建数据加载器
        loader = StockDataLoader()
        
        # 获取最新数据文件
        latest_file = loader.get_latest_data_file()
        if not latest_file:
            print("❌ 未找到JSON数据文件")
            return False
        
        print(f"✅ 找到数据文件: {latest_file}")
        
        # 加载数据
        data = loader.load_data()
        if not data:
            print("❌ 数据加载失败")
            return False
        
        print(f"✅ 数据加载成功")
        print(f"   包含股票数量: {len(data.get('stocks', {}))}")
        print(f"   生成时间: {data.get('timestamp', 'unknown')}")
        print(f"   数据版本: {data.get('data_version', 'unknown')}")
        
        # 检查数据结构
        required_keys = ['timestamp', 'data_version', 'stocks']
        for key in required_keys:
            if key not in data:
                print(f"❌ 缺少必要的数据字段: {key}")
                return False
        
        print("✅ 数据结构验证通过")
        return True
        
    except Exception as e:
        logger.error(f"JSON数据加载测试失败: {e}")
        print(f"❌ 测试失败: {e}")
        return False

def test_stock_data_access():
    """测试股票数据访问功能"""
    print("\n🧪 测试股票数据访问功能")
    print("=" * 50)
    
    try:
        loader = StockDataLoader()
        data = loader.load_data()
        
        if not data:
            print("❌ 数据加载失败")
            return False
        
        # 获取所有股票列表
        symbols = loader.get_all_stocks()
        if not symbols:
            print("❌ 没有股票数据")
            return False
        
        print(f"✅ 获取股票列表成功: {len(symbols)} 只股票")
        
        # 测试获取单个股票数据
        test_symbol = symbols[0]
        stock_data = loader.get_stock_data(test_symbol)
        
        if not stock_data:
            print(f"❌ 无法获取 {test_symbol} 的数据")
            return False
        
        print(f"✅ 成功获取 {test_symbol} 的数据")
        
        # 检查股票数据结构
        required_stock_keys = ['basic_info', 'market_environment', 'strategy']
        for key in required_stock_keys:
            if key not in stock_data:
                print(f"❌ 股票数据缺少必要字段: {key}")
                return False
        
        print("✅ 股票数据结构验证通过")
        
        # 显示股票基本信息
        basic_info = stock_data['basic_info']
        print(f"   {test_symbol} 基本信息:")
        print(f"     当前价格: ${basic_info.get('current_price', 0):.2f}")
        print(f"     涨跌幅: {basic_info.get('price_change_pct', 0):+.2f}%")
        print(f"     RSI: {basic_info.get('rsi', 0):.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"股票数据访问测试失败: {e}")
        print(f"❌ 测试失败: {e}")
        return False

def test_ai_integration():
    """测试AI集成功能"""
    print("\n🧪 测试AI集成功能")
    print("=" * 50)
    
    try:
        loader = StockDataLoader()
        data = loader.load_data()
        
        if not data:
            print("❌ 数据加载失败")
            return False
        
        # 测试AI输入格式化
        symbols = loader.get_all_stocks()
        if not symbols:
            print("❌ 没有股票数据")
            return False
        
        # 测试完整AI输入格式化
        ai_input_full = loader.format_for_ai_input()
        if not ai_input_full:
            print("❌ AI输入格式化失败")
            return False
        
        print(f"✅ AI输入格式化成功")
        print(f"   完整数据长度: {len(ai_input_full)} 字符")
        
        # 测试部分股票AI输入格式化
        ai_input_partial = loader.format_for_ai_input(symbols[:2])
        if not ai_input_partial:
            print("❌ 部分股票AI输入格式化失败")
            return False
        
        print(f"✅ 部分股票AI输入格式化成功")
        print(f"   部分数据长度: {len(ai_input_partial)} 字符")
        
        # 测试股票摘要
        test_symbol = symbols[0]
        summary = loader.get_stock_summary(test_symbol)
        if not summary:
            print(f"❌ 无法生成 {test_symbol} 的摘要")
            return False
        
        print(f"✅ 股票摘要生成成功")
        print(f"   {test_symbol} 摘要预览:")
        print(f"   {summary[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"AI集成测试失败: {e}")
        print(f"❌ 测试失败: {e}")
        return False

def test_portfolio_and_macro_data():
    """测试投资组合和宏观数据功能"""
    print("\n🧪 测试投资组合和宏观数据功能")
    print("=" * 50)
    
    try:
        loader = StockDataLoader()
        data = loader.load_data()
        
        if not data:
            print("❌ 数据加载失败")
            return False
        
        # 测试投资组合汇总
        portfolio_summary = loader.get_portfolio_summary()
        if portfolio_summary:
            print("✅ 投资组合汇总数据获取成功")
            print(f"   总价值: ${portfolio_summary.get('total_value', 0):,.2f}")
            print(f"   股票配置: {portfolio_summary.get('stock_allocation', 0):.2f}%")
        else:
            print("⚠️ 投资组合汇总数据不可用")
        
        # 测试宏观分析数据
        macro_analysis = loader.get_macro_analysis()
        if macro_analysis:
            print("✅ 宏观分析数据获取成功")
            print(f"   宏观得分: {macro_analysis.get('macro_score', 0):.2f}/1.00")
            print(f"   环境建议: {macro_analysis.get('recommendation', '无')}")
        else:
            print("⚠️ 宏观分析数据不可用")
        
        # 测试数据统计
        stats = loader.get_data_statistics()
        if stats:
            print("✅ 数据统计获取成功")
            print(f"   总股票数: {stats.get('total_stocks', 0)}")
            print(f"   分析覆盖: {stats.get('analysis_coverage', {})}")
        else:
            print("❌ 数据统计获取失败")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"投资组合和宏观数据测试失败: {e}")
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 JSON数据功能完整测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("JSON数据生成", test_json_data_generation),
        ("JSON数据加载", test_json_data_loading),
        ("股票数据访问", test_stock_data_access),
        ("AI集成功能", test_ai_integration),
        ("投资组合和宏观数据", test_portfolio_and_macro_data)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"{test_name} 测试异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试结果
    print("\n📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！JSON数据功能正常工作")
        print("\n💡 使用建议:")
        print("   1. 运行智能日报生成器生成JSON数据")
        print("   2. 使用StockDataLoader加载和访问数据")
        print("   3. 使用format_for_ai_input()生成AI输入格式")
        print("   4. 将数据用于AI分析和决策支持")
    else:
        print("⚠️ 部分测试失败，请检查系统配置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    print("\n🧪 读取ADBE和TSLA的数据演示\n" + "="*50)
    loader = StockDataLoader()
    loader.load_data()
    adbe_data = loader.get_stock_data("ADBE")
    tsla_data = loader.get_stock_data("TSLA")
    print("ADBE:")
    print(json.dumps(adbe_data, ensure_ascii=False, indent=2))
    print("\nTSLA:")
    print(json.dumps(tsla_data, ensure_ascii=False, indent=2)) 