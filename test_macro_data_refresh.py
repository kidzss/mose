#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试宏观数据刷新功能
验证修复后的数据更新机制是否正常工作
"""

import sys
import os
from datetime import datetime
import logging

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_macro_factor_analyzer():
    """测试宏观因子分析器的数据刷新功能"""
    print("=" * 60)
    print("测试宏观因子分析器数据刷新功能")
    print("=" * 60)
    
    try:
        from analysis.macro_factor_analyzer import MacroFactorAnalyzer
        analyzer = MacroFactorAnalyzer()
        
        # 第一次获取数据
        print("\n1. 第一次获取宏观数据...")
        start_time = datetime.now()
        macro_score_1 = analyzer.calculate_macro_score(force_refresh=True)
        end_time = datetime.now()
        duration_1 = (end_time - start_time).total_seconds()
        
        print(f"宏观得分: {macro_score_1.get('macro_score', 0):.3f}")
        print(f"建议: {macro_score_1.get('recommendation', '无')}")
        print(f"耗时: {duration_1:.2f}秒")
        
        # 第二次获取数据（应该使用缓存）
        print("\n2. 第二次获取宏观数据（使用缓存）...")
        start_time = datetime.now()
        macro_score_2 = analyzer.calculate_macro_score(force_refresh=False)
        end_time = datetime.now()
        duration_2 = (end_time - start_time).total_seconds()
        
        print(f"宏观得分: {macro_score_2.get('macro_score', 0):.3f}")
        print(f"建议: {macro_score_2.get('recommendation', '无')}")
        print(f"耗时: {duration_2:.2f}秒")
        
        # 第三次强制刷新
        print("\n3. 第三次强制刷新宏观数据...")
        start_time = datetime.now()
        macro_score_3 = analyzer.calculate_macro_score(force_refresh=True)
        end_time = datetime.now()
        duration_3 = (end_time - start_time).total_seconds()
        
        print(f"宏观得分: {macro_score_3.get('macro_score', 0):.3f}")
        print(f"建议: {macro_score_3.get('recommendation', '无')}")
        print(f"耗时: {duration_3:.2f}秒")
        
        # 验证数据是否更新
        print("\n4. 数据更新验证:")
        print(f"第一次 vs 第二次: {'相同' if macro_score_1.get('macro_score') == macro_score_2.get('macro_score') else '不同'}")
        print(f"第二次 vs 第三次: {'相同' if macro_score_2.get('macro_score') == macro_score_3.get('macro_score') else '不同'}")
        print(f"缓存效果: 第二次比第一次快 {((duration_1 - duration_2) / duration_1 * 100):.1f}%")
        
    except ImportError as e:
        print(f"❌ 导入宏观因子分析器失败: {e}")
    except Exception as e:
        print(f"❌ 测试宏观因子分析器失败: {e}")

def test_portfolio_macro_integration():
    """测试投资组合宏观集成器的数据刷新功能"""
    print("\n" + "=" * 60)
    print("测试投资组合宏观集成器数据刷新功能")
    print("=" * 60)
    
    try:
        from analysis.portfolio_macro_integration import PortfolioMacroIntegration
        integration = PortfolioMacroIntegration()
        
        # 第一次生成报告
        print("\n1. 第一次生成宏观报告...")
        start_time = datetime.now()
        report_1 = integration.generate_macro_report(force_refresh=True)
        end_time = datetime.now()
        duration_1 = (end_time - start_time).total_seconds()
        
        if 'error' not in report_1:
            macro_score_1 = report_1.get('executive_summary', {}).get('macro_score', 0)
            print(f"宏观得分: {macro_score_1:.3f}")
            print(f"耗时: {duration_1:.2f}秒")
        else:
            print(f"报告生成失败: {report_1['error']}")
        
        # 第二次生成报告（使用缓存）
        print("\n2. 第二次生成宏观报告（使用缓存）...")
        start_time = datetime.now()
        report_2 = integration.generate_macro_report(force_refresh=False)
        end_time = datetime.now()
        duration_2 = (end_time - start_time).total_seconds()
        
        if 'error' not in report_2:
            macro_score_2 = report_2.get('executive_summary', {}).get('macro_score', 0)
            print(f"宏观得分: {macro_score_2:.3f}")
            print(f"耗时: {duration_2:.2f}秒")
        else:
            print(f"报告生成失败: {report_2['error']}")
        
        # 验证数据是否更新
        if 'error' not in report_1 and 'error' not in report_2:
            print("\n3. 数据更新验证:")
            print(f"两次报告得分: {'相同' if macro_score_1 == macro_score_2 else '不同'}")
            print(f"缓存效果: 第二次比第一次快 {((duration_1 - duration_2) / duration_1 * 100):.1f}%")
            
    except ImportError as e:
        print(f"❌ 导入投资组合宏观集成器失败: {e}")
    except Exception as e:
        print(f"❌ 测试投资组合宏观集成器失败: {e}")

def test_inflation_sector_analyzer():
    """测试通胀行业分析器的数据刷新功能"""
    print("\n" + "=" * 60)
    print("测试通胀行业分析器数据刷新功能")
    print("=" * 60)
    
    try:
        from analysis.inflation_sector_analyzer import InflationSectorAnalyzer
        analyzer = InflationSectorAnalyzer()
        
        # 第一次生成报告
        print("\n1. 第一次生成通胀分析报告...")
        start_time = datetime.now()
        report_1 = analyzer.generate_inflation_sector_report(force_refresh=True)
        end_time = datetime.now()
        duration_1 = (end_time - start_time).total_seconds()
        
        if report_1:
            inflation_env_1 = report_1.get('inflation_regime', {})
            print(f"通胀环境: {inflation_env_1.get('type', '未知')}")
            print(f"信心度: {inflation_env_1.get('confidence', 0):.1%}")
            print(f"耗时: {duration_1:.2f}秒")
        else:
            print("报告生成失败")
        
        # 第二次生成报告（使用缓存）
        print("\n2. 第二次生成通胀分析报告（使用缓存）...")
        start_time = datetime.now()
        report_2 = analyzer.generate_inflation_sector_report(force_refresh=False)
        end_time = datetime.now()
        duration_2 = (end_time - start_time).total_seconds()
        
        if report_2:
            inflation_env_2 = report_2.get('inflation_regime', {})
            print(f"通胀环境: {inflation_env_2.get('type', '未知')}")
            print(f"信心度: {inflation_env_2.get('confidence', 0):.1%}")
            print(f"耗时: {duration_2:.2f}秒")
        else:
            print("报告生成失败")
        
        # 验证数据是否更新
        if report_1 and report_2:
            print("\n3. 数据更新验证:")
            env_same = inflation_env_1.get('type') == inflation_env_2.get('type')
            print(f"通胀环境: {'相同' if env_same else '不同'}")
            print(f"缓存效果: 第二次比第一次快 {((duration_1 - duration_2) / duration_1 * 100):.1f}%")
            
    except ImportError as e:
        print(f"❌ 导入通胀行业分析器失败: {e}")
    except Exception as e:
        print(f"❌ 测试通胀行业分析器失败: {e}")

def main():
    """主测试函数"""
    print("开始测试宏观数据刷新功能...")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[:3]}...")
    
    try:
        # 测试宏观因子分析器
        test_macro_factor_analyzer()
        
        # 测试投资组合宏观集成器
        test_portfolio_macro_integration()
        
        # 测试通胀行业分析器
        test_inflation_sector_analyzer()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        print("\n修复说明:")
        print("1. 添加了force_refresh参数，可以强制刷新数据")
        print("2. 改进了缓存机制，1小时内使用缓存，超过1小时自动刷新")
        print("3. 智能日报生成器默认强制刷新，确保数据最新")
        print("4. 添加了更详细的日志输出，便于调试")
        print("5. 改进了错误处理，避免数据获取失败时系统崩溃")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        print(f"\n❌ 测试失败: {e}")

if __name__ == "__main__":
    main() 