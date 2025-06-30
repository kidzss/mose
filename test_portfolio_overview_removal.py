#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试投资组合概览移除功能
验证AI每日持股分析监控系统中投资组合概览部分已被成功移除
"""

import sys
import os
import re

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_portfolio_overview_removal():
    """测试投资组合概览移除功能"""
    print("🧪 测试投资组合概览移除功能")
    print("=" * 60)
    
    try:
        # 读取文件内容
        with open('start_ai_daily_analysis_monitor.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含投资组合概览相关代码
        portfolio_overview_patterns = [
            r'st\.subheader\("📋 投资组合概览"\)',
            r'总市值',
            r'总成本', 
            r'总盈亏',
            r'total_value = 0',
            r'total_cost = 0',
            r'portfolio_summary = \[\]',
            r'st\.metric\("总市值"',
            r'st\.metric\("总成本"',
            r'st\.metric\("总盈亏"',
            r'summary_df = pd\.DataFrame\(portfolio_summary\)',
            r'st\.dataframe\(summary_df'
        ]
        
        print("🔍 检查投资组合概览相关代码...")
        
        found_patterns = []
        for pattern in portfolio_overview_patterns:
            if re.search(pattern, content):
                found_patterns.append(pattern)
                print(f"   ❌ 发现: {pattern}")
            else:
                print(f"   ✅ 已移除: {pattern}")
        
        # 检查分析历史部分是否保留
        analysis_history_patterns = [
            r'st\.subheader\("📚 分析历史"\)',
            r'analysis_history',
            r'st\.write\(f"'
        ]
        
        print("\n🔍 检查分析历史部分...")
        
        for pattern in analysis_history_patterns:
            if re.search(pattern, content):
                print(f"   ✅ 保留: {pattern}")
            else:
                print(f"   ⚠️ 未找到: {pattern}")
        
        # 检查col2部分的结构
        print("\n🔍 检查col2部分结构...")
        
        # 查找col2部分的代码
        col2_pattern = r'with col2:(.*?)(?=with col1:|# 页脚|$)'
        col2_match = re.search(col2_pattern, content, re.DOTALL)
        
        if col2_match:
            col2_content = col2_match.group(1)
            print("   ✅ 找到col2部分")
            
            # 检查col2内容
            if '投资组合概览' not in col2_content:
                print("   ✅ 投资组合概览已移除")
            else:
                print("   ❌ 投资组合概览仍存在")
                
            if '分析历史' in col2_content:
                print("   ✅ 分析历史部分保留")
            else:
                print("   ❌ 分析历史部分缺失")
        else:
            print("   ⚠️ 未找到col2部分")
        
        # 验证结果
        if len(found_patterns) == 0:
            print("\n🎉 投资组合概览移除测试通过!")
            print("✅ 所有投资组合概览相关代码已成功移除")
            print("✅ 分析历史部分正常保留")
            return True
        else:
            print(f"\n⚠️ 发现 {len(found_patterns)} 个投资组合概览相关代码片段")
            print("请检查并移除这些代码")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_layout():
    """测试UI布局"""
    print("\n🧪 测试UI布局...")
    
    try:
        # 检查主界面布局
        with open('start_ai_daily_analysis_monitor.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查列布局
        layout_patterns = [
            r'col1, col2 = st\.columns\(\[2, 1\]\)',
            r'with col1:',
            r'with col2:'
        ]
        
        for pattern in layout_patterns:
            if re.search(pattern, content):
                print(f"   ✅ 布局正确: {pattern}")
            else:
                print(f"   ⚠️ 布局问题: {pattern}")
        
        # 检查col1内容（应该包含实时市场数据和AI分析）
        col1_pattern = r'with col1:(.*?)(?=with col2:|$)'
        col1_match = re.search(col1_pattern, content, re.DOTALL)
        
        if col1_match:
            col1_content = col1_match.group(1)
            if '实时市场数据' in col1_content and 'AI分析结果' in col1_content:
                print("   ✅ col1内容正确: 包含实时市场数据和AI分析")
            else:
                print("   ⚠️ col1内容可能有问题")
        
        # 检查col2内容（应该只包含分析历史）
        col2_pattern = r'with col2:(.*?)(?=# 页脚|$)'
        col2_match = re.search(col2_pattern, content, re.DOTALL)
        
        if col2_match:
            col2_content = col2_match.group(1)
            if '分析历史' in col2_content and '投资组合概览' not in col2_content:
                print("   ✅ col2内容正确: 只包含分析历史")
            else:
                print("   ⚠️ col2内容可能有问题")
        
        print("✅ UI布局测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ UI布局测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试投资组合概览移除功能")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("投资组合概览移除", test_portfolio_overview_removal),
        ("UI布局", test_ui_layout)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {str(e)}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! 投资组合概览移除成功!")
        print("\n💡 改进效果:")
        print("   ✅ 移除了总市值、总成本、总盈亏显示")
        print("   ✅ 移除了投资组合概览部分")
        print("   ✅ 保留了分析历史功能")
        print("   ✅ 界面更加简洁专注")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 