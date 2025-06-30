#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可配置AI每日持股分析监控系统测试脚本
Test script for Configurable AI Daily Holdings Analysis Monitor System
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# 导入测试模块
from start_configurable_ai_monitor import ConfigurableAIMonitor

async def test_configurable_ai_monitor():
    """测试可配置AI监控系统"""
    print("🧪 开始测试可配置AI每日持股分析监控系统")
    print("=" * 60)
    
    # 创建监控系统实例
    monitor = ConfigurableAIMonitor()
    
    # 测试1: 配置文件加载
    print("\n📋 测试1: 配置文件加载")
    print("-" * 30)
    
    config = monitor.load_portfolio_config()
    if config:
        print("✅ 配置文件加载成功")
        
        # 检查持仓
        positions = config.get('positions', {})
        if positions:
            print(f"📊 发现持仓股票: {list(positions.keys())}")
        else:
            print("⚠️ 未发现持仓股票")
        
        # 检查观察仓
        watchlist = config.get('watchlist', {})
        if watchlist:
            print(f"👀 发现观察仓股票: {list(watchlist.keys())}")
        else:
            print("⚠️ 未发现观察仓股票")
    else:
        print("❌ 配置文件加载失败")
        return False
    
    # 测试2: 实时数据获取
    print("\n📈 测试2: 实时数据获取")
    print("-" * 30)
    
    # 获取测试股票列表
    test_symbols = []
    if positions:
        test_symbols.extend(list(positions.keys())[:2])  # 取前2个持仓股票
    if watchlist:
        test_symbols.extend(list(watchlist.keys())[:2])  # 取前2个观察仓股票
    
    if not test_symbols:
        test_symbols = ['NVDA', 'AMD']  # 使用默认测试股票
    
    print(f"🔍 测试股票: {test_symbols}")
    
    real_time_data = monitor.get_real_time_data(test_symbols)
    if real_time_data:
        print(f"✅ 成功获取 {len(real_time_data)} 只股票的实时数据")
        for symbol, data in real_time_data.items():
            print(f"  📊 {symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)")
    else:
        print("❌ 实时数据获取失败")
        return False
    
    # 测试3: AI分析功能
    print("\n🤖 测试3: AI分析功能")
    print("-" * 30)
    
    if real_time_data:
        # 选择第一只股票进行AI分析测试
        test_symbol = list(real_time_data.keys())[0]
        test_data = real_time_data[test_symbol]
        
        print(f"🔍 测试AI分析: {test_symbol}")
        
        # 添加持仓信息（如果是持仓股票）
        if test_symbol in positions:
            position_info = positions[test_symbol]
            test_data['position_info'] = {
                'shares': position_info.get('shares', 0),
                'cost_basis': position_info.get('cost_basis', 0),
                'weight': position_info.get('weight', 0),
                'sector': position_info.get('sector', 'Unknown')
            }
            print(f"  📊 添加持仓信息: {position_info.get('shares', 0)}股, 成本${position_info.get('cost_basis', 0):.2f}")
        
        # 执行AI分析
        try:
            ai_result = await monitor.analyze_stock_with_ai(test_symbol, test_data, "quick")
            
            if ai_result:
                print("✅ AI分析成功")
                
                # 显示分析结果摘要
                action_suggestion = ai_result.get('action_suggestion', {})
                action = action_suggestion.get('action', 'N/A')
                reason = action_suggestion.get('reason', 'N/A')
                
                print(f"  🎯 操作建议: {action}")
                print(f"  📝 分析理由: {reason[:100]}...")
                
                # 检查分析内容
                ai_analysis = ai_result.get('ai_analysis', '')
                if ai_analysis:
                    print(f"  📊 AI分析内容长度: {len(ai_analysis)} 字符")
                else:
                    print("  ⚠️ AI分析内容为空")
                
            else:
                print("❌ AI分析失败")
                return False
                
        except Exception as e:
            print(f"❌ AI分析出错: {e}")
            return False
    
    # 测试4: 系统功能完整性
    print("\n🔧 测试4: 系统功能完整性")
    print("-" * 30)
    
    # 检查必要的属性
    required_attrs = ['ai_analyzer', 'daily_analyzer', 'analysis_history']
    for attr in required_attrs:
        if hasattr(monitor, attr):
            print(f"✅ {attr}: 正常")
        else:
            print(f"❌ {attr}: 缺失")
            return False
    
    # 检查方法
    required_methods = ['load_portfolio_config', 'get_real_time_data', 'analyze_stock_with_ai']
    for method in required_methods:
        if hasattr(monitor, method) and callable(getattr(monitor, method)):
            print(f"✅ {method}: 正常")
        else:
            print(f"❌ {method}: 缺失或不可调用")
            return False
    
    # 测试5: 历史记录功能
    print("\n📚 测试5: 历史记录功能")
    print("-" * 30)
    
    # 添加测试记录
    test_record = {
        'symbol': 'TEST',
        'timestamp': datetime.now(),
        'result': {'test': 'data'},
        'type': 'test'
    }
    
    monitor.analysis_history.append(test_record)
    print(f"✅ 添加测试记录: {len(monitor.analysis_history)} 条记录")
    
    # 检查记录内容
    if monitor.analysis_history:
        latest_record = monitor.analysis_history[-1]
        if latest_record['symbol'] == 'TEST':
            print("✅ 历史记录功能正常")
        else:
            print("❌ 历史记录功能异常")
            return False
    else:
        print("❌ 历史记录为空")
        return False
    
    print("\n🎉 所有测试通过！")
    print("=" * 60)
    return True

def test_config_file_loading():
    """测试配置文件加载功能"""
    print("\n📋 测试配置文件加载功能")
    print("-" * 30)
    
    monitor = ConfigurableAIMonitor()
    
    # 测试不同的配置文件路径
    test_paths = [
        'portfolio_config.json',
        'config/portfolio_config.json',
        'config/portfolio_config_latest.json'
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            print(f"✅ 配置文件存在: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    positions = config.get('positions', {})
                    watchlist = config.get('watchlist', {})
                    print(f"  📊 持仓股票: {len(positions)} 只")
                    print(f"  👀 观察仓股票: {len(watchlist)} 只")
            except Exception as e:
                print(f"  ❌ 配置文件读取失败: {e}")
        else:
            print(f"⚠️ 配置文件不存在: {path}")

def main():
    """主函数"""
    print("🚀 可配置AI每日持股分析监控系统测试")
    print("=" * 60)
    
    # 测试配置文件加载
    test_config_file_loading()
    
    # 运行主要测试
    success = asyncio.run(test_configurable_ai_monitor())
    
    if success:
        print("\n✅ 系统测试完成，所有功能正常")
        print("\n💡 使用建议:")
        print("  1. 运行 start_configurable_ai_monitor.bat 启动系统")
        print("  2. 在浏览器中访问 http://localhost:8504")
        print("  3. 在侧边栏选择要分析的股票")
        print("  4. 点击'批量AI分析'开始分析")
    else:
        print("\n❌ 系统测试失败，请检查配置和依赖")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 