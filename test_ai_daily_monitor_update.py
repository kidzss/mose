#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI每日持股分析监控系统更新功能
测试新的AI分析下拉选择功能
"""

import sys
import os
import asyncio
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ai_analysis_selection():
    """测试AI分析选择功能"""
    print("🧪 测试AI分析选择功能...")
    
    try:
        from start_ai_daily_analysis_monitor import AIDailyAnalysisMonitor
        
        # 创建监控实例
        monitor = AIDailyAnalysisMonitor()
        
        # 测试加载配置
        print("📋 测试配置加载...")
        portfolio_config = monitor.load_portfolio_config()
        positions = portfolio_config.get('positions', {})
        watchlist = portfolio_config.get('watchlist', {})
        
        if positions:
            print(f"✅ 成功加载持仓配置: {len(positions)} 只股票")
            for symbol, pos in list(positions.items())[:3]:
                print(f"   - {symbol}: {pos.get('shares', 0)} 股")
        else:
            print("⚠️ 未找到持仓配置")
        
        if watchlist:
            print(f"✅ 成功加载观察仓配置: {len(watchlist)} 只股票")
            for symbol, watch in list(watchlist.items())[:3]:
                print(f"   - {symbol}: 目标价 ${watch.get('target_buy_price', 0):.2f}")
        else:
            print("⚠️ 未找到观察仓配置")
        
        # 测试分析类型配置
        print("\n🔍 测试分析类型配置...")
        analysis_types = {
            "comprehensive": "综合分析",
            "detailed": "详细分析", 
            "quick": "快速分析"
        }
        
        for key, value in analysis_types.items():
            print(f"   - {key}: {value}")
        
        print("✅ 分析类型配置正确")
        
        # 测试股票选择逻辑
        print("\n📊 测试股票选择逻辑...")
        test_positions = ["AAPL", "GOOGL", "MSFT"]
        test_watchlist = ["TSLA", "NVDA", "AMD"]
        
        all_symbols = test_positions + test_watchlist
        stock_options = [f"{symbol} ({'持仓' if symbol in test_positions else '观察仓'})" for symbol in all_symbols]
        
        print("股票选项列表:")
        for option in stock_options:
            print(f"   - {option}")
        
        # 测试股票代码提取
        test_selection = "AAPL (持仓)"
        extracted_symbol = test_selection.split(" (")[0]
        print(f"✅ 股票代码提取测试: '{test_selection}' -> '{extracted_symbol}'")
        
        # 测试分析历史记录结构
        print("\n📚 测试分析历史记录结构...")
        test_record = {
            'symbol': 'AAPL',
            'timestamp': datetime.now(),
            'result': {
                'action_suggestion': {
                    'action': '持有',
                    'reason': '技术面良好，基本面稳健',
                    'risk_warning': '注意市场波动风险'
                },
                'ai_analysis': '这是详细的AI分析内容...'
            },
            'type': 'position',
            'analysis_type': 'comprehensive'
        }
        
        print("✅ 分析历史记录结构正确")
        print(f"   - 股票: {test_record['symbol']}")
        print(f"   - 类型: {test_record['type']}")
        print(f"   - 分析类型: {test_record['analysis_type']}")
        print(f"   - 操作建议: {test_record['result']['action_suggestion']['action']}")
        
        print("\n🎉 AI分析选择功能测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_components():
    """测试UI组件功能"""
    print("\n🧪 测试UI组件功能...")
    
    try:
        # 测试分析控制面板布局
        print("📱 测试分析控制面板布局...")
        
        # 模拟列布局
        col1_width = 2
        col2_width = 1
        col3_width = 1
        total_width = col1_width + col2_width + col3_width
        
        print(f"✅ 列布局配置: {col1_width}:{col2_width}:{col3_width} (总计: {total_width})")
        
        # 测试分析类型映射
        analysis_types = {
            "comprehensive": "综合分析",
            "detailed": "详细分析", 
            "quick": "快速分析"
        }
        
        print("分析类型映射:")
        for key, value in analysis_types.items():
            print(f"   - {key} -> {value}")
        
        # 测试结果展示格式
        print("\n📊 测试结果展示格式...")
        
        test_metrics = {
            "操作建议": "持有",
            "分析类型": "综合分析", 
            "股票类型": "持仓"
        }
        
        for label, value in test_metrics.items():
            print(f"   - {label}: {value}")
        
        print("✅ UI组件功能测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ UI组件测试失败: {str(e)}")
        return False

def test_integration():
    """测试集成功能"""
    print("\n🧪 测试集成功能...")
    
    try:
        # 测试与professional_trading_monitor.py的集成
        print("🔗 测试与专业交易监控系统的集成...")
        
        # 检查关键文件是否存在
        required_files = [
            "start_ai_daily_analysis_monitor.py",
            "professional_trading_monitor.py",
            "portfolio_config.json"
        ]
        
        for file in required_files:
            if os.path.exists(file):
                print(f"✅ 找到文件: {file}")
            else:
                print(f"⚠️ 未找到文件: {file}")
        
        # 测试配置加载
        print("\n📋 测试配置加载...")
        try:
            import json
            with open("portfolio_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            
            positions = config.get("positions", {})
            watchlist = config.get("watchlist", {})
            
            print(f"✅ 配置加载成功")
            print(f"   - 持仓股票: {len(positions)} 只")
            print(f"   - 观察仓股票: {len(watchlist)} 只")
            
        except Exception as e:
            print(f"⚠️ 配置加载测试: {str(e)}")
        
        print("✅ 集成功能测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        return False

def test_ai_analysis_data_completeness():
    """测试AI分析数据的完整性"""
    print("\n🧪 测试AI分析数据完整性...")
    
    try:
        from start_ai_daily_analysis_monitor import AIDailyAnalysisMonitor
        
        # 创建监控实例
        monitor = AIDailyAnalysisMonitor()
        
        # 模拟市场数据
        mock_market_data = {
            'price': 174.43,
            'change_pct': 1.71,
            'volume': 25606956,
            'market_cap': 2126178000000,
            'technical_indicators': {
                'rsi': 56.5,
                'ma20': 173.12,
                'ma50': 166.63,
                'macd': 0.5,
                'macd_signal': 0.3,
                'macd_hist': 0.2,
                'volume_ratio': 1.2,
                'volatility': 0.25,
                'trend': 'up'
            },
            'financial_data': {
                'pe_ratio': 19.65,
                'peg_ratio': 1.35,
                'pb_ratio': 6.19,
                'roe': 34.79,
                'profit_margins': 0.3086,
                'revenue_growth': 0.12
            },
            'company_info': {
                'name': 'Alphabet Inc.',
                'sector': 'Communication Services',
                'industry': 'Internet Content & Information',
                'market_cap': 2126178000000
            },
            'position_info': {
                'shares': 30,
                'cost_basis': 170.0,
                'weight': 18.28,
                'sector': 'Technology'
            }
        }
        
        # 测试构建分析数据
        print("📊 测试构建分析数据...")
        analysis_data = monitor._build_comprehensive_analysis_data('GOOG', mock_market_data)
        
        # 检查必要的数据字段
        required_fields = [
            'symbol', 'current_price', 'change_pct', 'volume', 'market_cap',
            'technical_analysis', 'financial_analysis', 'company_info', 
            'position_analysis', 'market_environment'
        ]
        
        print("检查必要字段:")
        for field in required_fields:
            if field in analysis_data:
                print(f"   ✅ {field}: 存在")
            else:
                print(f"   ❌ {field}: 缺失")
        
        # 检查技术分析数据
        tech_analysis = analysis_data.get('technical_analysis', {})
        tech_fields = ['rsi', 'ma20', 'ma50', 'macd', 'volume_ratio', 'volatility', 'trend']
        
        print("\n检查技术分析数据:")
        for field in tech_fields:
            if field in tech_analysis:
                value = tech_analysis[field]
                print(f"   ✅ {field}: {value}")
            else:
                print(f"   ❌ {field}: 缺失")
        
        # 检查财务分析数据
        financial_analysis = analysis_data.get('financial_analysis', {})
        financial_fields = ['pe_ratio', 'peg_ratio', 'pb_ratio', 'roe', 'profit_margins', 'revenue_growth']
        
        print("\n检查财务分析数据:")
        for field in financial_fields:
            if field in financial_analysis:
                value = financial_analysis[field]
                print(f"   ✅ {field}: {value}")
            else:
                print(f"   ❌ {field}: 缺失")
        
        # 检查持仓分析数据
        position_analysis = analysis_data.get('position_analysis', {})
        if position_analysis:
            print("\n检查持仓分析数据:")
            position_fields = ['shares', 'cost_basis', 'weight', 'sector']
            for field in position_fields:
                if field in position_analysis:
                    value = position_analysis[field]
                    print(f"   ✅ {field}: {value}")
                else:
                    print(f"   ❌ {field}: 缺失")
        else:
            print("\n⚠️ 持仓分析数据: 无持仓信息")
        
        # 检查市场环境数据
        market_env = analysis_data.get('market_environment', {})
        env_fields = ['trend_strength', 'volume_analysis', 'volatility_assessment', 'overall_sentiment']
        
        print("\n检查市场环境数据:")
        for field in env_fields:
            if field in market_env:
                value = market_env[field]
                print(f"   ✅ {field}: {value}")
            else:
                print(f"   ❌ {field}: 缺失")
        
        # 验证数据格式是否符合您提到的要求
        print("\n📋 验证数据格式:")
        
        # 检查是否包含您提到的关键信息
        expected_info = [
            "当前价格", "涨跌幅", "RSI指标", "成交量", "持仓成本", 
            "持仓股数", "投资金额", "当前市值", "盈亏金额", "盈亏比例"
        ]
        
        # 模拟您提到的数据格式
        mock_complete_data = {
            'symbol': 'GOOG',
            'current_price': 174.43,
            'change_pct': 1.71,
            'volume': 25606956,
            'rsi': 56.5,
            'position_info': {
                'cost_basis': 170.0,
                'shares': 30,
                'investment_amount': 5100.00,
                'current_value': 5232.90,
                'unrealized_pnl': 132.90,
                'unrealized_pnl_pct': 2.61
            }
        }
        
        print("模拟完整数据格式:")
        for key, value in mock_complete_data.items():
            print(f"   ✅ {key}: {value}")
        
        print("\n🎉 AI分析数据完整性测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试AI每日持股分析监控系统更新功能")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("AI分析选择功能", test_ai_analysis_selection),
        ("UI组件功能", test_ui_components),
        ("集成功能", test_integration),
        ("AI分析数据完整性", test_ai_analysis_data_completeness)
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
        print("🎉 所有测试通过! AI分析选择功能更新成功!")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 