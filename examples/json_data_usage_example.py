#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON数据使用示例
演示如何使用生成的JSON数据文件进行AI输入和其他分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stock_data_loader import StockDataLoader
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_json_data_usage():
    """演示JSON数据的使用方法"""
    
    print("🚀 JSON数据功能演示")
    print("=" * 60)
    
    # 1. 初始化数据加载器
    loader = StockDataLoader()
    
    # 2. 获取最新数据文件
    latest_file = loader.get_latest_data_file()
    if not latest_file:
        print("❌ 未找到JSON数据文件，请先运行日报生成器")
        return
    
    print(f"✅ 找到数据文件: {latest_file}")
    
    # 3. 加载数据
    data = loader.load_data()
    if not data:
        print("❌ 数据加载失败")
        return
    
    print(f"✅ 数据加载成功，包含 {len(data.get('stocks', {}))} 只股票")
    
    # 4. 获取数据统计信息
    stats = loader.get_data_statistics()
    print(f"\n📊 数据统计:")
    print(f"   生成时间: {stats.get('timestamp', 'unknown')}")
    print(f"   数据版本: {stats.get('data_version', 'unknown')}")
    print(f"   股票数量: {stats.get('total_stocks', 0)}")
    print(f"   包含投资组合汇总: {'是' if stats.get('has_portfolio_summary') else '否'}")
    print(f"   包含宏观分析: {'是' if stats.get('has_macro_analysis') else '否'}")
    
    # 5. 获取所有股票列表
    symbols = loader.get_all_stocks()
    print(f"\n📈 分析的股票: {symbols}")
    
    # 6. 获取投资组合汇总
    portfolio_summary = loader.get_portfolio_summary()
    if portfolio_summary:
        print(f"\n💰 投资组合汇总:")
        print(f"   总价值: ${portfolio_summary.get('total_value', 0):,.2f}")
        print(f"   股票配置: {portfolio_summary.get('stock_allocation', 0):.2f}%")
        print(f"   现金配置: {portfolio_summary.get('cash_allocation', 0):.2f}%")
    
    # 7. 获取宏观分析
    macro_analysis = loader.get_macro_analysis()
    if macro_analysis:
        print(f"\n🌍 宏观分析:")
        print(f"   宏观得分: {macro_analysis.get('macro_score', 0):.2f}/1.00")
        print(f"   环境建议: {macro_analysis.get('recommendation', '无')}")
    
    # 8. 演示单个股票数据获取
    if symbols:
        test_symbol = symbols[0]
        print(f"\n🔍 单个股票数据示例 ({test_symbol}):")
        
        stock_data = loader.get_stock_data(test_symbol)
        if stock_data:
            # 基本信息
            basic_info = stock_data.get('basic_info', {})
            print(f"   当前价格: ${basic_info.get('current_price', 0):.2f}")
            print(f"   涨跌幅: {basic_info.get('price_change_pct', 0):+.2f}%")
            print(f"   RSI: {basic_info.get('rsi', 0):.1f}")
            
            # 市场环境
            market_env = stock_data.get('market_environment', {})
            print(f"   市场环境: {market_env.get('trend', 'unknown')}")
            print(f"   置信度: {market_env.get('confidence', 0):.2f}")
            
            # 策略建议
            strategy = stock_data.get('strategy', {})
            print(f"   推荐策略: {strategy.get('recommended_strategy', 'unknown')}")
            print(f"   信号质量: {strategy.get('signal_quality', 0):.2f}")
            
            # 持仓分析（如果有）
            position = stock_data.get('position_analysis', {})
            if position:
                print(f"   持仓成本: ${position.get('cost_price', 0):.2f}")
                print(f"   持仓数量: {position.get('shares', 0):,.0f}")
                print(f"   盈亏: {position.get('pnl_percent', 0):+.2f}%")
    
    # 9. 演示AI输入格式化
    print(f"\n🤖 AI输入格式化示例:")
    print("-" * 40)
    
    # 只显示前3只股票的数据
    ai_input = loader.format_for_ai_input(symbols[:3])
    print(ai_input)
    
    # 10. 演示股票摘要
    if symbols:
        print(f"\n📋 股票摘要示例:")
        for symbol in symbols[:2]:  # 只显示前2只股票
            summary = loader.get_stock_summary(symbol)
            print(f"\n{summary}")
    
    # 11. 演示数据导出
    print(f"\n💾 数据导出功能:")
    print("   1. 原始JSON数据可直接用于程序处理")
    print("   2. format_for_ai_input() 方法生成AI友好的文本格式")
    print("   3. get_stock_data() 方法获取单个股票的完整数据")
    print("   4. get_stock_summary() 方法获取股票摘要信息")
    
    # 12. 演示高级用法
    print(f"\n🔧 高级用法示例:")
    
    # 获取特定分析类型的数据
    if symbols:
        stock_data = loader.get_stock_data(symbols[0])
        if stock_data:
            # 财务分析数据
            financial = stock_data.get('financial_analysis', {})
            if financial:
                print(f"   📊 {symbols[0]} 财务评分: {financial.get('total_score', 0):.1f}/100")
            
            # 流动性分析数据
            liquidity = stock_data.get('liquidity_analysis', {})
            if liquidity:
                print(f"   💧 {symbols[0]} 流动性评分: {liquidity.get('liquidity_score', 0):.1f}/100")
            
            # 增强分析数据
            enhanced = stock_data.get('enhanced_analysis', {})
            if enhanced:
                print(f"   🚀 {symbols[0]} 增强评分: {enhanced.get('total_score', 0):.3f}/1.0")

def demonstrate_ai_integration():
    """演示AI集成用法"""
    
    print(f"\n🤖 AI集成演示")
    print("=" * 60)
    
    loader = StockDataLoader()
    data = loader.load_data()
    
    if not data:
        print("❌ 数据加载失败")
        return
    
    symbols = loader.get_all_stocks()
    if not symbols:
        print("❌ 没有股票数据")
        return
    
    # 示例1: 为AI提供完整的投资组合分析
    print("📊 示例1: 完整投资组合分析")
    print("-" * 40)
    
    portfolio_analysis = loader.format_for_ai_input()
    print("AI输入提示词:")
    print("请基于以下股票分析数据，提供投资组合优化建议：")
    print(portfolio_analysis[:800] + "..." if len(portfolio_analysis) > 800 else portfolio_analysis)
    
    # 示例2: 为AI提供特定股票分析
    print(f"\n📈 示例2: 特定股票分析")
    print("-" * 40)
    
    if symbols:
        specific_stock = symbols[0]
        stock_data = loader.get_stock_data(specific_stock)
        if stock_data:
            print(f"AI输入提示词:")
            print(f"请分析 {specific_stock} 的投资价值：")
            print(f"当前价格: ${stock_data.get('basic_info', {}).get('current_price', 0):.2f}")
            print(f"市场环境: {stock_data.get('market_environment', {}).get('trend', 'unknown')}")
            print(f"推荐策略: {stock_data.get('strategy', {}).get('recommended_strategy', 'unknown')}")
            
            # 添加持仓信息
            position = stock_data.get('position_analysis', {})
            if position:
                print(f"持仓成本: ${position.get('cost_price', 0):.2f}")
                print(f"盈亏: {position.get('pnl_percent', 0):+.2f}%")
    
    # 示例3: 为AI提供宏观环境分析
    print(f"\n🌍 示例3: 宏观环境分析")
    print("-" * 40)
    
    macro_analysis = loader.get_macro_analysis()
    if macro_analysis:
        print("AI输入提示词:")
        print("请基于以下宏观环境数据，分析当前投资环境：")
        print(f"宏观得分: {macro_analysis.get('macro_score', 0):.2f}/1.00")
        print(f"环境建议: {macro_analysis.get('recommendation', '无')}")
    
    # 示例4: 为AI提供风险分析
    print(f"\n⚠️ 示例4: 风险分析")
    print("-" * 40)
    
    risk_analysis = []
    for symbol in symbols[:3]:  # 分析前3只股票
        stock_data = loader.get_stock_data(symbol)
        if stock_data:
            liquidity = stock_data.get('liquidity_analysis', {})
            if liquidity:
                risk_analysis.append({
                    'symbol': symbol,
                    'liquidity_score': liquidity.get('liquidity_score', 0),
                    'risk_level': liquidity.get('risk_level', 'unknown'),
                    'risk_warning': liquidity.get('risk_warning', '')
                })
    
    if risk_analysis:
        print("AI输入提示词:")
        print("请基于以下流动性风险数据，评估投资组合风险：")
        for risk in risk_analysis:
            print(f"{risk['symbol']}: 流动性评分 {risk['liquidity_score']:.1f}/100, 风险等级 {risk['risk_level']}")

def main():
    """主函数"""
    try:
        # 演示基本用法
        demonstrate_json_data_usage()
        
        # 演示AI集成
        demonstrate_ai_integration()
        
        print(f"\n✅ JSON数据功能演示完成")
        print(f"💡 提示:")
        print(f"   • 每次运行日报生成器都会更新JSON数据文件")
        print(f"   • 可以使用StockDataLoader类轻松访问数据")
        print(f"   • format_for_ai_input()方法生成AI友好的格式")
        print(f"   • 支持获取单个股票、投资组合汇总、宏观分析等")
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
        print(f"❌ 演示失败: {e}")

if __name__ == "__main__":
    main() 