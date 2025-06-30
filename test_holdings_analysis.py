#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试持仓分析修复
验证盈亏率计算是否正确
"""

import json
from enhanced_ai_data_integrator import EnhancedAIDataIntegrator

def test_holdings_analysis():
    """测试持仓分析"""
    print("🔍 测试持仓分析修复...")
    
    # 初始化数据整合器
    integrator = EnhancedAIDataIntegrator()
    
    # 测试股票列表（从portfolio_config.json中获取）
    test_symbols = ['NVDA', 'AMD', 'GOOG', 'PFE', 'MRK', 'BRK-B']
    
    for symbol in test_symbols:
        print(f"\n📊 测试 {symbol} 的持仓分析...")
        
        try:
            # 获取持仓分析数据
            holdings_data = integrator._get_holdings_analysis(symbol)
            
            if holdings_data:
                print(f"✅ {symbol} 持仓分析成功")
                print(f"  持股数量: {holdings_data.get('shares', 0)} 股")
                print(f"  成本价格: ${holdings_data.get('cost_basis', 0):.2f}")
                print(f"  当前价格: ${holdings_data.get('current_price', 0):.2f}")
                print(f"  投资金额: ${holdings_data.get('investment_amount', 0):.2f}")
                print(f"  未实现盈亏: ${holdings_data.get('unrealized_pnl', 0):+.2f}")
                print(f"  盈亏率: {holdings_data.get('pnl_pct', 0):+.2f}%")
                print(f"  持仓状态: {holdings_data.get('position_status', '未知')}")
                print(f"  行业板块: {holdings_data.get('sector', 'Unknown')}")
            else:
                print(f"❌ {symbol} 无持仓数据")
                
        except Exception as e:
            print(f"❌ 测试 {symbol} 时发生错误: {e}")
    
    print("\n🎯 持仓分析测试完成!")

def verify_portfolio_data():
    """验证portfolio_config.json中的数据"""
    print("\n📋 验证portfolio_config.json数据...")
    
    try:
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            portfolio_config = json.load(f)
        
        positions = portfolio_config.get('positions', {})
        
        for symbol, position in positions.items():
            if symbol in ['NVDA', 'AMD', 'GOOG', 'PFE', 'MRK', 'BRK-B']:
                print(f"\n📊 {symbol}:")
                print(f"  成本价: ${position.get('cost_basis', 0):.2f}")
                print(f"  当前价: ${position.get('current_price', 0):.2f}")
                print(f"  投资金额: ${position.get('investment_amount', 0):.2f}")
                print(f"  未实现盈亏: ${position.get('unrealized_pnl', 0):+.2f}")
                
                # 手动计算盈亏率
                investment_amount = position.get('investment_amount', 0)
                unrealized_pnl = position.get('unrealized_pnl', 0)
                if investment_amount > 0:
                    manual_pnl_pct = (unrealized_pnl / investment_amount) * 100
                    print(f"  手动计算盈亏率: {manual_pnl_pct:+.2f}%")
                    
                    # 判断状态
                    if manual_pnl_pct > 15:
                        status = '大幅盈利'
                    elif manual_pnl_pct > 5:
                        status = '盈利'
                    elif manual_pnl_pct > -5:
                        status = '小幅亏损'
                    else:
                        status = '大幅亏损'
                    print(f"  状态判断: {status}")
                
    except Exception as e:
        print(f"❌ 验证portfolio_config.json失败: {e}")

def main():
    """主测试函数"""
    print("🧪 持仓分析修复验证")
    print("=" * 50)
    
    # 验证原始数据
    verify_portfolio_data()
    
    # 测试持仓分析
    test_holdings_analysis()
    
    print("\n🎉 所有测试完成!")

if __name__ == "__main__":
    main() 