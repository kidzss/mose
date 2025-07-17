#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
我的Adobe买入价格分析 - 基于收益比超过20%的策略
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_my_adobe_buy_price():
    """我的Adobe买入价格分析"""
    
    print("🎨 我的Adobe买入价格分析")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 当前市场数据
    current_data = {
        'current_price': 364.18,
        'rsi': 35.5,
        'signal_strength': '6/7',
        'market_trend': '强势上升趋势',
        'strategy': 'trend_following',
        'signal_quality': 0.46,
        'volume': 2548800
    }
    
    # 分析师目标价
    analyst_target = 491.32  # 上涨空间25.4%
    
    # 我的分析师指引目标
    my_targets = {
        '2025_11_low': 407,
        '2025_11_high': 611,
        '2025_11_mid': 509,
        '2026_11_low': 459,
        '2026_11_high': 688,
        '2026_11_mid': 574
    }
    
    # 支撑位信息
    support_levels = {
        'small': (397, 412),
        'strong_1': (365, 388),
        'strong_2': (320, 353),
        'strong_3': (270, 310),
        'strong_4': (225, 262)
    }
    
    print(f"📊 当前状况:")
    print("-" * 25)
    print(f"• 当前价格: ${current_data['current_price']}")
    print(f"• RSI: {current_data['rsi']} (超卖)")
    print(f"• 信号强度: {current_data['signal_strength']}")
    print(f"• 市场趋势: {current_data['market_trend']}")
    print(f"• 分析师目标: ${analyst_target} (+25.4%)")
    
    print(f"\n🎯 我的目标价分析:")
    print("-" * 30)
    print(f"• 2025年11月: ${my_targets['2025_11_low']}-${my_targets['2025_11_high']}")
    print(f"• 2025年中位数: ${my_targets['2025_11_mid']}")
    print(f"• 2026年11月: ${my_targets['2026_11_low']}-${my_targets['2026_11_high']}")
    print(f"• 2026年中位数: ${my_targets['2026_11_mid']}")
    
    print(f"\n🛡️ 支撑位分析:")
    print("-" * 25)
    for level_name, (low, high) in support_levels.items():
        distance_low = ((current_data['current_price'] - low) / current_data['current_price']) * 100
        distance_high = ((current_data['current_price'] - high) / current_data['current_price']) * 100
        print(f"• {level_name}: ${low}-${high} (距离: {distance_low:+.1f}% 到 {distance_high:+.1f}%)")
    
    # 计算不同买入价格的收益比
    print(f"\n💰 收益比分析 (收益/风险 > 20%):")
    print("-" * 45)
    
    buy_scenarios = []
    
    # 不同买入价格情景
    buy_prices = [360, 350, 340, 330, 320, 310, 300]
    
    for buy_price in buy_prices:
        if buy_price >= current_data['current_price']:
            continue
            
        # 计算到最近支撑位的风险
        risk_to_support = ((current_data['current_price'] - buy_price) / current_data['current_price']) * 100
        
        # 计算不同目标的收益
        scenarios = [
            ("分析师目标", analyst_target),
            ("2025年低端", my_targets['2025_11_low']),
            ("2025年中位数", my_targets['2025_11_mid']),
            ("2025年高端", my_targets['2025_11_high']),
            ("2026年低端", my_targets['2026_11_low']),
            ("2026年中位数", my_targets['2026_11_mid']),
            ("2026年高端", my_targets['2026_11_high'])
        ]
        
        for scenario_name, target_price in scenarios:
            if target_price > buy_price:
                potential_return = ((target_price - buy_price) / buy_price) * 100
                risk_reward_ratio = potential_return / risk_to_support if risk_to_support > 0 else float('inf')
                
                if potential_return > 20:  # 收益超过20%
                    buy_scenarios.append({
                        'buy_price': buy_price,
                        'scenario': scenario_name,
                        'target_price': target_price,
                        'potential_return': potential_return,
                        'risk_to_support': risk_to_support,
                        'risk_reward_ratio': risk_reward_ratio
                    })
    
    # 按收益比排序
    buy_scenarios.sort(key=lambda x: x['risk_reward_ratio'], reverse=True)
    
    print("买入价格 | 目标 | 潜在收益 | 风险 | 收益比")
    print("-" * 60)
    
    for scenario in buy_scenarios[:15]:  # 显示前15个最佳机会
        print(f"${scenario['buy_price']:>6.0f} | {scenario['scenario']:<12} | {scenario['potential_return']:>6.1f}% | {scenario['risk_to_support']:>4.1f}% | {scenario['risk_reward_ratio']:>5.2f}")
    
    # 我的买入价格建议
    print(f"\n💡 我的买入价格建议:")
    print("-" * 30)
    
    # 找出最佳买入价格
    best_scenarios = {}
    for scenario in buy_scenarios:
        buy_price = scenario['buy_price']
        if buy_price not in best_scenarios or scenario['risk_reward_ratio'] > best_scenarios[buy_price]['risk_reward_ratio']:
            best_scenarios[buy_price] = scenario
    
    # 按买入价格排序
    sorted_buy_prices = sorted(best_scenarios.keys())
    
    print("🎯 我的买入价格区间:")
    print("1. 第一档买入: $360-365")
    print("   • 理由: 接近当前价格，风险较小")
    print("   • 目标: 分析师目标$491 (+25%)")
    print("   • 收益比: 优秀")
    
    print("\n2. 第二档买入: $340-350")
    print("   • 理由: 接近4月份低点，技术支撑")
    print("   • 目标: 2025年中位数$509 (+45%)")
    print("   • 收益比: 极佳")
    
    print("\n3. 第三档买入: $320-330")
    print("   • 理由: 强支撑位，安全边际高")
    print("   • 目标: 2025年高端$611 (+85%)")
    print("   • 收益比: 最佳")
    
    print("\n4. 第四档买入: $300-310")
    print("   • 理由: 极强支撑位，风险极低")
    print("   • 目标: 2026年高端$688 (+125%)")
    print("   • 收益比: 最优")
    
    # 我的真实买入价格
    print(f"\n🎯 我的真实买入价格:")
    print("-" * 25)
    
    print("基于收益比超过20%的要求，我的买入价格是:")
    print("🟢 主要买入价格: $340-350")
    print("   • 理由: 收益比最佳，风险可控")
    print("   • 预期收益: 45-85%")
    print("   • 风险: 到当前价格约4-7%")
    
    print("\n🟡 保守买入价格: $360-365")
    print("   • 理由: 接近当前价格，快速建仓")
    print("   • 预期收益: 25-35%")
    print("   • 风险: 到当前价格约0-1%")
    
    print("\n🔴 激进买入价格: $320-330")
    print("   • 理由: 强支撑位，最大收益")
    print("   • 预期收益: 85-125%")
    print("   • 风险: 到当前价格约9-12%")
    
    # 我的最终建议
    print(f"\n💭 我的最终建议:")
    print("-" * 25)
    
    print("基于所有信息分析，我的买入价格是:")
    print("🎯 主要买入价格: $340-350")
    
    print("\n理由:")
    print("✅ 收益比超过20% (45-85%收益 vs 4-7%风险)")
    print("✅ 接近4月份低点，技术支撑强")
    print("✅ 分析师长期看好，基本面优秀")
    print("✅ 当前RSI超卖，有反弹机会")
    print("✅ 风险可控，安全边际充足")
    
    print("\n操作策略:")
    print("• 如果价格回调到$340-350，果断买入")
    print("• 如果价格在$360附近企稳，也可以买入")
    print("• 分批建仓，控制风险")
    print("• 设置止损位在$320-330")
    
    print("\n目标设置:")
    print("• 短期目标: $407 (2025年低端)")
    print("• 中期目标: $509 (2025年中位数)")
    print("• 长期目标: $611 (2025年高端)")
    print("• 理想目标: $688 (2026年高端)")

if __name__ == "__main__":
    analyze_my_adobe_buy_price() 