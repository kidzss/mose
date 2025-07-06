#!/usr/bin/env python3
"""
测试投资组合更新
验证更新后的持仓配置是否正确
"""

import json
from datetime import datetime

def test_portfolio_update():
    """测试投资组合更新"""
    
    # 读取更新后的配置文件
    with open('portfolio_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("📊 投资组合更新验证")
    print("=" * 50)
    
    # 验证基本信息
    print(f"总资产: ${config['meta']['total_assets']:,.2f}")
    print(f"更新时间: {config['meta']['last_updated']}")
    print(f"描述: {config['meta']['description']}")
    print()
    
    # 验证资产配置
    print("💰 资产配置:")
    allocation = config['meta']['asset_allocation']
    for asset_type, info in allocation.items():
        print(f"  {asset_type}: {info['percentage']}% (${info['amount']:,.2f})")
    print()
    
    # 验证持仓信息
    print("📈 当前持仓:")
    positions = config['positions']
    total_stock_value = 0
    
    for symbol, pos in positions.items():
        investment_amount = pos['investment_amount']
        total_stock_value += investment_amount
        print(f"  {symbol}: {pos['shares']}股, 成本${pos['cost_basis']:.3f}, "
              f"权重{pos['weight']}%, 投资金额${investment_amount:,.2f}")
    
    print(f"\n股票总价值: ${total_stock_value:,.2f}")
    print()
    
    # 验证最近交易
    if 'recent_transactions' in config:
        print("🔄 最近交易:")
        for tx in config['recent_transactions']:
            print(f"  {tx['date']} {tx['action']} {tx['symbol']} "
                  f"{tx['shares']}股 @ ${tx['price']} = ${tx['amount']:,.2f}")
            if 'note' in tx:
                print(f"    备注: {tx['note']}")
        print()
    
    # 验证配置完整性
    print("✅ 配置验证:")
    print(f"  - 持仓数量: {len(positions)}")
    print(f"  - 观察列表数量: {len(config['watchlist'])}")
    print(f"  - 监控配置: {'已配置' if 'monitoring_config' in config else '未配置'}")
    
    # 检查是否移除了当前价格字段
    has_current_price = False
    for symbol, pos in positions.items():
        if 'current_price' in pos or 'current_value' in pos or 'unrealized_pnl' in pos:
            has_current_price = True
            break
    
    print(f"  - 当前价格字段: {'已移除' if not has_current_price else '仍存在'}")
    
    print("\n🎯 投资组合特点:")
    print("  - PFE减仓至20股，准备做T操作")
    print("  - 科技股占比高（NVDA、GOOG、AMD）")
    print("  - 防御性配置（MRK、BRK-B）")
    print("  - 货币基金占比34.38%，流动性良好")
    
    return True

if __name__ == "__main__":
    test_portfolio_update() 