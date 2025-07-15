#!/usr/bin/env python3
"""
验证投资组合配置更新
"""

import json

def verify_portfolio_config():
    """验证投资组合配置"""
    try:
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("📊 投资组合配置验证")
        print("="*60)
        
        # 检查基本信息
        meta = config.get('meta', {})
        print(f"💰 基本信息:")
        print(f"  总资产: ${meta.get('total_assets', 0):,.2f}")
        print(f"  最后更新: {meta.get('last_updated', 'Unknown')}")
        print(f"  描述: {meta.get('description', 'No description')}")
        
        # 检查资产配置
        asset_allocation = meta.get('asset_allocation', {})
        print(f"\n📈 资产配置:")
        for asset_type, info in asset_allocation.items():
            if isinstance(info, dict):
                percentage = info.get('percentage', 0)
                amount = info.get('amount', 0)
                print(f"  {asset_type}: {percentage:.2f}% (${amount:,.2f})")
        
        # 检查持仓
        positions = config.get('positions', {})
        print(f"\n📋 当前持仓:")
        if positions:
            for symbol, position in positions.items():
                shares = position.get('shares', 0)
                weight = position.get('weight', 0)
                print(f"  {symbol}: {shares}股 ({weight:.2f}%)")
        else:
            print("  无持仓信息")
        
        # 检查是否还有PFE
        if 'PFE' in positions:
            print(f"\n❌ 问题: PFE仍然在持仓中")
            return False
        else:
            print(f"\n✅ PFE已成功移除")
        
        # 检查是否还有TSLA
        if 'TSLA' in positions:
            print(f"\n❌ 问题: TSLA仍然在持仓中")
            return False
        else:
            print(f"\n✅ TSLA已成功移除")
        
        print(f"\n✅ 配置验证完成")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    verify_portfolio_config() 