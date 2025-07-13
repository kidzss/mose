#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持仓配置刷新工具
用于清除缓存并重新加载最新的配置信息
"""

import json
import os
import sys
from datetime import datetime

def refresh_portfolio_config():
    """刷新持仓配置"""
    print("🔄 开始刷新持仓配置...")
    
    # 检查配置文件是否存在
    config_paths = [
        'portfolio_config.json',
        'config/portfolio_config.json',
        'config/portfolio_config_latest.json'
    ]
    
    config_found = False
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"✅ 找到配置文件: {path}")
                    print(f"📅 最后更新: {config.get('meta', {}).get('last_updated', '未知')}")
                    print(f"💰 总资产: ${config.get('meta', {}).get('total_assets', 0):,.2f}")
                    
                    # 显示持仓信息
                    positions = config.get('positions', {})
                    print(f"📊 持仓股票数量: {len(positions)}")
                    
                    for symbol, position in positions.items():
                        shares = position.get('shares', 0)
                        cost_basis = position.get('cost_basis', 0)
                        weight = position.get('weight', 0)
                        print(f"   {symbol}: {shares}股 @ ${cost_basis:.2f} (占比{weight:.2f}%)")
                    
                    config_found = True
                    break
            except Exception as e:
                print(f"⚠️ 无法读取配置文件 {path}: {e}")
                continue
    
    if not config_found:
        print("❌ 未找到有效的配置文件")
        return False
    
    # 清除可能的缓存文件
    cache_files = [
        '.streamlit/cache',
        '__pycache__',
        '.cache'
    ]
    
    for cache_path in cache_files:
        if os.path.exists(cache_path):
            try:
                import shutil
                shutil.rmtree(cache_path)
                print(f"🗑️ 已清除缓存: {cache_path}")
            except Exception as e:
                print(f"⚠️ 无法清除缓存 {cache_path}: {e}")
    
    print("✅ 配置刷新完成!")
    return True

def verify_config_integrity():
    """验证配置完整性"""
    print("\n🔍 验证配置完整性...")
    
    try:
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必要字段
        required_fields = ['meta', 'positions', 'watchlist']
        missing_fields = []
        
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ 缺少必要字段: {missing_fields}")
            return False
        
        # 检查持仓数据
        positions = config.get('positions', {})
        for symbol, position in positions.items():
            required_position_fields = ['shares', 'cost_basis', 'weight']
            for field in required_position_fields:
                if field not in position:
                    print(f"⚠️ {symbol} 缺少字段: {field}")
        
        print("✅ 配置完整性验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔄 持仓配置刷新工具")
    print("=" * 60)
    
    # 刷新配置
    if refresh_portfolio_config():
        # 验证配置
        verify_config_integrity()
        
        print("\n💡 使用说明:")
        print("1. 如果使用Streamlit应用，请点击侧边栏的'清除缓存'按钮")
        print("2. 或者重启Streamlit应用以加载最新配置")
        print("3. 配置更新后，所有监控系统将使用最新数据")
        
        print(f"\n⏰ 刷新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("❌ 配置刷新失败")

if __name__ == "__main__":
    main() 