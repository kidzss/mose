#!/usr/bin/env python3
"""
CRM 实际交易计划 - 基于多重验证分析
"""

import yfinance as yf
from datetime import datetime

def get_crm_trading_plan():
    """获取CRM实时交易计划"""
    print("🎯 CRM (Salesforce) 实时交易计划")
    print("="*50)
    
    # 获取实时价格
    ticker = yf.Ticker('CRM')
    hist = ticker.history(period='2d')
    current_price = hist['Close'].iloc[-1]
    
    print(f"📊 当前价格: ${current_price:.2f}")
    print(f"📅 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 基于分析的价格策略
    ideal_buy_low = 260.00
    ideal_buy_high = 265.00
    acceptable_buy_high = 273.00
    stop_loss = 255.00
    take_profit_1 = 290.00
    take_profit_2 = 296.00
    
    print(f"\n💰 价格策略:")
    print(f"   🎯 理想买入区间: ${ideal_buy_low:.2f} - ${ideal_buy_high:.2f}")
    print(f"   ✅ 可接受区间: ${ideal_buy_low:.2f} - ${acceptable_buy_high:.2f}")
    print(f"   ⚠️  止损价位: ${stop_loss:.2f}")
    print(f"   🎯 第一止盈: ${take_profit_1:.2f} (+{((take_profit_1/current_price-1)*100):.1f}%)")
    print(f"   🎯 第二止盈: ${take_profit_2:.2f} (+{((take_profit_2/current_price-1)*100):.1f}%)")
    
    # 当前价格评估
    if current_price <= ideal_buy_high:
        price_status = "🟢 理想买入价格"
        action = "立即买入"
    elif current_price <= acceptable_buy_high:
        price_status = "🟡 可接受买入价格"
        action = "谨慎买入"
    else:
        price_status = "🔴 价格偏高"
        action = "等待回调"
    
    print(f"\n📈 当前价格评估: {price_status}")
    print(f"   建议操作: {action}")
    
    # 试水仓位建议
    print(f"\n📊 试水仓位建议:")
    
    if current_price <= ideal_buy_high:
        # 理想价格区间 - 可以较积极
        positions = [
            ("第1批", "10-15%", "立即买入"),
            ("第2批", "5-10%", "跌破$260时加仓"),
            ("第3批", "5%", "突破$275后追加")
        ]
        total_position = "20-30%"
    elif current_price <= acceptable_buy_high:
        # 可接受价格区间 - 较保守
        positions = [
            ("第1批", "5-10%", "当前价位小量试水"),
            ("第2批", "5-10%", "跌至$262以下加仓"),
            ("第3批", "5%", "确认突破后追加")
        ]
        total_position = "15-25%"
    else:
        # 价格偏高 - 等待
        positions = [
            ("观望", "0%", "等待回调至$270以下")
        ]
        total_position = "0%"
    
    print(f"   总仓位规划: {total_position}")
    for batch, size, condition in positions:
        print(f"   {batch}: {size} - {condition}")
    
    # 风险管理
    print(f"\n⚠️  风险管理:")
    print(f"   止损纪律: 跌破${stop_loss:.2f}必须止损")
    print(f"   仓位控制: 单只股票不超过30%")
    print(f"   分批策略: 分2-3次建仓，避免一次性all-in")
    
    # 监控要点
    print(f"\n👀 监控要点:")
    print(f"   技术位: 关注$275阻力位和$255支撑位")
    print(f"   信号变化: 每日监控TDI策略信号")
    print(f"   大盘走势: 关注科技股整体表现")
    
    return {
        'current_price': current_price,
        'price_status': price_status,
        'action': action,
        'ideal_range': [ideal_buy_low, ideal_buy_high],
        'stop_loss': stop_loss,
        'targets': [take_profit_1, take_profit_2]
    }

if __name__ == "__main__":
    get_crm_trading_plan() 