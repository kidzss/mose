#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的实时监控测试
"""

import sys
import os
import asyncio
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

async def simple_realtime_test():
    """简单的实时数据测试"""
    print("🚀 简单实时监控测试 (2分钟)")
    print("="*50)
    
    # 初始化数据接口
    data_interface = DataInterface()
    yahoo_source = data_interface.get_data_source('yahoo')
    
    symbols = ['AMD', 'NVDA', 'TSLA']
    update_count = 0
    
    print(f"📊 监控股票: {symbols}")
    print(f"⏰ 每30秒更新一次，总共4次更新")
    
    try:
        for i in range(4):  # 4次更新，每次30秒，总共2分钟
            update_count += 1
            current_time = datetime.now()
            
            print(f"\n🔄 第{update_count}次更新 - {current_time.strftime('%H:%M:%S')}")
            print("-" * 40)
            
            # 获取实时数据
            realtime_data = await yahoo_source.get_realtime_data(symbols, timeframe='1m')
            
            for symbol, df in realtime_data.items():
                if not df.empty:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    
                    current_price = float(latest['close'])
                    prev_price = float(prev['close'])
                    change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                    
                    print(f"📊 {symbol}:")
                    print(f"   💰 当前价格: ${current_price:.2f}")
                    print(f"   📈 变动: {change_pct:+.2f}%")
                    print(f"   📦 成交量: {int(latest['volume']):,}")
                    print(f"   📅 数据时间: {latest.name}")
                    
                    # 简单的信号判断
                    if abs(change_pct) > 1.0:
                        signal = "🟢 上涨" if change_pct > 0 else "🔴 下跌"
                        print(f"   ⚠️ 信号: {signal} (变动超过1%)")
                else:
                    print(f"❌ {symbol}: 无数据")
            
            # 如果不是最后一次，等待30秒
            if i < 3:
                print(f"\n⏳ 等待30秒后进行下一次更新...")
                await asyncio.sleep(30)
    
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 测试完成! 总共更新了{update_count}次")
    print(f"⏰ 结束时间: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(simple_realtime_test()) 