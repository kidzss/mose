#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试data模块的实时数据API
"""

import sys
import os
import asyncio
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

async def test_realtime_apis():
    """测试实时数据API"""
    print("🔍 测试data模块实时数据API")
    print("="*60)
    
    # 1. 初始化DataInterface
    data_interface = DataInterface()
    print(f"📊 可用数据源: {list(data_interface.data_sources.keys())}")
    
    # 2. 检查哪些数据源支持实时数据
    print(f"\n🌐 实时数据支持情况:")
    for name, source in data_interface.data_sources.items():
        has_realtime = hasattr(source, 'get_realtime_data')
        print(f"   {name}: {'✅ 支持实时数据' if has_realtime else '❌ 仅历史数据'}")
        
        # 显示数据源类型
        print(f"     类型: {type(source).__name__}")
        
        if has_realtime:
            print(f"     🔧 实时数据方法:")
            print(f"       - get_realtime_data()")
            print(f"       - subscribe_updates()")
            print(f"       - unsubscribe_updates()")
    
    # 3. 测试Yahoo Finance实时数据
    if 'yahoo' in data_interface.data_sources:
        yahoo_source = data_interface.get_data_source('yahoo')
        
        if hasattr(yahoo_source, 'get_realtime_data'):
            print(f"\n🎯 测试Yahoo Finance实时数据API:")
            try:
                symbols = ['AMD', 'NVDA', 'TSLA']
                print(f"   正在获取 {symbols} 的实时数据...")
                
                # 调用实时数据API
                realtime_data = await yahoo_source.get_realtime_data(symbols, timeframe='1m')
                
                print(f"   ✅ 成功获取实时数据!")
                for symbol, df in realtime_data.items():
                    if not df.empty:
                        latest = df.iloc[-1]
                        print(f"   📊 {symbol}:")
                        print(f"      💰 最新价格: ${latest['close']:.2f}")
                        print(f"      📅 数据时间: {latest.name}")
                        print(f"      📦 成交量: {int(latest['volume']):,}")
                        print(f"      📈 数据行数: {len(df)}")
                    else:
                        print(f"   ❌ {symbol}: 无数据")
                        
            except Exception as e:
                print(f"   ❌ Yahoo Finance实时数据测试失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 4. 查看DataInterface是否有实时数据的公共接口
    print(f"\n🔍 DataInterface实时数据接口检查:")
    interface_methods = [method for method in dir(data_interface) if not method.startswith('_')]
    realtime_methods = [method for method in interface_methods if 'realtime' in method.lower()]
    
    if realtime_methods:
        print(f"   ✅ 找到实时数据方法: {realtime_methods}")
    else:
        print(f"   ❌ DataInterface没有公开的实时数据方法")
        print(f"   💡 需要直接访问具体的数据源")
    
    # 5. 展示正确的实时数据使用方法
    print(f"\n💡 正确的实时数据使用方法:")
    print(f"""
    # 方法1: 直接使用Yahoo Finance数据源
    data_interface = DataInterface()
    yahoo_source = data_interface.get_data_source('yahoo')
    realtime_data = await yahoo_source.get_realtime_data(['AMD', 'NVDA', 'TSLA'])
    
    # 方法2: 或者创建专门的实时数据源
    from data.data_interface import YahooFinanceRealTimeSource
    realtime_source = YahooFinanceRealTimeSource()
    realtime_data = await realtime_source.get_realtime_data(['AMD', 'NVDA', 'TSLA'])
    """)
    
    print(f"\n⚠️ 重要说明:")
    print(f"   🔸 实时数据API都是异步的(async/await)")
    print(f"   🔸 需要安装yfinance依赖: pip install yfinance")
    print(f"   🔸 Yahoo Finance的'实时'数据实际上有几分钟延迟")
    print(f"   🔸 DataInterface默认使用MySQL数据源，不包含实时数据接口")

def test_sync_realtime_workaround():
    """测试同步方式获取实时数据的变通方法"""
    print(f"\n{'='*60}")
    print(f"🔧 同步方式获取实时数据的变通方法")
    print(f"{'='*60}")
    
    try:
        # 方法: 使用yfinance直接获取
        import yfinance as yf
        
        symbols = ['AMD', 'NVDA', 'TSLA'] 
        print(f"📊 使用yfinance直接获取当前数据:")
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            
            # 获取最新的历史数据
            hist = ticker.history(period='2d')
            if not hist.empty:
                latest = hist.iloc[-1]
                print(f"   💰 {symbol}: ${latest['Close']:.2f}")
                print(f"      📅 数据时间: {latest.name}")
                print(f"      📦 成交量: {int(latest['Volume']):,}")
            
            # 获取实时报价信息
            info = ticker.info
            if 'regularMarketPrice' in info:
                current_price = info['regularMarketPrice']
                print(f"      🕒 实时价格: ${current_price:.2f}")
                
    except ImportError:
        print(f"   ❌ 需要安装yfinance: pip install yfinance")
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")

async def main():
    """主函数"""
    await test_realtime_apis()
    test_sync_realtime_workaround()
    
    print(f"\n⏰ 测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main()) 