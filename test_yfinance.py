#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.yfinance_client import YFinanceClient

def main():
    print("🧪 测试yfinance客户端...")
    
    client = YFinanceClient()
    
    # 测试获取数据
    data = client.get_batch_financial_data(['AAPL', 'MSFT'], max_symbols=2)
    
    print(f"✅ 获取到 {len(data)} 只股票数据")
    
    for symbol, metrics in data.items():
        print(f"\n📊 {symbol}:")
        print(f"  ROE: {metrics['roe']:.1f}%")
        print(f"  市值: ${metrics['market_cap']:,.0f}")
        print(f"  PE比率: {metrics['pe_ratio']:.2f}")
        print(f"  债务权益比: {metrics['debt_to_equity']:.2f}")

if __name__ == "__main__":
    main() 