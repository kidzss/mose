#!/usr/bin/env python3
import yfinance as yf

def analyze_nvda_position():
    ticker = yf.Ticker('NVDA')
    hist = ticker.history(period='1mo')
    
    current_price = hist['Close'].iloc[-1]
    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]
    
    # 计算RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    
    print("🎯 NVDA 仓位管理分析")
    print("=" * 50)
    print(f"当前价格: ${current_price:.2f}")
    print(f"MA20: ${ma20:.2f}")
    print(f"MA50: ${ma50:.2f}")
    print(f"RSI: {rsi:.1f}")
    print()
    
    print("📊 仓位管理建议:")
    print("当前仓位: 18.29%")
    print("建议目标: 25% (需加仓6.7%)")
    print("最大目标: 30% (需谨慎考虑)")
    print()
    
    # 计算关键价位
    ma20_support = ma20
    pullback_5pct = current_price * 0.95
    pullback_8pct = current_price * 0.92
    
    print("💡 具体加仓策略和价位:")
    print("-" * 30)
    print("1. 【推荐】稳健分批策略:")
    print(f"   • 第一批加仓: ${pullback_5pct:.2f} - ${current_price-2:.2f} (加仓3%)")
    print(f"   • 第二批加仓: ${ma20_support:.2f} - ${pullback_8pct:.2f} (加仓3-4%)")
    print(f"   • 目标仓位: 25%")
    print()
    
    print("2. 激进追涨策略 (不推荐):")
    print(f"   • 突破确认: ${current_price+1:.2f}以上 (风险较高)")
    print()
    
    print("3. 保守等待策略:")
    print(f"   • 深度回调: ${ma20_support-3:.2f}以下 (可能等不到)")
    print()
    
    # RSI判断
    if rsi > 70:
        print("⚠️ 重要提醒: RSI={:.1f} 超买状态".format(rsi))
        print("建议: 等待回调再加仓，避免追高")
    elif rsi > 60:
        print("🟡 RSI={:.1f} 偏高，可小幅加仓".format(rsi))
    else:
        print("✅ RSI={:.1f} 健康区间，适合加仓".format(rsi))
    
    print()
    print("🎯 最终建议:")
    print("=" * 20)
    
    if current_price > ma20 * 1.05 and rsi > 70:
        print("🔴 当前不建议加仓")
        print(f"理由: 价格过高(比MA20高{((current_price/ma20-1)*100):.1f}%)且RSI超买")
        print(f"建议: 等待回调至${ma20:.2f}附近再考虑加仓")
    elif current_price > ma20 * 1.02:
        print("🟡 可以小幅加仓2-3%")
        print(f"建议价位: ${pullback_5pct:.2f} - ${current_price-1:.2f}")
        print("策略: 分批建仓，不要一次性投入")
    else:
        print("🟢 适合加仓")
        print("可以按计划加仓至25%")
    
    print()
    print("📋 操作建议:")
    print("• 总仓位控制在25%以内")
    print("• 分2-3次加仓，每次2-3%")
    print("• 设置止损位在$140以下")
    print("• 关注AI芯片行业动态")

if __name__ == "__main__":
    analyze_nvda_position() 