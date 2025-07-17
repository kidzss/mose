#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adobe技术面支撑位分析 - 360争夺战与340支撑位
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_adobe_technical_support():
    """分析Adobe技术面支撑位"""
    
    print("🎨 Adobe技术面支撑位分析")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取Adobe数据
    try:
        adbe = yf.Ticker("ADBE")
        hist = adbe.history(period="1y")
        info = adbe.info
        
        current_price = 364.18
        print(f"📊 当前价格: ${current_price}")
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        return
    
    # 计算技术指标
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    
    # 计算RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算布林带
    hist['BB_middle'] = hist['Close'].rolling(window=20).mean()
    bb_std = hist['Close'].rolling(window=20).std()
    hist['BB_upper'] = hist['BB_middle'] + (bb_std * 2)
    hist['BB_lower'] = hist['BB_middle'] - (bb_std * 2)
    
    # 获取最新数据
    latest = hist.iloc[-1]
    
    print(f"\n📈 技术指标现状:")
    print("-" * 30)
    print(f"• 20日均线: ${latest['MA20']:.2f}")
    print(f"• 50日均线: ${latest['MA50']:.2f}")
    print(f"• 200日均线: ${latest['MA200']:.2f}")
    print(f"• RSI(14): {latest['RSI']:.1f}")
    print(f"• 布林带上轨: ${latest['BB_upper']:.2f}")
    print(f"• 布林带下轨: ${latest['BB_lower']:.2f}")
    
    # 分析360争夺战
    print(f"\n⚔️ 360争夺战分析:")
    print("-" * 25)
    
    # 查看360附近的交易情况
    recent_data = hist.tail(30)  # 最近30天
    price_360_tests = recent_data[recent_data['Close'].between(358, 362)]
    
    if not price_360_tests.empty:
        print(f"• 360附近测试次数: {len(price_360_tests)}次")
        print(f"• 最近测试时间: {price_360_tests.index[-1].strftime('%Y-%m-%d')}")
        print(f"• 360附近成交量: {price_360_tests['Volume'].sum():,.0f}")
    else:
        print("• 360附近暂无明确测试")
    
    # 分析当前价格相对360的位置
    distance_from_360 = ((current_price - 360) / 360) * 100
    if abs(distance_from_360) < 2:
        print(f"• 当前处于360争夺区域 (距离360: {distance_from_360:+.1f}%)")
        print("• 360是关键心理支撑位")
    else:
        print(f"• 距离360: {distance_from_360:+.1f}%")
    
    # 分析340支撑位
    print(f"\n🛡️ 340支撑位分析:")
    print("-" * 25)
    
    # 查找4月份低点
    april_data = hist[hist.index.month == 4]
    if not april_data.empty:
        april_low = april_data['Low'].min()
        april_low_date = april_data[april_data['Low'] == april_low].index[0]
        print(f"• 4月份低点: ${april_low:.2f} ({april_low_date.strftime('%Y-%m-%d')})")
        
        # 计算当前价格相对4月低点的位置
        distance_from_april_low = ((current_price - april_low) / april_low) * 100
        print(f"• 距离4月低点: {distance_from_april_low:+.1f}%")
        
        if distance_from_april_low < 10:
            print("⚠️ 接近4月份低点，需要密切关注支撑")
        else:
            print("✅ 距离4月份低点还有一定空间")
    
    # 查找历史支撑位
    print(f"\n📊 历史支撑位分析:")
    print("-" * 25)
    
    # 计算过去6个月的低点
    six_month_low = hist.tail(180)['Low'].min()
    print(f"• 6个月低点: ${six_month_low:.2f}")
    
    # 计算过去3个月的低点
    three_month_low = hist.tail(90)['Low'].min()
    print(f"• 3个月低点: ${three_month_low:.2f}")
    
    # 计算过去1个月的低点
    one_month_low = hist.tail(30)['Low'].min()
    print(f"• 1个月低点: ${one_month_low:.2f}")
    
    # 分析支撑位强度
    print(f"\n💪 支撑位强度分析:")
    print("-" * 25)
    
    support_levels = [
        (360, "心理支撑位"),
        (350, "整数关口"),
        (340, "4月份低点附近"),
        (330, "技术支撑位"),
        (320, "强支撑位")
    ]
    
    for level, description in support_levels:
        distance = ((current_price - level) / current_price) * 100
        if distance > 0:
            print(f"• ${level:.0f} ({description}): 距离 {distance:+.1f}%")
        else:
            print(f"• ${level:.0f} ({description}): 已跌破 {abs(distance):.1f}%")
    
    # 技术面趋势分析
    print(f"\n📈 技术面趋势分析:")
    print("-" * 25)
    
    # 均线排列
    ma20 = latest['MA20']
    ma50 = latest['MA50']
    ma200 = latest['MA200']
    
    if current_price > ma20 > ma50 > ma200:
        trend = "多头排列 - 强势"
        trend_icon = "🟢"
    elif current_price > ma20 > ma50:
        trend = "短期强势"
        trend_icon = "🟡"
    elif current_price < ma20 < ma50:
        trend = "空头排列 - 弱势"
        trend_icon = "🔴"
    else:
        trend = "震荡整理"
        trend_icon = "🟡"
    
    print(f"{trend_icon} 均线趋势: {trend}")
    
    # RSI分析
    rsi = latest['RSI']
    if rsi < 30:
        rsi_status = "超卖 - 可能反弹"
        rsi_icon = "🟢"
    elif rsi < 40:
        rsi_status = "偏低 - 有反弹机会"
        rsi_icon = "🟡"
    elif rsi > 70:
        rsi_status = "超买 - 注意回调"
        rsi_icon = "🔴"
    else:
        rsi_status = "中性区间"
        rsi_icon = "🟡"
    
    print(f"{rsi_icon} RSI状态: {rsi_status}")
    
    # 布林带位置
    bb_position = (current_price - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
    if bb_position < 0.2:
        bb_status = "接近下轨 - 超卖"
        bb_icon = "🟢"
    elif bb_position < 0.4:
        bb_status = "偏下轨 - 有反弹机会"
        bb_icon = "🟡"
    elif bb_position > 0.8:
        bb_status = "接近上轨 - 超买"
        bb_icon = "🔴"
    else:
        bb_status = "中轨附近"
        bb_icon = "🟡"
    
    print(f"{bb_icon} 布林带位置: {bb_status} ({bb_position:.1%})")
    
    # 下跌风险分析
    print(f"\n⚠️ 下跌风险分析:")
    print("-" * 25)
    
    # 计算潜在下跌空间
    scenarios = [
        (360, "跌破360心理位"),
        (350, "跌破350整数位"),
        (340, "跌破340支撑位"),
        (330, "跌破330技术位"),
        (320, "跌破320强支撑")
    ]
    
    print("📉 不同下跌情景分析:")
    for target_price, scenario in scenarios:
        if current_price > target_price:
            drop_pct = ((current_price - target_price) / current_price) * 100
            print(f"• {scenario}: -{drop_pct:.1f}% (${target_price:.0f})")
    
    # 我的建议
    print(f"\n💡 我的技术面建议:")
    print("-" * 25)
    
    print("🎯 当前状况:")
    if current_price > 360:
        print("• 360争夺战正在进行中")
        print("• 需要观察是否能站稳360")
    else:
        print("• 已跌破360，关注340支撑")
    
    print("\n📊 支撑位重要性排序:")
    print("1. 360 - 心理支撑位 (最重要)")
    print("2. 340 - 4月份低点附近 (强支撑)")
    print("3. 330 - 技术支撑位")
    print("4. 320 - 强支撑位")
    
    print("\n🤔 我的判断:")
    if rsi < 35:
        print("• RSI超卖，短期有反弹机会")
    else:
        print("• RSI中性，等待明确信号")
    
    if current_price > ma20:
        print("• 价格在20日均线上方，短期趋势尚可")
    else:
        print("• 价格跌破20日均线，短期趋势转弱")
    
    print("\n📈 操作建议:")
    print("• 如果跌破360，关注340支撑")
    print("• 340附近是重要支撑位，可以考虑分批买入")
    print("• 设置止损位在320-330区间")
    print("• 等待技术面企稳信号")
    
    # 风险提示
    print(f"\n🚨 风险提示:")
    print("-" * 20)
    print("• 技术面尚未明确企稳")
    print("• 360争夺战结果不确定")
    print("• 340支撑位需要验证")
    print("• 建议分批建仓，控制风险")
    print("• 密切关注成交量变化")

if __name__ == "__main__":
    analyze_adobe_technical_support() 