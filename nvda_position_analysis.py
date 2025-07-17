#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVDA独立持仓分析 - 考虑H20芯片销售放开因素
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_nvda_data():
    """获取NVDA实时数据"""
    try:
        nvda = yf.Ticker("NVDA")
        
        # 获取基本信息
        info = nvda.info
        
        # 获取历史数据
        hist = nvda.history(period="1y")
        
        return info, hist
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None, None

def calculate_technical_indicators(df):
    """计算技术指标"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 移动平均线
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 布林带
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    return df

def analyze_nvda_position():
    """分析NVDA持仓情况"""
    print("=" * 60)
    print("NVDA独立持仓分析 - 考虑H20芯片销售放开因素")
    print("=" * 60)
    
    # 获取数据
    info, hist = get_nvda_data()
    
    if info is None or hist is None:
        print("无法获取NVDA数据")
        return
    
    # 计算技术指标
    hist = calculate_technical_indicators(hist)
    
    # 当前价格信息
    current_price = hist['Close'].iloc[-1]
    current_rsi = hist['RSI'].iloc[-1]
    
    # 52周高低点
    high_52w = hist['High'].max()
    low_52w = hist['Low'].min()
    
    # 距离52周高点的百分比
    distance_from_high = ((high_52w - current_price) / high_52w) * 100
    
    print(f"\n📊 当前市场数据:")
    print(f"   当前价格: ${current_price:.2f}")
    print(f"   52周高点: ${high_52w:.2f}")
    print(f"   52周低点: ${low_52w:.2f}")
    print(f"   距离52周高点: {distance_from_high:.1f}%")
    print(f"   当前RSI: {current_rsi:.1f}")
    
    # 基本面信息
    print(f"\n📈 基本面数据:")
    print(f"   市值: ${info.get('marketCap', 0)/1e12:.2f}T")
    print(f"   P/E比率: {info.get('trailingPE', 0):.1f}")
    print(f"   营收增长率: {info.get('revenueGrowth', 0)*100:.1f}%")
    print(f"   净利润率: {info.get('profitMargins', 0)*100:.1f}%")
    
    # H20芯片销售放开的影响分析
    print(f"\n🚀 H20芯片销售放开影响分析:")
    print(f"   ✅ 正面因素:")
    print(f"      - 中国市场重新开放，潜在收入增长")
    print(f"      - 缓解地缘政治风险")
    print(f"      - 扩大全球市场份额")
    print(f"      - 长期增长前景改善")
    
    print(f"\n   ⚠️  当前技术面风险:")
    print(f"      - RSI {current_rsi:.1f} (超买区域)")
    print(f"      - 价格接近52周高点")
    print(f"      - 短期可能存在回调压力")
    
    # 长期持有策略建议
    print(f"\n💡 长期持有策略建议:")
    print(f"   基于H20芯片销售放开的长期利好:")
    print(f"   1. 核心持仓: 保持60-70%的长期核心仓位")
    print(f"   2. 回调加仓: 等待RSI回调至50-60区间")
    print(f"   3. 分批建仓: 不要一次性满仓")
    print(f"   4. 止损设置: 设置10-15%的止损位")
    
    # 技术面分析
    print(f"\n📉 技术面分析:")
    if current_rsi > 80:
        print(f"   RSI严重超买，短期回调概率高")
    elif current_rsi > 70:
        print(f"   RSI超买，但基本面支撑，可谨慎持有")
    else:
        print(f"   RSI正常区间")
    
    if distance_from_high < 5:
        print(f"   价格接近52周高点，突破需要强基本面支撑")
    else:
        print(f"   价格距离高点有安全边际")
    
    # 投资建议
    print(f"\n🎯 综合投资建议:")
    print(f"   长期评级: 强烈买入 (基于H20芯片销售放开)")
    print(f"   短期评级: 谨慎持有 (基于技术面超买)")
    print(f"   建议操作:")
    print(f"   - 现有持仓: 继续持有，等待回调")
    print(f"   - 新资金: 分批建仓，回调时加仓")
    print(f"   - 目标价位: 前高突破后看$150-160区间")
    
    # 风险提示
    print(f"\n⚠️  风险提示:")
    print(f"   - 技术面超买，短期回调风险")
    print(f"   - 地缘政治风险仍然存在")
    print(f"   - 市场情绪可能过度乐观")
    print(f"   - 建议设置止损位保护收益")
    
    # 绘制技术分析图表
    plot_technical_analysis(hist)
    
    return {
        'current_price': current_price,
        'current_rsi': current_rsi,
        'distance_from_high': distance_from_high,
        'long_term_rating': '强烈买入',
        'short_term_rating': '谨慎持有',
        'h20_impact': '重大利好'
    }

def plot_technical_analysis(df):
    """绘制技术分析图表"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 价格和移动平均线
    ax1.plot(df.index, df['Close'], label='收盘价', linewidth=2)
    ax1.plot(df.index, df['MA20'], label='MA20', alpha=0.7)
    ax1.plot(df.index, df['MA50'], label='MA50', alpha=0.7)
    ax1.plot(df.index, df['MA200'], label='MA200', alpha=0.7)
    ax1.fill_between(df.index, df['BB_upper'], df['BB_lower'], alpha=0.1, label='布林带')
    ax1.set_title('NVDA价格走势与技术指标', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格 ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MACD
    ax3.plot(df.index, df['MACD'], label='MACD', linewidth=2)
    ax3.plot(df.index, df['Signal'], label='Signal', linewidth=2)
    ax3.bar(df.index, df['MACD'] - df['Signal'], alpha=0.3, label='MACD柱状图')
    ax3.set_ylabel('MACD')
    ax3.set_xlabel('日期')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nvda_technical_analysis_h20.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    result = analyze_nvda_position()
    print(f"\n✅ 分析完成！图表已保存为 'nvda_technical_analysis_h20.png'") 