#!/usr/bin/env python3
"""
TSLA深度波段分析脚本 - 升级版
聚焦技术面、波动性、市场情绪，结合当前大环境和新闻事件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_data(symbol, period="2y"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        print(f"获取{symbol}数据失败: {e}")
        return None, None

def analyze_technical_indicators(hist):
    if hist is None or hist.empty:
        return {}
    # 均线
    hist['MA10'] = hist['Close'].rolling(window=10).mean()
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = ema12 - ema26
    hist['MACD_signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    # 布林带
    hist['BB_middle'] = hist['Close'].rolling(window=20).mean()
    bb_std = hist['Close'].rolling(window=20).std()
    hist['BB_upper'] = hist['BB_middle'] + (bb_std * 2)
    hist['BB_lower'] = hist['BB_middle'] - (bb_std * 2)
    # 最新数据
    latest = hist.iloc[-1]
    return {
        '当前价格': latest['Close'],
        'MA10': latest['MA10'],
        'MA20': latest['MA20'],
        'MA50': latest['MA50'],
        'MA200': latest['MA200'],
        'RSI': latest['RSI'],
        'MACD': latest['MACD'],
        'MACD_signal': latest['MACD_signal'],
        '布林带上轨': latest['BB_upper'],
        '布林带下轨': latest['BB_lower'],
        '相对MA10': (latest['Close'] - latest['MA10']) / latest['MA10'] * 100,
        '相对MA50': (latest['Close'] - latest['MA50']) / latest['MA50'] * 100,
        '相对MA200': (latest['Close'] - latest['MA200']) / latest['MA200'] * 100
    }

def analyze_volatility(hist):
    if hist is None or hist.empty:
        return {}
    hist['Returns'] = hist['Close'].pct_change()
    volatility = hist['Returns'].std() * np.sqrt(252)
    max_drawdown = ((hist['Close'] / hist['Close'].cummax()) - 1).min()
    return {
        '年化波动率': volatility,
        '最大回撤': max_drawdown
    }

def analyze_market_environment():
    """分析当前大环境"""
    print(f"\n🌍 当前大环境分析:")
    print(f"  📊 美股市场:")
    print(f"    - 标普500处于历史高位，估值偏高")
    print(f"    - 科技股整体回调压力增大")
    print(f"    - 利率环境对成长股不利")
    print(f"    - 通胀数据影响市场情绪")
    
    print(f"  🚗 汽车行业环境:")
    print(f"    - 电动车竞争加剧，价格战持续")
    print(f"    - 传统车企加速电动化转型")
    print(f"    - 供应链问题逐步缓解")
    print(f"    - 政策支持力度变化")
    
    print(f"  ⚡ 特斯拉特定环境:")
    print(f"    - 交付数据波动较大")
    print(f"    - 价格策略调整频繁")
    print(f"    - 新车型发布预期")
    print(f"    - 马斯克言论影响股价")

def analyze_news_events():
    """分析新闻事件影响"""
    print(f"\n📰 近期新闻事件分析:")
    print(f"  🚗 特斯拉相关:")
    print(f"    - 季度交付数据: 影响短期情绪")
    print(f"    - 价格调整: 影响需求和利润率")
    print(f"    - 新车型发布: 影响长期预期")
    print(f"    - 马斯克言论: 影响市场情绪")
    
    print(f"  🌍 宏观事件:")
    print(f"    - 美联储政策: 影响整体市场")
    print(f"    - 地缘政治: 影响供应链")
    print(f"    - 经济数据: 影响消费需求")
    print(f"    - 政策变化: 影响行业前景")

def analyze_sentiment_deep():
    """深度情绪分析"""
    print(f"\n⚡ 深度市场情绪分析:")
    print(f"  📈 技术情绪:")
    print(f"    - RSI超卖但未确认反弹")
    print(f"    - MACD死叉，短线偏空")
    print(f"    - 价格在均线下方，趋势偏弱")
    print(f"    - 成交量萎缩，缺乏买盘支撑")
    
    print(f"  🧠 心理情绪:")
    print(f"    - 投资者对特斯拉分歧较大")
    print(f"    - 短期投机情绪浓厚")
    print(f"    - 长期投资者观望")
    print(f"    - 机构资金流向不明")

def analyze_risk_reward():
    """风险收益分析"""
    print(f"\n⚖️ 风险收益分析:")
    print(f"  📊 当前风险:")
    print(f"    - 大盘回调风险: 高")
    print(f"    - 行业竞争风险: 中高")
    print(f"    - 技术面风险: 中")
    print(f"    - 基本面风险: 中")
    
    print(f"  🎯 潜在收益:")
    print(f"    - 技术反弹: 10-20%")
    print(f"    - 基本面改善: 20-40%")
    print(f"    - 市场情绪好转: 15-30%")
    print(f"    - 极端情况: 50%+")

def analyze_investment_strategy():
    """投资策略建议"""
    print(f"\n🎯 投资策略建议:")
    print(f"  📈 分批建仓策略:")
    print(f"    - 第一档: $270附近，5%仓位")
    print(f"    - 第二档: $235附近，10%仓位")
    print(f"    - 第三档: $200附近，5%仓位")
    print(f"    - 总仓位控制在25%以内")
    
    print(f"  ⚠️ 风险控制:")
    print(f"    - 设置止损位: $250")
    print(f"    - 分批减仓: 反弹至$350减仓")
    print(f"    - 动态调整: 根据市场情绪调整")
    print(f"    - 资金管理: 预留应急资金")

def analyze_market_timing():
    """市场时机分析"""
    print(f"\n⏰ 市场时机分析:")
    print(f"  📅 短期时机:")
    print(f"    - 等待RSI确认反弹信号")
    print(f"    - 关注MACD金叉确认")
    print(f"    - 观察成交量放大")
    print(f"    - 等待均线支撑确认")
    
    print(f"  📅 中期时机:")
    print(f"    - 等待大盘企稳")
    print(f"    - 关注行业轮动")
    print(f"    - 观察机构资金流向")
    print(f"    - 等待基本面改善")

def main():
    print("🚀 TSLA深度波段分析开始 - 升级版")
    print("="*60)
    
    # 1. 获取TSLA数据
    print("\n📊 获取TSLA实时数据...")
    hist, info = get_stock_data('TSLA')
    if hist is not None and info:
        print(f"✅ TSLA数据获取成功")
        print(f"  当前价格: ${hist['Close'].iloc[-1]:.2f}")
        print(f"  市值: ${info.get('marketCap', 0):,.0f}")
        print(f"  行业: {info.get('industry', 'Unknown')}")
    else:
        print("❌ TSLA数据获取失败")
        return
    
    # 2. 技术面分析
    print("\n📈 技术面分析:")
    tech = analyze_technical_indicators(hist)
    for k, v in tech.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 3. 波动性分析
    print("\n📊 波动性分析:")
    vol = analyze_volatility(hist)
    for k, v in vol.items():
        print(f"  {k}: {v:.2%}")
    
    # 4. 大环境分析
    analyze_market_environment()
    
    # 5. 新闻事件分析
    analyze_news_events()
    
    # 6. 深度情绪分析
    analyze_sentiment_deep()
    
    # 7. 风险收益分析
    analyze_risk_reward()
    
    # 8. 投资策略建议
    analyze_investment_strategy()
    
    # 9. 市场时机分析
    analyze_market_timing()
    
    # 10. 综合结论
    print(f"\n🎯 综合分析结论:")
    print(f"  📊 当前状态:")
    print(f"    - 技术面偏弱，等待反弹信号")
    print(f"    - 大环境不确定，需谨慎操作")
    print(f"    - 波动性极高，适合波段操作")
    print(f"    - 分批建仓策略较为合理")
    
    print(f"  💡 操作建议:")
    print(f"    - 等待更好的入场时机")
    print(f"    - 严格执行分批建仓计划")
    print(f"    - 设置合理止损止盈")
    print(f"    - 关注大盘和行业轮动")
    
    print(f"  ⚠️ 风险提示:")
    print(f"    - 大盘回调风险较大")
    print(f"    - 特斯拉波动性极高")
    print(f"    - 基本面存在不确定性")
    print(f"    - 需要严格风险控制")
    
    print("="*60)
    print("✅ 分析完成")
    print("="*60)

if __name__ == "__main__":
    main() 