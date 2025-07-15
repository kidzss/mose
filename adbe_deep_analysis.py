#!/usr/bin/env python3
"""
ADBE深度分析脚本
包含实时数据获取、同行业对比、历史表现分析和市场环境研究
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import seaborn as sns
# from monitor.enhanced_stock_analyzer import EnhancedStockAnalyzer
# from data.data_interface import DataInterface

def get_stock_data(symbol, period="2y"):
    """获取股票数据"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        print(f"获取{symbol}数据失败: {e}")
        return None, None

def calculate_valuation_metrics(info):
    """计算估值指标"""
    if not info:
        return {}
    
    metrics = {}
    
    # 基础估值指标
    if info.get('trailingPE'):
        metrics['PE'] = info['trailingPE']
    if info.get('priceToBook'):
        metrics['PB'] = info['priceToBook']
    if info.get('pegRatio'):
        metrics['PEG'] = info['pegRatio']
    
    # 财务指标
    if info.get('profitMargins'):
        metrics['净利润率'] = info['profitMargins'] * 100
    if info.get('returnOnEquity'):
        metrics['ROE'] = info['returnOnEquity'] * 100
    if info.get('revenueGrowth'):
        metrics['营收增长率'] = info['revenueGrowth'] * 100
    
    return metrics

def analyze_technical_indicators(hist):
    """分析技术指标"""
    if hist is None or hist.empty:
        return {}
    
    # 计算移动平均线
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
    
    return {
        '当前价格': latest['Close'],
        'MA20': latest['MA20'],
        'MA50': latest['MA50'],
        'MA200': latest['MA200'],
        'RSI': latest['RSI'],
        '布林带上轨': latest['BB_upper'],
        '布林带下轨': latest['BB_lower'],
        '相对MA20位置': (latest['Close'] - latest['MA20']) / latest['MA20'] * 100,
        '相对MA50位置': (latest['Close'] - latest['MA50']) / latest['MA50'] * 100,
        '相对MA200位置': (latest['Close'] - latest['MA200']) / latest['MA200'] * 100
    }

def compare_with_peers(main_symbol, peer_symbols):
    """与同行业公司对比"""
    print(f"\n{'='*60}")
    print("📊 同行业公司对比分析")
    print(f"{'='*60}")
    
    comparison_data = {}
    
    # 分析主要股票
    print(f"\n🎯 分析目标股票: {main_symbol}")
    hist, info = get_stock_data(main_symbol)
    if hist is not None and info:
        comparison_data[main_symbol] = {
            '估值': calculate_valuation_metrics(info),
            '技术': analyze_technical_indicators(hist),
            '基本信息': {
                '市值': info.get('marketCap', 0),
                '行业': info.get('industry', 'Unknown'),
                '52周最高': info.get('fiftyTwoWeekHigh', 0),
                '52周最低': info.get('fiftyTwoWeekLow', 0)
            }
        }
    
    # 分析同行业公司
    for symbol in peer_symbols:
        print(f"\n📈 分析同行业公司: {symbol}")
        hist, info = get_stock_data(symbol)
        if hist is not None and info:
            comparison_data[symbol] = {
                '估值': calculate_valuation_metrics(info),
                '技术': analyze_technical_indicators(hist),
                '基本信息': {
                    '市值': info.get('marketCap', 0),
                    '行业': info.get('industry', 'Unknown'),
                    '52周最高': info.get('fiftyTwoWeekHigh', 0),
                    '52周最低': info.get('fiftyTwoWeekLow', 0)
                }
            }
    
    return comparison_data

def analyze_historical_performance(symbol, period="5y"):
    """分析历史表现"""
    print(f"\n{'='*60}")
    print(f"📈 {symbol}历史表现分析")
    print(f"{'='*60}")
    
    hist, info = get_stock_data(symbol, period)
    if hist is None or hist.empty:
        print("无法获取历史数据")
        return
    
    # 计算历史表现指标
    hist['Returns'] = hist['Close'].pct_change()
    hist['Cumulative_Returns'] = (1 + hist['Returns']).cumprod()
    
    # 计算年化收益率
    total_days = (hist.index[-1] - hist.index[0]).days
    total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (365 / total_days) - 1
    
    # 计算波动率
    volatility = hist['Returns'].std() * np.sqrt(252)
    
    # 计算夏普比率（假设无风险利率为2%）
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / volatility
    
    # 计算最大回撤
    hist['Peak'] = hist['Close'].expanding().max()
    hist['Drawdown'] = (hist['Close'] - hist['Peak']) / hist['Peak']
    max_drawdown = hist['Drawdown'].min()
    
    print(f"📊 历史表现指标:")
    print(f"  总收益率: {total_return:.2%}")
    print(f"  年化收益率: {annual_return:.2%}")
    print(f"  年化波动率: {volatility:.2%}")
    print(f"  夏普比率: {sharpe_ratio:.2f}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    
    # 分析估值历史
    if info:
        current_pe = info.get('trailingPE', 0)
        current_pb = info.get('priceToBook', 0)
        
        print(f"\n💰 当前估值水平:")
        print(f"  当前PE: {current_pe:.2f}")
        print(f"  当前PB: {current_pb:.2f}")
        
        # 简单的估值判断
        if current_pe > 30:
            print(f"  ⚠️ PE偏高，可能存在估值风险")
        elif current_pe < 15:
            print(f"  ✅ PE偏低，可能存在投资机会")
        else:
            print(f"  📊 PE处于合理区间")
    
    return {
        '年化收益率': annual_return,
        '波动率': volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '当前PE': info.get('trailingPE', 0) if info else 0,
        '当前PB': info.get('priceToBook', 0) if info else 0
    }

def analyze_market_environment():
    """分析市场环境"""
    print(f"\n{'='*60}")
    print("🌍 市场环境分析")
    print(f"{'='*60}")
    
    # 获取主要指数数据
    indices = {
        'SPY': '标普500ETF',
        'QQQ': '纳斯达克100ETF',
        'XLK': '科技股ETF',
        'XLF': '金融股ETF'
    }
    
    market_data = {}
    
    for symbol, name in indices.items():
        print(f"\n📊 分析{name}({symbol}):")
        hist, info = get_stock_data(symbol, "6m")
        if hist is not None and not hist.empty:
            # 计算近期表现
            recent_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            recent_volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            
            market_data[symbol] = {
                '近期收益率': recent_return,
                '近期波动率': recent_volatility,
                '当前价格': hist['Close'].iloc[-1]
            }
            
            print(f"  近期收益率: {recent_return:.2%}")
            print(f"  近期波动率: {recent_volatility:.2%}")
            
            # 市场环境判断
            if recent_return > 0.1:
                print(f"  🟢 强势上涨")
            elif recent_return > 0:
                print(f"  🟡 温和上涨")
            elif recent_return > -0.1:
                print(f"  🟠 温和下跌")
            else:
                print(f"  🔴 大幅下跌")
    
    return market_data

def main():
    """主函数"""
    print("🚀 ADBE深度分析开始")
    print("="*60)
    
    # 1. 获取ADBE实时数据
    print("\n📊 获取ADBE实时数据...")
    adbe_hist, adbe_info = get_stock_data('ADBE')
    
    if adbe_hist is not None and adbe_info:
        print(f"✅ ADBE数据获取成功")
        print(f"  当前价格: ${adbe_hist['Close'].iloc[-1]:.2f}")
        print(f"  市值: ${adbe_info.get('marketCap', 0):,.0f}")
        print(f"  行业: {adbe_info.get('industry', 'Unknown')}")
    else:
        print("❌ ADBE数据获取失败")
        return
    
    # 2. 同行业对比分析
    peer_symbols = ['MSFT', 'CRM', 'WDAY', 'ORCL', 'INTU']
    comparison_data = compare_with_peers('ADBE', peer_symbols)
    
    # 3. 历史表现分析
    historical_data = analyze_historical_performance('ADBE')
    
    # 4. 市场环境分析
    market_data = analyze_market_environment()
    
    # 5. 综合分析结论
    print(f"\n{'='*60}")
    print("🎯 综合分析结论")
    print(f"{'='*60}")
    
    # 基于数据的投资建议
    if historical_data:
        pe_ratio = historical_data['当前PE']
        annual_return = historical_data['年化收益率']
        sharpe_ratio = historical_data['夏普比率']
        
        print(f"\n📊 基于数据的投资建议:")
        
        # 估值分析
        if pe_ratio > 30:
            print(f"  ⚠️ 估值风险: PE({pe_ratio:.1f})偏高")
        elif pe_ratio < 20:
            print(f"  ✅ 估值机会: PE({pe_ratio:.1f})偏低")
        else:
            print(f"  📊 估值合理: PE({pe_ratio:.1f})")
        
        # 历史表现分析
        if annual_return > 0.15:
            print(f"  ✅ 历史表现优秀: 年化收益率{annual_return:.1%}")
        elif annual_return > 0.08:
            print(f"  📊 历史表现良好: 年化收益率{annual_return:.1%}")
        else:
            print(f"  ⚠️ 历史表现一般: 年化收益率{annual_return:.1%}")
        
        # 风险调整收益
        if sharpe_ratio > 1.0:
            print(f"  ✅ 风险调整收益优秀: 夏普比率{sharpe_ratio:.2f}")
        elif sharpe_ratio > 0.5:
            print(f"  📊 风险调整收益良好: 夏普比率{sharpe_ratio:.2f}")
        else:
            print(f"  ⚠️ 风险调整收益一般: 夏普比率{sharpe_ratio:.2f}")
    
    # 市场环境建议
    if market_data:
        tech_performance = market_data.get('XLK', {}).get('近期收益率', 0)
        print(f"\n🌍 市场环境分析:")
        print(f"  科技股近期表现: {tech_performance:.1%}")
        
        if tech_performance > 0.1:
            print(f"  🟢 科技股强势，有利于ADBE")
        elif tech_performance > 0:
            print(f"  🟡 科技股温和上涨，ADBE可能跟随")
        else:
            print(f"  🔴 科技股下跌，ADBE可能承压")
    
    print(f"\n{'='*60}")
    print("✅ 分析完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 