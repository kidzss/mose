"""
波动率指标模块测试

测试volatility.py中各种波动率指标的计算结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os
import sys

# 添加项目根目录到路径，确保能导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 导入测试的模块
from strategy.indicators.volatility import (
    atr, historical_volatility, bollinger_bandwidth, 
    keltner_channel_width, volatility_ratio, chaikin_volatility,
    garman_klass_volatility, ulcer_index
)


def generate_test_data(length: int = 200) -> Dict[str, pd.Series]:
    """
    生成测试用的价格数据
    
    参数:
        length: 数据长度
        
    返回:
        包含OHLCV数据的字典
    """
    # 生成日期索引
    dates = pd.date_range('2020-01-01', periods=length)
    
    # 生成基础价格
    np.random.seed(42)  # 固定随机种子以便结果可重现
    price = 100 + np.random.randn(length).cumsum()
    
    # 生成随机波动，制造一些高低点
    volatility = np.abs(np.random.randn(length)) * 2
    
    # 生成OHLCV数据
    high = price + volatility
    low = price - volatility
    close = price + np.random.randn(length) * 0.5
    open_price = price - np.random.randn(length) * 0.5
    volume = np.random.randint(1000, 10000, length)
    
    # 创建Series
    return {
        'open': pd.Series(open_price, index=dates),
        'high': pd.Series(high, index=dates),
        'low': pd.Series(low, index=dates),
        'close': pd.Series(close, index=dates),
        'volume': pd.Series(volume, index=dates)
    }


def test_atr():
    """测试ATR计算"""
    print("测试ATR计算...")
    data = generate_test_data()
    
    # 计算ATR
    atr_values = atr(data['high'], data['low'], data['close'])
    
    # 验证
    assert isinstance(atr_values, pd.Series), "ATR结果应为Series类型"
    assert not atr_values.empty, "ATR结果不应为空"
    assert not atr_values.isnull().all(), "ATR结果不应全为NaN"
    assert atr_values.min() >= 0, "ATR值应大于等于0"
    
    print("ATR测试通过！")
    
    # 绘制图表
    plt.figure(figsize=(12, 6))
    
    # 绘制价格图
    ax1 = plt.subplot(211)
    ax1.plot(data['close'], label='收盘价')
    ax1.set_title('价格数据')
    ax1.legend()
    
    # 绘制ATR图
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(atr_values, label='ATR', color='orange')
    ax2.set_title('ATR (14)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'volatility_atr_test.png'))
    plt.close()


def test_historical_volatility():
    """测试历史波动率计算"""
    print("测试历史波动率计算...")
    data = generate_test_data()
    
    # 计算历史波动率
    hv_annual = historical_volatility(data['close'], annualize=True)
    hv_daily = historical_volatility(data['close'], annualize=False)
    
    # 验证
    assert isinstance(hv_annual, pd.Series), "历史波动率结果应为Series类型"
    assert not hv_annual.empty, "历史波动率结果不应为空"
    assert not hv_annual.isnull().all(), "历史波动率结果不应全为NaN"
    assert hv_annual.min() >= 0, "历史波动率值应大于等于0"
    assert (hv_annual > hv_daily).all(), "年化波动率应大于日波动率"
    
    print("历史波动率测试通过！")
    
    # 绘制图表
    plt.figure(figsize=(12, 6))
    
    # 绘制价格图
    ax1 = plt.subplot(211)
    ax1.plot(data['close'], label='收盘价')
    ax1.set_title('价格数据')
    ax1.legend()
    
    # 绘制波动率图
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(hv_annual, label='年化历史波动率', color='red')
    ax2.plot(hv_daily, label='日历史波动率', color='blue')
    ax2.set_title('历史波动率 (20)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'volatility_hv_test.png'))
    plt.close()


def test_bollinger_bandwidth():
    """测试布林带宽度计算"""
    print("测试布林带宽度计算...")
    data = generate_test_data()
    
    # 计算布林带宽度
    bandwidth = bollinger_bandwidth(data['close'])
    
    # 验证
    assert isinstance(bandwidth, pd.Series), "布林带宽度结果应为Series类型"
    assert not bandwidth.empty, "布林带宽度结果不应为空"
    assert not bandwidth.isnull().all(), "布林带宽度结果不应全为NaN"
    assert bandwidth.min() >= 0, "布林带宽度值应大于等于0"
    
    print("布林带宽度测试通过！")
    
    # 绘制图表
    plt.figure(figsize=(12, 6))
    
    # 绘制价格图
    ax1 = plt.subplot(211)
    ax1.plot(data['close'], label='收盘价')
    ax1.set_title('价格数据')
    ax1.legend()
    
    # 绘制布林带宽度图
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(bandwidth, label='布林带宽度', color='purple')
    ax2.set_title('布林带宽度 (20, 2.0)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'volatility_bbw_test.png'))
    plt.close()


def test_multiple_volatility_indicators():
    """测试多种波动率指标"""
    print("测试多种波动率指标...")
    data = generate_test_data()
    
    # 计算各种波动率指标
    atr_vals = atr(data['high'], data['low'], data['close'])
    hv = historical_volatility(data['close'])
    bbw = bollinger_bandwidth(data['close'])
    kcw = keltner_channel_width(data['high'], data['low'], data['close'])
    vr = volatility_ratio(data['close'])
    cv = chaikin_volatility(data['high'], data['low'])
    gkv = garman_klass_volatility(data['open'], data['high'], data['low'], data['close'])
    ui = ulcer_index(data['close'])
    
    # 验证每个指标
    for name, indicator in [
        ('ATR', atr_vals),
        ('历史波动率', hv),
        ('布林带宽度', bbw),
        ('Keltner通道宽度', kcw),
        ('波动率比率', vr),
        ('Chaikin波动率', cv),
        ('Garman-Klass波动率', gkv),
        ('Ulcer指数', ui)
    ]:
        assert isinstance(indicator, pd.Series), f"{name}结果应为Series类型"
        assert not indicator.empty, f"{name}结果不应为空"
        assert not indicator.isnull().all(), f"{name}结果不应全为NaN"
    
    print("多种波动率指标测试通过！")
    
    # 绘制综合图表
    plt.figure(figsize=(15, 10))
    
    # 绘制价格图
    ax1 = plt.subplot(311)
    ax1.plot(data['close'], label='收盘价')
    ax1.set_title('价格数据')
    ax1.legend()
    
    # 绘制主要波动率指标
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.plot(atr_vals, label='ATR', color='orange')
    ax2.plot(hv, label='历史波动率', color='red')
    ax2.plot(gkv, label='Garman-Klass波动率', color='green')
    ax2.set_title('主要波动率指标')
    ax2.legend()
    
    # 绘制其他波动率指标
    ax3 = plt.subplot(313, sharex=ax1)
    ax3.plot(bbw, label='布林带宽度', color='purple')
    ax3.plot(kcw, label='Keltner通道宽度', color='blue')
    ax3.plot(cv, label='Chaikin波动率', color='brown')
    ax3.plot(ui, label='Ulcer指数', color='black')
    ax3.set_title('其他波动率指标')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'volatility_multi_test.png'))
    plt.close()


if __name__ == "__main__":
    test_atr()
    test_historical_volatility()
    test_bollinger_bandwidth()
    test_multiple_volatility_indicators()
    
    print("所有波动率指标测试完成！") 