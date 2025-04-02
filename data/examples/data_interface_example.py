#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一数据接口使用示例
展示如何使用DataInterface从不同数据源获取数据
"""

import os
import sys
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_interface import DataInterface
from config.data_config import DataConfig, MySQLConfig, FutuConfig


def basic_usage_example():
    """基本使用示例"""
    # 使用默认配置创建数据接口实例
    data_interface = DataInterface()
    
    # 获取AAPL从2023年开始的历史数据
    start_date = '2023-01-01'
    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    
    print(f"获取AAPL从{start_date}到{end_date}的数据...")
    
    try:
        aapl_data = data_interface.get_historical_data('AAPL', start_date, end_date)
        
        # 打印数据基本信息
        print(f"数据时间范围: {aapl_data.index[0]} 到 {aapl_data.index[-1]}")
        print(f"数据点数量: {len(aapl_data)}")
        print(f"数据列: {aapl_data.columns.tolist()}")
        print("\n前5行数据:")
        print(aapl_data.head())
        
        # 绘制收盘价走势图
        plt.figure(figsize=(12, 6))
        plt.plot(aapl_data.index, aapl_data['close'])
        plt.title('AAPL股价走势')
        plt.xlabel('日期')
        plt.ylabel('价格 ($)')
        plt.grid(True)
        plt.savefig('aapl_price.png')
        print("已保存价格走势图到 aapl_price.png")
        
    except Exception as e:
        print(f"获取数据失败: {e}")


def strategy_data_example():
    """获取策略数据示例"""
    data_interface = DataInterface()
    
    # 获取策略所需的AAPL数据（包含技术指标）
    try:
        aapl_strategy_data = data_interface.get_data_for_strategy('AAPL')
        
        # 打印策略数据基本信息
        print("\n策略数据信息:")
        print(f"数据时间范围: {aapl_strategy_data.index[0]} 到 {aapl_strategy_data.index[-1]}")
        print(f"数据点数量: {len(aapl_strategy_data)}")
        print(f"数据列: {aapl_strategy_data.columns.tolist()}")
        
        # 绘制价格和移动平均线
        plt.figure(figsize=(12, 6))
        plt.plot(aapl_strategy_data.index, aapl_strategy_data['close'], label='收盘价')
        plt.plot(aapl_strategy_data.index, aapl_strategy_data['ma20'], label='20日均线', linewidth=1.5)
        plt.plot(aapl_strategy_data.index, aapl_strategy_data['ma60'], label='60日均线', linewidth=1.5)
        plt.fill_between(aapl_strategy_data.index, 
                        aapl_strategy_data['upper_band'],
                        aapl_strategy_data['lower_band'], 
                        alpha=0.2, color='gray', label='布林带')
        plt.title('AAPL股价和技术指标')
        plt.xlabel('日期')
        plt.ylabel('价格 ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('aapl_indicators.png')
        print("已保存技术指标图到 aapl_indicators.png")
        
    except Exception as e:
        print(f"获取策略数据失败: {e}")


def multiple_symbols_example():
    """获取多个股票数据示例"""
    data_interface = DataInterface()
    
    # 股票列表
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # 获取多个股票的数据
    try:
        # 获取近90天数据
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=90)
        
        print(f"\n获取多个股票数据: {', '.join(symbols)}")
        data_dict = data_interface.get_multiple_symbols_data(
            symbols, start_date, end_date
        )
        
        # 打印每个股票的数据信息
        for symbol, df in data_dict.items():
            print(f"{symbol} 数据点数量: {len(df)}")
        
        # 归一化价格并绘制比较图
        plt.figure(figsize=(12, 6))
        
        for symbol, df in data_dict.items():
            if not df.empty:
                # 归一化价格（第一天=100）
                normalized = df['close'] / df['close'].iloc[0] * 100
                plt.plot(df.index, normalized, label=symbol)
        
        plt.title('股票价格比较 (基准化)')
        plt.xlabel('日期')
        plt.ylabel('价格 (归一化)')
        plt.legend()
        plt.grid(True)
        plt.savefig('stocks_comparison.png')
        print("已保存股票比较图到 stocks_comparison.png")
        
    except Exception as e:
        print(f"获取多股票数据失败: {e}")


def custom_config_example():
    """自定义配置示例"""
    # 创建自定义数据配置
    custom_config = DataConfig(
        default_source='mysql',
        mysql=MySQLConfig(
            host='localhost',
            port=3306,
            user='root',
            password='',
            database='mose'
        ),
        default_lookback_days=120  # 自定义回溯天数
    )
    
    # 使用自定义配置创建数据接口
    data_interface = DataInterface(
        config=custom_config.get_all_configs()
    )
    
    try:
        # 使用自定义配置获取数据
        print("\n使用自定义配置获取NVDA数据...")
        nvda_data = data_interface.get_data_for_strategy('NVDA')
        
        print(f"NVDA数据点数量: {len(nvda_data)}")
        print(f"自定义回溯天数: {custom_config.default_lookback_days}")
        
    except Exception as e:
        print(f"使用自定义配置获取数据失败: {e}")


def search_symbols_example():
    """股票搜索示例"""
    data_interface = DataInterface()
    
    # 搜索股票
    search_terms = ['Apple', 'Micro', 'Google']
    
    print("\n搜索股票示例:")
    for term in search_terms:
        try:
            print(f"搜索 '{term}':")
            results = data_interface.search_symbols(term)
            
            if results:
                for i, stock in enumerate(results[:5], 1):  # 最多显示5个结果
                    print(f"  {i}. {stock.get('symbol')} - {stock.get('name')} ({stock.get('exchange')})")
            else:
                print(f"  未找到匹配 '{term}' 的股票")
                
        except Exception as e:
            print(f"  搜索出错: {e}")
        
        print()


if __name__ == "__main__":
    print("=" * 50)
    print("数据接口使用示例")
    print("=" * 50)
    
    # 运行各个示例
    basic_usage_example()
    strategy_data_example()
    multiple_symbols_example()
    custom_config_example()
    search_symbols_example()
    
    print("\n示例运行完成！") 