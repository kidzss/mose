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
from data.data_validator import DataValidator
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
        # 第一次调用会从数据源获取数据
        print("第一次调用get_historical_data...")
        aapl_data = data_interface.get_historical_data('AAPL', start_date, end_date)
        
        # 第二次调用会使用缓存
        print("第二次调用get_historical_data（使用缓存）...")
        aapl_data_cached = data_interface.get_historical_data('AAPL', start_date, end_date)
        
        # 验证数据质量
        print("\n验证数据质量...")
        validated_data, report = DataValidator.validate_data(aapl_data)
        
        print("\n数据验证报告:")
        print(f"- 原始数据行数: {report['original_rows']}")
        print(f"- 处理后行数: {report['processed_rows']}")
        
        if report['missing_values']:
            print(f"- 缺失值: {report['missing_values']}")
        
        if report['outliers']:
            print(f"- 异常值: {report['outliers']}")
            
        if report['invalid_prices']:
            print(f"- 不合理价格: {report['invalid_prices']}")
            
        if report['gaps']:
            print(f"- 数据缺失区间: {report['gaps']}")
        
        # 打印数据基本信息
        print("\n数据基本信息:")
        print(f"数据时间范围: {validated_data.index[0]} 到 {validated_data.index[-1]}")
        print(f"数据点数量: {len(validated_data)}")
        print(f"数据列: {validated_data.columns.tolist()}")
        print("\n前5行数据:")
        print(validated_data.head())
        
        # 绘制收盘价和技术指标
        plt.figure(figsize=(15, 10))
        
        # 绘制主图
        plt.subplot(2, 1, 1)
        plt.plot(validated_data.index, validated_data['close'], label='收盘价')
        plt.plot(validated_data.index, validated_data['ma20'], label='20日均线')
        plt.plot(validated_data.index, validated_data['upper_band'], '--', label='上轨')
        plt.plot(validated_data.index, validated_data['lower_band'], '--', label='下轨')
        plt.title('AAPL股价走势与技术指标')
        plt.legend()
        plt.grid(True)
        
        # 绘制副图（成交量）
        plt.subplot(2, 1, 2)
        plt.bar(validated_data.index, validated_data['volume'], label='成交量')
        plt.plot(validated_data.index, validated_data['volume_ma'], 'r', label='成交量MA20')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('aapl_analysis.png')
        print("已保存分析图表到 aapl_analysis.png")
        
    except Exception as e:
        print(f"获取数据失败: {e}")


def strategy_data_example():
    """策略数据使用示例"""
    data_interface = DataInterface()
    
    try:
        # 获取策略所需数据
        print("\n获取策略数据...")
        strategy_data = data_interface.get_data_for_strategy('AAPL', lookback_days=120)
        
        print(f"\n策略数据包含以下指标:")
        print(f"技术指标: {[col for col in strategy_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
        
        # 计算一些基本的统计信息
        print("\n基本统计信息:")
        print(f"平均日收益率: {strategy_data['returns'].mean():.4%}")
        print(f"收益率标准差: {strategy_data['returns'].std():.4%}")
        print(f"最大日收益率: {strategy_data['returns'].max():.4%}")
        print(f"最小日收益率: {strategy_data['returns'].min():.4%}")
        
    except Exception as e:
        print(f"获取策略数据失败: {e}")


def multiple_symbols_example():
    """多股票数据示例"""
    data_interface = DataInterface()
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    
    try:
        print(f"\n获取多个股票的数据: {symbols}")
        data_dict = data_interface.get_multiple_symbols_data(symbols, start_date, end_date)
        
        # 计算每个股票的基本统计信息
        for symbol, data in data_dict.items():
            # 验证数据
            validated_data, report = DataValidator.validate_data(data)
            
            print(f"\n{symbol} 统计信息:")
            print(f"数据点数量: {len(validated_data)}")
            print(f"平均收盘价: ${validated_data['close'].mean():.2f}")
            print(f"平均日成交量: {validated_data['volume'].mean():.0f}")
            
            if not report['validation_passed']:
                print(f"警告: 数据验证未通过")
                if report['invalid_prices']:
                    print(f"发现不合理价格: {report['invalid_prices']}")
                if report['gaps']:
                    print(f"发现数据缺失: {report['gaps']}")
        
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