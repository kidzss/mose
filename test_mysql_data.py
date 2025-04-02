#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试MySQL数据源连接和数据检索
"""

import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data.data_interface import DataInterface, MySQLDataSource
from config.data_config import default_data_config

def test_mysql_connection():
    """测试MySQL连接"""
    print("测试MySQL连接...")
    try:
        # 使用默认配置创建MySQL数据源
        mysql_source = MySQLDataSource()
        print("MySQL连接成功!")
        return True
    except Exception as e:
        print(f"MySQL连接失败: {e}")
        return False

def test_get_stock_data():
    """测试获取股票数据"""
    print("\n测试获取股票数据...")
    
    # 创建数据接口
    data_interface = DataInterface()
    
    # 获取当前日期和一年前的日期
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365)
    
    # 测试股票代码列表
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    for symbol in test_symbols:
        print(f"\n获取 {symbol} 数据:")
        try:
            # 获取历史数据
            df = data_interface.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                print(f"未找到 {symbol} 的数据")
            else:
                print(f"数据点数量: {len(df)}")
                print(f"日期范围: {df.index[0]} 到 {df.index[-1]}")
                print(f"数据列: {df.columns.tolist()}")
                print("\n前5行数据:")
                print(df.head())
                
        except Exception as e:
            print(f"获取数据失败: {e}")

def get_available_symbols():
    """获取数据库中可用的股票代码"""
    print("\n获取数据库中可用的股票代码...")
    
    try:
        # 创建MySQL数据源
        mysql_source = MySQLDataSource()
        
        # 获取连接
        with mysql_source.get_connection() as conn:
            with conn.cursor() as cursor:
                # 查询不同的股票代码
                cursor.execute("SELECT DISTINCT Code FROM stock_code_time LIMIT 10")
                symbols = [row['Code'] for row in cursor.fetchall()]
                
                if symbols:
                    print(f"找到 {len(symbols)} 个股票代码:")
                    for i, symbol in enumerate(symbols, 1):
                        print(f"{i}. {symbol}")
                    
                    # 返回第一个股票代码用于后续测试
                    return symbols
                else:
                    print("未找到任何股票代码")
                    return []
                    
    except Exception as e:
        print(f"获取股票代码失败: {e}")
        return []

def test_specific_symbol(symbol):
    """测试特定股票代码的数据检索"""
    if not symbol:
        print("\n没有可用的股票代码进行测试")
        return
        
    print(f"\n测试特定股票 {symbol} 的数据检索...")
    
    # 创建数据接口
    data_interface = DataInterface()
    
    # 获取当前日期和一年前的日期
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365)
    
    try:
        # 获取历史数据
        df = data_interface.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            print(f"未找到 {symbol} 的数据")
        else:
            print(f"数据点数量: {len(df)}")
            print(f"日期范围: {df.index[0]} 到 {df.index[-1]}")
            print(f"数据列: {df.columns.tolist()}")
            print("\n前5行数据:")
            print(df.head())
            
            # 创建用于策略的数据
            print("\n获取策略数据（包含技术指标）...")
            strategy_df = data_interface.get_data_for_strategy(
                symbol=symbol,
                lookback_days=120
            )
            
            print(f"策略数据点数量: {len(strategy_df)}")
            print(f"策略数据列: {strategy_df.columns.tolist()}")
            print("\n前5行策略数据:")
            print(strategy_df.head())
            
    except Exception as e:
        print(f"测试特定股票失败: {e}")

def compare_tables_data(symbol):
    """分析stock_code_time表中的数据"""
    if not symbol:
        print("\n没有可用的股票代码进行分析")
        return
        
    print(f"\n分析 stock_code_time 表中 {symbol} 的数据...")
    
    # 创建MySQL数据源
    mysql_source = MySQLDataSource()
    
    # 获取最近一年的数据
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365)  # 分析一年的数据
    
    try:
        # 从 stock_code_time 表获取数据
        with mysql_source.get_connection() as conn:
            cursor = conn.cursor()
            
            # 从 stock_code_time 表查询
            query = """
            SELECT Date, Code, Open, High, Low, Close, Volume, AdjClose
            FROM stock_code_time
            WHERE Code = %s AND Date BETWEEN %s AND %s
            ORDER BY Date
            """
            
            cursor.execute(query, (
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ))
            
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            
            if not rows:
                print(f"stock_code_time 表中未找到 {symbol} 的数据")
                return
                
            df = pd.DataFrame(list(rows), columns=columns)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 分析数据
            print(f"数据点数量: {len(df)}")
            print(f"日期范围: {df.index[0]} 到 {df.index[-1]}")
            
            # 计算统计信息
            stats = df.describe()
            print("\n基本统计信息:")
            print(stats)
            
            # 检查缺失值
            missing = df.isnull().sum()
            print("\n缺失值统计:")
            print(missing)
            
            # 绘制收盘价走势图
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'])
            plt.title(f'{symbol} 收盘价走势')
            plt.xlabel('日期')
            plt.ylabel('价格')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{symbol}_price_trend.png')
            print(f"已保存价格走势图到 {symbol}_price_trend.png")
            
    except Exception as e:
        print(f"分析表数据失败: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("MySQL数据源测试")
    print("=" * 50)
    
    # 测试MySQL连接
    if test_mysql_connection():
        # 获取可用的股票代码
        available_symbols = get_available_symbols()
        
        if available_symbols:
            # 测试特定股票代码
            test_specific_symbol(available_symbols[0])
            
            # 分析数据
            for symbol in available_symbols[:2]:  # 只取前两个符号测试
                compare_tables_data(symbol)
        
        # 测试预定义的股票列表
        test_get_stock_data()
    
    print("\n测试完成!") 