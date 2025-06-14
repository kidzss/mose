#!/usr/bin/env python3
"""
数据更新诊断脚本
分析为什么数据还是12号的，以及为什么系统提示不需要更新
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta, date
import pandas as pd
import pymysql
from data.data_updater import MarketDataUpdater, DB_CONFIG
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_data_update_issue():
    """诊断数据更新问题"""
    print("🔍 开始诊断数据更新问题")
    print("=" * 60)
    
    # 1. 检查当前时间
    now = datetime.now()
    print(f"📅 当前时间: {now}")
    print(f"📅 当前日期: {now.date()}")
    print(f"📅 当前是周几: {now.strftime('%A')}")
    
    # 2. 检查数据库中最新的数据日期
    print("\n📊 检查数据库中最新的数据日期...")
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        
        with conn.cursor() as cursor:
            # 检查stock_time_code表的最新日期
            cursor.execute("SELECT MAX(Date) FROM stock_time_code")
            latest_time_code = cursor.fetchone()[0]
            print(f"📈 stock_time_code表最新日期: {latest_time_code}")
            
            # 检查stock_code_time表的最新日期
            cursor.execute("SELECT MAX(Date) FROM stock_code_time")
            latest_code_time = cursor.fetchone()[0]
            print(f"📈 stock_code_time表最新日期: {latest_code_time}")
            
            # 检查最近几天的数据分布
            print("\n📊 最近5天的数据分布:")
            for i in range(5):
                check_date = (now.date() - timedelta(days=i)).strftime('%Y-%m-%d')
                cursor.execute("SELECT COUNT(*) FROM stock_time_code WHERE Date = %s", (check_date,))
                count = cursor.fetchone()[0]
                print(f"  {check_date}: {count} 条记录")
                
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return
    
    finally:
        conn.close()
    
    # 3. 测试一只股票的数据更新逻辑
    print("\n🧪 测试股票数据更新逻辑...")
    test_symbol = "AAPL"  # 使用苹果股票作为测试
    
    try:
        updater = MarketDataUpdater(DB_CONFIG)
        
        # 获取最后更新时间
        last_update = updater.get_last_update_time(test_symbol)
        print(f"📅 {test_symbol} 最后更新时间: {last_update}")
        
        if last_update:
            # 计算时间差
            if isinstance(last_update, date) and not isinstance(last_update, datetime):
                last_update = datetime.combine(last_update, datetime.min.time())
            
            time_diff = datetime.now() - last_update
            print(f"⏰ 距离最后更新: {time_diff.days} 天 {time_diff.seconds//3600} 小时")
            
            # 检查是否在24小时内
            if time_diff.days < 1:
                print("⚠️  系统会跳过更新，因为数据在24小时内已更新")
            else:
                print("✅ 系统会尝试更新数据")
        
        # 4. 检查Yahoo Finance是否有最新数据
        print("\n🌐 检查Yahoo Finance是否有最新数据...")
        import yfinance as yf
        
        ticker = yf.Ticker(test_symbol)
        
        # 获取最近3天的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="1d",
            prepost=False,
            actions=True
        )
        
        if not df.empty:
            print(f"📊 Yahoo Finance返回了 {len(df)} 条记录")
            print(f"📅 数据日期范围: {df.index.min()} 到 {df.index.max()}")
            
            # 检查是否有今天的数据
            today_str = now.date().strftime('%Y-%m-%d')
            if today_str in df.index.strftime('%Y-%m-%d').values:
                print(f"✅ Yahoo Finance有今天({today_str})的数据")
            else:
                print(f"❌ Yahoo Finance没有今天({today_str})的数据")
                
            # 检查是否有昨天的数据
            yesterday_str = (now.date() - timedelta(days=1)).strftime('%Y-%m-%d')
            if yesterday_str in df.index.strftime('%Y-%m-%d').values:
                print(f"✅ Yahoo Finance有昨天({yesterday_str})的数据")
            else:
                print(f"❌ Yahoo Finance没有昨天({yesterday_str})的数据")
        else:
            print("❌ Yahoo Finance没有返回数据")
        
        # 5. 检查是否是交易日
        print("\n📅 检查交易日信息...")
        today = now.date()
        yesterday = today - timedelta(days=1)
        
        # 简单的交易日检查（周末不是交易日）
        if today.weekday() >= 5:  # 周六=5, 周日=6
            print(f"⚠️  今天({today})是周末，不是交易日")
        else:
            print(f"✅ 今天({today})是工作日")
            
        if yesterday.weekday() >= 5:
            print(f"⚠️  昨天({yesterday})是周末，不是交易日")
        else:
            print(f"✅ 昨天({yesterday})是工作日")
        
        # 6. 分析问题原因
        print("\n🔍 问题分析:")
        print("-" * 40)
        
        if latest_time_code and latest_time_code < now.date():
            days_behind = (now.date() - latest_time_code).days
            print(f"📊 数据库数据落后 {days_behind} 天")
            
            if days_behind == 1:
                print("💡 可能原因:")
                print("  1. 昨天是周末，没有交易数据")
                print("  2. Yahoo Finance还没有更新昨天的数据")
                print("  3. 数据更新脚本没有运行")
            elif days_behind > 1:
                print("💡 可能原因:")
                print("  1. 数据更新脚本长时间没有运行")
                print("  2. Yahoo Finance API问题")
                print("  3. 网络连接问题")
        else:
            print("✅ 数据库数据是最新的")
        
        # 7. 建议解决方案
        print("\n💡 建议解决方案:")
        print("-" * 40)
        print("1. 强制更新数据:")
        print("   updater.update_stock_data(force_update=True)")
        print()
        print("2. 检查Yahoo Finance API状态")
        print("3. 检查网络连接")
        print("4. 检查数据更新脚本是否正常运行")
        
    except Exception as e:
        print(f"❌ 诊断过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_data_update_issue() 