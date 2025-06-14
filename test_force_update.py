#!/usr/bin/env python3
"""
强制更新数据脚本
使用force_update=True来获取最新数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_updater import MarketDataUpdater, DB_CONFIG
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def force_update_data():
    """强制更新数据"""
    print("🚀 开始强制更新数据")
    print("=" * 60)
    
    try:
        # 创建更新器
        updater = MarketDataUpdater(DB_CONFIG)
        
        # 获取当前时间
        now = datetime.now()
        print(f"📅 当前时间: {now}")
        
        # 强制更新所有股票数据
        print("🔄 开始强制更新...")
        result = updater.update_stock_data(force_update=True)
        
        print("\n📊 更新结果:")
        print(f"  总计: {result['total']} 只股票")
        print(f"  更新: {result['updated']} 只")
        print(f"  跳过: {result['skipped']} 只")
        print(f"  失败: {result['failed']} 只")
        print(f"  新增记录: {result['new_records_count']} 条")
        
        # 检查更新后的最新日期
        print("\n📈 检查更新后的最新数据日期...")
        import pymysql
        
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MAX(Date) FROM stock_time_code")
                latest_date = cursor.fetchone()[0]
                print(f"📅 更新后最新日期: {latest_date}")
                
                if latest_date:
                    days_behind = (now.date() - latest_date).days
                    if days_behind == 0:
                        print("✅ 数据已是最新！")
                    else:
                        print(f"⚠️  数据仍落后 {days_behind} 天")
                        
        finally:
            conn.close()
            
    except Exception as e:
        print(f"❌ 强制更新失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    force_update_data() 