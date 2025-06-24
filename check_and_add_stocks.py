#!/usr/bin/env python3
"""
检查并添加股票到MySQL数据库
"""

import mysql.connector
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDatabaseManager:
    """股票数据库管理器"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '123456',
            'database': 'mose'
        }
        
        # 目标股票列表
        self.target_stocks = [
            'BRK-B',   # 伯克希尔
            'SPY',     # 标普500 ETF
            'GS',      # 高盛
            'ABT',     # 雅培
            '^VIX',    # 恐慌指数
            'ORCL',    # 甲骨文
            'COST',    # 好市多
            'XLK',     # 科技ETF
            'IBM',     # IBM
            'PLTR',    # Palantir
            'MRK',     # 默克
            # 额外优质股票
            'JPM',     # 摩根大通
            'JNJ',     # 强生
            'WMT',     # 沃尔玛
            'KO',      # 可口可乐
            'PG',      # 宝洁
            'V',       # Visa
            'MA',      # 万事达
            'UNH',     # 联合健康
            'HD',      # 家得宝
            'DIS',     # 迪士尼
            'NFLX',    # 奈飞
            'CRM',     # Salesforce
            'ADBE',    # Adobe
            'INTC',    # 英特尔
            'CSCO',    # 思科
            'BAC',     # 美国银行
            'WFC',     # 富国银行
            'XOM',     # 埃克森美孚
            'CVX',     # 雪佛龙
            'QQQ',     # 纳斯达克100 ETF
            'IWM',     # 罗素2000 ETF
            'GLD',     # 黄金ETF
            'TLT',     # 长期国债ETF
        ]
        
        logger.info("📊 股票数据库管理器初始化完成")
    
    def connect_db(self):
        """连接数据库"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return None
    
    def check_existing_stocks(self):
        """检查数据库中已存在的股票"""
        conn = self.connect_db()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            existing_stocks = []
            for table in tables:
                table_name = table[0]
                # 检查是否是股票表（通常以股票代码命名）
                if len(table_name) <= 6 and table_name.replace('-', '').replace('^', '').isalnum():
                    existing_stocks.append(table_name)
            
            cursor.close()
            conn.close()
            
            logger.info(f"✅ 发现 {len(existing_stocks)} 个股票表")
            return existing_stocks
            
        except Exception as e:
            logger.error(f"❌ 检查股票表失败: {e}")
            conn.close()
            return []
    
    def get_stock_info(self, symbol):
        """获取股票基本信息"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            logger.warning(f"获取{symbol}信息失败: {e}")
            return None
    
    def create_stock_table(self, symbol):
        """为股票创建数据表"""
        conn = self.connect_db()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # 处理特殊字符（如^VIX -> VIX_INDEX）
            table_name = symbol.replace('^', '').replace('-', '_')
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `id` int NOT NULL AUTO_INCREMENT,
                `date` date NOT NULL,
                `open` decimal(10,4) DEFAULT NULL,
                `high` decimal(10,4) DEFAULT NULL,
                `low` decimal(10,4) DEFAULT NULL,
                `close` decimal(10,4) DEFAULT NULL,
                `adj_close` decimal(10,4) DEFAULT NULL,
                `volume` bigint DEFAULT NULL,
                `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
                `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`id`),
                UNIQUE KEY `date` (`date`),
                KEY `idx_date` (`date`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            
            cursor.close()
            conn.close()
            
            logger.info(f"✅ 创建股票表: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 创建{symbol}表失败: {e}")
            conn.close()
            return False
    
    def insert_stock_data(self, symbol, days=365):
        """插入股票历史数据"""
        conn = self.connect_db()
        if not conn:
            return False
        
        try:
            # 获取股票数据
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"⚠️ {symbol} 无历史数据")
                conn.close()
                return False
            
            cursor = conn.cursor()
            table_name = symbol.replace('^', '').replace('-', '_')
            
            # 准备插入数据
            insert_sql = f"""
            INSERT IGNORE INTO `{table_name}` 
            (date, open, high, low, close, adj_close, volume) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            data_to_insert = []
            for date, row in hist.iterrows():
                data_to_insert.append((
                    date.strftime('%Y-%m-%d'),
                    float(row['Open']) if not pd.isna(row['Open']) else None,
                    float(row['High']) if not pd.isna(row['High']) else None,
                    float(row['Low']) if not pd.isna(row['Low']) else None,
                    float(row['Close']) if not pd.isna(row['Close']) else None,
                    float(row['Close']) if not pd.isna(row['Close']) else None,  # 使用Close作为adj_close
                    int(row['Volume']) if not pd.isna(row['Volume']) else 0
                ))
            
            cursor.executemany(insert_sql, data_to_insert)
            conn.commit()
            
            inserted_rows = cursor.rowcount
            cursor.close()
            conn.close()
            
            logger.info(f"✅ {symbol} 插入 {inserted_rows} 条数据")
            return True
            
        except Exception as e:
            logger.error(f"❌ 插入{symbol}数据失败: {e}")
            conn.close()
            return False
    
    def add_stock_to_database(self, symbol):
        """添加股票到数据库（创建表+插入数据）"""
        logger.info(f"🔄 处理股票: {symbol}")
        
        # 创建表
        if not self.create_stock_table(symbol):
            return False
        
        # 插入数据
        if not self.insert_stock_data(symbol):
            return False
        
        return True
    
    def check_and_add_stocks(self):
        """检查并添加缺失的股票"""
        # 检查现有股票
        existing_stocks = self.check_existing_stocks()
        
        # 标准化现有股票名称用于比较
        existing_normalized = []
        for stock in existing_stocks:
            normalized = stock.replace('_', '-').upper()
            if not normalized.startswith('^'):
                # 检查是否需要添加^前缀（如VIX_INDEX -> ^VIX）
                if 'VIX' in normalized and 'INDEX' in stock:
                    normalized = '^VIX'
            existing_normalized.append(normalized)
        
        print("=" * 80)
        print("📊 MySQL股票数据库检查报告")
        print(f"📅 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        print(f"\n📋 数据库中已存在的股票 ({len(existing_stocks)}个):")
        print("-" * 60)
        for i, stock in enumerate(existing_stocks, 1):
            print(f"{i:2d}. {stock}")
        
        # 检查缺失的股票
        missing_stocks = []
        for target_stock in self.target_stocks:
            target_normalized = target_stock.upper()
            if target_normalized not in existing_normalized:
                missing_stocks.append(target_stock)
        
        print(f"\n❌ 缺失的目标股票 ({len(missing_stocks)}个):")
        print("-" * 60)
        if missing_stocks:
            for i, stock in enumerate(missing_stocks, 1):
                stock_info = self.get_stock_info(stock)
                if stock_info:
                    print(f"{i:2d}. {stock:<8} - {stock_info['name'][:40]}")
                else:
                    print(f"{i:2d}. {stock:<8} - 获取信息失败")
        else:
            print("✅ 所有目标股票都已存在！")
        
        # 询问是否添加缺失股票
        if missing_stocks:
            print(f"\n💡 建议添加的优质股票:")
            print("-" * 60)
            
            # 按类别分组显示
            categories = {
                '防御性股票': ['BRK-B', 'JNJ', 'PG', 'KO', 'WMT'],
                '金融股票': ['JPM', 'GS', 'BAC', 'WFC', 'V', 'MA'],
                '医疗股票': ['ABT', 'MRK', 'UNH'],
                '科技股票': ['ORCL', 'IBM', 'PLTR', 'CRM', 'ADBE', 'INTC', 'CSCO'],
                '消费股票': ['COST', 'HD', 'DIS', 'NFLX'],
                '能源股票': ['XOM', 'CVX'],
                'ETF基金': ['SPY', 'QQQ', 'XLK', 'IWM', 'GLD', 'TLT'],
                '市场指标': ['^VIX']
            }
            
            for category, stocks in categories.items():
                category_missing = [s for s in stocks if s in missing_stocks]
                if category_missing:
                    print(f"\n{category}:")
                    for stock in category_missing:
                        stock_info = self.get_stock_info(stock)
                        if stock_info:
                            print(f"  • {stock:<8} - {stock_info['name'][:40]}")
            
            print(f"\n🚀 开始添加缺失股票...")
            print("-" * 60)
            
            success_count = 0
            for stock in missing_stocks:
                if self.add_stock_to_database(stock):
                    success_count += 1
                else:
                    logger.error(f"❌ 添加{stock}失败")
            
            print(f"\n✅ 成功添加 {success_count}/{len(missing_stocks)} 个股票")
            
        print("\n" + "=" * 80)
        print("📋 数据库更新完成")
        print("=" * 80)

def main():
    """主函数"""
    manager = StockDatabaseManager()
    manager.check_and_add_stocks()

if __name__ == "__main__":
    main() 