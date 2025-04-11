from data.data_updater import MarketDataUpdater, DB_CONFIG
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text
import sys
import pandas_market_calendars as mcal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    def __init__(self, db_config):
        """初始化数据质量检查器"""
        self.db_config = db_config
        self.engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        # 获取NYSE日历
        self.nyse = mcal.get_calendar('NYSE')
    
    def get_stock_first_date(self, symbol):
        """获取股票最早的数据日期"""
        query = text("""
            SELECT MIN(Date) 
            FROM stock_time_code 
            WHERE Code = :symbol
            AND Volume > 0  -- 确保是有效的交易日
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol}).scalar()
            return result.strftime('%Y-%m-%d') if result else None

    def get_trading_days(self, start_date, end_date):
        """获取指定日期范围内的实际交易日"""
        try:
            schedule = self.nyse.schedule(start_date=start_date, end_date=end_date)
            trading_days = mcal.date_range(schedule, frequency='1D')
            return trading_days.strftime('%Y-%m-%d').tolist()
        except Exception as e:
            logger.error(f"获取交易日历出错: {str(e)}")
            return []
    
    def check_missing_dates(self, symbol):
        """检查缺失的交易日数据"""
        # 获取股票的数据范围
        query = text("""
            SELECT 
                MIN(Date) as first_date,
                MAX(Date) as last_date,
                COUNT(DISTINCT Date) as total_days
            FROM stock_time_code 
            WHERE Code = :symbol
            AND Volume > 0  -- 确保是有效的交易日
            AND Date <= CURDATE()  -- 只检查到今天为止的数据
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol}).fetchone()
            if not result or not result[0]:
                return []
            
            first_date = result[0].strftime('%Y-%m-%d') if isinstance(result[0], datetime) else result[0]
            last_date = result[1].strftime('%Y-%m-%d') if isinstance(result[1], datetime) else result[1]
            total_days = result[2]
            
            logger.info(f"股票 {symbol} 数据范围: {first_date} 到 {last_date}, 总天数: {total_days}")
        
        # 获取该时间范围内的所有交易日
        trading_days = self.get_trading_days(first_date, last_date)
        if not trading_days:
            return []
        
        logger.info(f"股票 {symbol} 交易日历天数: {len(trading_days)}")
        
        # 获取实际的数据日期
        query = text("""
            SELECT Date FROM stock_time_code 
            WHERE Code = :symbol 
            AND Date BETWEEN :start_date AND :end_date
            AND Volume > 0  -- 只考虑有交易的日期
            ORDER BY Date
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                "symbol": symbol,
                "start_date": first_date,
                "end_date": last_date
            })
            dates = [row[0].strftime('%Y-%m-%d') if isinstance(row[0], datetime) else row[0] 
                    for row in result]
        
        logger.info(f"股票 {symbol} 实际数据天数: {len(dates)}")
        
        # 确保日期格式统一，都转换为字符串进行比较
        trading_days_set = set(trading_days)
        dates_set = set(dates)
        
        # 计算缺失的交易日
        missing_dates = sorted(list(trading_days_set - dates_set))  # 使用集合运算找出缺失的日期
        
        # 添加调试信息
        if missing_dates:
            logger.info(f"股票 {symbol} 缺失的具体日期: {missing_dates[:5]}...")
            logger.info(f"股票 {symbol} 缺失天数: {len(missing_dates)}")
            logger.info(f"股票 {symbol} 第一个实际数据日期: {dates[0] if dates else 'N/A'}")
            logger.info(f"股票 {symbol} 最后一个实际数据日期: {dates[-1] if dates else 'N/A'}")
            
            # 添加更多调试信息以验证日期范围
            logger.info(f"股票 {symbol} 交易日历范围: {trading_days[0]} 到 {trading_days[-1]}")
            logger.info(f"股票 {symbol} 实际数据比例: {len(dates)}/{len(trading_days)} = {len(dates)/len(trading_days):.2%}")
        
        return missing_dates
    
    def check_data_anomalies(self, symbol, lookback_days=30):
        """检查数据异常"""
        # 使用最近的已完成交易日
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_days)
        
        query = text("""
            SELECT Date, Open, High, Low, Close, Volume, AdjClose
            FROM stock_time_code 
            WHERE Code = :symbol 
            AND Date BETWEEN :start_date AND :end_date
            AND Volume > 0  -- 只检查有交易的日期
            ORDER BY Date
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "symbol": symbol,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            })
        
        if df.empty:
            return []
            
        anomalies = []
        
        # 检查价格为0或null（排除AdjClose，因为它可能合理地为空）
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                zero_prices = df[df[col] <= 0].Date.tolist()
                null_prices = df[df[col].isnull()].Date.tolist()
                if zero_prices:
                    anomalies.append(f"{col}价格为0: {[d.strftime('%Y-%m-%d') if isinstance(d, datetime) else d for d in zero_prices]}")
                if null_prices:
                    anomalies.append(f"{col}价格为空: {[d.strftime('%Y-%m-%d') if isinstance(d, datetime) else d for d in null_prices]}")
        
        # 检查成交量异常（排除节假日）
        trading_days = set(self.get_trading_days(start_date.strftime('%Y-%m-%d'), 
                                               end_date.strftime('%Y-%m-%d')))
        df['Date_Str'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x)
        df_trading = df[df['Date_Str'].isin(trading_days)]
        
        zero_volume = df_trading[df_trading['Volume'] <= 0]['Date'].tolist()
        if zero_volume:
            anomalies.append(f"成交量为0: {[d.strftime('%Y-%m-%d') if isinstance(d, datetime) else d for d in zero_volume]}")
        
        # 检查价格逻辑关系
        invalid_prices = df[
            (df['High'] < df['Low']) | 
            (df['Open'] > df['High']) | 
            (df['Open'] < df['Low']) |
            (df['Close'] > df['High']) |
            (df['Close'] < df['Low'])
        ]['Date'].tolist()
        
        if invalid_prices:
            anomalies.append(f"价格逻辑关系错误: {[d.strftime('%Y-%m-%d') if isinstance(d, datetime) else d for d in invalid_prices]}")
        
        # 检查价格跳跃（排除已知的分红、拆股等事件）
        df['price_change'] = df['Close'].pct_change()
        large_changes = df[abs(df['price_change']) > 0.2]['Date'].tolist()
        if large_changes:
            # 检查是否有对应的分红或拆股信息
            dates_str = ', '.join([f"'{d.strftime('%Y-%m-%d')}'" if isinstance(d, datetime) else f"'{d}'" 
                                 for d in large_changes])
            div_split_query = text(f"""
                SELECT Date, Dividends, StockSplits
                FROM stock_time_code
                WHERE Code = :symbol
                AND Date IN ({dates_str})
                AND (Dividends > 0 OR StockSplits != 1)
            """)
            
            with self.engine.connect() as conn:
                div_split_dates = pd.read_sql(div_split_query, conn, params={"symbol": symbol})
            
            # 排除有分红或拆股的日期
            if not div_split_dates.empty:
                large_changes = [d for d in large_changes 
                               if d not in div_split_dates['Date'].tolist()]
            
            if large_changes:  # 如果还有异常日期
                anomalies.append(f"价格异常跳跃(>20%): {[d.strftime('%Y-%m-%d') if isinstance(d, datetime) else d for d in large_changes]}")
        
        return anomalies
    
    def check_duplicate_records(self, symbol):
        """检查重复记录"""
        query = text("""
            SELECT Date, COUNT(*) as count
            FROM stock_time_code
            WHERE Code = :symbol
            GROUP BY Date
            HAVING count > 1
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol})
            duplicates = [(row[0].strftime('%Y-%m-%d') if isinstance(row[0], datetime) else row[0], 
                          row[1]) for row in result]
        
        return duplicates
    
    def check_data_consistency(self, symbol):
        """检查两个表之间的数据一致性"""
        query = text("""
            SELECT t1.Date, 
                   t1.Close as time_code_close, 
                   t2.Close as code_time_close,
                   t1.Volume as time_code_volume,
                   t2.Volume as code_time_volume
            FROM stock_time_code t1
            LEFT JOIN stock_code_time t2 
            ON t1.Code = t2.Code AND t1.Date = t2.Date
            WHERE t1.Code = :symbol
            AND (ABS(t1.Close - t2.Close) > 0.01  -- 允许0.01的误差
                 OR ABS(t1.Volume - t2.Volume) > 0)
        """)
        
        with self.engine.connect() as conn:
            result = pd.read_sql(query, conn, params={"symbol": symbol})
        
        return result
    
    def get_all_symbols(self):
        """获取所有股票代码"""
        query = text("""
            SELECT DISTINCT t.Code 
            FROM stock_time_code t
            WHERE EXISTS (
                SELECT 1 
                FROM stock_time_code 
                WHERE Code = t.Code 
                AND Date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                AND Volume > 0  -- 确保是活跃交易的股票
            )
            AND t.Code NOT LIKE '^%'  -- 排除指数
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result]

def main():
    """主函数：运行所有数据质量检查"""
    checker = DataQualityChecker(DB_CONFIG)
    symbols = checker.get_all_symbols()
    
    print(f"\n=== 开始检查 {len(symbols)} 只股票的数据质量 ===\n")
    
    # 存储问题数据
    problems = {}
    
    for symbol in symbols:
        try:
            symbol_problems = []
            
            # 1. 检查缺失日期
            missing_dates = checker.check_missing_dates(symbol)
            if len(missing_dates) > 5:  # 只报告缺失超过5天的情况
                symbol_problems.append(f"缺失 {len(missing_dates)} 个交易日数据")
            
            # 2. 检查数据异常
            anomalies = checker.check_data_anomalies(symbol)
            if anomalies:
                symbol_problems.extend(anomalies)
            
            # 3. 检查重复记录
            duplicates = checker.check_duplicate_records(symbol)
            if duplicates:
                symbol_problems.append(f"存在 {len(duplicates)} 个重复日期")
            
            # 4. 检查数据一致性
            inconsistencies = checker.check_data_consistency(symbol)
            if not inconsistencies.empty:
                symbol_problems.append(f"两表数据不一致: {len(inconsistencies)} 条记录")
            
            if symbol_problems:
                problems[symbol] = symbol_problems
                print(f"\n股票 {symbol} 存在以下问题：")
                for problem in symbol_problems:
                    print(f"  - {problem}")
                    
        except Exception as e:
            logger.error(f"检查股票 {symbol} 时出错: {str(e)}")
            continue
    
    print(f"\n=== 数据质量检查完成 ===")
    print(f"检查了 {len(symbols)} 只股票")
    print(f"发现 {len(problems)} 只股票存在问题")
    
    # 保存问题报告
    with open('data_quality_report.txt', 'w') as f:
        f.write(f"数据质量检查报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"检查的股票数量: {len(symbols)}\n")
        f.write(f"发现问题的股票数量: {len(problems)}\n\n")
        
        for symbol, issues in problems.items():
            f.write(f"\n股票 {symbol} 的问题：\n")
            for issue in issues:
                f.write(f"  - {issue}\n")
    
    return problems

if __name__ == '__main__':
    main() 