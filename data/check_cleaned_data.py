from data.tests.check_data_quality import DataQualityChecker, DB_CONFIG
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_quality_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """运行数据质量检查"""
    checker = DataQualityChecker(DB_CONFIG)
    symbols = checker.get_all_symbols()
    
    logger.info(f"\n=== 开始检查 {len(symbols)} 只股票的数据质量 ===\n")
    
    # 存储问题数据
    problems = {}
    total_records = 0
    total_trading_days = 0
    
    for symbol in symbols:
        try:
            symbol_problems = []
            
            # 1. 检查缺失日期
            missing_dates = checker.check_missing_dates(symbol)
            if missing_dates:
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
            
            # 5. 获取数据统计
            first_date = checker.get_stock_first_date(symbol)
            if first_date:
                total_records += 1
                total_trading_days += len(checker.nyse.valid_days(start_date=first_date, end_date=datetime.now()))
            
            if symbol_problems:
                problems[symbol] = symbol_problems
                logger.info(f"\n股票 {symbol} 存在以下问题：")
                for problem in symbol_problems:
                    logger.info(f"  - {problem}")
                    
        except Exception as e:
            logger.error(f"检查股票 {symbol} 时出错: {str(e)}")
            continue
    
    # 输出总结报告
    logger.info(f"\n=== 数据质量检查完成 ===")
    logger.info(f"检查了 {len(symbols)} 只股票")
    logger.info(f"发现 {len(problems)} 只股票存在问题")
    logger.info(f"总记录数: {total_records}")
    logger.info(f"总交易日数: {total_trading_days}")
    
    # 输出问题股票列表
    if problems:
        logger.info("\n问题股票列表:")
        for symbol, symbol_problems in problems.items():
            logger.info(f"\n{symbol}:")
            for problem in symbol_problems:
                logger.info(f"  - {problem}")

if __name__ == "__main__":
    main() 