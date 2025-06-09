from data_updater import MarketDataUpdater, DB_CONFIG

def update_missing_stocks():
    """更新缺失交易日数据的股票"""
    # 创建市场数据更新器
    updater = MarketDataUpdater(DB_CONFIG)
    
    # 指定需要更新的股票列表（缺失数据的股票）
    missing_stocks = [
        "ASML", "AZN", "CCEP", "ILMN", "MELI", "MRVL", "WDAY", 
        "TEAM", "TTD", "MDB", "ZS", "PDD", "TMDX", "DDOG", "ARM"
    ]
    
    # 强制更新数据（忽略最后更新时间）
    report = updater.update_stock_data(symbols=missing_stocks, force_update=True)
    
    # 输出更新报告
    print("\n=== 股票数据更新报告 ===")
    print(f"总计: {report['total']} 只股票")
    print(f"更新成功: {report['updated']} 只")
    print(f"跳过: {report['skipped']} 只")
    print(f"失败: {report['failed']} 只")
    
    print("\n详细信息:")
    for symbol, status in report['details'].items():
        print(f"{symbol}: {status}")

if __name__ == "__main__":
    update_missing_stocks() 