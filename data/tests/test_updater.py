from data.data_updater import MarketDataUpdater, DB_CONFIG
import logging
import time

# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_data_updater():
    """测试数据更新器的各种场景"""
    
    # 创建更新器实例
    updater = MarketDataUpdater(DB_CONFIG)
    
    # 测试场景1：正常更新（不强制更新）
    print('\n=== 测试场景1：正常更新（不强制更新）===')
    print('预期行为：如果今天的数据已存在且完整，将跳过更新')
    updater.update_stock_data(symbols=['AAPL', 'MSFT'], force_update=False)
    
    # 等待一段时间，避免API调用过于频繁
    time.sleep(2)
    
    # 测试场景2：强制更新
    print('\n=== 测试场景2：强制更新 ===')
    print('预期行为：无论数据是否存在，都会重新获取今天的数据')
    updater.update_stock_data(symbols=['AAPL', 'MSFT'], force_update=True)
    
    # 等待一段时间
    time.sleep(2)
    
    # 测试场景3：检查数据完整性
    print('\n=== 测试场景3：检查数据完整性 ===')
    print('检查AAPL的今天数据是否完整')
    from datetime import datetime
    today_str = datetime.now().strftime('%Y-%m-%d')
    is_complete = updater.is_data_complete('AAPL', today_str)
    print(f'AAPL今天的数据完整性: {"完整" if is_complete else "不完整"}')
    
    # 测试场景4：市场状态检查
    print('\n=== 测试场景4：市场状态检查 ===')
    is_closed = updater.is_market_closed()
    print(f'当前市场状态: {"已收盘" if is_closed else "交易中"}')

if __name__ == '__main__':
    test_data_updater() 