import sys
import os
import logging
from datetime import datetime
import pandas as pd
from monitor.alert_system import AlertSystem
from monitor.data_fetcher import DataFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_alert.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_trading_alert():
    """测试交易警报"""
    try:
        # 初始化数据获取器
        data_fetcher = DataFetcher()
        
        # 获取测试股票数据
        symbol = "GOOG"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = data_fetcher.get_historical_data(symbol, start_date, end_date)
        
        if data.empty:
            logger.error(f"无法获取 {symbol} 的数据")
            return
            
        # 初始化警报系统
        alert_system = AlertSystem()
        
        # 生成测试警报
        current_price = data['Close'].iloc[-1]
        indicators = {
            'cost_basis': current_price * 0.9,  # 假设成本价比当前价格低10%
            'price_change': -0.05,  # 假设价格下跌5%
            'RSI': 75,  # 假设RSI超买
            'MACD': 0.5,
            'MACD_Signal': 0.3,
            'volume': data['Volume'].iloc[-1],
            'volume_ma20': data['Volume'].rolling(20).mean().iloc[-1],
            'stop_loss': 0.15,
            'weight': 0.25
        }
        
        # 发送警报
        alert_system.send_alert(
            stock=symbol,
            alert_type="技术指标预警",
            message=f"{symbol} RSI超买，建议减仓",
            price=current_price,
            indicators=indicators
        )
        
        logger.info("测试警报已发送")
        
    except Exception as e:
        logger.error(f"测试警报时出错: {e}")

if __name__ == "__main__":
    test_trading_alert() 