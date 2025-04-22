import logging
import time
import asyncio
from datetime import datetime, timedelta
import pytz
from typing import Dict, List
import json
import pandas as pd

from monitor.trading_monitor import TradingMonitor, AlertSystem
from monitor.strategy_monitor import StrategyMonitor
from config.trading_config import TradingConfig, EmailConfig, DatabaseConfig
from data.data_interface import YahooFinanceRealTimeSource
from test_monitor import default_config  # 导入默认配置

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketMonitor:
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        
    def is_market_open(self) -> bool:
        """检查市场是否开盘"""
        now = datetime.now(self.ny_tz)
        
        # 检查是否是工作日
        if now.weekday() >= 5:  # 5是周六，6是周日
            return False
            
        # 检查是否在交易时间内（9:30 - 16:00）
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close

def calculate_stop_loss_price(cost_basis: float, stop_loss: float) -> float:
    """计算止损价格
    
    Args:
        cost_basis: 成本价
        stop_loss: 止损设置（可以是百分比或具体价格）
        
    Returns:
        float: 止损价格
    """
    if isinstance(stop_loss, float) and stop_loss < 1.0:  # 如果是百分比形式
        return cost_basis * (1 - stop_loss)
    else:  # 如果是具体价格
        return stop_loss

async def send_test_email():
    """发送测试邮件验证止损价格计算"""
    try:
        # 初始化组件
        data_source = YahooFinanceRealTimeSource()
        alert_system = AlertSystem(default_config)
        
        # 加载持仓配置
        with open('monitor/configs/portfolio_config.json', 'r') as f:
            portfolio_config = json.load(f)
            
        # 获取GOOG的实时数据
        real_time_data = await data_source.get_realtime_data(['GOOG'])
        
        if 'GOOG' in real_time_data:
            stock_data = real_time_data['GOOG']
            latest_data = stock_data.iloc[-1]
            position = portfolio_config['positions']['GOOG']
            
            # 计算止损价格
            stop_loss_price = calculate_stop_loss_price(
                position['cost_basis'],
                position['stop_loss']
            )
            
            # 计算价格变化百分比
            price_change = ((latest_data['close'] - position['cost_basis']) / position['cost_basis'] * 100)
            
            message = "止损价格测试报告：\n\n"
            message += f"股票代码: GOOG\n"
            message += f"警报类型: stop_loss_test\n\n"
            message += f"价格信息:\n"
            message += f"- 当前价格: {latest_data['close']:.2f}\n"
            message += f"- 成本价格: {position['cost_basis']:.2f}\n"
            message += f"- 价格变化: {price_change:.2f}%\n\n"
            message += f"风险控制:\n"
            message += f"- 止损设置: {position['stop_loss']}\n"
            message += f"- 止损价格: {stop_loss_price:.2f}\n"
            message += f"- 仓位权重: {position['weight']*100:.2f}%\n"
            message += f"- 持仓数量: {position['shares']}\n"
            message += f"- 持仓市值: {position['shares'] * latest_data['close']:.2f}\n"
            
            # 发送测试邮件
            alert_system.send_alert(
                stock="GOOG",
                alert_type="stop_loss_test",
                message=message,
                price=latest_data['close'],
                indicators={}
            )
            
            logger.info("测试邮件已发送")
        else:
            logger.error("无法获取GOOG的实时数据")
        
    except Exception as e:
        logger.error(f"发送测试邮件时出错: {str(e)}")
        raise

async def main():
    """主函数"""
    try:
        # 发送测试邮件
        await send_test_email()
        
        # 初始化组件
        market_monitor = MarketMonitor()
        data_source = YahooFinanceRealTimeSource()
        alert_system = AlertSystem(default_config)
        
        # 加载持仓配置
        with open('monitor/configs/portfolio_config.json', 'r') as f:
            portfolio_config = json.load(f)
            
        monitored_stocks = list(portfolio_config['positions'].keys())
        
        logger.info(f"开始监控股票: {', '.join(monitored_stocks)}")
        
        last_regular_update = datetime.now()
        last_non_trading_update = datetime.now()
        
        while True:
            try:
                # 获取当前时间
                now = datetime.now()
                is_market_open = market_monitor.is_market_open()
                
                # 获取实时数据
                real_time_data = await data_source.get_realtime_data(monitored_stocks)
                
                # 检查每个股票
                for stock in monitored_stocks:
                    if stock in real_time_data:
                        stock_data = real_time_data[stock]
                        latest_data = stock_data.iloc[-1]
                        position = portfolio_config['positions'][stock]
                        
                        # 计算止损价格
                        stop_loss_price = calculate_stop_loss_price(
                            position['cost_basis'],
                            position['stop_loss']
                        )
                        
                        # 检查是否有交易信号
                        if is_market_open:
                            # 市场开盘时，检查交易信号
                            signals = check_trading_signals(stock, latest_data)
                            if signals:
                                # 有信号时立即发送通知
                                alert_system.send_alert(
                                    stock=stock,
                                    alert_type="trading_signal",
                                    message=generate_signal_message(signals),
                                    price=latest_data['close'],
                                    indicators={
                                        'cost_basis': position['cost_basis'],
                                        'stop_loss_price': stop_loss_price,
                                        'weight': position['weight'],
                                        **latest_data.to_dict()
                                    }
                                )
                
                # 处理常规更新
                if is_market_open:
                    # 市场开盘时，每10-15分钟发送一次常规更新
                    if (now - last_regular_update).total_seconds() >= 15 * 60:
                        send_regular_update(alert_system, monitored_stocks, real_time_data, portfolio_config)
                        last_regular_update = now
                else:
                    # 市场闭市时，每4小时发送一次更新
                    if (now - last_non_trading_update).total_seconds() >= 4 * 60 * 60:
                        send_regular_update(alert_system, monitored_stocks, real_time_data, portfolio_config)
                        last_non_trading_update = now
                
                # 等待下一次检查
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"监控循环中出错: {str(e)}")
                await asyncio.sleep(60)
                
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise

def check_trading_signals(stock: str, data: pd.Series) -> Dict:
    """检查交易信号"""
    signals = {}
    
    # 检查价格突破
    if data['close'] > data['BB_upper']:
        signals['price_breakout'] = 'upper'
    elif data['close'] < data['BB_lower']:
        signals['price_breakout'] = 'lower'
        
    # 检查RSI
    if data['RSI'] > 70:
        signals['rsi'] = 'overbought'
    elif data['RSI'] < 30:
        signals['rsi'] = 'oversold'
        
    # 检查MACD
    if data['MACD'] > data['Signal']:
        signals['macd'] = 'bullish'
    elif data['MACD'] < data['Signal']:
        signals['macd'] = 'bearish'
        
    return signals

def generate_signal_message(signals: Dict) -> str:
    """生成信号消息"""
    message = "检测到以下交易信号：\n"
    
    for signal_type, signal_value in signals.items():
        if signal_type == 'price_breakout':
            message += f"- 价格突破布林带{'上轨' if signal_value == 'upper' else '下轨'}\n"
        elif signal_type == 'rsi':
            message += f"- RSI指标{'超买' if signal_value == 'overbought' else '超卖'}\n"
        elif signal_type == 'macd':
            message += f"- MACD指标{'看涨' if signal_value == 'bullish' else '看跌'}\n"
            
    return message

def send_regular_update(alert_system: AlertSystem, stocks: List[str], data: Dict[str, pd.DataFrame], portfolio_config: Dict):
    """发送常规更新"""
    message = "市场状态更新：\n\n"
    
    for stock in stocks:
        if stock in data:
            stock_data = data[stock]
            latest_data = stock_data.iloc[-1]
            position = portfolio_config['positions'][stock]
            
            # 计算止损价格
            stop_loss_price = calculate_stop_loss_price(
                position['cost_basis'],
                position['stop_loss']
            )
            
            message += f"{stock}:\n"
            message += f"- 当前价格: {latest_data['close']:.2f}\n"
            message += f"- 成本价格: {position['cost_basis']:.2f}\n"
            message += f"- 价格变化: {((latest_data['close'] - position['cost_basis']) / position['cost_basis'] * 100):.2f}%\n"
            message += f"- 止损价格: {stop_loss_price:.2f}\n"
            message += f"- 仓位权重: {position['weight']*100:.2f}%\n"
            message += f"- 成交量: {latest_data['volume']}\n"
            message += f"- RSI: {latest_data['RSI']:.2f}\n"
            message += f"- MACD: {latest_data['MACD']:.2f}\n\n"
            
    alert_system.send_alert(
        stock="MARKET_UPDATE",
        alert_type="regular_update",
        message=message,
        price=0,
        indicators={}
    )

if __name__ == "__main__":
    asyncio.run(main()) 