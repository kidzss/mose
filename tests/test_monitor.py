import logging
import time
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from typing import Dict
import yfinance as yf
import traceback

from monitor.trading_monitor import TradingMonitor, AlertSystem
from monitor.strategy_monitor import StrategyMonitor, default_config
from config.trading_config import TradingConfig, EmailConfig, DatabaseConfig
from data.data_interface import YahooFinanceRealTimeSource, DataInterface
from strategy.combined_strategy import CombinedStrategy
from strategy.momentum_strategy import MomentumStrategy
from strategy.mean_reversion_strategy import MeanReversionStrategy
from strategy.bollinger_bands_strategy import BollingerBandsStrategy
from strategy.breakout_strategy import BreakoutStrategy
from monitor.portfolio_monitor import PortfolioMonitor
from monitor.market_monitor import MarketMonitor
from monitor.data_fetcher import DataFetcher
from monitor.technical_analysis import TechnicalAnalysis
from monitor.alert_system import AlertSystem
from monitor.report_generator import ReportGenerator
from monitor.notification_system import NotificationSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def get_test_data(symbol: str = None):
    """获取测试数据"""
    # 创建数据源
    data_source = YahooFinanceRealTimeSource()
    
    if symbol:
        # 获取单个股票的实时数据
        data = await data_source.get_realtime_data([symbol])
        if symbol in data:
            return data[symbol].iloc[-1]
        return None
    else:
        # 获取所有测试股票的数据
        test_stocks = ['GOOG', 'TSLA', 'AMD', 'NVDA', 'PFE', 'MSFT', 'TMDX']
        return await data_source.get_realtime_data(test_stocks)

# 初始化配置
trading_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'kidzss@gmail.com',
    'sender_password': 'wlkp dbbz xpgk rkhy',
    'recipient_email': 'kidzss@gmail.com',
    'notification_threshold': {
        'price_change': 0.05,
        'volume_change': 2.0,
        'market_volatility': 0.02,
        'risk_level': 'high'
    }
}

def load_portfolio_config():
    """加载持仓配置"""
    with open('monitor/configs/portfolio_config.json', 'r') as f:
        return json.load(f)

def get_strategy_signals(data: pd.DataFrame) -> Dict:
    """获取各个策略的信号"""
    signals = {}
    
    try:
        # 初始化策略
        momentum = MomentumStrategy()
        mean_reversion = MeanReversionStrategy()
        bollinger = BollingerBandsStrategy()
        breakout = BreakoutStrategy()
        combined = CombinedStrategy()
        
        # 计算各个策略的信号
        momentum_data = momentum.calculate_indicators(data.copy())
        mean_reversion_data = mean_reversion.calculate_indicators(data.copy())
        bollinger_data = bollinger.calculate_indicators(data.copy())
        breakout_data = breakout.calculate_indicators(data.copy())
        combined_data = combined.calculate_indicators(data.copy())
        
        signals['momentum'] = momentum.generate_signals(momentum_data)
        signals['mean_reversion'] = mean_reversion.generate_signals(mean_reversion_data)
        signals['bollinger'] = bollinger.generate_signals(bollinger_data)
        signals['breakout'] = pd.DataFrame({'signal': breakout.generate_signals(breakout_data)})
        signals['combined'] = combined.generate_signals(combined_data)
        
        return signals
    except Exception as e:
        logger.error(f"获取策略信号时出错: {str(e)}")
        # 返回空信号
        empty_signal = pd.DataFrame({'signal': [0]})
        return {
            'momentum': empty_signal,
            'mean_reversion': empty_signal,
            'bollinger': empty_signal,
            'breakout': empty_signal,
            'combined': empty_signal
        }

async def test_monitor_system(trading_config):
    """测试监控系统"""
    print("开始监控系统测试...")
    
    # 初始化监控系统
    monitor = TradingMonitor()
    monitor.config = trading_config
    
    # 获取测试数据
    test_stocks = ['GOOG', 'TSLA', 'AMD', 'NVDA', 'PFE', 'MSFT', 'TMDX']
    for symbol in test_stocks:
        # 只传递股票代码
        monitor.monitor_stock(symbol)
    
    print("监控系统测试完成")

def get_default_config():
    """获取默认配置"""
    return {
        'price_alert_threshold': 0.05,
        'loss_alert_threshold': 0.05,
        'profit_target': 0.25,
        'stop_loss': 0.15,
        'check_interval': 60,
        'email_notifications': True,
        'update_interval': 60,
        'notification_threshold': 0.05,  # Added notification threshold
        'risk_thresholds': {
            'volatility': 0.02,
            'concentration': 0.3,
            'var': 0.1
        },
        'notification_settings': {
            'email': True,
            'slack': False,
            'telegram': False
        },
        'sector_specific_settings': {
            'semiconductor': {
                'stop_loss': 0.08,
                'price_alert_threshold': 0.03,
                'volatility_threshold': 0.03
            },
            'tech': {
                'stop_loss': 0.12,
                'price_alert_threshold': 0.04,
                'volatility_threshold': 0.025
            },
            'healthcare': {
                'stop_loss': 0.1,
                'price_alert_threshold': 0.04,
                'volatility_threshold': 0.02
            },
            'automotive': {
                'stop_loss': 0.15,
                'price_alert_threshold': 0.05,
                'volatility_threshold': 0.03
            }
        }
    }

async def test_strategy_monitor(trading_config):
    """测试策略监控器"""
    print("开始策略监控测试...")
    try:
        # 初始化策略监控器
        strategy_monitor = StrategyMonitor()
        
        # 设置配置
        config = {
            'notification_threshold': {
                'price_change': 0.05,
                'volume_change': 2.0,
                'market_volatility': 0.02,
                'risk_level': 'high'
            }
        }
        config.update(trading_config)
        strategy_monitor.config = config
        
        # 获取测试数据
        test_data = await get_test_data()
        
        # 运行策略监控
        await strategy_monitor.monitor_strategies(test_data)
        print("策略监控测试完成")
        return True
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        traceback.print_exc()
        return False

async def test_notification_system(trading_config):
    """测试通知系统"""
    print("开始通知系统测试...")
    
    # 初始化通知系统
    alert_system = AlertSystem()
    alert_system.config = trading_config
    
    # 获取测试数据
    test_stocks = ['GOOG', 'TSLA', 'AMD', 'NVDA', 'PFE', 'MSFT', 'TMDX']
    for symbol in test_stocks:
        data = await get_test_data(symbol)
        if data is not None:
            # 将数据转换为DataFrame格式
            df = pd.DataFrame({
                'close': [data.get('close', 0)],
                'volume': [data.get('volume', 0)],
                'open': [data.get('open', 0)],
                'high': [data.get('high', 0)],
                'low': [data.get('low', 0)]
            })
            
            # 使用 generate_alerts 方法
            position = {
                'cost_basis': data.get('close', 0),
                'weight': 0.1
            }
            alerts = alert_system.generate_alerts(symbol, position, df)
            if alerts:
                for alert in alerts:
                    alert_system.send_alert(
                        stock=symbol,
                        alert_type=alert['type'],
                        message=alert['message'],
                        price=data.get('close', 0),
                        indicators={
                            'cost_basis': position['cost_basis'],
                            'weight': position['weight'],
                            'RSI': data.get('RSI', 0),
                            'MACD': data.get('MACD', 0),
                            'MACD_Signal': data.get('MACD_Signal', 0),
                            'volume': data.get('volume', 0),
                            'volume_ma20': data.get('volume_ma20', 0)
                        }
                    )
    
    print("通知系统测试完成")

async def main():
    """主函数"""
    # 加载配置
    portfolio_config = load_portfolio_config()
    
    # 合并配置
    trading_config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': 'kidzss@gmail.com',
        'sender_password': 'wlkp dbbz xpgk rkhy',
        'recipient_email': 'kidzss@gmail.com',
        'notification_threshold': {
            'price_change': 0.05,
            'volume_change': 2.0,
            'market_volatility': 0.02,
            'risk_level': 'high'
        }
    }
    
    # 合并持仓配置
    trading_config.update(portfolio_config)
    
    # 运行测试
    await test_monitor_system(trading_config)
    await test_notification_system(trading_config)
    await test_strategy_monitor(trading_config)
    
    print("所有测试完成")

if __name__ == "__main__":
    asyncio.run(main())

def _get_signal_explanation(self, signal: float) -> str:
    """获取策略信号的解释"""
    if signal > 0:
        return f"买入信号 (强度: {signal:.2f})"
    elif signal < 0:
        return f"卖出信号 (强度: {abs(signal):.2f})"
    else:
        return "观望信号"

def _get_rsi_explanation(self, rsi: float) -> str:
    """获取RSI指标的解释"""
    if rsi > 70:
        return f"超买区域 ({rsi:.2f})，可能面临回调风险"
    elif rsi < 30:
        return f"超卖区域 ({rsi:.2f})，可能存在反弹机会"
    else:
        return f"中性区域 ({rsi:.2f})，市场相对平衡"

def _get_macd_explanation(self, macd: float, signal: float) -> str:
    """获取MACD指标的解释"""
    if macd > signal:
        return f"金叉形态，上涨动能增强"
    elif macd < signal:
        return f"死叉形态，下跌动能增强"
    else:
        return f"趋势不明朗，等待方向确认"

def _get_bollinger_explanation(self, price: float, upper: float, lower: float) -> str:
    """获取布林带指标的解释"""
    if price > upper:
        return f"价格突破上轨，可能超买"
    elif price < lower:
        return f"价格跌破下轨，可能超卖"
    else:
        return f"价格在通道内运行，市场相对稳定"

def _get_ma_explanation(self, price: float, ma20: float) -> str:
    """获取均线指标的解释"""
    if price > ma20:
        return f"价格在20日均线上方，短期趋势向上"
    elif price < ma20:
        return f"价格在20日均线下方，短期趋势向下"
    else:
        return f"价格接近20日均线，趋势不明朗"

def _get_volume_explanation(self, volume: float, volume_ma20: float) -> str:
    """获取成交量指标的解释"""
    if volume > volume_ma20 * 1.5:
        return f"成交量显著放大，市场活跃度增加"
    elif volume < volume_ma20 * 0.5:
        return f"成交量萎缩，市场活跃度降低"
    else:
        return f"成交量正常，市场活跃度适中"

def test_monitor():
    # 初始化数据获取器
    data_fetcher = DataFetcher()
    
    # 测试股票列表
    symbols = ['AMD', 'NVDA', 'PFE', 'MSFT', 'TMDX']
    
    # 设置时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("开始测试数据获取...")
    
    for symbol in symbols:
        print(f"\n测试 {symbol} 的数据获取:")
        try:
            # 获取数据
            data = data_fetcher.get_historical_data(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if not data.empty:
                print(f"数据形状: {data.shape}")
                print(f"列名: {data.columns.tolist()}")
                print(f"前5行数据:\n{data.head()}")
            else:
                print(f"未获取到 {symbol} 的数据")
            
        except Exception as e:
            print(f"获取 {symbol} 数据时出错: {str(e)}")

if __name__ == "__main__":
    test_monitor() 