import logging
import time
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from typing import Dict

from monitor.trading_monitor import TradingMonitor, AlertSystem
from monitor.strategy_monitor import StrategyMonitor
from config.trading_config import TradingConfig, EmailConfig, DatabaseConfig
from data.data_interface import YahooFinanceRealTimeSource, DataInterface
from strategy.combined_strategy import CombinedStrategy
from strategy.momentum_strategy import MomentumStrategy
from strategy.mean_reversion_strategy import MeanReversionStrategy
from strategy.bollinger_bands_strategy import BollingerBandsStrategy
from strategy.breakout_strategy import BreakoutStrategy

# 创建默认配置
default_config = TradingConfig(
    notification_settings={
        'email': True,
        'slack': False,
        'telegram': False
    },
    price_alert_threshold=0.05,
    loss_alert_threshold=0.05,
    profit_target=0.25,
    stop_loss=0.15,
    check_interval=60,
    update_interval=60,
    email_notifications=True,
    risk_thresholds={
        'volatility': 0.02,
        'concentration': 0.3,
        'var': 0.1
    },
    sector_specific_settings={
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
            'stop_loss': 0.10,
            'price_alert_threshold': 0.04,
            'volatility_threshold': 0.02
        },
        'automotive': {
            'stop_loss': 0.15,
            'price_alert_threshold': 0.05,
            'volatility_threshold': 0.03
        }
    },
    email=EmailConfig(
        sender_password="wlkp dbbz xpgk rkhy",
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        sender_email="kidzss@gmail.com",
        receiver_emails=["kidzss@gmail.com"]
    ),
    database=DatabaseConfig(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database="mose"
    )
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_portfolio_config():
    """加载持仓配置"""
    try:
        with open('monitor/configs/portfolio_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载持仓配置失败: {str(e)}")
        return {"positions": {}, "monitor_config": {}}

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

async def test_trading_monitor():
    """测试交易监控器"""
    try:
        logger.info("开始测试交易监控器...")
        
        # 加载持仓配置
        portfolio_config = load_portfolio_config()
        monitor_config = portfolio_config.get('monitor_config', {})
        
        # 获取需要监控的股票
        stocks_to_monitor = set(portfolio_config.get('positions', {}).keys())
        
        logger.info(f"需要监控的股票: {list(stocks_to_monitor)}")
        logger.info(f"监控配置: {monitor_config}")
        
        # 创建监控器实例
        monitor = TradingMonitor()
        
        # 创建实时数据源
        data_source = YahooFinanceRealTimeSource()
        
        # 获取实时数据
        real_time_data = await data_source.get_realtime_data(list(stocks_to_monitor))
        
        # 测试单个股票监控
        for stock in stocks_to_monitor:
            if stock in real_time_data:
                stock_data = real_time_data[stock]
                logger.info(f"股票 {stock} 实时数据: {stock_data.iloc[-1].to_dict()}")
            monitor.monitor_stock(stock)
            logger.info(f"完成股票 {stock} 的监控测试")
        
        logger.info("持仓股票监控测试完成")
        
    except Exception as e:
        logger.error(f"交易监控器测试失败: {str(e)}")

async def test_strategy_monitor():
    """测试策略监控器"""
    try:
        logger.info("开始测试策略监控器...")
        
        # 创建策略监控器实例，使用默认配置
        strategy_monitor = StrategyMonitor(config_path="config/strategy_config.json")
        
        # 测试监控功能
        strategy_monitor.start_monitoring()
        await asyncio.sleep(5)  # 运行5秒
        strategy_monitor.stop_monitoring()
        
        logger.info("策略监控器测试完成")
        
    except Exception as e:
        logger.error(f"策略监控器测试失败: {str(e)}")

async def test_notification_system():
    """测试通知系统"""
    try:
        logger.info("开始测试通知系统...")
        
        # 加载持仓配置
        portfolio_config = load_portfolio_config()
        
        # 创建警报系统实例
        alert_system = AlertSystem(default_config)
        
        # 创建数据接口
        data_interface = DataInterface()
        
        # 创建实时数据源
        data_source = YahooFinanceRealTimeSource()
        
        # 测试交易信号通知
        for stock, position in portfolio_config.get('positions', {}).items():
            try:
                # 获取实时数据
                real_time_data = await data_source.get_realtime_data([stock])
                if stock not in real_time_data:
                    logger.warning(f"无法获取 {stock} 的实时数据")
                    continue
                    
                # 获取最新的实时数据
                latest_data = real_time_data[stock].iloc[-1]
                current_price = latest_data['close']
                cost_basis = position.get('cost_basis', 0)
                
                # 计算价格变化百分比
                price_change = (current_price - cost_basis) / cost_basis if cost_basis > 0 else 0
                
                # 获取止损价格
                stop_loss = position.get('stop_loss', 0)
                if isinstance(stop_loss, float) and stop_loss < 1.0:  # 如果是百分比形式
                    stop_loss_price = cost_basis * (1 - stop_loss)
                else:  # 如果是具体价格
                    stop_loss_price = stop_loss
                
                # 获取历史数据用于计算技术指标
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)
                hist_data = data_interface.get_historical_data(stock, start_date, end_date)
                
                if hist_data is None or hist_data.empty:
                    logger.warning(f"无法获取 {stock} 的历史数据")
                    continue
                
                # 获取各个策略的信号
                strategy_signals = get_strategy_signals(hist_data)
                
                # 生成消息
                message = f"""
实时价格: {current_price:.2f}
成本价: {cost_basis:.2f}
变化: {price_change:.2%}
止损价格: {stop_loss_price:.2f}

策略信号:
- 动量策略: {alert_system._get_signal_explanation(strategy_signals['momentum'].iloc[-1]['signal'])}
- 均值回归: {alert_system._get_signal_explanation(strategy_signals['mean_reversion'].iloc[-1]['signal'])}
- 布林带策略: {alert_system._get_signal_explanation(strategy_signals['bollinger'].iloc[-1]['signal'])}
- 突破策略: {alert_system._get_signal_explanation(strategy_signals['breakout'].iloc[-1]['signal'])}
- 组合策略: {alert_system._get_signal_explanation(strategy_signals['combined'].iloc[-1]['signal'])}

技术指标分析:
- RSI: {alert_system._get_rsi_explanation(latest_data.get('RSI', 0))}
- MACD: {alert_system._get_macd_explanation(latest_data.get('MACD', 0), latest_data.get('Signal', 0))}
- 布林带: {alert_system._get_bollinger_explanation(current_price, latest_data.get('BB_upper', 0), latest_data.get('BB_lower', 0))}
- 20日均线: {alert_system._get_ma_explanation(current_price, latest_data.get('MA20', 0))}
- 成交量: {alert_system._get_volume_explanation(latest_data.get('volume', 0), latest_data.get('volume_ma20', 0))}
"""
                
                alert_system.send_alert(
                    stock=stock,
                    alert_type="strategy_update",
                    message=message,
                    price=current_price,
                    indicators={
                        # 基本指标
                        "cost_basis": cost_basis,
                        "price_change": price_change,
                        "stop_loss": stop_loss_price,
                        "weight": position.get('weight', 0),
                        
                        # 技术指标
                        "RSI": latest_data.get('RSI', 0),
                        "MACD": latest_data.get('MACD', 0),
                        "Signal": latest_data.get('Signal', 0),
                        "BB_upper": latest_data.get('BB_upper', 0),
                        "BB_lower": latest_data.get('BB_lower', 0),
                        "MA20": latest_data.get('MA20', 0),
                        "volume_ma20": latest_data.get('Volume_MA20', 0),
                        
                        # 策略信号
                        "momentum_signal": float(strategy_signals['momentum'].iloc[-1]['signal']),
                        "mean_reversion_signal": float(strategy_signals['mean_reversion'].iloc[-1]['signal']),
                        "bollinger_signal": float(strategy_signals['bollinger'].iloc[-1]['signal']),
                        "breakout_signal": float(strategy_signals['breakout'].iloc[-1]['signal']),
                        "combined_signal": float(strategy_signals['combined'].iloc[-1]['signal'])
                    }
                )
                logger.info(f"完成股票 {stock} 的通知测试")
            except Exception as e:
                logger.error(f"处理股票 {stock} 时出错: {str(e)}")
            
            await asyncio.sleep(1)  # 避免请求过于频繁
        
        logger.info("交易信号通知测试完成")
        
    except Exception as e:
        logger.error(f"通知系统测试失败: {str(e)}")

async def main():
    """主函数"""
    logger.info("开始监控系统测试...")
    
    # 运行各个测试
    await test_trading_monitor()
    await test_strategy_monitor()
    await test_notification_system()
    
    logger.info("所有测试完成")

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