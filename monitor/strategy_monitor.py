import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import json
from datetime import datetime, timedelta
import threading
import time

from strategy.combined_strategy import CombinedStrategy
from strategy.tdi_strategy import TDIStrategy
from monitor.market_monitor import MarketMonitor
from data.data_interface import YahooFinanceRealTimeSource
from monitor.notification_manager import NotificationManager

logger = logging.getLogger(__name__)

class StrategyMonitor:
    """策略监控器，整合短期和长期策略"""
    
    def __init__(self, config_path: str = "config/strategy_config.json", portfolio_path: str = "config/portfolio_config.json"):
        """
        初始化策略监控器
        
        Args:
            config_path: 策略配置文件路径
            portfolio_path: 持仓和观察股票配置文件路径
        """
        self.config = self._load_config(config_path)
        self.portfolio_config = self._load_portfolio_config(portfolio_path)
        
        # 初始化策略
        self.combined_strategy = CombinedStrategy(parameters=self.config.get('combined_strategy', {}))
        self.tdi_strategy = TDIStrategy(params=self.config.get('tdi_strategy', {}))
        
        # 初始化监控器
        self.market_monitor = MarketMonitor()
        
        # 监控状态
        self.is_running = False
        self.monitor_thread = None
        
        # 初始化数据源
        self.data_source = YahooFinanceRealTimeSource()
        
        # Initialize notification manager with default config
        self.notification_manager = NotificationManager(default_config)
        
        # Initialize watchlist with stocks from config
        for stock in self.config.get('monitored_stocks', []):
            self._process_stock(stock)
        
        logger.info("策略监控器初始化完成")
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return {}
            
    def _load_portfolio_config(self, portfolio_path: str) -> Dict:
        """加载持仓和观察股票配置"""
        try:
            with open(portfolio_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载持仓配置时出错: {str(e)}")
            return {'holdings': [], 'watchlist': []}
            
    def get_monitoring_symbols(self) -> List[str]:
        """获取需要监控的股票代码列表"""
        # 合并持仓股和观察股票
        holdings_symbols = [holding['symbol'] for holding in self.portfolio_config.get('holdings', [])]
        watchlist_symbols = self.portfolio_config.get('watchlist', [])
        
        # 去重
        return list(set(holdings_symbols + watchlist_symbols))
        
    def start_monitoring(self) -> None:
        """
        启动监控
        """
        if self.is_running:
            logger.warning("监控已经在运行中")
            return
            
        try:
            # 获取需要监控的股票
            symbols = self.get_monitoring_symbols()
            
            # 启动监控线程
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info(f"开始监控 {len(symbols)} 只股票")
            logger.info(f"持仓股: {[holding['symbol'] for holding in self.portfolio_config.get('holdings', [])]}")
            logger.info(f"观察股票: {self.portfolio_config.get('watchlist', [])}")
            
        except Exception as e:
            logger.error(f"启动监控时出错: {str(e)}")
            self.is_running = False
            
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self.is_running:
            logger.warning("监控未在运行")
            return
            
        try:
            self.is_running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
                
            logger.info("停止监控")
            
        except Exception as e:
            logger.error(f"停止监控时出错: {str(e)}")
            
    def _monitoring_loop(self) -> None:
        """监控循环"""
        last_regular_update = datetime.now()
        last_non_trading_update = datetime.now()
        
        while self.is_running:
            try:
                # 获取当前时间
                now = datetime.now()
                is_trading_hours = self._is_trading_hours(now)
                
                # 获取持仓和观察股票状态
                portfolio_status = self.get_portfolio_status()
                watchlist_status = self.get_watchlist_status()
                
                # 短期策略监控（每分钟）
                self._monitor_short_term_signals()
                
                # 长期策略监控（每天收盘后）
                if now.hour == 15 and now.minute == 0:  # 假设15:00是收盘时间
                    self._monitor_long_term_signals()
                
                # 处理常规更新
                if is_trading_hours:
                    # 交易时间：每10-15分钟发送一次常规更新
                    if (now - last_regular_update).total_seconds() >= 15 * 60:
                        self.notification_manager.send_regular_update(portfolio_status, watchlist_status)
                        last_regular_update = now
                else:
                    # 非交易时间：每4小时发送一次更新
                    if (now - last_non_trading_update).total_seconds() >= 4 * 60 * 60:
                        self.notification_manager.send_regular_update(portfolio_status, watchlist_status)
                        last_non_trading_update = now
                
                # 等待下一次检查
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"监控循环中出错: {str(e)}")
                time.sleep(60)
                
    def _is_trading_hours(self, dt: datetime) -> bool:
        """
        检查当前是否在交易时间内
        
        参数:
            dt: 当前时间
        """
        # 检查是否为工作日
        if dt.weekday() >= 5:  # 5和6是周六和周日
            return False
            
        # 检查是否在交易时间内（假设交易时间为9:30-16:00）
        current_time = dt.time()
        market_open = datetime.strptime("09:30:00", "%H:%M:%S").time()
        market_close = datetime.strptime("16:00:00", "%H:%M:%S").time()
        
        return market_open <= current_time <= market_close
        
    def _monitor_short_term_signals(self) -> None:
        """监控短期信号"""
        try:
            # 获取实时数据
            current_data = self.data_source.get_current_data()
            if not current_data:
                return
                
            for symbol, data in current_data.items():
                # 运行TDI策略
                signals = self.tdi_strategy.generate_signals(data)
                
                # 检查是否有交易信号
                if signals['signal'].iloc[-1] != 0:
                    self._process_short_term_signal(symbol, data, signals)
                    
        except Exception as e:
            logger.error(f"监控短期信号时出错: {str(e)}")
            
    def _monitor_long_term_signals(self) -> None:
        """监控长期信号"""
        try:
            # 获取日线数据
            daily_data = self.data_source.get_historical_data()
            if not daily_data:
                return
                
            for symbol, data in daily_data.items():
                # 运行组合策略
                signals = self.combined_strategy.generate_signals(data)
                
                # 检查是否有交易信号
                if signals['signal'].iloc[-1] != 0:
                    self._process_long_term_signal(symbol, data, signals)
                    
        except Exception as e:
            logger.error(f"监控长期信号时出错: {str(e)}")
            
    def _process_short_term_signal(self, symbol: str, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        """处理短期信号"""
        try:
            signal = signals['signal'].iloc[-1]
            price = data['close'].iloc[-1]
            confidence = signals.get('confidence', {}).get(symbol, None)
            
            # 检查是否是持仓股
            is_holding = any(holding['symbol'] == symbol for holding in self.portfolio_config.get('holdings', []))
            
            if signal > 0:
                message = f"短期买入信号 - {symbol} @ {price:.2f}"
                if is_holding:
                    message += " (持仓股)"
                logger.info(message)
                self.notification_manager.send_trading_signal(
                    stock=symbol,
                    signal_type="short_term_buy",
                    price=price,
                    indicators=signals.to_dict(),
                    confidence=confidence,
                    time_frame="short"
                )
                
            elif signal < 0:
                message = f"短期卖出信号 - {symbol} @ {price:.2f}"
                if is_holding:
                    message += " (持仓股)"
                logger.info(message)
                self.notification_manager.send_trading_signal(
                    stock=symbol,
                    signal_type="short_term_sell",
                    price=price,
                    indicators=signals.to_dict(),
                    confidence=confidence,
                    time_frame="short"
                )
                
        except Exception as e:
            logger.error(f"处理短期信号时出错: {str(e)}")
            
    def _process_long_term_signal(self, symbol: str, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        """处理长期信号"""
        try:
            signal = signals['signal'].iloc[-1]
            price = data['close'].iloc[-1]
            confidence = signals.get('confidence', {}).get(symbol, None)
            
            # 检查是否是持仓股
            is_holding = any(holding['symbol'] == symbol for holding in self.portfolio_config.get('holdings', []))
            
            if signal > 0:
                message = f"长期买入信号 - {symbol} @ {price:.2f}"
                if is_holding:
                    message += " (持仓股)"
                logger.info(message)
                self.notification_manager.send_trading_signal(
                    stock=symbol,
                    signal_type="long_term_buy",
                    price=price,
                    indicators=signals.to_dict(),
                    confidence=confidence,
                    time_frame="long"
                )
                
            elif signal < 0:
                message = f"长期卖出信号 - {symbol} @ {price:.2f}"
                if is_holding:
                    message += " (持仓股)"
                logger.info(message)
                self.notification_manager.send_trading_signal(
                    stock=symbol,
                    signal_type="long_term_sell",
                    price=price,
                    indicators=signals.to_dict(),
                    confidence=confidence,
                    time_frame="long"
                )
                
        except Exception as e:
            logger.error(f"处理长期信号时出错: {str(e)}")
            
    def get_monitoring_status(self) -> Dict:
        """获取监控状态"""
        return {
            'is_running': self.is_running,
            'market_state': self.market_monitor._analyze_market_state({}) if self.market_monitor else None
        }
        
    def get_portfolio_status(self) -> Dict:
        """获取当前持仓状态"""
        try:
            current_data = self.data_source.get_current_data()
            if not current_data:
                return {}
                
            portfolio_status = {
                'holdings': [],
                'total_value': 0.0,
                'total_cost': 0.0,
                'total_pnl': 0.0
            }
            
            for holding in self.portfolio_config.get('holdings', []):
                symbol = holding['symbol']
                shares = holding['shares']
                avg_cost = holding['avg_cost']
                
                if symbol in current_data:
                    current_price = current_data[symbol]['close'].iloc[-1]
                    current_value = shares * current_price
                    cost_basis = shares * avg_cost
                    pnl = current_value - cost_basis
                    
                    portfolio_status['holdings'].append({
                        'symbol': symbol,
                        'shares': shares,
                        'avg_cost': avg_cost,
                        'current_price': current_price,
                        'current_value': current_value,
                        'pnl': pnl,
                        'pnl_percent': (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                    })
                    
                    portfolio_status['total_value'] += current_value
                    portfolio_status['total_cost'] += cost_basis
                    portfolio_status['total_pnl'] += pnl
                    
            return portfolio_status
            
        except Exception as e:
            logger.error(f"获取持仓状态时出错: {str(e)}")
            return {}
            
    def get_watchlist_status(self) -> Dict:
        """获取观察股票状态"""
        try:
            current_data = self.data_source.get_current_data()
            if not current_data:
                return {}
                
            watchlist_status = []
            
            for symbol in self.portfolio_config.get('watchlist', []):
                if symbol in current_data:
                    current_price = current_data[symbol]['close'].iloc[-1]
                    # 获取信号
                    signals = self.combined_strategy.generate_signals(current_data[symbol])
                    current_signal = signals['signal'].iloc[-1]
                    signal_type = signals['signal_type'].iloc[-1] if 'signal_type' in signals else ''
                    
                    # 解析信号类型和时间周期
                    time_frame = 'N/A'
                    if signal_type:
                        parts = signal_type.split('_')
                        if len(parts) >= 2:
                            time_frame = parts[0]  # 获取时间周期
                    
                    # 获取置信度
                    confidence = signals.get('confidence', {}).get(symbol, 'N/A')
                    
                    watchlist_status.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'signal': current_signal,
                        'time_frame': time_frame,
                        'confidence': confidence
                    })
                    
            return {'watchlist': watchlist_status}
            
        except Exception as e:
            logger.error(f"获取观察股票状态时出错: {str(e)}")
            return {} 