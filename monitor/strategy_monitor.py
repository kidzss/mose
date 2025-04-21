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
from monitor.real_time_monitor import RealTimeMonitor
from data.data_interface import YahooFinanceRealTimeSource

logger = logging.getLogger(__name__)

class StrategyMonitor:
    """策略监控器，整合短期和长期策略"""
    
    def __init__(self, config_path: str = "config/strategy_config.json"):
        """
        初始化策略监控器
        
        Args:
            config_path: 策略配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 初始化策略
        self.combined_strategy = CombinedStrategy(parameters=self.config.get('combined_strategy', {}))
        self.tdi_strategy = TDIStrategy(params=self.config.get('tdi_strategy', {}))
        
        # 初始化监控器
        self.market_monitor = MarketMonitor()
        self.real_time_monitor = None  # 将在start_monitoring时初始化
        
        # 监控状态
        self.is_running = False
        self.monitor_thread = None
        
        # 初始化数据源
        self.data_source = YahooFinanceRealTimeSource()
        
        logger.info("策略监控器初始化完成")
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return {}
            
    def start_monitoring(self, symbols: List[str]) -> None:
        """
        启动监控
        
        Args:
            symbols: 要监控的股票代码列表
        """
        if self.is_running:
            logger.warning("监控已经在运行中")
            return
            
        try:
            # 初始化实时监控器
            self.real_time_monitor = RealTimeMonitor(
                symbols=symbols,
                config=self.config.get('monitor_settings', {})
            )
            
            # 启动监控线程
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # 启动实时监控
            self.real_time_monitor.start_monitoring()
            
            logger.info(f"开始监控 {len(symbols)} 只股票")
            
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
            if self.real_time_monitor:
                self.real_time_monitor.stop_monitoring()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
                
            logger.info("停止监控")
            
        except Exception as e:
            logger.error(f"停止监控时出错: {str(e)}")
            
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_running:
            try:
                # 获取当前时间
                now = datetime.now()
                
                # 短期策略监控（每分钟）
                self._monitor_short_term_signals()
                
                # 长期策略监控（每天收盘后）
                if now.hour == 15 and now.minute == 0:  # 假设15:00是收盘时间
                    self._monitor_long_term_signals()
                
                # 等待下一次检查
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"监控循环中出错: {str(e)}")
                time.sleep(60)
                
    def _monitor_short_term_signals(self) -> None:
        """监控短期信号"""
        try:
            if not self.real_time_monitor:
                return
                
            # 获取实时数据
            current_data = self.real_time_monitor.current_data
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
            if not self.real_time_monitor:
                return
                
            # 获取日线数据
            daily_data = self.real_time_monitor.get_historical_data()
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
            
            if signal > 0:
                message = f"短期买入信号 - {symbol} @ {price:.2f}"
                logger.info(message)
                # TODO: 发送通知
                
            elif signal < 0:
                message = f"短期卖出信号 - {symbol} @ {price:.2f}"
                logger.info(message)
                # TODO: 发送通知
                
        except Exception as e:
            logger.error(f"处理短期信号时出错: {str(e)}")
            
    def _process_long_term_signal(self, symbol: str, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        """处理长期信号"""
        try:
            signal = signals['signal'].iloc[-1]
            price = data['close'].iloc[-1]
            
            if signal > 0:
                message = f"长期买入信号 - {symbol} @ {price:.2f}"
                logger.info(message)
                # TODO: 发送通知
                
            elif signal < 0:
                message = f"长期卖出信号 - {symbol} @ {price:.2f}"
                logger.info(message)
                # TODO: 发送通知
                
        except Exception as e:
            logger.error(f"处理长期信号时出错: {str(e)}")
            
    def get_monitoring_status(self) -> Dict:
        """获取监控状态"""
        return {
            'is_running': self.is_running,
            'real_time_monitor_status': self.real_time_monitor.get_monitoring_status() if self.real_time_monitor else None,
            'market_state': self.market_monitor._analyze_market_state({}) if self.market_monitor else None
        } 