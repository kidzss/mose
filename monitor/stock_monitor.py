import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any
import logging
import asyncio
from data.data_interface import DataInterface, RealTimeDataSource
from data.data_validator import DataValidator
from monitor.notification_manager import NotificationManager
import json
import pytz
import time as time_module
from data.data_loader import DataLoader
from monitor.strategy_manager import StrategyManager
import threading
import os
from pathlib import Path
import time

from .data_fetcher import DataFetcher
from .report_generator import ReportGenerator
from .market_monitor import MarketMonitor
from .stock_manager import StockManager
from .alert_system import AlertSystem

logger = logging.getLogger(__name__)

class StockMonitor:
    def __init__(
        self,
        data_fetcher: DataFetcher,
        strategy_manager: StrategyManager,
        report_generator: ReportGenerator,
        market_monitor: MarketMonitor,
        stock_manager: StockManager,
        check_interval: int = 300,  # 检查间隔（秒）
        max_alerts: int = 100,      # 最大警报数量
        mode: str = 'prod',         # 运行模式：'prod' 或 'dev'
        config: Optional[Dict[str, Any]] = None
    ):
        # 初始化组件
        self.data_fetcher = data_fetcher
        self.strategy_manager = strategy_manager
        self.report_generator = report_generator
        self.market_monitor = market_monitor
        self.stock_manager = stock_manager
        self.alert_system = AlertSystem(config)
        
        # 设置参数
        self.check_interval = check_interval
        self.max_alerts = max_alerts
        self.mode = mode
        
        # 加载配置
        self.config = self._load_config(config)
        
        # 初始化监控状态
        self.is_running = False
        self.alerts = []
        self.last_check_time = None
        
        self.logger = logging.getLogger(__name__)
        
        logger.info(f"StockMonitor初始化完成，运行模式: {mode}")
        
    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict:
        """加载配置文件"""
        try:
            if config:
                return config
                
            config_path = Path(__file__).parent / 'config' / 'monitor_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
            
    def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            logger.warning("监控已经在运行中")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("监控已启动")
        
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_running:
            logger.warning("监控未在运行")
            return
            
        self.is_running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        logger.info("监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 获取当前时间
                current_time = time.time()
                
                # 检查是否需要执行监控
                if (self.last_check_time is None or 
                    current_time - self.last_check_time >= self.check_interval):
                    
                    # 执行监控
                    self._execute_monitoring()
                    
                    # 更新最后检查时间
                    self.last_check_time = current_time
                    
                # 休眠一段时间
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(5)  # 出错后等待一段时间再继续
                
    def _execute_monitoring(self):
        """执行监控"""
        try:
            # 获取所有监控的股票
            symbols = self.stock_manager.get_all_symbols()
            
            # 获取市场监控结果
            market_results = self.market_monitor.monitor_market(symbols)
            
            # 分析每只股票
            for symbol in symbols:
                # 获取股票数据
                data = self.stock_manager.get_stock_data(symbol)
                if isinstance(data, pd.Series):
                    data = pd.DataFrame(data).T
                if data.empty:
                    continue
                    
                # 使用策略管理器分析股票
                analysis_result = self.strategy_manager.analyze_stock(data, symbol)
                
                # 添加市场监控结果
                analysis_result['market_monitoring'] = {
                    'news': market_results['news'].get(symbol, {}),
                    'social_sentiment': market_results['social_sentiment'].get(symbol, {}),
                    'sector_rotation': market_results['sector_rotation'],
                    'money_flow': market_results['money_flow'].get(symbol, {})
                }
                
                # 生成报告
                report = self.report_generator.generate_report(
                    analysis_result,
                    data,
                    symbol
                )
                
                # 检查是否需要生成警报
                self._check_alerts(symbol, analysis_result, report)
                
        except Exception as e:
            logger.error(f"执行监控失败: {e}")
            
    def _check_alerts(self, symbol: str, analysis_result: Dict, report: Dict):
        """
        检查是否需要生成警报
        :param symbol: 股票代码
        :param analysis_result: 分析结果
        :param report: 报告
        """
        try:
            # 检查风险水平
            if analysis_result.get('risk_level') == 'high':
                self._add_alert(symbol, '高风险警报', report)
                
            # 检查市场情绪
            market_monitoring = analysis_result.get('market_monitoring', {})
            news = market_monitoring.get('news', {})
            social_sentiment = market_monitoring.get('social_sentiment', {})
            if (news.get('sentiment') == 'negative' and 
                social_sentiment.get('sentiment') == 'negative'):
                self._add_alert(symbol, '负面市场情绪警报', report)
                
            # 检查资金流向
            money_flow = market_monitoring.get('money_flow', {})
            if money_flow.get('trend') == 'outflow' and money_flow.get('mfi', 50) < 20:
                self._add_alert(symbol, '资金流出警报', report)
                
        except Exception as e:
            logger.error(f"检查警报失败: {e}")
            
    def _add_alert(self, symbol: str, alert_type: str, report: Dict):
        """
        添加警报
        :param symbol: 股票代码
        :param alert_type: 警报类型
        :param report: 报告
        """
        try:
            # 创建警报
            alert = {
                'symbol': symbol,
                'type': alert_type,
                'timestamp': time.time(),
                'report': report
            }
            
            # 添加警报
            self.alerts.append(alert)
            
            # 限制警报数量
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
                
            logger.info(f"添加警报: {symbol} - {alert_type}")
            
        except Exception as e:
            logger.error(f"添加警报失败: {e}")
            
    def get_alerts(self) -> List[Dict]:
        """
        获取警报列表
        :return: 警报列表
        """
        return self.alerts
        
    def clear_alerts(self):
        """清除警报"""
        self.alerts = []
        logger.info("警报已清除")

    def _load_positions(self) -> Dict[str, Dict]:
        """加载持仓数据"""
        try:
            # 从data模块获取持仓数据
            self.positions = self.data_loader.load_positions()
            return self.positions
        except Exception as e:
            self.logger.error(f"加载持仓数据失败: {e}")
            return {}
        
    def load_watchlist(self):
        """加载监控列表和观察列表"""
        try:
            # 从portfolio_config.json获取当前持仓
            with open('monitor/configs/portfolio_config.json', 'r') as f:
                config = json.load(f)
                self.monitored_stocks = set(config['positions'].keys())
                
            # 从watchlist.json加载观察列表
            with open('monitor/configs/watchlist.json', 'r') as f:
                config = json.load(f)
                self.watchlist_stocks = set(config['watchlist'])
                
            self.logger.info(f"已加载持仓股票: {self.monitored_stocks}")
            self.logger.info(f"已加载观察列表: {self.watchlist_stocks}")
        except Exception as e:
            self.logger.error(f"加载监控列表失败: {str(e)}")
            
    def add_stock(self, symbol: str):
        """添加要监控的股票"""
        if symbol not in self.monitored_stocks:
            self.monitored_stocks.add(symbol)
            self.logger.info(f"添加监控股票: {symbol}")
            
    def remove_stock(self, symbol: str):
        """移除监控的股票"""
        if symbol in self.monitored_stocks:
            self.monitored_stocks.remove(symbol)
            self.logger.info(f"移除监控股票: {symbol}")
            
    async def get_stock_analysis(self, symbol: str) -> Dict:
        """
        获取单个股票的综合分析
        :param symbol: 股票代码
        :return: 包含各种分析指标的字典
        """
        try:
            # 获取实时数据
            if isinstance(self.data_interface, RealTimeDataSource):
                # 使用1分钟数据
                realtime_data = await self.data_interface.get_realtime_data([symbol], timeframe='1m')
                df = realtime_data.get(symbol)
            else:
                # 如果数据源不支持实时数据，获取最近的数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # 获取最近30天的数据用于计算指标
                df = self.data_interface.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1m'  # 使用1分钟数据
                )
            
            if df is None or df.empty:
                return {
                    "symbol": symbol,
                    "error": "无法获取数据",
                    "current_price": 0.0,
                    "price_change": "0.00%",
                    "volume_change": "0.00%",
                    "trend": "无数据",
                    "volume_status": "无数据",
                    "rsi": None,
                    "rsi_status": "无数据",
                    "macd_status": "无数据",
                    "position_size": 0,
                    "avg_price": 0.0,
                    "position_pnl": "0.00%",
                    "position_pnl_amount": "$0.00",
                    "score": 0,
                    "risk_level": "high",
                    "stop_loss_price": None,
                    "take_profit_price": None,
                    "recommendations": ["无法获取数据，请检查数据源"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            # 验证数据
            validated_df, report = self.data_validator.validate_data(df)
            
            # 忽略adj_close的缺失值警告
            if 'missing_values' in report and 'adj_close' in report['missing_values']:
                del report['missing_values']['adj_close']
                if not report['missing_values']:
                    del report['missing_values']
                    
            if not report['validation_passed']:
                return {
                    "symbol": symbol,
                    "error": "数据验证失败",
                    "current_price": 0.0,
                    "price_change": "0.00%",
                    "volume_change": "0.00%",
                    "trend": "数据无效",
                    "volume_status": "数据无效",
                    "rsi": None,
                    "rsi_status": "数据无效",
                    "macd_status": "数据无效",
                    "position_size": 0,
                    "avg_price": 0.0,
                    "position_pnl": "0.00%",
                    "position_pnl_amount": "$0.00",
                    "score": 0,
                    "risk_level": "high",
                    "stop_loss_price": None,
                    "take_profit_price": None,
                    "recommendations": ["数据验证失败，请检查数据质量"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            # 获取最新数据
            latest_data = validated_df.iloc[-1]
            prev_data = validated_df.iloc[-2] if len(validated_df) > 1 else latest_data
            
            # 计算基本指标
            current_price = latest_data['close']
            price_change = (current_price - prev_data['close']) / prev_data['close']
            volume_change = ((latest_data['volume'] - prev_data['volume']) / prev_data['volume']) if prev_data['volume'] > 0 else 0
            
            # 获取策略分析结果
            strategy_analysis = self.strategy_manager.analyze_stock(validated_df, symbol)
            
            # 获取持仓信息
            position_info = self.positions.get(symbol, {})
            avg_price = position_info.get('avg_price', 0.0)
            position_size = position_info.get('size', 0)
            
            # 计算持仓盈亏
            if avg_price > 0 and position_size > 0:
                position_pnl = (current_price - avg_price) / avg_price
                position_pnl_amount = (current_price - avg_price) * position_size
            else:
                position_pnl = 0
                position_pnl_amount = 0
                
            # 合并分析结果
            analysis_result = {
                "symbol": symbol,
                "current_price": current_price,
                "price_change": f"{price_change:.2%}",
                "volume_change": f"{volume_change:.2%}",
                "trend": strategy_analysis['signals'].get('NiuniuV3', {}).get('signals', {}).get('trend', '未知'),
                "volume_status": "放量" if volume_change > 0.5 else "缩量",
                "rsi": strategy_analysis['signals'].get('NiuniuV3', {}).get('signals', {}).get('rsi', None),
                "rsi_status": self._get_rsi_status(strategy_analysis['signals'].get('NiuniuV3', {}).get('signals', {}).get('rsi', None)),
                "macd_status": strategy_analysis['signals'].get('NiuniuV3', {}).get('signals', {}).get('macd_status', '未知'),
                "position_size": position_size,
                "avg_price": avg_price,
                "position_pnl": f"{position_pnl:.2%}",
                "position_pnl_amount": f"${position_pnl_amount:.2f}",
                "score": strategy_analysis['score'],
                "risk_level": strategy_analysis['risk_level'],
                "stop_loss_price": strategy_analysis.get('stop_loss_price'),
                "take_profit_price": strategy_analysis.get('take_profit_price'),
                "recommendations": strategy_analysis['recommendations'],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"分析股票 {symbol} 失败: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "current_price": 0.0,
                "price_change": "0.00%",
                "volume_change": "0.00%",
                "trend": "错误",
                "volume_status": "错误",
                "rsi": None,
                "rsi_status": "错误",
                "macd_status": "错误",
                "position_size": 0,
                "avg_price": 0.0,
                "position_pnl": "0.00%",
                "position_pnl_amount": "$0.00",
                "score": 0,
                "risk_level": "high",
                "stop_loss_price": None,
                "take_profit_price": None,
                "recommendations": [f"分析失败: {str(e)}"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
    def _get_rsi_status(self, rsi: Optional[float]) -> str:
        """
        获取RSI状态
        :param rsi: RSI值
        :return: RSI状态描述
        """
        if rsi is None:
            return "未知"
        if rsi > self.config['monitor_config']['alert_thresholds']['rsi_overbought']:
            return "超买"
        if rsi < self.config['monitor_config']['alert_thresholds']['rsi_oversold']:
            return "超卖"
        return "正常"
            
    async def monitor_stocks(self):
        """监控所有股票"""
        while True:
            try:
                # 获取当前时间（卡尔加里时区）
                mt_tz = pytz.timezone('America/Edmonton')  # 使用埃德蒙顿时区（与卡尔加里相同）
                now = datetime.now(mt_tz)
                
                # 检查是否在交易时间内
                if not await self.is_market_open(now):
                    # 市场已收盘，生成每日总结报告
                    self.logger.info("市场已收盘，生成每日总结报告...")
                    analysis_results = []
                    
                    # 分析持仓股票
                    for symbol in self.monitored_stocks:
                        try:
                            result = await self.get_stock_analysis(symbol)
                            if result:
                                analysis_results.append(result)
                        except Exception as e:
                            self.logger.error(f"分析股票 {symbol} 失败: {str(e)}")
                            
                    # 分析观察列表股票
                    for symbol in self.watchlist_stocks:
                        try:
                            result = await self.get_stock_analysis(symbol)
                            if result:
                                analysis_results.append(result)
                        except Exception as e:
                            self.logger.error(f"分析股票 {symbol} 失败: {str(e)}")
                            
                    # 生成并发送每日总结
                    summary = self.generate_daily_summary(analysis_results)
                    html_report = self._generate_html_report(analysis_results, summary=True)
                    self.notification_manager.send_batch_alerts([{
                        "type": "daily_summary",
                        "message": html_report
                    }])
                    
                    # 计算到下一个交易日的等待时间（使用卡尔加里时间）
                    next_open = await self.get_next_market_open(now)
                    sleep_time = (next_open - now).total_seconds()
                    
                    # 确保等待时间合理（不超过18小时）
                    if sleep_time <= 0 or sleep_time > 18 * 3600:
                        self.logger.warning(f"等待时间 {sleep_time/3600:.2f} 小时超出合理范围，将调整为12小时")
                        sleep_time = 12 * 3600
                        
                    self.logger.info(f"市场已收盘，等待 {sleep_time/3600:.2f} 小时后重新开始监控...")
                    await asyncio.sleep(sleep_time)
                    continue
                    
                # 市场开盘时的操作
                # 更新持仓数据
                await self.update_positions()
                
                # 检查警报
                alerts = await self.check_alerts()
                if alerts:
                    self.notification_manager.send_batch_alerts(alerts)
                
                # 生成报告
                analysis_results = []
                for symbol in self.monitored_stocks:
                    try:
                        result = await self.get_stock_analysis(symbol)
                        if result:
                            analysis_results.append(result)
                    except Exception as e:
                        self.logger.error(f"分析股票 {symbol} 失败: {str(e)}")
                        
                if analysis_results:
                    html_report = self._generate_html_report(analysis_results)
                    self.logger.info(f"当前持仓总价值: ${sum(r.get('current_price', 0) * r.get('position_size', 0) for r in analysis_results):.2f}")
                
                # 等待下一次更新
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"监控过程中发生错误: {str(e)}")
                await asyncio.sleep(60)  # 发生错误时等待1分钟后重试
                
    async def is_market_open(self, now):
        """检查市场是否开盘（使用卡尔加里时间）"""
        # 检查是否是工作日（周一至周五）
        if now.weekday() >= 5:  # 5是周六，6是周日
            return False
            
        # 检查是否在交易时间内（7:30 AM - 2:00 PM MT）
        market_open = time(7, 30)  # 卡尔加里时间 7:30 AM
        market_close = time(14, 0)  # 卡尔加里时间 2:00 PM
        
        current_time = now.time()
        return market_open <= current_time <= market_close
        
    async def get_next_market_open(self, now):
        """获取下一个市场开盘时间（使用卡尔加里时间）"""
        # 计算到下一个交易日的天数
        days_to_add = 1
        if now.weekday() == 4:  # 周五
            days_to_add = 3
        elif now.weekday() == 5:  # 周六
            days_to_add = 2
        elif now.weekday() == 6:  # 周日
            days_to_add = 1
            
        # 计算下一个交易日的开盘时间（卡尔加里时间 7:30 AM）
        next_open = now + timedelta(days=days_to_add)
        next_open = next_open.replace(hour=7, minute=30, second=0, microsecond=0)
        
        return next_open
        
    def generate_daily_summary(self, analysis_results):
        """生成每日总结报告"""
        if not analysis_results:
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "positions": [],
                "watchlist": [],
                "market_summary": {
                    "total_positions": 0,
                    "profitable_positions": 0,
                    "losing_positions": 0,
                    "total_pnl": 0.0
                }
            }
            
        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "positions": [],
            "watchlist": [],
            "market_summary": {
                "total_positions": len([r for r in analysis_results if r["symbol"] in self.monitored_stocks]),
                "profitable_positions": 0,
                "losing_positions": 0,
                "total_pnl": 0.0
            }
        }
        
        for result in analysis_results:
            if "error" in result:
                continue
                
            try:
                position_data = {
                    "symbol": result["symbol"],
                    "current_price": result.get("current_price", 0.0),
                    "price_change": result.get("price_change", "0.00%"),
                    "position_pnl": result.get("position_pnl", "0.00%"),
                    "position_pnl_amount": result.get("position_pnl_amount", "$0.00"),
                    "recommendations": result.get("recommendations", ["无数据"])
                }
                
                if result["symbol"] in self.monitored_stocks:
                    summary["positions"].append(position_data)
                    pnl = float(position_data["position_pnl"].strip('%')) if position_data["position_pnl"] else 0
                    if pnl > 0:
                        summary["market_summary"]["profitable_positions"] += 1
                    else:
                        summary["market_summary"]["losing_positions"] += 1
                    pnl_amount = float(position_data["position_pnl_amount"].strip('$')) if position_data["position_pnl_amount"] else 0
                    summary["market_summary"]["total_pnl"] += pnl_amount
                else:
                    summary["watchlist"].append(position_data)
            except Exception as e:
                self.logger.error(f"处理股票 {result.get('symbol', 'unknown')} 的总结数据时出错: {str(e)}")
                continue
                
        return summary
        
    def _generate_html_report(self, analysis_results, summary=False):
        """
        生成HTML格式的报告
        :param analysis_results: 分析结果列表
        :param summary: 是否为总结报告
        :return: HTML格式的报告
        """
        # 基础样式
        style = """
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: auto; }
            .header { background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
            .section { margin-bottom: 30px; }
            .stock-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .stock-card {
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 5px;
                background-color: white;
            }
            .stock-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .stock-title { font-size: 1.2em; font-weight: bold; }
            .price-info { text-align: right; }
            .indicator-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                margin: 10px 0;
            }
            .indicator { padding: 5px; border-radius: 3px; background-color: #f8f9fa; }
            .recommendations { margin-top: 15px; }
            .recommendation-item { margin: 5px 0; padding: 5px; border-radius: 3px; }
            .position-info { margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            .risk-info { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .high { background-color: #dc3545; color: white; }
            .normal { background-color: #28a745; color: white; }
            .warning { background-color: #ffc107; color: black; }
            .info { background-color: #17a2b8; color: white; }
            .positive { color: #28a745; }
            .negative { color: #dc3545; }
            .neutral { color: #6c757d; }
            .error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
        """
        
        # 报告头部
        header = f"""
        <div class="container">
            <div class="header">
                <h2>{'股票监控日报' if summary else '股票实时监控报告'}</h2>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # 生成股票卡片
        stock_cards = []
        for result in analysis_results:
            if 'error' in result:
                card = self._generate_error_card(result)
            else:
                card = self._generate_stock_card(result)
            stock_cards.append(card)
            
        # 组装报告
        report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{'股票监控日报' if summary else '股票实时监控报告'}</title>
            {style}
        </head>
        <body>
            {header}
            <div class="stock-grid">
                {''.join(stock_cards)}
            </div>
        </div>
        </body>
        </html>
        """
        
        return report
        
    def _generate_stock_card(self, result: Dict) -> str:
        """
        生成单个股票的卡片HTML
        :param result: 股票分析结果
        :return: HTML字符串
        """
        # 获取基本信息
        symbol = result['symbol']
        current_price = result['current_price']
        price_change = result['price_change']
        volume_change = result['volume_change']
        
        # 价格变化的样式
        price_class = 'positive' if float(price_change.strip('%')) > 0 else 'negative' if float(price_change.strip('%')) < 0 else 'neutral'
        
        # 风险等级样式
        risk_class = {
            'high': 'high',
            'normal': 'normal',
            'warning': 'warning'
        }.get(result['risk_level'], 'info')
        
        # 生成指标网格
        indicators = f"""
        <div class="indicator-grid">
            <div class="indicator">趋势: {result['trend']}</div>
            <div class="indicator">成交量: {result['volume_status']}</div>
            <div class="indicator">RSI: {result['rsi_status']}</div>
            <div class="indicator">MACD: {result['macd_status']}</div>
        </div>
        """
        
        # 生成持仓信息
        position_info = ""
        if result['position_size'] > 0:
            pnl_class = 'positive' if float(result['position_pnl'].strip('%')) > 0 else 'negative'
            position_info = f"""
            <div class="position-info">
                <div>持仓数量: {result['position_size']}</div>
                <div>平均成本: ${result['avg_price']:.2f}</div>
                <div>持仓盈亏: <span class="{pnl_class}">{result['position_pnl']} ({result['position_pnl_amount']})</span></div>
            </div>
            """
            
        # 生成风险信息
        risk_info = f"""
        <div class="risk-info {risk_class}">
            <div>风险等级: {result['risk_level'].upper()}</div>
            <div>止损价: ${result['stop_loss_price']:.2f if result['stop_loss_price'] else 'N/A'}</div>
            <div>止盈价: ${result['take_profit_price']:.2f if result['take_profit_price'] else 'N/A'}</div>
            <div>建议仓位: {result['position_size']*100:.0f}%</div>
        </div>
        """
        
        # 生成建议列表
        recommendations = '<div class="recommendations">'
        for rec in result['recommendations']:
            rec_class = 'info'
            if '买入' in rec:
                rec_class = 'normal'
            elif '卖出' in rec or '风险' in rec:
                rec_class = 'high'
            elif '观望' in rec:
                rec_class = 'warning'
            recommendations += f'<div class="recommendation-item {rec_class}">{rec}</div>'
        recommendations += '</div>'
        
        # 组装卡片
        return f"""
        <div class="stock-card">
            <div class="stock-header">
                <div class="stock-title">{symbol}</div>
                <div class="price-info">
                    <div>${current_price:.2f}</div>
                    <div class="{price_class}">{price_change}</div>
                </div>
            </div>
            {indicators}
            {position_info}
            {risk_info}
            {recommendations}
        </div>
        """
        
    def _generate_error_card(self, result: Dict) -> str:
        """
        生成错误卡片的HTML
        :param result: 错误结果
        :return: HTML字符串
        """
        return f"""
        <div class="stock-card">
            <div class="stock-header">
                <div class="stock-title">{result['symbol']}</div>
            </div>
            <div class="error">
                {result['error']}
            </div>
        </div>
        """

    async def update_positions(self):
        """更新持仓数据"""
        try:
            # 从data模块获取最新的持仓数据
            self.positions = self.data_loader.load_positions()
            self.logger.info("持仓数据已更新")
        except Exception as e:
            self.logger.error(f"更新持仓数据失败: {e}")
            # 如果更新失败，保持原有持仓数据不变 

    async def check_alerts(self) -> List[Dict]:
        """
        检查所有监控股票和观察列表的警报条件
        :return: 警报列表
        """
        alerts = []
        
        try:
            # 检查持仓股票
            for symbol in self.monitored_stocks:
                try:
                    analysis = await self.get_stock_analysis(symbol)
                    if not analysis or "error" in analysis:
                        continue
                        
                    # 检查价格变化
                    price_change = float(analysis["price_change"].strip('%')) / 100
                    if abs(price_change) > 0.02:  # 2%的价格变化
                        alerts.append({
                            "type": "price_alert",
                            "symbol": symbol,
                            "message": f"{symbol} 价格变化 {analysis['price_change']}，当前价格 ${analysis['current_price']:.2f}",
                            "severity": "warning" if price_change < 0 else "info"
                        })
                        
                    # 检查RSI状态
                    if analysis["rsi_status"] == "超买":
                        alerts.append({
                            "type": "rsi_alert",
                            "symbol": symbol,
                            "message": f"{symbol} RSI超买 ({analysis['rsi']:.2f})，注意回调风险",
                            "severity": "warning"
                        })
                    elif analysis["rsi_status"] == "超卖":
                        alerts.append({
                            "type": "rsi_alert",
                            "symbol": symbol,
                            "message": f"{symbol} RSI超卖 ({analysis['rsi']:.2f})，可能有反弹机会",
                            "severity": "info"
                        })
                        
                    # 检查MACD信号
                    if analysis["macd_status"] == "金叉":
                        alerts.append({
                            "type": "macd_alert",
                            "symbol": symbol,
                            "message": f"{symbol} MACD金叉，短期趋势向好",
                            "severity": "info"
                        })
                    elif analysis["macd_status"] == "死叉":
                        alerts.append({
                            "type": "macd_alert",
                            "symbol": symbol,
                            "message": f"{symbol} MACD死叉，短期趋势向弱",
                            "severity": "warning"
                        })
                        
                except Exception as e:
                    self.logger.error(f"检查股票 {symbol} 警报时出错: {e}")
                    
            # 检查观察列表股票
            for symbol in self.watchlist_stocks:
                try:
                    analysis = await self.get_stock_analysis(symbol)
                    if not analysis or "error" in analysis:
                        continue
                        
                    # 检查价格变化（观察列表使用更高的阈值）
                    price_change = float(analysis["price_change"].strip('%')) / 100
                    if abs(price_change) > 0.05:  # 5%的价格变化
                        alerts.append({
                            "type": "watchlist_alert",
                            "symbol": symbol,
                            "message": f"观察列表 {symbol} 价格变化 {analysis['price_change']}，当前价格 ${analysis['current_price']:.2f}",
                            "severity": "warning" if price_change < 0 else "info"
                        })
                        
                except Exception as e:
                    self.logger.error(f"检查观察列表股票 {symbol} 警报时出错: {e}")
                    
        except Exception as e:
            self.logger.error(f"检查警报时发生错误: {e}")
            
        return alerts 

    def analyze_market_bottom(self):
        """分析市场底部信号"""
        try:
            # 获取SPY数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 获取一年的数据
            spy_data = self.data_fetcher.get_historical_data('SPY', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if spy_data.empty:
                self.logger.error("无法获取SPY数据")
                return None
            
            # 计算技术指标
            indicators = self._calculate_technical_indicators(spy_data)
            
            # 分析市场状态
            market_state = self._analyze_market_state(indicators)
            
            # 生成分析报告
            report = self._generate_analysis_report(market_state, indicators)
            
            # 发送通知
            self._send_notification(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"市场底部分析失败: {str(e)}")
            return None
        
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """计算技术指标"""
        try:
            indicators = {}
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            indicators['macd'] = exp1 - exp2
            indicators['signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            
            # 移动平均线
            indicators['sma20'] = data['Close'].rolling(window=20).mean()
            indicators['sma50'] = data['Close'].rolling(window=50).mean()
            indicators['sma200'] = data['Close'].rolling(window=200).mean()
            
            # 成交量变化
            indicators['volume_change'] = data['Volume'].pct_change()
            
            return indicators
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {str(e)}")
            return {}
        
    def _analyze_market_state(self, indicators: Dict) -> Dict:
        """分析市场状态"""
        try:
            state = {
                'is_bottom': False,
                'signals': [],
                'risk_level': 'info'
            }
            
            # 检查RSI超卖
            if not indicators['rsi'].empty and float(indicators['rsi'].iloc[-1]) < 30:
                state['signals'].append('RSI超卖')
                state['risk_level'] = 'warning'
            
            # 检查MACD金叉
            if (not indicators['macd'].empty and not indicators['signal'].empty and
                len(indicators['macd']) >= 2 and len(indicators['signal']) >= 2 and
                float(indicators['macd'].iloc[-2]) < float(indicators['signal'].iloc[-2]) and
                float(indicators['macd'].iloc[-1]) > float(indicators['signal'].iloc[-1])):
                state['signals'].append('MACD金叉')
                state['risk_level'] = 'opportunity'
            
            # 检查价格与移动平均线的关系
            if (not indicators['sma20'].empty and not indicators['sma50'].empty and
                not indicators['sma200'].empty and
                float(indicators['sma20'].iloc[-1]) > float(indicators['sma50'].iloc[-1]) and
                float(indicators['sma50'].iloc[-1]) > float(indicators['sma200'].iloc[-1])):
                state['signals'].append('均线多头排列')
                state['risk_level'] = 'opportunity'
            
            # 检查成交量放大
            if not indicators['volume_change'].empty and float(indicators['volume_change'].iloc[-1]) > 1.5:
                state['signals'].append('成交量放大')
                state['risk_level'] = 'warning'
            
            # 综合判断市场底部
            if len(state['signals']) >= 2 and 'RSI超卖' in state['signals']:
                state['is_bottom'] = True
                state['risk_level'] = 'opportunity'
            
            return state
        except Exception as e:
            self.logger.error(f"分析市场状态失败: {str(e)}")
            return {}
        
    def _generate_analysis_report(self, market_state: Dict, indicators: Dict) -> Dict:
        """生成分析报告"""
        try:
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_state': market_state,
                'indicators': {
                    'rsi': round(indicators['rsi'].iloc[-1], 2),
                    'macd': round(indicators['macd'].iloc[-1], 2),
                    'signal': round(indicators['signal'].iloc[-1], 2),
                    'sma20': round(indicators['sma20'].iloc[-1], 2),
                    'sma50': round(indicators['sma50'].iloc[-1], 2),
                    'sma200': round(indicators['sma200'].iloc[-1], 2),
                    'volume_change': round(indicators['volume_change'].iloc[-1] * 100, 2)
                }
            }
            return report
        except Exception as e:
            self.logger.error(f"生成分析报告失败: {str(e)}")
            return {}
        
    def _send_notification(self, report: Dict):
        """发送通知"""
        try:
            if not report or 'market_state' not in report:
                self.logger.warning("没有有效的报告数据，跳过发送通知")
                return
            
            # 构建HTML格式的邮件内容
            html_content = f"""
            <h2>市场底部分析报告</h2>
            <p>时间: {report['timestamp']}</p>
            <h3>市场状态</h3>
            <ul>
                <li>是否底部: {'是' if report['market_state'].get('is_bottom', False) else '否'}</li>
                <li>风险等级: {report['market_state'].get('risk_level', 'unknown')}</li>
                <li>信号: {', '.join(report['market_state'].get('signals', []))}</li>
            </ul>
            <h3>技术指标</h3>
            <ul>
                <li>RSI: {report['indicators'].get('rsi', 'N/A')}</li>
                <li>MACD: {report['indicators'].get('macd', 'N/A')}</li>
                <li>Signal: {report['indicators'].get('signal', 'N/A')}</li>
                <li>SMA20: {report['indicators'].get('sma20', 'N/A')}</li>
                <li>SMA50: {report['indicators'].get('sma50', 'N/A')}</li>
                <li>SMA200: {report['indicators'].get('sma200', 'N/A')}</li>
                <li>成交量变化: {report['indicators'].get('volume_change', 'N/A')}%</li>
            </ul>
            """
            
            # 发送邮件
            self.alert_system.send_email(
                subject="市场底部分析报告",
                body=html_content,
                is_html=True
            )
            
            self.logger.info("市场底部分析报告已发送")
            
        except Exception as e:
            self.logger.error(f"发送通知失败: {str(e)}") 