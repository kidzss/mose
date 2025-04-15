import pandas as pd
import numpy as np
import datetime as dt
import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from .market_monitor import MarketMonitor, Alert
from .notification_manager import NotificationManager
from data.data_interface import YahooFinanceRealTimeSource
import yfinance as yf
import schedule
import time
import asyncio
import json

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    cost_basis: float
    weight: float
    current_price: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    avg_price: float = 0.0
    shares: float = 0.0

class PortfolioMonitor:
    """投资组合监控类"""
    
    def __init__(self, positions: Dict[str, Dict], monitor_config: Dict):
        """
        初始化投资组合监控器
        
        Args:
            positions: 持仓信息字典，包含每个股票的成本基础和权重
            monitor_config: 监控配置，包含各种阈值和设置
        """
        self.logger = logging.getLogger("PortfolioMonitor")
        
        # 初始化数据源
        self.data_source = YahooFinanceRealTimeSource()
        
        # 初始化市场监控器
        self.market_monitor = MarketMonitor()
        
        # 初始化通知管理器
        self.notification_manager = NotificationManager()
        
        # 初始化实时数据源
        self.realtime_data_source = YahooFinanceRealTimeSource()
        
        # 设置默认配置
        self.config = {
            "price_alert_threshold": 0.02,    # 价格变动提醒阈值
            "loss_alert_threshold": 0.05,     # 亏损提醒阈值
            "profit_target": 0.10,            # 止盈目标
            "stop_loss": 0.15,                # 止损线
            "check_interval": 300,            # 检查间隔（秒）
            "email_notifications": True       # 是否发送邮件通知
        }
        if monitor_config:
            self.config.update(monitor_config)
            
        # 初始化持仓
        self.positions = positions
        self.current_prices = {}
        self.historical_data = {}
        self.alerts = []
        self.last_update = None
        self.total_value = 0.0  # 添加总市值属性
        
        # 监控状态
        self.is_running = False
        self.monitor_thread = None
        
        # 初始化数据
        self.update_positions()
        
    async def update_positions(self) -> None:
        """更新所有持仓的当前价格和历史数据"""
        try:
            # 从portfolio_config.json获取持仓信息
            with open('monitor/configs/portfolio_config.json', 'r') as f:
                config = json.load(f)
                positions = config['positions']
            
            # 获取实时数据
            symbols = list(positions.keys())
            data = await self.realtime_data_source.get_realtime_data(symbols, timeframe='1m')
            
            total_value = 0.0
            for symbol, df in data.items():
                if not df.empty:
                    # 获取最新价格
                    current_price = df['close'].iloc[-1]
                    self.current_prices[symbol] = current_price
                    # 获取历史数据用于分析
                    self.historical_data[symbol] = df
                    # 计算持仓市值
                    shares = positions[symbol]['shares']
                    total_value += current_price * shares
                    
            self.total_value = total_value  # 更新总市值
            self.last_update = dt.datetime.now()
            self.logger.info("Successfully updated portfolio positions with real-time data")
        except Exception as e:
            self.logger.error(f"Error updating positions with real-time data: {str(e)}")
            
    def calculate_portfolio_value(self) -> float:
        """计算当前投资组合总价值"""
        total_value = 0
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                shares = position['shares']
                value = shares * self.current_prices[symbol]
                total_value += value
        return total_value
        
    def calculate_returns(self) -> Dict[str, float]:
        """计算每个持仓的收益率"""
        returns = {}
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                avg_price = position['avg_price']
                current_price = self.current_prices[symbol]
                returns[symbol] = (current_price - avg_price) / avg_price
        return returns
        
    def set_monitoring_parameters(self, config: Dict) -> None:
        """
        设置监控参数
        
        Args:
            config: 包含监控参数的字典
        """
        if config:
            self.config.update(config)
            self.logger.info("Updated monitoring parameters")

    def check_alerts(self) -> List[str]:
        """检查是否需要发出警报"""
        alerts = []
        returns = self.calculate_returns()
        
        # 定义行业分类
        sectors = {
            'AMD': 'semiconductor',
            'NVDA': 'semiconductor',
            'GOOG': 'tech',
            'MSFT': 'tech',
            'TSLA': 'automotive',
            'PFE': 'healthcare',
            'TMDX': 'healthcare'
        }
        
        for symbol, ret in returns.items():
            position = self.positions[symbol]
            current_price = self.current_prices[symbol]
            avg_price = position['avg_price']
            shares = position['shares']
            
            # 获取行业特定设置
            sector = sectors.get(symbol)
            stop_loss = self.config.get('sector_specific_settings', {}).get(sector, {}).get('stop_loss', self.config['stop_loss'])
            price_alert_threshold = self.config.get('sector_specific_settings', {}).get(sector, {}).get('price_alert_threshold', self.config['price_alert_threshold'])
            
            # 计算持仓市值
            position_value = current_price * shares
            
            # 检查止损
            if ret < -stop_loss:
                alerts.append(f"Stop Loss Alert: {symbol} has dropped {ret*100:.2f}% below entry price")
                
            # 检查止盈
            if ret > self.config['profit_target']:
                alerts.append(f"Profit Target Alert: {symbol} has gained {ret*100:.2f}% above entry price")
                
            # 检查价格变动
            if abs(ret) > price_alert_threshold:
                alerts.append(f"Price Movement Alert: {symbol} has moved {ret*100:.2f}% from entry price")
                
            # 检查亏损
            if ret < -self.config['loss_alert_threshold']:
                alerts.append(f"Loss Alert: {symbol} is down {ret*100:.2f}%, consider cutting losses")
                
            # 特别关注半导体行业的关税相关新闻
            if sector == 'semiconductor':
                alerts.append(f"Semiconductor Alert: {symbol} may be affected by trade policy changes")
                
        return alerts
        
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """计算投资组合的在险价值(VaR)"""
        portfolio_returns = []
        for symbol in self.positions:
            if symbol in self.historical_data:
                returns = self.historical_data[symbol]['Close'].pct_change().dropna()
                weight = self.positions[symbol]['weight']
                portfolio_returns.append(returns * weight)
                
        if portfolio_returns:
            portfolio_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            return abs(var)
        return 0.0
        
    def analyze_portfolio_risk(self) -> Dict:
        """分析投资组合风险"""
        risk_metrics = {
            'var_95': self.calculate_var(0.95),
            'volatility': {},
            'correlation': {},
            'concentration_risk': {}
        }
        
        # 计算波动率
        for symbol in self.positions:
            if symbol in self.historical_data:
                returns = self.historical_data[symbol]['Close'].pct_change().dropna()
                risk_metrics['volatility'][symbol] = returns.std()
                
        # 计算相关性
        returns_data = {}
        for symbol in self.positions:
            if symbol in self.historical_data:
                returns_data[symbol] = self.historical_data[symbol]['Close'].pct_change().dropna()
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            risk_metrics['correlation'] = returns_df.corr().to_dict()
            
        # 计算集中度风险
        total_weight = sum(position['weight'] for position in self.positions.values())
        for symbol, position in self.positions.items():
            risk_metrics['concentration_risk'][symbol] = position['weight'] / total_weight
            
        return risk_metrics
        
    def generate_portfolio_report(self) -> str:
        """生成投资组合报告"""
        total_value = self.total_value
        returns = self.calculate_returns()
        
        report = f"Portfolio Report (as of {self.last_update})\n"
        report += f"Total Portfolio Value: ${total_value:,.2f}\n\n"
        
        report += "Individual Positions:\n"
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                shares = position['shares']
                value = shares * current_price
                ret = returns[symbol]
                
                report += f"{symbol}:\n"
                report += f"  Shares: {shares}\n"
                report += f"  Current Price: ${current_price:.2f}\n"
                report += f"  Position Value: ${value:.2f}\n"
                report += f"  Return: {ret*100:.2f}%\n"
                
        return report
        
    def get_portfolio_recommendations(self) -> List[str]:
        """基于当前投资组合状态生成建议"""
        recommendations = []
        risk_metrics = self.analyze_portfolio_risk()
        returns = self.calculate_returns()
        
        # 检查高波动率
        for symbol, volatility in risk_metrics['volatility'].items():
            if volatility > 0.02:  # 2% 日波动率阈值
                recommendations.append(
                    f"Consider reducing position in {symbol} due to high volatility ({volatility*100:.2f}%)"
                )
                
        # 检查集中度风险
        for symbol, concentration in risk_metrics['concentration_risk'].items():
            if concentration > 0.3:  # 30% 集中度阈值
                recommendations.append(
                    f"Consider diversifying away from {symbol} due to high concentration ({concentration*100:.2f}%)"
                )
                
        # 检查表现不佳的持仓
        for symbol, ret in returns.items():
            if ret < -0.1:  # -10% 收益率阈值
                recommendations.append(
                    f"Review position in {symbol} due to poor performance ({ret*100:.2f}%)"
                )
                
        return recommendations

    def send_notifications(self, alerts: List[str]) -> None:
        """发送通知"""
        if not alerts:
            return
            
        if self.config['email_notifications']:
            try:
                message = "\n".join(alerts)
                self.notification_manager.send_email(
                    subject="Portfolio Alerts",
                    body=message
                )
                self.logger.info("Successfully sent email notifications")
            except Exception as e:
                self.logger.error(f"Failed to send email notifications: {str(e)}")

    async def monitor_stocks(self) -> None:
        """监控股票"""
        try:
            # 更新持仓数据
            await self.update_positions()
            
            # 检查警报
            alerts = self.check_alerts()
            
            # 发送通知
            self.send_notifications(alerts)
            
            # 生成报告
            report = self.generate_portfolio_report()
            self.logger.info(report)
            
        except Exception as e:
            self.logger.error(f"监控过程中发生错误: {str(e)}")
            raise

    def start_monitoring(self, interval_minutes: int = 5):
        """开始监控
        
        Args:
            interval_minutes: 检查间隔（分钟）
        """
        try:
            # 使用 schedule 库设置定时任务
            schedule.every(interval_minutes).minutes.do(lambda: asyncio.run(self.monitor_stocks()))
            
            # 运行调度器
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"启动监控时发生错误: {str(e)}")
            raise 