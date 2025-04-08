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
        
        # 监控状态
        self.is_running = False
        self.monitor_thread = None
        
        # 初始化数据
        self.update_positions()
        
    def update_positions(self) -> None:
        """更新所有持仓的当前价格和历史数据"""
        try:
            for symbol in self.positions.keys():
                ticker = yf.Ticker(symbol)
                
                # 获取当前价格
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                self.current_prices[symbol] = current_price
                
                # 获取历史数据用于分析
                hist_data = ticker.history(period='1y')
                self.historical_data[symbol] = hist_data
                
            self.last_update = dt.datetime.now()
            self.logger.info("Successfully updated portfolio positions")
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            
    def calculate_portfolio_value(self) -> float:
        """计算当前投资组合总价值"""
        total_value = 0
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                shares = position['weight'] * 100  # 假设总投资为100单位
                value = shares * self.current_prices[symbol]
                total_value += value
        return total_value
        
    def calculate_returns(self) -> Dict[str, float]:
        """计算每个持仓的收益率"""
        returns = {}
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                cost_basis = position['cost_basis']
                current_price = self.current_prices[symbol]
                returns[symbol] = (current_price - cost_basis) / cost_basis
        return returns
        
    def check_alerts(self) -> List[str]:
        """检查是否需要触发任何警报"""
        new_alerts = []
        returns = self.calculate_returns()
        
        for symbol, ret in returns.items():
            # 止损警报
            if ret <= -self.config['stop_loss']:
                alert = f"Stop Loss Alert: {symbol} has lost {ret*100:.2f}% of its value"
                new_alerts.append(alert)
                
            # 获利目标警报
            elif ret >= self.config['profit_target']:
                alert = f"Profit Target Alert: {symbol} has gained {ret*100:.2f}% in value"
                new_alerts.append(alert)
                
            # 价格变动警报
            if symbol in self.historical_data:
                daily_return = self.historical_data[symbol]['Close'].pct_change().iloc[-1]
                if abs(daily_return) >= self.config['price_alert_threshold']:
                    alert = f"Price Movement Alert: {symbol} moved {daily_return*100:.2f}% today"
                    new_alerts.append(alert)
                    
        return new_alerts
        
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
        total_value = self.calculate_portfolio_value()
        returns = self.calculate_returns()
        
        report = f"Portfolio Report (as of {self.last_update})\n"
        report += f"Total Portfolio Value: ${total_value:,.2f}\n\n"
        
        report += "Individual Positions:\n"
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                shares = position['weight'] * 100
                value = shares * current_price
                ret = returns[symbol]
                
                report += f"{symbol}:\n"
                report += f"  Shares: {shares:.2f}\n"
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