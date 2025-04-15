from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .strategy_base import Strategy, MarketRegime
from data.market_sentiment import MarketSentimentData
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TDIStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("TDI Strategy", params)
        self.market_sentiment = MarketSentimentData()
        self.trade_history = []
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.max_drawdown = 0.0
        self.current_losing_streak = 0
        
        # 默认参数
        self.default_params = {
            'ma_short': 20,
            'ma_long': 50,
            'rsi_period': 14,
            'atr_period': 14,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0,
            'base_position_size': 0.1,
            'max_position_size': 0.3,
            'volatility_threshold': 25,
            'max_daily_loss': 0.02,
            'max_total_loss': 0.1
        }
        
        # 合并用户参数
        if params:
            self.default_params.update(params)
            
        # 初始化自适应参数
        self.adaptive_params = self.default_params.copy()
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        """
        df = data.copy()
        
        # 计算移动平均线
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=self.adaptive_params['atr_period']).mean()
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        """
        # 首先计算指标
        df = self.calculate_indicators(data)
        
        # 生成信号
        df['signal'] = 0.0
        
        for i in range(1, len(df)):
            current_data = df.iloc[i]
            df.loc[df.index[i], 'signal'] = self.generate_signal(current_data)
        
        return df
        
    def generate_signal(self, current_data: pd.Series) -> float:
        """
        根据当前数据生成信号
        """
        # 趋势判断
        trend_up = current_data['MA20'] > current_data['MA50']
        
        # RSI超买超卖
        rsi_overbought = current_data['RSI'] > 70
        rsi_oversold = current_data['RSI'] < 30
        
        # 生成信号
        if trend_up and rsi_oversold:
            return 1.0  # 买入信号
        elif not trend_up and rsi_overbought:
            return -1.0  # 卖出信号
        else:
            return 0.0  # 无信号
        
    def calculate_position_size(self, data: pd.DataFrame, volatility_high: bool) -> float:
        """计算仓位大小"""
        base_size = self.adaptive_params['base_position_size']
        max_size = self.adaptive_params['max_position_size']
        
        # 根据波动率调整仓位
        if volatility_high:
            position_size = base_size * 0.5
        else:
            position_size = base_size
            
        # 根据当前亏损情况调整仓位
        if self.current_losing_streak > 0:
            position_size *= (1 - 0.1 * self.current_losing_streak)
            
        # 确保仓位大小在合理范围内
        position_size = max(0.01, min(position_size, max_size))
        
        return position_size
        
    def _check_risk_limits(self, current_price: float) -> bool:
        """检查风险限制"""
        # 检查每日最大亏损
        if self.total_pnl < -self.adaptive_params['max_daily_loss'] * current_price:
            logger.warning("Daily loss limit reached")
            return True
            
        # 检查总最大亏损
        if self.total_pnl < -self.adaptive_params['max_total_loss'] * current_price:
            logger.warning("Total loss limit reached")
            return True
            
        return False
        
    def update_trade_stats(self, trade_result: Dict[str, Any]):
        """更新交易统计"""
        self.trade_history.append(trade_result)
        self.total_pnl += trade_result['pnl']
        
        if trade_result['pnl'] > 0:
            self.win_count += 1
            self.current_losing_streak = 0
        else:
            self.loss_count += 1
            self.current_losing_streak += 1
            
        # 更新最大回撤
        current_drawdown = abs(trade_result['pnl']) / trade_result['entry_price']
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # 自适应调整参数
        self._adjust_parameters(trade_result)
        
        logger.info(f"Trade updated - PnL: {trade_result['pnl']:.2f}, "
                   f"Win Rate: {self.win_count/(self.win_count + self.loss_count):.2%}, "
                   f"Max Drawdown: {self.max_drawdown:.2%}")
                   
    def _adjust_parameters(self, trade_result: Dict[str, Any]):
        """自适应调整参数"""
        # 根据交易结果调整止损和止盈
        if trade_result['pnl'] < 0:
            # 亏损时减小止损距离
            self.adaptive_params['stop_loss_atr'] *= 0.95
        else:
            # 盈利时增加止盈距离
            self.adaptive_params['take_profit_atr'] *= 1.05
            
        # 根据波动率调整仓位大小
        vix = self.market_sentiment.get_vix(datetime.now().strftime('%Y-%m-%d'))
        if vix > 30:
            self.adaptive_params['base_position_size'] *= 0.8
        elif vix < 15:
            self.adaptive_params['base_position_size'] *= 1.1
            
        # 确保参数在合理范围内
        self.adaptive_params['stop_loss_atr'] = max(1.5, min(self.adaptive_params['stop_loss_atr'], 3.0))
        self.adaptive_params['take_profit_atr'] = max(2.0, min(self.adaptive_params['take_profit_atr'], 4.0))
        self.adaptive_params['base_position_size'] = max(0.05, min(self.adaptive_params['base_position_size'], 0.2))
        
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """提取信号组件"""
        return {
            'trend': data['MA20'] - data['MA50'],
            'momentum': data['RSI'],
            'volatility': data['ATR']
        }
        
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """获取信号元数据"""
        return {
            'trend': {
                'name': 'Moving Average Trend',
                'description': 'Difference between MA20 and MA50',
                'range': [-np.inf, np.inf],
                'interpretation': 'Positive values indicate uptrend'
            },
            'momentum': {
                'name': 'RSI',
                'description': 'Relative Strength Index',
                'range': [0, 100],
                'interpretation': 'Above 50 indicates bullish momentum'
            },
            'volatility': {
                'name': 'ATR',
                'description': 'Average True Range',
                'range': [0, np.inf],
                'interpretation': 'Higher values indicate higher volatility'
            }
        }
        
    def optimize_parameters(self, data: pd.DataFrame, param_ranges: Dict[str, List[float]]) -> Dict[str, float]:
        """优化策略参数"""
        best_params = self.default_params.copy()
        best_sharpe = -np.inf
        
        # 生成参数组合
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for params in param_combinations:
            # 更新参数
            self.adaptive_params.update(params)
            
            # 运行回测
            results = self.run_backtest(data)
            
            # 计算夏普比率
            sharpe = self._calculate_sharpe_ratio(results)
            
            # 更新最佳参数
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params.copy()
                
        return best_params
        
    def _generate_param_combinations(self, param_ranges: Dict[str, List[float]]) -> List[Dict[str, float]]:
        """生成参数组合"""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = list(itertools.product(*param_values))
        
        return [dict(zip(param_names, combo)) for combo in combinations]
        
    def _calculate_sharpe_ratio(self, results: pd.DataFrame) -> float:
        """计算夏普比率"""
        returns = results['pnl'].pct_change()
        return np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0 