import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from strategy.strategy_base import Strategy


class CPGWStrategy(Strategy):
    """
    CPGW (长庄股王) 策略
    
    该策略基于长庄线、游资线和主力线的交叉关系生成买卖信号。
    
    买入条件:
    1. 长庄线 < 12 且 主力线 < 8 且 (游资线 < 7.2 或 前一天主力线 < 5) 且 (主力线 > 前一天主力线 或 游资线 > 前一天游资线)
    2. 或 长庄线 < 8 且 主力线 < 7 且 游资线 < 15 且 游资线 > 前一天游资线
    3. 或 长庄线 < 10 且 主力线 < 7 且 游资线 < 1
    
    卖出条件:
    主力线 < 前一天主力线 且 前一天主力线 > 80 且 (前一天游资线 > 95 或 前两天游资线 > 95) 且 长庄线 > 60 且
    游资线 < 83.5 且 游资线 < 主力线 且 游资线 < 主力线 + 4
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化CPGW策略
        
        参数:
            parameters: 策略参数字典，可包含以下键:
                - long_window: 长庄线窗口大小，默认34
                - hot_money_window: 游资线窗口大小，默认14
                - main_force_window: 主力线窗口大小，默认34
                - main_force_span: 主力线EMA平滑参数，默认4
        """
        default_params = {
            'long_window': 34,
            'hot_money_window': 14,
            'main_force_window': 34,
            'main_force_span': 4,
            'stop_loss_pct': 5.0,
            'take_profit_pct': 15.0,
            'max_position_size': 0.2
        }

        # 合并默认参数和传入的参数
        if parameters:
            default_params.update(parameters)

        super().__init__("CPGWStrategy", default_params)
        self.logger = logging.getLogger(__name__)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算CPGW策略所需的技术指标
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了技术指标的DataFrame
        """
        if data is None or data.empty:
            self.logger.warning("数据为空，无法计算指标")
            return data

        # 检查必要的列是否存在
        required_columns = ['close']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"数据中缺少 {col} 列")
                return data

        try:
            # 获取参数
            long_window = self.parameters['long_window']
            hot_money_window = self.parameters['hot_money_window']
            main_force_window = self.parameters['main_force_window']
            main_force_span = self.parameters['main_force_span']

            # 计算指标 A、B 和 D
            data['A'] = data['close'].rolling(window=long_window).apply(
                lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10)
            )
            data['A'] = data['A'].rolling(window=19).mean()

            data['B'] = data['close'].rolling(window=hot_money_window).apply(
                lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10)
            )

            data['D'] = data['close'].rolling(window=main_force_window).apply(
                lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10)
            )
            data['D'] = data['D'].ewm(span=main_force_span).mean()

            # 计算长庄线、游资线和主力线
            data['Long_Line'] = data['A'] + 100
            data['Hot_Money_Line'] = data['B'] + 100
            data['Main_Force_Line'] = data['D'] + 100

            return data
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        根据CPGW策略生成交易信号
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            添加了信号列的DataFrame，其中:
            1: 买入信号
            0: 持有/无信号
            -1: 卖出信号
        """
        if data is None or data.empty:
            self.logger.warning("数据为空，无法生成信号")
            return data

        # 检查必要的列是否存在
        required_columns = ['Long_Line', 'Hot_Money_Line', 'Main_Force_Line']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"数据中缺少 {col} 列")
                return data

        try:
            # 初始化信号列
            data['signal'] = 0

            # 买卖信号定义
            sell_condition = (
                    (data['Main_Force_Line'] < data['Main_Force_Line'].shift(1)) &
                    (data['Main_Force_Line'].shift(1) > 80) &
                    ((data['Hot_Money_Line'].shift(1) > 95) | (data['Hot_Money_Line'].shift(2) > 95)) &
                    (data['Long_Line'] > 60) &
                    (data['Hot_Money_Line'] < 83.5) &
                    (data['Hot_Money_Line'] < data['Main_Force_Line']) &
                    (data['Hot_Money_Line'] < data['Main_Force_Line'] + 4)
            )

            buy_condition1 = (
                    (data['Long_Line'] < 12) &
                    (data['Main_Force_Line'] < 8) &
                    ((data['Hot_Money_Line'] < 7.2) | (data['Main_Force_Line'].shift(1) < 5)) &
                    ((data['Main_Force_Line'] > data['Main_Force_Line'].shift(1)) |
                     (data['Hot_Money_Line'] > data['Hot_Money_Line'].shift(1)))
            )

            buy_condition2 = (
                    (data['Long_Line'] < 8) &
                    (data['Main_Force_Line'] < 7) &
                    (data['Hot_Money_Line'] < 15) &
                    (data['Hot_Money_Line'] > data['Hot_Money_Line'].shift(1))
            )

            buy_condition3 = (
                    (data['Long_Line'] < 10) &
                    (data['Main_Force_Line'] < 7) &
                    (data['Hot_Money_Line'] < 1)
            )

            buy_condition = buy_condition1 | buy_condition2 | buy_condition3

            # 设置信号
            data.loc[buy_condition, 'signal'] = 1
            data.loc[sell_condition, 'signal'] = -1

            return data
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return data

    def get_market_regime(self, data: pd.DataFrame) -> str:
        """
        根据市场指标判断当前市场环境
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            市场环境类型: "bullish", "bearish", "sideways", "volatile"
        """
        if data is None or data.empty:
            return "unknown"

        try:
            # 获取最近的数据点
            recent_data = data.iloc[-20:]

            # 计算主力线和游资线的平均值
            avg_main_force = recent_data['Main_Force_Line'].mean() if 'Main_Force_Line' in recent_data.columns else 50
            avg_hot_money = recent_data['Hot_Money_Line'].mean() if 'Hot_Money_Line' in recent_data.columns else 50

            # 计算收盘价的波动率
            volatility = recent_data['close'].pct_change().std() * 100 if 'close' in recent_data.columns else 1.5

            # 判断市场环境
            if avg_main_force > 70 and avg_hot_money > 70:
                return "bullish"
            elif avg_main_force < 30 and avg_hot_money < 30:
                return "bearish"
            elif volatility > 2.5:
                return "volatile"
            else:
                return "sideways"
        except Exception as e:
            self.logger.error(f"判断市场环境时出错: {e}")
            return "normal"

    def get_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """
        根据信号和市场环境确定仓位大小
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            signal: 交易信号 (1: 买入, -1: 卖出, 0: 无信号)
            
        返回:
            仓位大小 (0.0 - 1.0)
        """
        if signal == 0:
            return 0.0

        max_position = self.parameters.get('max_position_size', 0.2)

        try:
            # 获取市场环境
            market_regime = self.get_market_regime(data)

            # 根据市场环境调整仓位
            if market_regime == "bullish":
                return max_position
            elif market_regime == "bearish":
                return max_position * 0.5 if signal == 1 else max_position
            elif market_regime == "volatile":
                return max_position * 0.7
            else:  # sideways or unknown
                return max_position * 0.8
        except Exception as e:
            self.logger.error(f"计算仓位大小时出错: {e}")
            return max_position * 0.5

    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止损价格
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            entry_price: 入场价格
            position: 仓位方向 (1: 多头, -1: 空头)
            
        返回:
            止损价格
        """
        stop_loss_pct = self.parameters.get('stop_loss_pct', 5.0)

        if position == 1:  # 多头
            return entry_price * (1 - stop_loss_pct / 100)
        elif position == -1:  # 空头
            return entry_price * (1 + stop_loss_pct / 100)
        else:
            return 0.0

    def get_take_profit(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止盈价格
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            entry_price: 入场价格
            position: 仓位方向 (1: 多头, -1: 空头)
            
        返回:
            止盈价格
        """
        take_profit_pct = self.parameters.get('take_profit_pct', 15.0)

        if position == 1:  # 多头
            return entry_price * (1 + take_profit_pct / 100)
        elif position == -1:  # 空头
            return entry_price * (1 - take_profit_pct / 100)
        else:
            return 0.0

    def should_adjust_stop_loss(self, data: pd.DataFrame, current_price: float,
                                stop_loss: float, position: int) -> float:
        """
        判断是否应该调整止损价格（追踪止损）
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            current_price: 当前价格
            stop_loss: 当前止损价格
            position: 仓位方向 (1: 多头, -1: 空头)
            
        返回:
            新的止损价格
        """
        if data is None or data.empty:
            return stop_loss

        try:
            # 获取最近的数据
            recent_data = data.iloc[-5:]

            # 计算ATR (Average True Range)
            high = recent_data['High'] if 'High' in recent_data.columns else recent_data['close'] * 1.01
            low = recent_data['Low'] if 'Low' in recent_data.columns else recent_data['close'] * 0.99
            close = recent_data['Close'] if 'Close' in recent_data.columns else recent_data['close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.mean()

            # 根据ATR调整止损
            if position == 1:  # 多头
                new_stop = current_price - 2 * atr
                return max(new_stop, stop_loss)  # 只上调止损，不下调
            elif position == -1:  # 空头
                new_stop = current_price + 2 * atr
                return min(new_stop, stop_loss)  # 只下调止损，不上调
            else:
                return stop_loss
        except Exception as e:
            self.logger.error(f"调整止损时出错: {e}")
            return stop_loss

    def optimize_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
            data: 包含价格数据的DataFrame
            
        返回:
            优化后的参数字典
        """
        # 这里可以实现参数优化逻辑，如网格搜索或遗传算法
        # 简单起见，这里返回默认参数
        return self.parameters

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        返回:
            包含策略信息的字典
        """
        return {
            "name": self.name,
            "version": "1.0.0",
            "description": "CPGW (长庄股王) 策略，基于长庄线、游资线和主力线的交叉关系生成买卖信号",
            "parameters": self.parameters,
            "author": "System",
            "creation_date": "2023-01-01",
            "last_modified_date": "2023-12-31",
            "risk_level": "medium",
            "performance_metrics": {
                "sharpe_ratio": None,
                "max_drawdown": None,
                "win_rate": None
            },
            "suitable_market_regimes": ["bullish", "sideways"],
            "tags": ["technical", "trend-following", "momentum"]
        }
