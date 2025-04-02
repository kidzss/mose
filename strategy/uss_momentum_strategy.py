import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from .strategy_base import Strategy
from strategy_optimizer.utils.technical_indicators import calculate_technical_indicators

class MomentumStrategy(Strategy):
    """
    动量交易策略
    
    策略说明:
    1. 买入条件: 
       - 价格突破N日高点
       - RSI大于50（上升趋势）
       - MACD柱状图为正（上升动量）
       - 如果risk_control=True，则还需要价格大于SMA_200（长期上升趋势）
    
    2. 卖出条件:
       - 价格跌破M日低点
       - 或者RSI跌破30（超卖区域）
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化动量策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - high_period: 突破周期，默认20
                - low_period: 跌破周期，默认10
                - rsi_period: RSI周期，默认14
                - rsi_upper: RSI上限，默认70
                - rsi_lower: RSI下限，默认30
                - macd_fast: MACD快线，默认12
                - macd_slow: MACD慢线，默认26
                - macd_signal: MACD信号线，默认9
                - sma_period: 长期均线周期，默认200
                - risk_control: 是否启用风险控制，默认True
        """
        default_params = {
            'high_period': 20,
            'low_period': 10,
            'rsi_period': 14,
            'rsi_upper': 70,
            'rsi_lower': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'sma_period': 200,
            'risk_control': True
        }
        
        # 合并参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('MomentumStrategy', default_params)
        self.logger.info(f"初始化动量策略，参数: {default_params}")
        self.version = '1.0.0'
        
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        根据技术指标生成交易信号
        
        参数:
            data: 包含技术指标的DataFrame
            
        返回:
            添加了交易信号的DataFrame
        """
        if data is None or data.empty:
            self.logger.warning("数据为空，无法生成信号")
            return np.zeros(0)
            
        try:
            # 检查必需的列是否存在
            required_columns = [
                'high_20', 'low_10', 'SMA_200', 'RSI', 'MACD', 'MACD_signal'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"缺少必需的列: {', '.join(missing_columns)}")
                return np.zeros(len(data))
            
            # 初始化信号列
            signals = np.zeros(len(data))
            position = np.zeros(len(data))
            
            # 使用较短的预热期
            warmup_period = 50  # 使用最长的技术指标周期作为预热期
            if len(data) <= warmup_period:
                return signals
            
            for i in range(warmup_period, len(data)):
                # 更新持仓状态
                position[i] = position[i-1]
                
                # 如果持有多头，检查止损止盈
                if position[i-1] == 1:
                    # 计算止损价和止盈价
                    entry_price = data['close'].iloc[position.nonzero()[0][-1]]
                    stop_loss_price = entry_price * 0.75  # 25%的止损
                    profit_target_price = entry_price * 1.5  # 50%的止盈
                    
                    # 检查止损和止盈
                    if data['close'].iloc[i] <= stop_loss_price:  # 触发止损
                        signals[i] = -1
                        position[i] = 0
                        continue
                    elif data['close'].iloc[i] >= profit_target_price:  # 触发止盈
                        signals[i] = -1
                        position[i] = 0
                        continue
                
                # 买入条件
                if position[i-1] == 0:  # 仅在空仓时考虑买入
                    # 主要条件
                    main_conditions = [
                        data['close'].iloc[i] > data['SMA_200'].iloc[i],  # 价格在长期均线上方
                        data['RSI'].iloc[i] > 40,  # RSI大于40（降低门槛）
                        data['MACD'].iloc[i] > data['MACD_signal'].iloc[i],  # MACD金叉
                    ]
                    
                    # 辅助条件
                    aux_conditions = [
                        data['volume'].iloc[i] > data['volume'].rolling(window=20).mean().iloc[i],  # 成交量放大
                        data['close'].iloc[i] > data['high_20'].iloc[i-1],  # 突破20日高点
                        data['close'].pct_change().rolling(window=20).std().iloc[i] < 0.03,  # 波动率相对正常
                    ]
                    
                    # 生成买入信号：满足所有主要条件和至少2个辅助条件
                    if all(main_conditions) and sum(aux_conditions) >= 2:
                        signals[i] = 1
                        position[i] = 1
                        continue
                
                # 卖出条件
                elif position[i-1] == 1:  # 仅在持仓时考虑卖出
                    # 主要条件
                    main_conditions = [
                        data['RSI'].iloc[i] < 35,  # RSI超卖（调整阈值）
                        data['MACD'].iloc[i] < data['MACD_signal'].iloc[i],  # MACD死叉
                        data['close'].iloc[i] < data['low_10'].iloc[i-1]  # 跌破10日低点
                    ]
                    
                    # 辅助条件
                    aux_conditions = [
                        data['volume'].iloc[i] > data['volume'].rolling(window=20).mean().iloc[i] * 1.3,  # 成交量明显放大
                        data['close'].pct_change().rolling(window=20).std().iloc[i] > 0.04  # 波动率增加
                    ]
                    
                    # 生成卖出信号：满足至少2个主要条件和1个辅助条件
                    if sum(main_conditions) >= 2 and any(aux_conditions):
                        signals[i] = -1
                        position[i] = 0
                        continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {str(e)}")
            return np.zeros(len(data))
        
    def get_market_regime(self, data: pd.DataFrame) -> str:
        """
        判断当前市场环境
        
        参数:
            data: 市场数据
            
        返回:
            市场环境类型: 'trend', 'range', 'volatile'
        """
        if data is None or data.empty or len(data) < 20:
            return 'unknown'
            
        # 计算ADX作为趋势强度指标
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            # 计算+DI和-DI
            high = data['high']
            low = data['low']
            close = data['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff()
            
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
            minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
            
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            adx = dx.rolling(window=14).mean()
            
            # 获取最新ADX值
            current_adx = adx.iloc[-1]
            
            # 计算波动率
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # 根据ADX和波动率判断市场环境
            if current_adx > 25:  # 强趋势
                return 'trend'
            elif volatility > 0.02:  # 高波动
                return 'volatile'
            else:
                return 'range'
        
        return 'unknown'
        
    def get_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """
        计算仓位大小 - 基于波动率调整
        
        参数:
            data: 市场数据
            signal: 信号(1, 0, -1)
            
        返回:
            仓位大小(0.0-1.0)
        """
        if signal == 0 or data is None or data.empty or len(data) < 20:
            return 0.0
            
        # 计算波动率
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # 目标波动率
            target_vol = 0.01  # 1%
            
            # 波动率调整系数
            vol_factor = min(target_vol / volatility, 1.0) if volatility > 0 else 1.0
            
            # 根据RSI调整仓位
            rsi = data['RSI'].iloc[-1]
            rsi_factor = 1.0
            
            if rsi > 70:  # 超买区域
                rsi_factor = 0.7
            elif rsi < 30:  # 超卖区域
                rsi_factor = 0.8
                
            return abs(signal) * vol_factor * rsi_factor
            
        return float(abs(signal))
        
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止损价格 - 使用ATR方法
        
        参数:
            data: 市场数据
            entry_price: 入场价格
            position: 仓位方向(1=多, -1=空)
            
        返回:
            止损价格
        """
        if data is None or data.empty or len(data) < 14:
            return super().get_stop_loss(data, entry_price, position)
            
        # 计算ATR
        high = data['high'].iloc[-14:]
        low = data['low'].iloc[-14:]
        close = data['close'].iloc[-14:]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.mean()
        
        # 使用2.5倍ATR作为止损距离
        if position > 0:
            return entry_price - 2.5 * atr
        elif position < 0:
            return entry_price + 2.5 * atr
            
        return 0.0
        
    def optimize_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
            data: 历史数据
            
        返回:
            优化后的参数字典
        """
        self.logger.info("开始优化动量策略参数")
        
        # 这里可以实现网格搜索或其他优化方法
        # 简单起见，这里只测试几组参数
        
        best_params = self.parameters.copy()
        best_return = -float('inf')
        
        # 测试不同的参数组合
        for high_period in [10, 20, 30]:
            for low_period in [5, 10, 15]:
                for rsi_period in [9, 14, 21]:
                    test_params = {
                        'high_period': high_period,
                        'low_period': low_period,
                        'rsi_period': rsi_period,
                        'rsi_upper': 70,
                        'rsi_lower': 30,
                        'macd_fast': 12,
                        'macd_slow': 26,
                        'macd_signal': 9,
                        'sma_period': 200,
                        'risk_control': True
                    }
                    
                    # 保存原参数
                    original_params = self.parameters.copy()
                    
                    # 设置测试参数
                    self.parameters = test_params
                    
                    # 生成信号
                    result = self.generate_signals(data.copy())
                    
                    # 计算收益
                    if 'signal' in result.columns and 'close' in result.columns:
                        # 计算策略收益
                        result['returns'] = result['close'].pct_change()
                        result['strategy_returns'] = result['signal'].shift(1) * result['returns']
                        
                        # 计算累积收益
                        cumulative_return = (1 + result['strategy_returns'].fillna(0)).cumprod().iloc[-1] - 1
                        
                        if cumulative_return > best_return:
                            best_return = cumulative_return
                            best_params = test_params.copy()
                            
                    # 恢复原参数
                    self.parameters = original_params
        
        self.logger.info(f"优化完成，最佳参数: {best_params}，收益率: {best_return:.2%}")
        return best_params

    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取动量策略的核心信号组件
        
        分解为以下核心信号组件:
        - price_breakout: 价格突破信号
        - rsi_momentum: RSI动量信号
        - macd_momentum: MACD动量信号
        - trend_filter: 趋势过滤信号
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        # 确保数据包含所需指标
        if not self._validate_input_data(data):
            return {"composite": pd.Series(0, index=data.index)}
        
        # 获取参数
        high_period = self.parameters['high_period']
        low_period = self.parameters['low_period']
        rsi_period = self.parameters['rsi_period']
        rsi_upper = self.parameters['rsi_upper']
        rsi_lower = self.parameters['rsi_lower']
        sma_period = self.parameters['sma_period']
        
        signals = {}
        
        # 1. 价格突破信号 (-1 到 1)
        high_n = data['high'].rolling(window=high_period).max()
        low_n = data['low'].rolling(window=low_period).min()
        
        # 归一化到 [-1, 1] 范围
        # 1表示突破高点，-1表示跌破低点，0表示在区间内
        price_position = (data['close'] - low_n) / (high_n - low_n + 1e-10) * 2 - 1
        signals['price_breakout'] = price_position.clip(-1, 1)
        
        # 2. RSI动量信号 (-1 到 1)
        # 转换RSI从[0,100]到[-1,1]
        if 'RSI' in data.columns:
            rsi_signal = (data['RSI'] / 50) - 1
            signals['rsi_momentum'] = rsi_signal.clip(-1, 1)
        else:
            # 尝试计算RSI
            try:
                import talib
                rsi = talib.RSI(data['close'], timeperiod=rsi_period)
                rsi_signal = (rsi / 50) - 1
                signals['rsi_momentum'] = rsi_signal.clip(-1, 1)
            except:
                signals['rsi_momentum'] = pd.Series(0, index=data.index)
        
        # 3. MACD动量信号 (-1 到 1)
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            # 计算MACD柱状图并归一化
            macd_hist = data['macd'] - data['macd_signal']
            # 使用绝对值的滚动最大值来归一化
            abs_max = macd_hist.abs().rolling(window=60).max()
            macd_norm = macd_hist / (abs_max + 1e-10)
            signals['macd_momentum'] = macd_norm.clip(-1, 1)
        else:
            signals['macd_momentum'] = pd.Series(0, index=data.index)
        
        # 4. 趋势过滤信号 (-1 到 1)
        if 'sma_200' in data.columns:
            sma_200 = data['sma_200']
        else:
            # 尝试计算SMA
            try:
                import talib
                sma_200 = talib.SMA(data['close'], timeperiod=sma_period)
            except:
                sma_200 = data['close'].rolling(window=sma_period).mean()
                
        # 计算价格相对于长期均线的位置
        trend_signal = (data['close'] / sma_200 - 1) * 5  # 放大差异
        signals['trend_filter'] = trend_signal.clip(-1, 1)
        
        # 5. 生成复合信号
        # 将各个组件加权组合
        signals['composite'] = (
            signals['price_breakout'] * 0.3 + 
            signals['rsi_momentum'] * 0.3 + 
            signals['macd_momentum'] * 0.2 + 
            signals['trend_filter'] * 0.2
        )
        
        # 填充NaN值
        for key in signals:
            signals[key] = signals[key].fillna(0)
            
        return signals
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "price_breakout": {
                "type": "breakout",
                "time_scale": "medium",
                "description": f"价格突破{self.parameters['high_period']}日高点或跌破{self.parameters['low_period']}日低点的信号",
                "min_value": -1,
                "max_value": 1
            },
            "rsi_momentum": {
                "type": "momentum",
                "time_scale": "short",
                "description": f"基于{self.parameters['rsi_period']}日RSI的动量信号",
                "min_value": -1,
                "max_value": 1
            },
            "macd_momentum": {
                "type": "momentum",
                "time_scale": "medium",
                "description": "基于MACD柱状图的动量信号",
                "min_value": -1,
                "max_value": 1
            },
            "trend_filter": {
                "type": "trend",
                "time_scale": "long",
                "description": f"基于{self.parameters['sma_period']}日均线的趋势过滤信号",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "medium",
                "description": "动量策略的综合信号",
                "min_value": -1,
                "max_value": 1
            }
        }
        
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据是否包含所需的列
        
        参数:
            data: 输入数据
            
        返回:
            数据是否有效
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"输入数据缺少必要的列: {col}")
                return False
                
        return True 