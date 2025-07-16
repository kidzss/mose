import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

from .strategy_base import Strategy, MarketRegime

logger = logging.getLogger(__name__)

class MarketForecastStrategy(Strategy):
    """
    Market Forecast策略 - 市场预测策略
    
    专业的市场预测策略基于多个时间周期的价格变化率，
    通过综合短期、中期和长期的市场趋势来生成交易信号。
    
    特点：
    1. 多时间周期分析（短期、中期、长期）
    2. 动态权重调整
    3. 市场状态识别
    4. 信号确认机制
    5. 风险管理集成
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化Market Forecast策略
        
        参数:
            parameters: 策略参数字典，可包含：
                - short_length: 短期周期，默认3
                - medium_length: 中期周期，默认14
                - long_length: 长期周期，默认30
                - buy_threshold: 买入阈值，默认70
                - sell_threshold: 卖出阈值，默认30
                - volume_weight: 成交量权重，默认0.2
                - volatility_weight: 波动率权重，默认0.1
                - stop_loss: 止损比例，默认0.05
                - take_profit: 止盈比例，默认0.15
        """
        # 设置默认参数
        default_params = {
            'short_length': 3,      # 短期周期
            'medium_length': 14,    # 中期周期
            'long_length': 30,      # 长期周期
            'buy_threshold': 70,    # 买入阈值
            'sell_threshold': 30,   # 卖出阈值
            'volume_weight': 0.2,   # 成交量权重
            'volatility_weight': 0.1, # 波动率权重
            'stop_loss': 0.05,      # 止损比例
            'take_profit': 0.15,    # 止盈比例
            'signal_confirmation': True, # 信号确认
            'dynamic_weights': True  # 动态权重调整
        }
        
        # 更新默认参数
        if parameters:
            default_params.update(parameters)
        
        # 验证参数
        self._validate_parameters(default_params)
        
        # 初始化基类
        super().__init__('MarketForecastStrategy', default_params)
        
        logger.info(f"✅ Market Forecast策略初始化完成，参数: {default_params}")
    
    def _validate_parameters(self, params: Dict[str, Any]):
        """验证策略参数"""
        if params['short_length'] >= params['medium_length']:
            raise ValueError("短期周期必须小于中期周期")
        if params['medium_length'] >= params['long_length']:
            raise ValueError("中期周期必须小于长期周期")
        if params['buy_threshold'] <= params['sell_threshold']:
            raise ValueError("买入阈值必须大于卖出阈值")
        if not (0 <= params['volume_weight'] <= 1):
            raise ValueError("成交量权重必须在0-1之间")
        if not (0 <= params['volatility_weight'] <= 1):
            raise ValueError("波动率权重必须在0-1之间")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        
        # 标准化列名，确保使用小写
        column_mapping = {
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        }
        
        # 重命名列（如果存在）
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # 确保使用小写列名
        close_col = 'close' if 'close' in df.columns else 'Close'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        
        short_length = self.parameters['short_length']
        medium_length = self.parameters['medium_length']
        long_length = self.parameters['long_length']
        
        # 计算短期、中期和长期的变化率
        df['mf_short_change'] = df[close_col].pct_change(short_length) * 100
        df['mf_medium_change'] = df[close_col].pct_change(medium_length) * 100
        df['mf_long_change'] = df[close_col].pct_change(long_length) * 100
        
        # 计算成交量指标
        df['volume_ma'] = df[volume_col].rolling(window=20).mean()
        df['volume_ratio'] = df[volume_col] / df['volume_ma']
        
        # 计算波动率指标
        df['volatility'] = df[close_col].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # 计算Market Forecast指标 (三个周期变化率的归一化)
        max_value = 100
        min_value = -100
        
        df['mf_short_norm'] = (df['mf_short_change'] - min_value) / (max_value - min_value) * 100
        df['mf_medium_norm'] = (df['mf_medium_change'] - min_value) / (max_value - min_value) * 100
        df['mf_long_norm'] = (df['mf_long_change'] - min_value) / (max_value - min_value) * 100
        
        # 动态权重调整
        if self.parameters['dynamic_weights']:
            weights = self._calculate_dynamic_weights(df)
        else:
            weights = {'short': 0.4, 'medium': 0.3, 'long': 0.3}
        
        # 计算总体的Market Forecast指标
        df['mf_indicator'] = (
            df['mf_short_norm'] * weights['short'] + 
            df['mf_medium_norm'] * weights['medium'] + 
            df['mf_long_norm'] * weights['long']
        )
        
        # 添加成交量权重
        volume_weight = self.parameters['volume_weight']
        df['mf_indicator'] = df['mf_indicator'] * (1 + volume_weight * (df['volume_ratio'] - 1))
        
        # 添加波动率权重
        volatility_weight = self.parameters['volatility_weight']
        df['mf_indicator'] = df['mf_indicator'] * (1 - volatility_weight * df['volatility'])
        
        # 确保指标在合理范围内
        df['mf_indicator'] = df['mf_indicator'].clip(0, 100)
        
        return df
    
    def _calculate_dynamic_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算动态权重"""
        # 基于市场状态调整权重
        volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.2
        
        if volatility > 0.3:  # 高波动市场
            return {'short': 0.3, 'medium': 0.4, 'long': 0.3}
        elif volatility < 0.1:  # 低波动市场
            return {'short': 0.5, 'medium': 0.3, 'long': 0.2}
        else:  # 正常波动市场
            return {'short': 0.4, 'medium': 0.3, 'long': 0.3}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame
        """
        # 计算技术指标
        df = self.calculate_indicators(data)
        buy_threshold = self.parameters['buy_threshold']
        sell_threshold = self.parameters['sell_threshold']
        
        # 初始化信号列
        df['signal'] = 0
        
        # 计算Market Forecast买入/卖出条件
        # 买入条件：指标上穿买入阈值
        buy_condition = (df['mf_indicator'] > buy_threshold) & (df['mf_indicator'].shift(1) <= buy_threshold)
        # 卖出条件：指标下穿卖出阈值
        sell_condition = (df['mf_indicator'] < sell_threshold) & (df['mf_indicator'].shift(1) >= sell_threshold)
        
        # 信号确认机制
        if self.parameters['signal_confirmation']:
            # 买入确认：短期和中期趋势一致
            buy_confirmation = (
                (df['mf_short_norm'] > 50) & 
                (df['mf_medium_norm'] > 50) &
                (df['volume_ratio'] > 1.2)
            )
            buy_condition = buy_condition & buy_confirmation
            
            # 卖出确认：短期和中期趋势一致
            sell_confirmation = (
                (df['mf_short_norm'] < 50) & 
                (df['mf_medium_norm'] < 50) &
                (df['volume_ratio'] > 1.2)
            )
            sell_condition = sell_condition & sell_confirmation
        
        # 生成信号
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # 填充NaN值
        df = df.fillna(0)
        
        return df
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        result = self.calculate_indicators(data)
        
        # 提取关键组件
        components = {
            'short_change': result.get('mf_short_change', pd.Series()),
            'medium_change': result.get('mf_medium_change', pd.Series()),
            'long_change': result.get('mf_long_change', pd.Series()),
            'short_norm': result.get('mf_short_norm', pd.Series()),
            'medium_norm': result.get('mf_medium_norm', pd.Series()),
            'long_norm': result.get('mf_long_norm', pd.Series()),
            'mf_indicator': result.get('mf_indicator', pd.Series()),
            'volume_ratio': result.get('volume_ratio', pd.Series()),
            'volatility': result.get('volatility', pd.Series()),
            'price': result.get('close', pd.Series())
        }
        
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            'mf_short_norm': {
                'name': '短期市场预测',
                'description': f"{self.parameters['short_length']}日收益率归一化指标",
                'type': 'momentum',
                'time_scale': 'short',
                'min_value': 0,
                'max_value': 100
            },
            'mf_medium_norm': {
                'name': '中期市场预测',
                'description': f"{self.parameters['medium_length']}日收益率归一化指标",
                'type': 'momentum',
                'time_scale': 'medium',
                'min_value': 0,
                'max_value': 100
            },
            'mf_long_norm': {
                'name': '长期市场预测',
                'description': f"{self.parameters['long_length']}日收益率归一化指标",
                'type': 'momentum',
                'time_scale': 'long',
                'min_value': 0,
                'max_value': 100
            },
            'mf_indicator': {
                'name': '综合市场预测指标',
                'description': '综合短期、中期和长期市场预测的加权平均',
                'type': 'composite',
                'time_scale': 'multi',
                'min_value': 0,
                'max_value': 100
            },
            'volume_ratio': {
                'name': '成交量比率',
                'description': '当前成交量与20日平均成交量的比值',
                'type': 'volume',
                'time_scale': 'short',
                'min_value': 0,
                'max_value': float('inf')
            },
            'volatility': {
                'name': '波动率',
                'description': '20日年化波动率',
                'type': 'volatility',
                'time_scale': 'medium',
                'min_value': 0,
                'max_value': float('inf')
            }
        }
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """获取市场环境"""
        try:
            df = self.calculate_indicators(data)
            if df.empty:
                return MarketRegime.UNKNOWN
            
            latest_indicator = df['mf_indicator'].iloc[-1]
            volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.2
            
            if latest_indicator > 70 and volatility < 0.2:
                return MarketRegime.BULLISH
            elif latest_indicator < 30 and volatility < 0.2:
                return MarketRegime.BEARISH
            elif volatility > 0.3:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logger.error(f"获取市场环境时出错: {e}")
            return MarketRegime.UNKNOWN
    
    def get_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """计算仓位大小"""
        if signal == 0:
            return 0.0
        
        try:
            df = self.calculate_indicators(data)
            if df.empty:
                return 0.1  # 默认10%仓位
            
            # 基于信号强度和波动率调整仓位
            indicator = df['mf_indicator'].iloc[-1]
            volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.2
            
            # 基础仓位
            base_position = 0.2
            
            # 根据指标强度调整
            if signal > 0:  # 买入信号
                strength_factor = (indicator - 50) / 50  # 0-1之间
            else:  # 卖出信号
                strength_factor = (50 - indicator) / 50  # 0-1之间
            
            # 根据波动率调整
            volatility_factor = max(0.5, 1 - volatility)
            
            position_size = base_position * strength_factor * volatility_factor
            
            return min(max(position_size, 0.05), 0.3)  # 限制在5%-30%之间
            
        except Exception as e:
            logger.error(f"计算仓位大小时出错: {e}")
            return 0.1
    
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """计算止损价格"""
        stop_loss_pct = self.parameters['stop_loss']
        
        if position == 1:  # 多头
            return entry_price * (1 - stop_loss_pct)
        elif position == -1:  # 空头
            return entry_price * (1 + stop_loss_pct)
        else:
            return 0.0
    
    def get_take_profit(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """计算止盈价格"""
        take_profit_pct = self.parameters['take_profit']
        
        if position == 1:  # 多头
            return entry_price * (1 + take_profit_pct)
        elif position == -1:  # 空头
            return entry_price * (1 - take_profit_pct)
        else:
            return 0.0 