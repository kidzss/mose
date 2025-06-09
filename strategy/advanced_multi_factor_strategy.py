import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

from .strategy_base import Strategy
from .indicators.indicators import TechnicalIndicators
from .indicators.volatility import atr, historical_volatility, bollinger_bandwidth, keltner_channel_width
from .indicators.volume import volume_sma, on_balance_volume, chaikin_money_flow, money_flow_index
from .indicators.oscillators import stochastic_oscillator, williams_r, roc, cci, awesome_oscillator
from .indicators.trend_strength import aroon, vortex_indicator, supertrend
from .indicators.support_resistance import pivot_points_traditional, price_channels, donchian_channels

class AdvancedMultiFactorStrategy(Strategy):
    """
    高级多因子策略
    
    策略说明:
    1. 结合多种技术指标构建综合市场分析框架
    2. 基于五大类指标计算独立因子:
       - 趋势因子: 评估价格趋势方向和强度
       - 动量因子: 评估价格变化的速度和力度
       - 波动率因子: 评估市场波动和稳定性
       - 成交量因子: 评估交易活跃度和资金流向
       - 支撑阻力因子: 评估价格在关键水平的表现
    3. 各因子独立生成信号，然后通过加权方式合成最终信号
    
    买入条件:
    - 综合信号大于买入阈值
    - 波动率处于适中或走低水平
    - 成交量确认价格走势
    
    卖出条件:
    - 综合信号小于卖出阈值
    - 价格触及或突破关键阻力位
    - 出现反转形态
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化高级多因子策略
        
        参数:
            parameters: 策略参数字典
        """
        default_params = {
            # 趋势指标参数
            'adx_period': 14,
            'adx_threshold': 25,
            'aroon_period': 14,
            'supertrend_factor': 3.0,
            'supertrend_period': 10,
            
            # 动量指标参数
            'roc_fast_period': 5,
            'roc_slow_period': 14,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # 波动率指标参数
            'atr_period': 14,
            'bbands_period': 20,
            'bbands_std': 2.0,
            'keltner_period': 20,
            'keltner_atr_factor': 1.5,
            
            # 成交量指标参数
            'volume_ma_period': 20,
            'cmf_period': 20,
            'mfi_period': 14,
            'mfi_overbought': 80,
            'mfi_oversold': 20,
            
            # 支撑阻力参数
            'donchian_period': 20,
            'pivot_type': 'traditional',
            
            # 信号生成参数
            'trend_weight': 0.3,
            'momentum_weight': 0.25,
            'volatility_weight': 0.15,
            'volume_weight': 0.2,
            'sr_weight': 0.1,
            'buy_threshold': 0.3,
            'sell_threshold': -0.3
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__('AdvancedMultiFactorStrategy', default_params)
        self.logger.info(f"初始化高级多因子策略，参数: {default_params}")
        self.version = '1.0.0'
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        try:
            if data is None or data.empty:
                self.logger.warning("数据为空，无法计算指标")
                return pd.DataFrame()
            
            # 复制数据以避免修改原始数据
            df = data.copy()
            
            # ====== 计算趋势指标 ======
            # ADX指标
            adx_result = TechnicalIndicators.calculate_adx(
                df['high'], df['low'], df['close'], 
                window=self.parameters['adx_period']
            )
            df['adx'] = adx_result['adx']
            df['plus_di'] = adx_result['plus_di']
            df['minus_di'] = adx_result['minus_di']
            
            # Aroon指标
            aroon_result = aroon(df['high'], df['low'], period=self.parameters['aroon_period'])
            df['aroon_up'] = aroon_result['aroon_up']
            df['aroon_down'] = aroon_result['aroon_down']
            df['aroon_osc'] = aroon_result['aroon_osc']
            
            # Supertrend指标
            supertrend_result = supertrend(
                df['high'], df['low'], df['close'],
                period=self.parameters['supertrend_period'],
                multiplier=self.parameters['supertrend_factor']
            )
            df['supertrend'] = supertrend_result['supertrend']
            df['supertrend_direction'] = supertrend_result['trend']
            
            # ====== 计算动量指标 ======
            # ROC指标
            df['roc_fast'] = roc(df['close'], period=self.parameters['roc_fast_period'])
            df['roc_slow'] = roc(df['close'], period=self.parameters['roc_slow_period'])
            
            # RSI指标
            df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'], window=self.parameters['rsi_period'])
            
            # 随机指标
            stoch_result = stochastic_oscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_result['k']
            df['stoch_d'] = stoch_result['d']
            
            # CCI指标
            df['cci'] = cci(df['high'], df['low'], df['close'], period=self.parameters['adx_period'])
            
            # ====== 计算波动率指标 ======
            # ATR
            df['atr'] = atr(df['high'], df['low'], df['close'], window=self.parameters['atr_period'])
            
            # 历史波动率
            df['volatility'] = historical_volatility(df['close'], window=20)
            
            # 布林带
            bb_result = TechnicalIndicators.calculate_bb(
                df['close'], 
                window=self.parameters['bbands_period'], 
                num_std=self.parameters['bbands_std']
            )
            df['bb_upper'] = bb_result['upper']
            df['bb_middle'] = bb_result['middle']
            df['bb_lower'] = bb_result['lower']
            df['bb_width'] = bollinger_bandwidth(df['close'], window=self.parameters['bbands_period'])
            
            # Keltner通道宽度
            df['keltner_width'] = keltner_channel_width(
                df['high'], df['low'], df['close'], 
                window=self.parameters['keltner_period'], 
                atr_mult=self.parameters['keltner_atr_factor']
            )
            
            # ====== 计算成交量指标 ======
            # 成交量SMA
            df['volume_sma'] = volume_sma(df['volume'], window=self.parameters['volume_ma_period'])
            
            # 成交量比例
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # OBV
            df['obv'] = on_balance_volume(df['close'], df['volume'])
            
            # CMF
            df['cmf'] = chaikin_money_flow(
                df['high'], df['low'], df['close'], df['volume'], 
                period=self.parameters['cmf_period']
            )
            
            # MFI
            df['mfi'] = money_flow_index(
                df['high'], df['low'], df['close'], df['volume'], 
                period=self.parameters['mfi_period']
            )
            
            # ====== 计算支撑阻力位 ======
            # 唐奇安通道
            donchian_result = donchian_channels(df['high'], df['low'], period=self.parameters['donchian_period'])
            df['donchian_upper'] = donchian_result['upper']
            df['donchian_middle'] = donchian_result['middle']
            df['donchian_lower'] = donchian_result['lower']
            
            # 价格通道
            pc_result = price_channels(df['high'], df['low'], period=self.parameters['donchian_period'])
            df['pc_upper'] = pc_result['upper']
            df['pc_lower'] = pc_result['lower']
            
            # 枢轴点
            pivot_result = pivot_points_traditional(df['high'], df['low'], df['close'])
            df['pivot'] = pivot_result['pivot']
            df['support1'] = pivot_result['support1']
            df['support2'] = pivot_result['support2']
            df['resistance1'] = pivot_result['resistance1']
            df['resistance2'] = pivot_result['resistance2']
            
            # ====== 计算各因子得分 ======
            # 趋势因子计算（范围：-1到1，正值表示上升趋势，负值表示下降趋势）
            df['trend_factor'] = self._calculate_trend_factor(df)
            
            # 动量因子计算（范围：-1到1，正值表示动量向上，负值表示动量向下）
            df['momentum_factor'] = self._calculate_momentum_factor(df)
            
            # 波动率因子计算（范围：-1到1，正值表示波动率降低，负值表示波动率上升）
            df['volatility_factor'] = self._calculate_volatility_factor(df)
            
            # 成交量因子计算（范围：-1到1，正值表示成交量支持价格走势，负值表示成交量不支持）
            df['volume_factor'] = self._calculate_volume_factor(df)
            
            # 支撑阻力因子计算（范围：-1到1，正值表示价格远离阻力靠近支撑，负值表示相反）
            df['sr_factor'] = self._calculate_support_resistance_factor(df)
            
            # 计算综合因子（加权平均）
            df['composite_factor'] = (
                df['trend_factor'] * self.parameters['trend_weight'] +
                df['momentum_factor'] * self.parameters['momentum_weight'] +
                df['volatility_factor'] * self.parameters['volatility_weight'] +
                df['volume_factor'] * self.parameters['volume_weight'] +
                df['sr_factor'] * self.parameters['sr_weight']
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return data
    
    def _calculate_trend_factor(self, df: pd.DataFrame) -> pd.Series:
        """计算趋势因子"""
        # ADX趋势强度组件
        adx_norm = (df['adx'] - 25) / 25  # 归一化ADX，25以上表示强趋势
        adx_norm = adx_norm.clip(0, 1)  # 限制在0-1范围
        
        # 趋势方向组件
        trend_direction = np.where(df['plus_di'] > df['minus_di'], 1, -1)
        
        # Aroon振荡器组件
        aroon_norm = df['aroon_osc'] / 100  # 归一化为-1到1
        
        # Supertrend方向组件
        supertrend_dir = df['supertrend_direction']
        
        # 综合趋势因子
        trend_factor = (
            adx_norm * trend_direction * 0.4 +  # ADX方向组件
            aroon_norm * 0.3 +                  # Aroon组件
            supertrend_dir * 0.3                # Supertrend组件
        )
        
        return trend_factor
    
    def _calculate_momentum_factor(self, df: pd.DataFrame) -> pd.Series:
        """计算动量因子"""
        # ROC组件
        roc_fast_norm = df['roc_fast'] / 10  # 归一化ROC
        roc_fast_norm = roc_fast_norm.clip(-1, 1)  # 限制在-1到1范围
        
        roc_slow_norm = df['roc_slow'] / 15  # 归一化ROC
        roc_slow_norm = roc_slow_norm.clip(-1, 1)  # 限制在-1到1范围
        
        # RSI组件
        rsi_norm = (df['rsi'] - 50) / 50  # 归一化为-1到1
        
        # 随机指标组件
        stoch_norm = (df['stoch_k'] - 50) / 50  # 归一化为-1到1
        
        # CCI组件
        cci_norm = df['cci'] / 200  # 归一化CCI
        cci_norm = cci_norm.clip(-1, 1)  # 限制在-1到1范围
        
        # 综合动量因子
        momentum_factor = (
            roc_fast_norm * 0.25 +  # 短期ROC
            roc_slow_norm * 0.15 +  # 长期ROC
            rsi_norm * 0.3 +        # RSI
            stoch_norm * 0.15 +     # 随机指标
            cci_norm * 0.15         # CCI
        )
        
        return momentum_factor
    
    def _calculate_volatility_factor(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率因子"""
        # 布林带宽度组件 - 宽度增加为负分，减少为正分
        bb_width_change = -df['bb_width'].pct_change(5)
        bb_width_norm = bb_width_change.clip(-1, 1)  # 限制在-1到1范围
        
        # ATR变化组件 - ATR增加为负分，减少为正分
        atr_change = -df['atr'].pct_change(5)
        atr_norm = atr_change.clip(-1, 1)  # 限制在-1到1范围
        
        # 历史波动率组件 - 波动率增加为负分，减少为正分
        vol_change = -df['volatility'].pct_change(5)
        vol_norm = vol_change.clip(-1, 1)  # 限制在-1到1范围
        
        # 布林带位置组件 - 接近中轨为正，远离为负
        bb_pos = 1 - 2 * abs((df['close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_lower'] + 1e-10) - 0.5)
        
        # 综合波动率因子
        volatility_factor = (
            bb_width_norm * 0.3 +  # 布林带宽度变化
            atr_norm * 0.3 +       # ATR变化
            vol_norm * 0.2 +       # 历史波动率变化
            bb_pos * 0.2           # 布林带位置
        )
        
        return volatility_factor
    
    def _calculate_volume_factor(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量因子"""
        # 成交量比率组件 - 成交量增加且价格上涨为正，成交量增加且价格下跌为负
        vol_ratio_norm = (df['volume_ratio'] - 1) * np.sign(df['close'].pct_change())
        vol_ratio_norm = vol_ratio_norm.clip(-1, 1)  # 限制在-1到1范围
        
        # OBV变化组件
        obv_change = df['obv'].pct_change(5)
        obv_norm = obv_change.clip(-1, 1)  # 限制在-1到1范围
        
        # CMF组件
        cmf_norm = df['cmf'].clip(-1, 1)  # 限制在-1到1范围
        
        # MFI组件
        mfi_norm = (df['mfi'] - 50) / 50  # 归一化为-1到1
        
        # 综合成交量因子
        volume_factor = (
            vol_ratio_norm * 0.25 +  # 成交量比率
            obv_norm * 0.25 +        # OBV变化
            cmf_norm * 0.25 +        # CMF
            mfi_norm * 0.25          # MFI
        )
        
        return volume_factor
    
    def _calculate_support_resistance_factor(self, df: pd.DataFrame) -> pd.Series:
        """计算支撑阻力因子"""
        # 价格与最近支撑位的距离
        support_dist = (df['close'] - df['support1']) / df['close']
        support_norm = 1 - support_dist.clip(0, 1)  # 越接近支撑位，值越大
        
        # 价格与最近阻力位的距离
        resist_dist = (df['resistance1'] - df['close']) / df['close']
        resist_norm = 1 - resist_dist.clip(0, 1)  # 越接近阻力位，值越大
        
        # 唐奇安通道位置
        donchian_pos = (df['close'] - df['donchian_lower']) / (df['donchian_upper'] - df['donchian_lower'] + 1e-10)
        donchian_norm = 1 - 2 * donchian_pos  # 接近下轨为1，接近上轨为-1
        
        # 综合支撑阻力因子
        sr_factor = (
            support_norm * 0.3 +     # 支撑位距离
            -resist_norm * 0.3 +     # 阻力位距离（负号使接近阻力为负分）
            donchian_norm * 0.4      # 唐奇安通道位置
        )
        
        return sr_factor
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame，其中:
            1 = 买入信号
            0 = 持有/无信号
            -1 = 卖出信号
        """
        try:
            # 计算技术指标
            df = self.calculate_indicators(data)
            
            # 初始化信号列
            df['signal'] = 0
            
            # 买入条件
            buy_condition = (
                (df['composite_factor'] > self.parameters['buy_threshold']) &  # 综合因子大于买入阈值
                (df['trend_factor'] > 0) &  # 趋势向上
                (df['volume_factor'] > 0)    # 成交量支持
            )
            
            # 卖出条件
            sell_condition = (
                (df['composite_factor'] < self.parameters['sell_threshold']) &  # 综合因子小于卖出阈值
                (df['trend_factor'] < 0)  # 趋势向下
            )
            
            # 生成信号
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return data.assign(signal=0)
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        df = self.calculate_indicators(data)
        
        return {
            "trend_factor": df['trend_factor'],
            "momentum_factor": df['momentum_factor'],
            "volatility_factor": df['volatility_factor'],
            "volume_factor": df['volume_factor'],
            "sr_factor": df['sr_factor'],
            "composite": df['composite_factor']
        }
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "trend_factor": {
                "type": "trend",
                "time_scale": "medium",
                "description": "趋势方向和强度因子",
                "min_value": -1,
                "max_value": 1
            },
            "momentum_factor": {
                "type": "momentum",
                "time_scale": "short",
                "description": "价格动量因子",
                "min_value": -1,
                "max_value": 1
            },
            "volatility_factor": {
                "type": "volatility",
                "time_scale": "short",
                "description": "波动率因子",
                "min_value": -1,
                "max_value": 1
            },
            "volume_factor": {
                "type": "volume",
                "time_scale": "short",
                "description": "成交量和资金流向因子",
                "min_value": -1,
                "max_value": 1
            },
            "sr_factor": {
                "type": "support_resistance",
                "time_scale": "medium",
                "description": "支撑阻力因子",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "medium",
                "description": "多因子策略综合信号",
                "min_value": -1,
                "max_value": 1
            }
        }
