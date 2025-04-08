import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    try:
        # 复制数据框
        df = df.copy()
        logger.info(f"输入数据点数量: {len(df)}")
        
        # 检查数据是否足够
        if len(df) < 200:  # 需要至少200个数据点来计算大多数指标
            logger.warning(f"数据点数量不足: {len(df)}")
            return pd.DataFrame()
            
        # 检查必要的列是否存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"缺少必要的列: {missing_columns}")
            return pd.DataFrame()
            
        # 检查数据类型并尝试转换
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"列 {col} 不是数值类型，尝试转换")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.error(f"转换列 {col} 为数值类型时出错: {e}")
                    return pd.DataFrame()
                
        # 处理基础数据中的缺失值和异常值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 检查每列的缺失值情况
        nan_info = df[required_columns].isna().sum()
        if nan_info.any():
            logger.warning(f"基础数据中的缺失值情况:\n{nan_info}")
        
        # 对基础数据使用更保守的填充策略
        for col in required_columns:
            if df[col].isna().any():
                # 首先尝试用前后值的平均值填充
                df[col] = df[col].fillna(method='ffill')
                temp = df[col].fillna(method='bfill')
                df[col] = (df[col] + temp) / 2
                
                # 如果还有NaN，使用该列的中位数填充
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.warning(f"列 {col} 使用中位数 {median_val} 填充了 {df[col].isna().sum()} 个缺失值")
        
        # 如果基础数据中仍然有缺失值，返回空数据框
        if df[required_columns].isnull().any().any():
            logger.warning("基础数据中存在无法填充的缺失值")
            return pd.DataFrame()
        
        # 趋势指标
        df = add_trend_indicators(df)
        if df.empty:
            logger.warning("添加趋势指标后数据为空")
            return pd.DataFrame()
        logger.info(f"添加趋势指标后数据点数量: {len(df)}")
        
        # 动量指标
        df = add_momentum_indicators(df)
        if df.empty:
            logger.warning("添加动量指标后数据为空")
            return pd.DataFrame()
        logger.info(f"添加动量指标后数据点数量: {len(df)}")
        
        # 波动率指标
        df = add_volatility_indicators(df)
        if df.empty:
            logger.warning("添加波动率指标后数据为空")
            return pd.DataFrame()
        logger.info(f"添加波动率指标后数据点数量: {len(df)}")
        
        # 成交量指标
        df = add_volume_indicators(df)
        if df.empty:
            logger.warning("添加成交量指标后数据为空")
            return pd.DataFrame()
        logger.info(f"添加成交量指标后数据点数量: {len(df)}")
        
        # 高级成交量指标
        df = add_advanced_volume_indicators(df)
        if df.empty:
            logger.warning("添加高级成交量指标后数据为空")
            return pd.DataFrame()
        logger.info(f"添加高级成交量指标后数据点数量: {len(df)}")
        
        # 高级指标
        df = add_advanced_indicators(df)
        if df.empty:
            logger.warning("添加高级指标后数据为空")
            return pd.DataFrame()
        logger.info(f"添加高级指标后数据点数量: {len(df)}")
        
        # 自定义指标
        df = add_custom_indicators(df)
        if df.empty:
            logger.warning("添加自定义指标后数据为空")
            return pd.DataFrame()
        logger.info(f"添加自定义指标后数据点数量: {len(df)}")
        
        # 计算volume_consensus相关的信号
        # 1. VWAP突破信号
        df['vwap_cross'] = np.where(df['close'] > df['vwap'], 1, -1)
        
        # 2. CMF信号
        df['cmf_signal'] = np.where(df['cmf'] > 0.05, 1,
                                  np.where(df['cmf'] < -0.05, -1, 0))
        
        # 3. KVO信号
        df['kvo_signal'] = np.where(df['kvo'] > 0, 1,
                                  np.where(df['kvo'] < 0, -1, 0))
        
        # 4. EMV信号
        df['emv_signal'] = np.where(df['emv'] > df['emv'].rolling(window=14).mean(), 1, -1)
        
        # 5. PVT趋势确认
        df['pvt_signal'] = np.where(df['pvt'] > df['pvt'].shift(1), 1, -1)
        
        # 计算volume_consensus
        df['volume_consensus'] = (
            df['vwap_cross'] + 
            df['cmf_signal'] + 
            df['kvo_signal'] + 
            df['emv_signal'] + 
            df['pvt_signal']
        ) / 5.0
        
        # 确保volume_consensus不包含NaN值
        df['volume_consensus'] = df['volume_consensus'].fillna(0)
        
        # 记录volume_consensus的统计信息
        logger.info(f"volume_consensus 统计: mean={df['volume_consensus'].mean():.4f}, std={df['volume_consensus'].std():.4f}, min={df['volume_consensus'].min():.4f}, max={df['volume_consensus'].max():.4f}")
        
        # 最终处理
        # 1. 替换无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. 对每个指标分别进行填充
        indicator_columns = [col for col in df.columns if col not in required_columns]
        
        # 检查每个指标列的缺失值情况
        nan_info = df[indicator_columns].isna().sum()
        nan_columns = nan_info[nan_info > 0]
        if not nan_columns.empty:
            logger.warning(f"指标列中的缺失值情况:\n{nan_columns}")
        
        # 对每个指标列分别进行填充
        for col in indicator_columns:
            if df[col].isnull().any():
                # 首先尝试用前后值的平均值填充
                df[col] = df[col].fillna(method='ffill')
                temp = df[col].fillna(method='bfill')
                df[col] = (df[col] + temp) / 2
                
                # 如果仍然有NaN，使用该列的中位数填充
                if df[col].isnull().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):  # 如果中位数也是NaN
                        df[col] = df[col].fillna(0)
                        logger.warning(f"列 {col} 使用0填充了所有缺失值")
                    else:
                        df[col] = df[col].fillna(median_val)
                        logger.warning(f"列 {col} 使用中位数 {median_val} 填充了缺失值")
                
        # 3. 检查是否所有列都是数值类型
        non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_cols:
            logger.warning(f"以下列不是数值类型: {non_numeric_cols}")
            # 尝试转换非数值列
            for col in non_numeric_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                except Exception as e:
                    logger.error(f"转换列 {col} 为数值类型时出错: {e}")
                    return pd.DataFrame()
        
        # 4. 最终检查数据是否为空
        if df.empty:
            logger.warning("最终处理后数据为空")
            return pd.DataFrame()
            
        # 5. 检查数据点数量是否仍然足够
        if len(df) < 200:
            logger.warning(f"最终处理后数据点数量不足: {len(df)}")
            return pd.DataFrame()
        
        # 6. 最终检查是否还有任何NaN值
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            logger.warning(f"最终数据中仍有NaN值的列: {nan_cols}")
            logger.warning(f"每列的NaN值数量:\n{df[nan_cols].isnull().sum()}")
            # 使用0填充任何剩余的NaN值
            df = df.fillna(0)
        
        logger.info(f"最终数据点数量: {len(df)}")
        return df
        
    except Exception as e:
        logger.error(f"计算技术指标时出错: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        return pd.DataFrame()

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加趋势指标"""
    try:
        # 简单移动平均线 (减少长期均线)
        for period in [5, 10, 20, 50]:  # 移除100和200日均线
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            
        # 指数移动平均线
        for period in [5, 10, 20]:  # 移除50日均线
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
        # MACD (使用较短的周期)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=8,  # 原为12
            slowperiod=17,  # 原为26
            signalperiod=9
        )
        
        # 抛物线转向指标 (SAR)
        df['sar'] = talib.SAR(df['high'], df['low'])
        
        # 趋势方向指数 (ADX)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 移动平均收敛发散指标 (PPO)
        df['ppo'] = talib.PPO(df['close'])
        
        return df
        
    except Exception as e:
        logger.error(f"添加趋势指标时出错: {e}")
        return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加动量指标"""
    try:
        # RSI
        for period in [6, 12, 24]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
            
        # 随机指标
        df['slowk'], df['slowd'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        
        # 威廉指标
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 动量指标
        df['mom'] = talib.MOM(df['close'], timeperiod=10)
        
        # ROC - 变动率指标
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        
        # CCI - 顺势指标
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
        
    except Exception as e:
        logger.error(f"添加动量指标时出错: {e}")
        return df

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加波动率指标"""
    try:
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        
        # ATR - 真实波动幅度均值
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 标准差
        df['stddev'] = talib.STDDEV(df['close'], timeperiod=20)
        
        # 历史波动率
        returns = np.log(df['close'] / df['close'].shift(1))
        annualization_factor = np.full(len(df), np.sqrt(252))
        df['volatility'] = returns.rolling(window=20).std().mul(annualization_factor)
        
        # Keltner通道
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['keltner_middle'] = talib.SMA(typical_price, timeperiod=20)
        df['keltner_upper'] = df['keltner_middle'] + 2 * df['atr']
        df['keltner_lower'] = df['keltner_middle'] - 2 * df['atr']
        
        return df
        
    except Exception as e:
        logger.error(f"添加波动率指标时出错: {e}")
        return df

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加成交量指标"""
    try:
        # 成交量SMA
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_sma'] = df['volume_sma'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # OBV - 能量潮指标
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv'] = df['obv'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # A/D - 累积/派发线
        df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['ad'] = df['ad'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # VWAP - 成交量加权平均价格
        volume_sum = df['volume'].rolling(window=20).sum()
        price_volume_sum = (df['close'] * df['volume']).rolling(window=20).sum()
        df['vwap'] = np.where(volume_sum != 0, price_volume_sum / volume_sum, df['close'])
        df['vwap'] = df['vwap'].fillna(method='ffill').fillna(method='bfill').fillna(df['close'])
        
        # CMF - 蔡金货币流量指标
        high_low_range = df['high'] - df['low']
        money_flow_multiplier = np.where(
            high_low_range != 0,
            ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_range,
            0
        )
        money_flow_volume = money_flow_multiplier * df['volume']
        volume_sum = df['volume'].rolling(window=20).sum()
        df['cmf'] = np.where(
            volume_sum != 0,
            money_flow_volume.rolling(window=20).sum() / volume_sum,
            0
        )
        df['cmf'] = df['cmf'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # PVT - 价格成交量趋势
        close_pct_change = df['close'].pct_change()
        df['pvt'] = (close_pct_change * df['volume']).fillna(0)
        df['pvt'] = df['pvt'].cumsum()
        df['pvt'] = df['pvt'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 记录指标计算结果
        volume_cols = ['volume_sma', 'obv', 'ad', 'vwap', 'cmf', 'pvt']
        for col in volume_cols:
            if col in df.columns:
                logger.info(f"{col} 统计: mean={df[col].mean():.4f}, std={df[col].std():.4f}, min={df[col].min():.4f}, max={df[col].max():.4f}")
                if df[col].isnull().any():
                    logger.warning(f"{col} 包含 {df[col].isnull().sum()} 个NaN值")
        
        return df
        
    except Exception as e:
        logger.error(f"添加成交量指标时出错: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return df

def add_advanced_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加高级成交量指标"""
    try:
        # EMV - 简易波动指标
        mid_point = (df['high'] + df['low']) / 2
        mid_point_prev = mid_point.shift(1)
        distance = mid_point - mid_point_prev
        
        box_ratio = np.where(
            (df['high'] - df['low']) != 0,
            (df['volume'] / 100000000) / (df['high'] - df['low']),
            0
        )
        
        df['emv'] = np.where(box_ratio != 0, distance / box_ratio, 0)
        df['emv'] = df['emv'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        df['emv_ma'] = df['emv'].rolling(window=14).mean()
        df['emv_ma'] = df['emv_ma'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # KVO - Klinger成交量震荡器
        trend = (df['high'] + df['low'] + df['close']) / 3
        trend_diff = trend - trend.shift(1)
        volume_force = df['volume'] * abs(trend_diff) * np.where(trend_diff > 0, 1, -1)
        volume_force = volume_force.fillna(0)
        
        # 使用talib的EMA，并处理可能的NaN值
        ema_34 = talib.EMA(volume_force, timeperiod=34)
        ema_55 = talib.EMA(volume_force, timeperiod=55)
        
        ema_34 = pd.Series(ema_34).fillna(method='ffill').fillna(method='bfill').fillna(0)
        ema_55 = pd.Series(ema_55).fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        df['kvo'] = ema_34 - ema_55
        df['kvo'] = df['kvo'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 记录指标计算结果
        advanced_volume_cols = ['emv', 'emv_ma', 'kvo']
        for col in advanced_volume_cols:
            if col in df.columns:
                logger.info(f"{col} 统计: mean={df[col].mean():.4f}, std={df[col].std():.4f}, min={df[col].min():.4f}, max={df[col].max():.4f}")
                if df[col].isnull().any():
                    logger.warning(f"{col} 包含 {df[col].isnull().sum()} 个NaN值")
        
        return df
        
    except Exception as e:
        logger.error(f"添加高级成交量指标时出错: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return df

def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加高级指标"""
    try:
        # TEMA - 三重指数移动平均线
        df['tema'] = talib.TEMA(df['close'], timeperiod=20)
        
        # TRIX - 三重指数平滑移动平均指标
        df['trix'] = talib.TRIX(df['close'], timeperiod=30)
        
        # ULTOSC - 终极波动指标
        df['ultosc'] = talib.ULTOSC(
            df['high'],
            df['low'],
            df['close'],
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28
        )
        
        # HT_DCPERIOD - 希尔伯特变换-主导周期
        df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])
        
        # HT_TRENDMODE - 希尔伯特变换-趋势模式
        df['ht_trendmode'] = talib.HT_TRENDMODE(df['close'])
        
        # BETA - 贝塔系数
        df['beta'] = talib.BETA(df['high'], df['low'], timeperiod=5)
        
        return df
        
    except Exception as e:
        logger.error(f"添加高级指标时出错: {e}")
        return df

def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加自定义指标"""
    try:
        # 价格动量 (5天)
        df['price_momentum'] = df['close'].pct_change(periods=5)
        
        # 价格波动范围
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # 成交量冲击 (5天)
        df['volume_impact'] = (df['volume'] * df['price_range']).rolling(
            window=5,
            min_periods=1  # 允许部分窗口计算
        ).mean()
        
        # 趋势一致性 (使用较短期的均线)
        sma_trends = [
            df['close'] > df[f'sma_{period}']
            for period in [5, 10, 20]  # 移除50日均线
        ]
        df['trend_consensus'] = sum(sma_trends) / len(sma_trends)
        
        # 动量发散 (使用较短期的RSI)
        df['momentum_divergence'] = (
            df['rsi_6'] - df['rsi_12']  # 使用较短期的RSI
        )
        
        # 市场效率系数 (使用较短期)
        price_change = abs(df['close'] - df['close'].shift(10))  # 改为10天
        price_path = df['high'].rolling(window=10, min_periods=1).max() - df['low'].rolling(window=10, min_periods=1).min()
        df['market_efficiency'] = price_change / price_path
        
        # 移除一些计算复杂的指标
        # 移除了: volatility_ratio, trend_strength
        
        return df
        
    except Exception as e:
        logger.error(f"添加自定义指标时出错: {e}")
        return df 