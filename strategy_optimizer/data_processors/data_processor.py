import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from strategy_optimizer.utils.technical_indicators import calculate_technical_indicators
from strategy_optimizer.data_processors.data_enhancer import DataEnhancer
from ..market_state import MarketState, create_market_features

from strategy.strategy_factory import (
    GoldTriangleStrategy,
    MomentumStrategy,
    TDIStrategy,
    MarketForecastStrategy,
    CPGWStrategy,
    StrategyFactory,
    VolumeStrategy
)

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器"""
    def __init__(self, db_connection: str = 'mysql+pymysql://root@localhost/mose'):
        self.engine = create_engine(db_connection)
        self.scaler = StandardScaler()
        self.data_enhancer = DataEnhancer()  # 添加DataEnhancer实例
        self.market_state = MarketState(self.engine)
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票数据并计算技术指标
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            包含价格数据和技术指标的DataFrame
        """
        try:
            # 获取股票数据
            market_data = self.market_state.get_market_data([symbol], start_date, end_date)
            if not market_data or symbol not in market_data:
                self.logger.warning(f"无法获取股票数据: {symbol}")
                return pd.DataFrame()
                
            df = market_data[symbol]
            
            # 重命名列
            column_map = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            df = df.rename(columns=column_map)
            
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 删除Code列，避免在计算中使用
            if 'Code' in df.columns:
                df = df.drop('Code', axis=1)
            
            # 计算基本技术指标
            df = self._calculate_required_indicators(df)
            
            # 打印指标统计信息
            for col in df.columns:
                non_null_count = df[col].count()
                self.logger.info(f"{col} 非空值数量: {non_null_count}")
            
            # 打印数据点数量
            self.logger.info(f"处理后的数据点数量: {len(df)}")
            
            # 打印技术指标的统计信息
            indicator_columns = [
                'high_20', 'low_10', 'SMA_200',  # 价格指标
                'RSI', 'MACD', 'MACD_signal',  # 动量指标
                'MA1', 'MA2'  # TDI指标
            ]
            
            for col in indicator_columns:
                if col in df.columns:
                    stats = df[col].describe()
                    self.logger.info(f"{col} 统计: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                                   f"min={stats['min']:.4f}, max={stats['max']:.4f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票数据时出错: {str(e)}")
            return pd.DataFrame()
            
    def _get_market_index_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """获取单个市场指数数据"""
        try:
            query = """
            SELECT 
                Date as date,
                Close as close
            FROM stock_time_code
            WHERE Code = %s
            AND Date BETWEEN %s AND %s
            ORDER BY Date ASC
            """
            
            df = pd.read_sql_query(
                query,
                self.engine,
                params=(symbol, start_date, end_date)
            )
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"获取市场指数 {symbol} 数据时出错: {str(e)}")
            return pd.DataFrame()
            
    def get_market_indices_data(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """获取市场指数数据"""
        try:
            # 扩展市场指数列表
            indices = {
                'core_indices': ['QQQ', 'SPY', 'VIX', 'TLT'],  # 核心指数
                'sector_etfs': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI'],  # 主要行业ETF
            }
            
            market_data = {}
            
            # 获取所有指数数据
            for category, symbols in indices.items():
                for symbol in symbols:
                    query = """
                    SELECT 
                        Date as date,
                        Open as open,
                        High as high,
                        Low as low,
                        Close as close,
                        Volume as volume
                    FROM stock_time_code
                    WHERE Code = %s
                    AND Date BETWEEN %s AND %s
                    ORDER BY Date ASC
                    """
                    
                    df = pd.read_sql_query(
                        query,
                        self.engine,
                        params=(symbol, start_date, end_date)
                    )
                    
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        
                        # 添加基础技术指标
                        df[f'{symbol}_return'] = df['close'].pct_change()
                        
                        # 多时间窗口指标
                        for window in [5, 20, 60]:
                            # 趋势指标
                            df[f'{symbol}_ma_{window}'] = df['close'].rolling(window=window).mean()
                            df[f'{symbol}_trend_{window}'] = (df['close'] - df[f'{symbol}_ma_{window}']) / df[f'{symbol}_ma_{window}']
                            
                            # 波动率指标
                            df[f'{symbol}_volatility_{window}'] = df['close'].pct_change().rolling(window).std() * np.full(len(df), np.sqrt(252))
                            
                            # 成交量指标
                            df[f'{symbol}_volume_ma_{window}'] = df['volume'].rolling(window).mean()
                            df[f'{symbol}_volume_trend_{window}'] = (df['volume'] - df[f'{symbol}_volume_ma_{window}']) / df[f'{symbol}_volume_ma_{window}']
                        
                        # 特殊指标计算
                        if symbol in ['QQQ', 'SPY']:
                            # 市场宽度指标（使用价格位置）
                            df[f'{symbol}_price_position'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
                            
                            # 资金流向指标
                            df[f'{symbol}_mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
                            
                            # 趋势强度
                            df[f'{symbol}_adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                            
                        elif symbol == 'VIX':
                            # VIX特殊指标
                            df['vix_ma_ratio'] = df['close'] / df['close'].rolling(window=20).mean()
                            df['vix_term_structure'] = df['close'] / df['close'].shift(20)
                            
                        elif symbol == 'TLT':
                            # 债券相关指标
                            df['tlt_yield_trend'] = -1 * df['close'].pct_change()  # 价格与收益率反向
                            df['tlt_risk_appetite'] = df['close'].rolling(window=20).corr(market_data['SPY']['close'] if 'SPY' in market_data else pd.Series())
                        
                        market_data[symbol] = df
                        logger.info(f"获取到指数 {symbol} 的 {len(df)} 条记录")
                    else:
                        logger.warning(f"获取指数 {symbol} 的数据为空")
            
            # 计算市场综合指标
            if 'SPY' in market_data and 'VIX' in market_data:
                spy_data = market_data['SPY']
                vix_data = market_data['VIX']
                
                # 风险情绪指标
                spy_data['risk_sentiment'] = -1 * (
                    (spy_data['close'].pct_change().rolling(20).std() * np.full(len(spy_data), np.sqrt(252))) *
                    (vix_data['close'] / vix_data['close'].rolling(20).mean())
                )
                
                # 更新SPY数据
                market_data['SPY'] = spy_data
            
            return market_data
            
        except Exception as e:
            logger.error(f"获取市场指数数据时出错: {e}")
            return {}
            
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据质量
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            bool: 数据是否通过验证
        """
        try:
            # 检查必需列是否存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"缺少必需列: {[col for col in required_columns if col not in df.columns]}")
                return False
            
            # 检查数据量是否足够
            min_rows = 250  # 至少需要250个交易日的数据
            if len(df) < min_rows:
                logger.warning(f"数据量不足: {len(df)} < {min_rows}")
                return False
            
            # 检查缺失值比例
            max_missing_ratio = 0.1  # 最大允许10%的缺失值
            missing_ratio = df[required_columns].isnull().mean()
            if (missing_ratio > max_missing_ratio).any():
                logger.warning(f"缺失值比例过高: {missing_ratio[missing_ratio > max_missing_ratio]}")
                return False
            
            # 检查异常值
            # 使用3倍标准差作为异常值判断标准
            for col in ['open', 'high', 'low', 'close']:
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)][col]
                if len(outliers) > len(df) * 0.01:  # 允许1%的异常值
                    logger.warning(f"{col}列异常值过多: {len(outliers)}")
                    return False
            
            # 检查成交量异常
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > len(df) * 0.05:  # 允许5%的零成交量
                logger.warning(f"零成交量天数过多: {zero_volume}")
                return False
            
            # 检查价格数据的合理性
            if not (df['high'] >= df['low']).all():
                logger.warning("存在最高价低于最低价的情况")
                return False
            
            if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
                logger.warning("存在收盘价或开盘价高于最高价的情况")
                return False
            
            if not ((df['low'] <= df['open']) & (df['low'] <= df['close'])).all():
                logger.warning("存在收盘价或开盘价低于最低价的情况")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证时出错: {e}")
            return False

    def prepare_features(self, df: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """
        准备特征数据，简化版本
        """
        try:
            # 计算技术指标
            df = self._calculate_required_indicators(df)
            
            # 选择基本特征列
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_20', 'SMA_50', 'SMA_200',
                'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_middle', 'BB_lower'
            ]
            
            # 确保所有特征列存在
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"缺少特征列: {missing_columns}")
                return np.array([])
            
            # 数据标准化
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(df[feature_columns])
            
            # 创建序列数据
            sequences = []
            for i in range(len(normalized_data) - sequence_length + 1):
                sequence = normalized_data[i:(i + sequence_length)]
                sequences.append(sequence)
            
            if not sequences:
                logger.warning("没有生成有效的序列数据")
                return np.array([])
            
            return np.array(sequences)
            
        except Exception as e:
            logger.error(f"准备特征时出错: {e}")
            return np.array([])

    def prepare_targets(self, df: pd.DataFrame, sequence_length: int = None) -> np.ndarray:
        """
        准备目标变量
        
        参数:
            df: 包含价格数据的DataFrame
            sequence_length: 序列长度，如果提供则对齐特征序列
            
        返回:
            目标变量数组，形状为(样本数, 策略数)
        """
        try:
            # 初始化策略工厂
            factory = StrategyFactory()
            strategies = factory.create_all_strategies()
            
            # 计算每个策略的信号
            signals = {}
            for name, strategy in strategies.items():
                try:
                    # 确保策略使用的是相同的DataFrame
                    strategy_df = df.copy()
                    signal = strategy.generate_signals(strategy_df)
                    
                    # 确保信号是一维数组
                    if signal is not None:
                        if isinstance(signal, pd.DataFrame):
                            if 'signal' in signal.columns:
                                signal = signal['signal'].values
                            else:
                                signal = signal.iloc[:, 0].values
                        elif isinstance(signal, pd.Series):
                            signal = signal.values
                            
                        # 确保信号是浮点数类型
                        signal = signal.astype(float)
                        
                        # 处理无效值
                        signal = np.nan_to_num(signal, nan=0.0)
                        
                        if len(signal) > 0:
                            signals[name] = signal
                            self.logger.info(f"策略 {name} 生成了 {len(signal)} 个信号")
                        else:
                            self.logger.warning(f"策略 {name} 生成了空信号")
                    else:
                        self.logger.warning(f"策略 {name} 没有生成有效信号")
                except Exception as e:
                    self.logger.error(f"策略 {name} 生成信号时出错: {str(e)}")
                    continue
            
            if not signals:
                self.logger.error("没有策略生成有效信号")
                return np.array([])
            
            # 确保所有信号长度一致
            signal_lengths = {name: len(signal) for name, signal in signals.items()}
            min_length = min(signal_lengths.values())
            
            # 截取所有信号到最小长度
            for name in signals:
                signals[name] = signals[name][-min_length:]
            
            # 将所有策略的信号组合成目标变量矩阵
            target_matrix = []
            for name in sorted(signals.keys()):
                target_matrix.append(signals[name])
            
            # 转置矩阵使形状为(样本数, 策略数)
            targets = np.array(target_matrix).T
            
            # 如果提供了序列长度，则对齐特征序列
            if sequence_length is not None:
                if len(targets) < sequence_length:
                    self.logger.error(f"目标变量长度({len(targets)})小于序列长度({sequence_length})")
                    return np.array([])
                    
                # 调整目标变量长度以匹配特征序列
                targets = targets[sequence_length-1:]
            
            # 打印目标变量的形状和统计信息
            self.logger.info(f"目标变量形状: {targets.shape}")
            self.logger.info(f"目标变量统计: mean={np.mean(targets):.4f}, std={np.std(targets):.4f}, "
                           f"min={np.min(targets):.4f}, max={np.max(targets):.4f}")
            
            # 检查目标变量是否包含无效值
            if np.isnan(targets).any() or np.isinf(targets).any():
                self.logger.warning("目标变量包含无效值，将被替换为0")
                targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            
            return targets
            
        except Exception as e:
            self.logger.error(f"准备目标变量时出错: {str(e)}")
            return np.array([])

    def _calculate_strategy_signal(self, df: pd.DataFrame, strategy_name: str) -> np.ndarray:
        """根据策略名称计算策略信号"""
        try:
            # Calculate required technical indicators first
            df = self._calculate_required_indicators(df)
            
            if strategy_name == "GoldTriangleStrategy":
                return self._calculate_gold_triangle_signal(df)
            elif strategy_name == "MomentumStrategy":
                return self._calculate_momentum_signal(df)
            elif strategy_name == "TDIStrategy":
                return self._calculate_tdi_signal(df)
            elif strategy_name == "MarketForecastStrategy":
                return self._calculate_market_forecast_signal(df)
            elif strategy_name == "CPGWStrategy":
                return self._calculate_cpgw_signal(df)
            elif strategy_name == "VolumeStrategy":
                return self._calculate_volume_signal(df)
            elif strategy_name == "NiuniuStrategy":
                return self._calculate_niuniu_signal(df)
            else:
                logger.warning(f"未知的策略名称: {strategy_name}")
                return None
        except Exception as e:
            logger.error(f"计算策略信号时出错: {e}")
            return None

    def _calculate_gold_triangle_signal(self, df: pd.DataFrame) -> np.ndarray:
        """计算黄金三角策略信号"""
        try:
            # 使用不同的最小周期计算移动平均线
            ma5 = df['close'].rolling(window=5, min_periods=1).mean()
            ma10 = df['close'].rolling(window=10, min_periods=2).mean()
            ma20 = df['close'].rolling(window=20, min_periods=3).mean()
            
            # 计算趋势方向
            ma5_trend = ma5.diff()
            ma10_trend = ma10.diff()
            ma20_trend = ma20.diff()
            
            # 生成信号
            buy_signal = (
                (ma5 > ma10) & (ma10 > ma20) &  # 均线多头排列
                (ma5_trend > 0) & (ma10_trend > 0) & (ma20_trend > 0)  # 所有均线向上
            )
            
            sell_signal = (
                (ma5 < ma10) & (ma10 < ma20) &  # 均线空头排列
                (ma5_trend < 0) & (ma10_trend < 0) & (ma20_trend < 0)  # 所有均线向下
            )
            
            # 转换为数值信号
            signals = pd.Series(0, index=df.index)
            signals[buy_signal] = 1
            signals[sell_signal] = -1
            
            # 记录信号统计
            signal_counts = signals.value_counts()
            logger.info(f"黄金三角策略信号分布:\n{signal_counts}")
            
            return signals.values
            
        except Exception as e:
            logger.error(f"计算黄金三角策略信号时出错: {e}")
            return None

    def _calculate_momentum_signal(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        计算动量策略信号
        1 = 买入
        -1 = 卖出
        0 = 不操作
        
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: 返回计算后的DataFrame和信号数组
        """
        try:
            # 创建DataFrame的副本以避免SettingWithCopyWarning
            df = df.copy()
            
            # 确保所有必需的列都已计算完成
            required_columns = ['high_20', 'low_10', 'SMA_200', 'RSI', 'MACD', 'MACD_signal']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return df, np.zeros(len(df))
                if df[col].isna().any():
                    logger.warning(f"Column {col} contains NaN values")
                    df[col] = df[col].ffill().fillna(0)
            
            # 初始化信号和持仓
            signals = np.zeros(len(df))
            position = np.zeros(len(df))
            
            # 使用较短的预热期
            warmup_period = 50  # 使用最长的技术指标周期作为预热期
            if len(df) <= warmup_period:
                return df, signals
            
            for i in range(warmup_period, len(df)):
                # 更新持仓状态
                position[i] = position[i-1]
                
                # 如果持有多头，检查止损止盈
                if position[i-1] == 1:
                    # 计算止损价和止盈价
                    entry_price = df['close'].iloc[position.nonzero()[0][-1]]
                    stop_loss_price = entry_price * 0.75  # 25%的止损
                    profit_target_price = entry_price * 1.5  # 50%的止盈
                    
                    # 检查止损和止盈
                    if df['close'].iloc[i] <= stop_loss_price:  # 触发止损
                        signals[i] = -1
                        position[i] = 0
                        continue
                    elif df['close'].iloc[i] >= profit_target_price:  # 触发止盈
                        signals[i] = -1
                        position[i] = 0
                        continue
                
                # 买入条件
                if position[i-1] == 0:  # 仅在空仓时考虑买入
                    # 主要条件
                    main_conditions = [
                        df['close'].iloc[i] > df['SMA_200'].iloc[i],  # 价格在长期均线上方
                        df['RSI'].iloc[i] > 40,  # RSI大于40（降低门槛）
                        df['MACD'].iloc[i] > df['MACD_signal'].iloc[i],  # MACD金叉
                    ]
                    
                    # 辅助条件
                    aux_conditions = [
                        df['volume'].iloc[i] > df['volume'].rolling(window=20).mean().iloc[i],  # 成交量放大
                        df['close'].iloc[i] > df['high_20'].iloc[i-1],  # 突破20日高点
                        df['close'].pct_change().rolling(window=20).std().iloc[i] < 0.03,  # 波动率相对正常
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
                        df['RSI'].iloc[i] < 35,  # RSI超卖（调整阈值）
                        df['MACD'].iloc[i] < df['MACD_signal'].iloc[i],  # MACD死叉
                        df['close'].iloc[i] < df['low_10'].iloc[i-1]  # 跌破10日低点
                    ]
                    
                    # 辅助条件
                    aux_conditions = [
                        df['volume'].iloc[i] > df['volume'].rolling(window=20).mean().iloc[i] * 1.3,  # 成交量明显放大
                        df['close'].pct_change().rolling(window=20).std().iloc[i] > 0.04  # 波动率增加
                    ]
                    
                    # 生成卖出信号：满足至少2个主要条件和1个辅助条件
                    if sum(main_conditions) >= 2 and any(aux_conditions):
                        signals[i] = -1
                        position[i] = 0
                        continue
            
            return df, signals
            
        except Exception as e:
            logger.error(f"计算动量策略信号时出错: {str(e)}")
            return df, np.zeros(len(df))

    def _calculate_tdi_signal(self, df: pd.DataFrame) -> np.ndarray:
        """计算TDI策略信号"""
        try:
            # 计算RSI
            rsi = talib.RSI(df['close'], timeperiod=13)
            
            # 计算RSI的移动平均
            rsi_ma = talib.SMA(rsi, timeperiod=7)
            signal_line = talib.SMA(rsi_ma, timeperiod=7)
            
            # 生成信号
            buy_signal = (rsi_ma > signal_line) & (rsi < 50)
            sell_signal = (rsi_ma < signal_line) & (rsi > 50)
            
            return buy_signal.astype(float) - sell_signal.astype(float)
            
        except Exception as e:
            logger.error(f"计算TDI策略信号时出错: {e}")
            return None

    def _calculate_market_forecast_signal(self, df: pd.DataFrame) -> np.ndarray:
        """计算市场预测策略信号"""
        try:
            # 计算EMV指标
            emv = talib.CMO(df['close'], timeperiod=14)
            
            # 计算CCI指标
            cci = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            
            # 计算ADX指标
            adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # 生成信号
            buy_signal = (emv > 0) & (cci > 100) & (adx > 25)
            sell_signal = (emv < 0) & (cci < -100) & (adx > 25)
            
            return buy_signal.astype(float) - sell_signal.astype(float)
            
        except Exception as e:
            logger.error(f"计算市场预测策略信号时出错: {e}")
            return None

    def _calculate_cpgw_signal(self, df: pd.DataFrame) -> np.ndarray:
        """计算CPGW策略信号"""
        try:
            # 计算TEMA指标
            tema = talib.TEMA(df['close'], timeperiod=20)
            
            # 计算ROC指标
            roc = talib.ROC(df['close'], timeperiod=10)
            
            # 计算ATR指标
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # 生成信号
            buy_signal = (df['close'] > tema) & (roc > 0) & (atr > atr.rolling(20).mean())
            sell_signal = (df['close'] < tema) & (roc < 0) & (atr < atr.rolling(20).mean())
            
            return buy_signal.astype(float) - sell_signal.astype(float)
            
        except Exception as e:
            logger.error(f"计算CPGW策略信号时出错: {e}")
            return None

    def _calculate_niuniu_signal(self, df: pd.DataFrame) -> np.ndarray:
        """计算牛牛策略信号
        
        牛牛策略基于主力成本线（牛线）和交易线的交叉关系生成买卖信号。
        同时结合成交量和市场趋势进行信号强度的调整。
        """
        try:
            # 计算 MID 值 (主力建仓均价)
            df['MID'] = (3 * df['close'] + df['low'] + df['open'] + df['high']) / 6
            
            # 计算牛线（主力成本线）- 使用20天加权移动平均
            weights = np.arange(20, 0, -1)  # 权重为20到1
            weighted_mid = pd.concat([df['MID'].shift(i) * weights[i] for i in range(20)], axis=1)
            df['Bull_Line'] = weighted_mid.sum(axis=1) / weights.sum()
            
            # 计算交易线 - 2日移动平均
            df['Trade_Line'] = df['Bull_Line'].rolling(window=2).mean()
            
            # 计算趋势强度
            df['trend'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
            
            # 计算成交量趋势
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_trend'] = df['volume'] / df['volume_ma']
            
            # 生成基础信号
            buy_signal = (df['Bull_Line'] > df['Trade_Line']) & (df['Bull_Line'].shift(1) <= df['Trade_Line'].shift(1))
            sell_signal = (df['Trade_Line'] > df['Bull_Line']) & (df['Trade_Line'].shift(1) <= df['Bull_Line'].shift(1))
            
            # 初始化信号数组
            signals = np.zeros(len(df))
            
            # 设置信号并根据趋势和成交量调整信号强度
            for i in range(len(df)):
                if buy_signal.iloc[i]:
                    strength = 1.0
                    # 上升趋势加强买入信号
                    if df['trend'].iloc[i] > 0:
                        strength *= (1 + min(df['trend'].iloc[i], 0.5))
                    # 放量确认买入信号
                    if df['volume_trend'].iloc[i] > 1.2:
                        strength *= 1.2
                    signals[i] = min(strength, 1.0)
                elif sell_signal.iloc[i]:
                    strength = -1.0
                    # 下降趋势加强卖出信号
                    if df['trend'].iloc[i] < 0:
                        strength *= (1 + min(abs(df['trend'].iloc[i]), 0.5))
                    # 放量确认卖出信号
                    if df['volume_trend'].iloc[i] > 1.2:
                        strength *= 1.2
                    signals[i] = max(strength, -1.0)
            
            return signals
            
        except Exception as e:
            logger.error(f"计算牛牛策略信号时出错: {e}")
            return None

    def _calculate_volume_signal(self, df: pd.DataFrame) -> np.ndarray:
        """计算成交量策略信号
        
        基于多个成交量指标综合分析，包括：
        1. VWAP（成交量加权平均价格）
        2. CMF（钱德勒资金流量）
        3. OBV（能量潮指标）
        4. PVT（价格成交量趋势）
        """
        try:
            # 1. 计算VWAP
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
            df['vwap'] = df['vwap'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # 2. 计算CMF
            mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mfv = mfm * df['volume']
            df['cmf'] = mfv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # 3. 计算OBV
            df['obv'] = np.where(df['close'] > df['close'].shift(1),
                                df['volume'],
                                np.where(df['close'] < df['close'].shift(1),
                                        -df['volume'], 0)).cumsum()
            
            # 4. 计算PVT
            df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
            df['pvt'] = df['pvt'].cumsum()
            
            # 计算各指标的信号
            vwap_signal = np.where(df['close'] > df['vwap'], 1, -1)
            cmf_signal = np.where(df['cmf'] > 0.05, 1, np.where(df['cmf'] < -0.05, -1, 0))
            obv_signal = np.where(df['obv'] > df['obv'].shift(1), 1, -1)
            pvt_signal = np.where(df['pvt'] > df['pvt'].shift(1), 1, -1)
            
            # 计算成交量趋势
            volume_ma = df['volume'].rolling(window=20).mean()
            volume_trend = df['volume'] / volume_ma
            
            # 综合信号
            signals = np.zeros(len(df))
            for i in range(len(df)):
                # 计算综合信号得分 (-4 到 4)
                score = vwap_signal[i] + cmf_signal[i] + obv_signal[i] + pvt_signal[i]
                
                # 根据成交量趋势调整信号强度
                if volume_trend.iloc[i] > 1.5:  # 显著放量
                    score *= 1.2
                elif volume_trend.iloc[i] < 0.7:  # 显著缩量
                    score *= 0.8
                
                # 归一化到 [-1, 1] 范围
                signals[i] = np.clip(score / 4, -1, 1)
            
            return signals
            
        except Exception as e:
            logger.error(f"计算成交量策略信号时出错: {e}")
            return None

    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算高级技术指标"""
        try:
            # 1. 趋势指标
            # DPO (Detrended Price Oscillator)
            df['dpo'] = talib.HT_TRENDLINE(df['close']) - df['close']
            
            # KAMA (Kaufman Adaptive Moving Average)
            df['kama'] = talib.KAMA(df['close'], timeperiod=30)
            
            # MAMA (MESA Adaptive Moving Average)
            df['mama'], df['fama'] = talib.MAMA(df['close'])
            
            # 2. 动量指标
            # PPO (Percentage Price Oscillator)
            df['ppo'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
            
            # Ultimate Oscillator
            df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
            
            # 3. 成交量指标
            # MFI (Money Flow Index)
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            
            # OBV (On Balance Volume)
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # 4. 价格模式指标
            # 蜡烛图模式
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            
            # 5. 市场预测策略指标
            # 计算20日移动平均线
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['Momentum_bottom_reversal'] = (df['close'] > df['ma20']) & (df['close'].shift(1) < df['ma20'].shift(1))
            
            # 6. CPGW策略指标
            # 计算指标 A、B 和 D
            df['A'] = df['close'].rolling(window=34).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10))
            df['A'] = df['A'].rolling(window=19).mean()
            df['B'] = df['close'].rolling(window=14).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10))
            df['D'] = df['close'].rolling(window=34).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10))
            df['D'] = df['D'].ewm(span=4).mean()
            
            # 计算长庄线、游资线和主力线
            df['Long_Line'] = df['A'] + 100
            df['Hot_Money_Line'] = df['B'] + 100
            df['Main_Force_Line'] = df['D'] + 100
            
            # 7. 牛牛策略指标
            # 计算 MID 值
            df['MID'] = (3 * df['close'] + df['low'] + df['open'] + df['high']) / 6
            
            # 计算牛线（主力成本线）
            weights = np.arange(20, 0, -1)  # 权重为20到1
            weighted_mid = pd.concat([df['MID'].shift(i) * weights[i] for i in range(20)], axis=1).sum(axis=1)
            df['Bull_Line'] = weighted_mid / weights.sum()
            
            # 计算买卖线
            df['Trade_Line'] = df['Bull_Line'].rolling(window=2).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"计算高级技术指标时出错: {e}")
            return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率特征"""
        try:
            # 1. 真实波动率
            df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
            
            # 2. 不同时间窗口的波动率
            for window in [5, 10, 20, 30]:
                # 收盘价波动率（年化）
                df[f'volatility_{window}d'] = df['close'].pct_change().rolling(window).std() * np.full(len(df), np.sqrt(252))
                # 成交量波动率
                df[f'volume_volatility_{window}d'] = df['volume'].pct_change().rolling(window).std()
                # 真实波动率的移动平均
                df[f'atr_{window}d'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window)
            
            # 3. 波动率比率
            df['volatility_ratio'] = df['volatility_5d'] / df['volatility_30d']
            
            # 4. 高低价波动
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['high_low_range_ma'] = df['high_low_range'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"添加波动率特征时出错: {e}")
            return df

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场情绪特征"""
        try:
            # 1. 价格动量
            df['price_momentum'] = df['close'].pct_change(periods=5)
            df['price_momentum_ma'] = df['price_momentum'].rolling(window=20).mean()
            
            # 2. 成交量动量
            df['volume_momentum'] = df['volume'].pct_change(periods=5)
            df['volume_momentum_ma'] = df['volume_momentum'].rolling(window=20).mean()
            
            # 3. 价格和成交量的相关性
            df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
            
            # 4. 趋势强度指标
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['di_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            df['di_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # 5. 市场效率系数
            df['market_efficiency'] = abs(df['close'].pct_change(20)) / (abs(df['close'].pct_change()).rolling(20).sum())
            
            # 6. 波动率指标
            df['bollinger_width'] = (
                talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)[0] -
                talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)[2]
            ) / df['close']
            
            return df
            
        except Exception as e:
            logger.error(f"添加市场情绪特征时出错: {e}")
            return df

    def _calculate_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基本必需的技术指标
        """
        if df is None or df.empty:
            self.logger.warning("数据为空，无法计算指标")
            return df
        
        try:
            # 检查必需的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"缺少必需的列: {', '.join(missing_columns)}")
                return df
            
            # 计算基本价格指标
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()
            
            # 计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # 计算布林带
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (std * 2)
            df['BB_lower'] = df['BB_middle'] - (std * 2)
            
            # 填充NaN值
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {str(e)}")
            return df

    def prepare_data(self, symbols: List[str], start_date: str, end_date: str, 
                     sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        参数:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            sequence_length: 序列长度
            
        返回:
            特征和目标变量的元组
        """
        try:
            all_features = []
            all_targets = []
            
            for symbol in symbols:
                # 获取股票数据
                df = self.get_stock_data(symbol, start_date, end_date)
                if df is None or df.empty:
                    self.logger.warning(f"无法获取股票数据: {symbol}")
                    continue
                    
                # 计算技术指标
                df = self._calculate_required_indicators(df)
                
                # 准备特征
                features = self.prepare_features(df, sequence_length)
                if features is None or len(features) == 0:
                    self.logger.warning(f"无法准备特征: {symbol}")
                    continue
                    
                # 准备目标变量
                targets = self.prepare_targets(df)
                if targets is None or len(targets) == 0:
                    self.logger.warning(f"无法准备目标变量: {symbol}")
                    continue
                    
                # 确保特征和目标变量的长度匹配
                min_length = min(len(features), len(targets))
                features = features[-min_length:]
                targets = targets[-min_length:]
                
                all_features.append(features)
                all_targets.append(targets)
            
            if not all_features or not all_targets:
                self.logger.error("没有有效的训练数据")
                return np.array([]), np.array([])
            
            # 合并所有股票的数据
            X = np.concatenate(all_features, axis=0)
            y = np.concatenate(all_targets, axis=0)
            
            # 打印数据统计信息
            self.logger.info(f"特征形状: {X.shape}")
            self.logger.info(f"目标变量形状: {y.shape}")
            self.logger.info(f"特征统计: mean={np.mean(X):.4f}, std={np.std(X):.4f}, min={np.min(X):.4f}, max={np.max(X):.4f}")
            self.logger.info(f"目标变量统计: mean={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"准备数据时出错: {str(e)}")
            return np.array([]), np.array([])

    def _get_signal_stats(self, signals: np.ndarray) -> Dict[str, Any]:
        """获取信号统计信息
        
        Args:
            signals: 信号数组
            
        Returns:
            统计信息字典
        """
        # 确保signals是numpy数组
        if isinstance(signals, pd.Series):
            signals = signals.values
        signals = np.array(signals, dtype=float)
        
        # 处理无效值
        signals = np.nan_to_num(signals, nan=0.0)
        
        return {
            'length': len(signals),
            'unique_values': len(np.unique(signals)),
            'mean': float(np.mean(signals)),
            'std': float(np.std(signals)),  # Calculate overall std
            'nan_count': int(np.isnan(signals).sum())
        }

    def _get_strategy_performance(self, start_date: str, end_date: str, 
                                symbols: List[str]) -> pd.DataFrame:
        """获取策略表现数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbols: 股票代码列表
            
        Returns:
            策略表现数据框
        """
        query = """
        SELECT date, symbol, return, volatility, sharpe, max_drawdown, weight
        FROM strategy_performance
        WHERE date BETWEEN %s AND %s
        AND symbol IN %s
        ORDER BY date, symbol
        """
        return pd.read_sql_query(query, self.engine, 
                               params=(start_date, end_date, tuple(symbols)),
                               index_col='date')
                               
    def _get_market_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取市场特征数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            市场特征数据框
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        market_features = []
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            market_state = self.market_state.calculate_market_state(date_str)
            if market_state:
                features = create_market_features(market_state)
                features.name = date
                market_features.append(features)
                
        return pd.DataFrame(market_features)
        
    def prepare_prediction_data(self, current_date: str, symbols: List[str]) -> pd.DataFrame:
        """准备预测数据
        
        Args:
            current_date: 当前日期
            symbols: 股票代码列表
            
        Returns:
            特征数据框
        """
        # 获取最近的策略表现
        recent_perf = self._get_recent_performance(current_date, symbols)
        
        # 获取当前市场状态
        market_state = self.market_state.calculate_market_state(current_date)
        market_features = create_market_features(market_state)
        
        # 构建预测特征
        features = []
        for symbol in symbols:
            symbol_perf = recent_perf[recent_perf['symbol'] == symbol]
            if not symbol_perf.empty:
                # 合并策略表现和市场特征
                feature_vector = pd.concat([
                    symbol_perf[['return', 'volatility', 'sharpe', 'max_drawdown']],
                    market_features
                ])
                
                # 添加股票特定特征
                feature_vector['symbol'] = symbol
                
                features.append(feature_vector)
                
        return pd.DataFrame(features)
        
    def _get_recent_performance(self, current_date: str, symbols: List[str]) -> pd.DataFrame:
        """获取最近的策略表现数据
        
        Args:
            current_date: 当前日期
            symbols: 股票代码列表
            
        Returns:
            最近的策略表现数据框
        """
        query = """
        SELECT symbol, return, volatility, sharpe, max_drawdown
        FROM strategy_performance
        WHERE date = (
            SELECT MAX(date)
            FROM strategy_performance
            WHERE date <= %s
        )
        AND symbol IN %s
        """
        return pd.read_sql_query(query, self.engine, 
                               params=(current_date, tuple(symbols)))

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """填充缺失值
        
        Args:
            df: 输入DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 创建DataFrame的副本
        df = df.copy()
        
        # 对每一列分别处理
        for col in df.columns:
            # 获取当前列的中位数
            median_val = df[col].median()
            # 记录NaN的数量
            nan_count = df[col].isna().sum()
            # 使用fillna方法填充NaN
            df[col] = df[col].fillna(median_val)
            logger.info(f"列 {col} 使用中位数 {median_val:.4f} 填充了 {nan_count} 个NaN值")
        
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        # 创建DataFrame的副本
        df = df.copy()
        
        # 使用滚动窗口计算均值和标准差
        rolling_mean = df.rolling(window=60, min_periods=1).mean()
        rolling_std = df.rolling(window=60, min_periods=1).std()
        
        # 将标准差为0的值替换为1，避免除以0
        rolling_std = rolling_std.replace(0, 1)
        
        # 标准化
        df = (df - rolling_mean) / rolling_std
        
        # 检查异常值
        for col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if abs(mean_val) > 5 or std_val > 5:
                logger.warning(f"特征 {col} 的标准化可能有问题: mean={mean_val:.4f}, std={std_val:.4f}")
        
        return df

    def get_symbols(self):
        return ["GOOG", "NVDA", "AMD", "TSLA", "AAPL", "ASML", "MSFT", "AMZN", "META", "GOOGL"]  # 默认列表作为后备