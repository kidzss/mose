from .strategy_base import Strategy
from strategy_optimizer.utils.technical_indicators import calculate_technical_indicators
import numpy as np
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class VolumeStrategy(Strategy):
    """基于交易量的策略"""
    
    def __init__(self, data=None):
        super().__init__(data)
        self.name = 'Volume'
        self.parameters = {
            'volume_consensus_threshold': 0.6  # 交易量共识阈值
        }
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        if data is None or data.empty:
            logger.warning("数据为空，无法计算指标")
            return pd.DataFrame()
            
        try:
            # 使用technical_indicators模块计算所有指标
            df = calculate_technical_indicators(data)
            if df.empty:
                logger.warning("计算技术指标后数据为空")
                return pd.DataFrame()
            
            # 记录所有可用的列
            logger.info(f"可用的指标列: {df.columns.tolist()}")
            
            # 确保所有需要的指标都存在
            required_indicators = ['volume_consensus']
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            if missing_indicators:
                logger.error(f"缺少以下必需的指标: {missing_indicators}")
                return pd.DataFrame()
            
            # 检查指标是否包含有效值并填充NaN
            for indicator in required_indicators:
                if df[indicator].isnull().all():
                    logger.error(f"指标 {indicator} 的所有值都是NaN")
                    return pd.DataFrame()
                df[indicator] = df[indicator].fillna(method='ffill').fillna(method='bfill').fillna(0)
                logger.info(f"{indicator} 统计: mean={df[indicator].mean():.4f}, std={df[indicator].std():.4f}, min={df[indicator].min():.4f}, max={df[indicator].max():.4f}")
            
            return df
            
        except Exception as e:
            logger.error(f"计算指标时出错: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return pd.DataFrame()
            
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        if df is None or df.empty:
            logger.warning("数据为空，无法生成信号")
            return pd.DataFrame()
            
        try:
            signals = pd.DataFrame(index=df.index)
            
            # 确保volume_consensus存在
            if 'volume_consensus' not in df.columns:
                logger.error("缺少volume_consensus列")
                return pd.DataFrame()
            
            # 生成信号
            consensus_threshold = self.parameters['volume_consensus_threshold']
            
            # 1. 基础信号
            signals['signal'] = np.where(df['volume_consensus'] > consensus_threshold, 1,
                                       np.where(df['volume_consensus'] < -consensus_threshold, -1, 0))
            
            # 2. 信号强度
            signals['signal_strength'] = abs(df['volume_consensus'])
            
            # 3. 确保信号强度在[0,1]范围内
            signals['signal_strength'] = signals['signal_strength'].clip(0, 1)
            
            # 填充可能的NaN值
            signals = signals.fillna(0)
            
            # 记录信号生成结果
            logger.info(f"生成了 {len(signals)} 个信号")
            logger.info(f"信号统计: mean={signals['signal'].mean():.4f}, std={signals['signal'].std():.4f}, min={signals['signal'].min():.4f}, max={signals['signal'].max():.4f}")
            logger.info(f"信号强度统计: mean={signals['signal_strength'].mean():.4f}, std={signals['signal_strength'].std():.4f}, min={signals['signal_strength'].min():.4f}, max={signals['signal_strength'].max():.4f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"生成信号时出错: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return pd.DataFrame()
            
    def get_latest_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        """获取最新的交易信号和信号强度"""
        if df is None or df.empty:
            return 0, 0.0
            
        try:
            signals = self.generate_signals(df)
            if signals.empty:
                return 0, 0.0
                
            latest_signal = signals['signal'].iloc[-1]
            latest_strength = signals['signal_strength'].iloc[-1]
            
            return latest_signal, latest_strength
            
        except Exception as e:
            logger.error(f"获取最新信号时出错: {e}")
            return 0, 0.0 