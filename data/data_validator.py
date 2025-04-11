import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        验证数据质量并返回处理后的数据和验证报告
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            处理后的DataFrame和验证报告
        """
        report = {
            'original_rows': len(data),
            'missing_values': {},
            'outliers': {},
            'invalid_prices': {},
            'gaps': [],
            'processed_rows': 0,
            'validation_passed': True
        }
        
        # 检查数据完整性
        if not DataValidator.validate_data_completeness(data):
            report['validation_passed'] = False
            return data, report
        
        # 检查缺失值
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            report['missing_values'] = missing_counts[missing_counts > 0].to_dict()
            logger.warning(f"发现缺失值: {report['missing_values']}")
            data = DataValidator._handle_missing_values(data)
        
        # 检查异常值
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                outliers = DataValidator._detect_outliers(data[col])
                report['outliers'][col] = len(outliers)
                if len(outliers) > 0:
                    logger.warning(f"在 {col} 列中发现 {len(outliers)} 个异常值")
                    data = DataValidator._handle_outliers(data, col)
        
        # 检查价格合理性
        invalid_prices = DataValidator._validate_price_logic(data)
        if invalid_prices:
            report['invalid_prices'] = invalid_prices
            # 修正不合理的数据
            for issue, dates in invalid_prices.items():
                for date in dates:
                    date_idx = pd.to_datetime(date)
                    if issue == 'high_lower_than_low':
                        # 交换最高价和最低价
                        data.loc[date_idx, ['high', 'low']] = data.loc[date_idx, ['low', 'high']].values
                    elif issue == 'high_lower_than_open':
                        # 将最高价设置为开盘价和收盘价中的较大值
                        data.loc[date_idx, 'high'] = max(data.loc[date_idx, 'open'], data.loc[date_idx, 'close'])
                    elif issue == 'high_lower_than_close':
                        # 将最高价设置为开盘价和收盘价中的较大值
                        data.loc[date_idx, 'high'] = max(data.loc[date_idx, 'open'], data.loc[date_idx, 'close'])
                    elif issue == 'low_higher_than_open':
                        # 将最低价设置为开盘价和收盘价中的较小值
                        data.loc[date_idx, 'low'] = min(data.loc[date_idx, 'open'], data.loc[date_idx, 'close'])
                    elif issue == 'low_higher_than_close':
                        # 将最低价设置为开盘价和收盘价中的较小值
                        data.loc[date_idx, 'low'] = min(data.loc[date_idx, 'open'], data.loc[date_idx, 'close'])
            
            # 重新验证价格逻辑
            invalid_prices = DataValidator._validate_price_logic(data)
            if invalid_prices:
                report['validation_passed'] = False
                logger.error(f"修正后仍然存在不合理的价格数据: {invalid_prices}")
            else:
                logger.info("成功修正不合理的价格数据")
        
        # 检查数据连续性
        gaps = DataValidator._check_data_continuity(data)
        if gaps:
            report['gaps'] = gaps
            logger.warning(f"发现数据缺失区间: {gaps}")
        
        # 添加技术指标
        data = DataValidator._add_technical_indicators(data)
        
        report['processed_rows'] = len(data)
        return data, report
    
    @staticmethod
    def _validate_price_logic(data: pd.DataFrame) -> Dict[str, List[str]]:
        """验证价格逻辑关系"""
        invalid_prices = {}
        
        # 检查high >= low
        invalid_hl = data[data['high'] < data['low']].index.tolist()
        if invalid_hl:
            invalid_prices['high_lower_than_low'] = [str(idx) for idx in invalid_hl]
        
        # 检查high >= open and high >= close
        invalid_ho = data[data['high'] < data['open']].index.tolist()
        invalid_hc = data[data['high'] < data['close']].index.tolist()
        if invalid_ho:
            invalid_prices['high_lower_than_open'] = [str(idx) for idx in invalid_ho]
        if invalid_hc:
            invalid_prices['high_lower_than_close'] = [str(idx) for idx in invalid_hc]
        
        # 检查low <= open and low <= close
        invalid_lo = data[data['low'] > data['open']].index.tolist()
        invalid_lc = data[data['low'] > data['close']].index.tolist()
        if invalid_lo:
            invalid_prices['low_higher_than_open'] = [str(idx) for idx in invalid_lo]
        if invalid_lc:
            invalid_prices['low_higher_than_close'] = [str(idx) for idx in invalid_lc]
        
        # 检查价格是否为负
        for col in ['open', 'high', 'low', 'close']:
            negative_prices = data[data[col] < 0].index.tolist()
            if negative_prices:
                invalid_prices[f'negative_{col}'] = [str(idx) for idx in negative_prices]
        
        return invalid_prices
    
    @staticmethod
    def _check_data_continuity(data: pd.DataFrame) -> List[Dict[str, str]]:
        """检查数据连续性，识别缺失的交易日"""
        gaps = []
        if len(data) < 2:
            return gaps
            
        # 确保索引是日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # 获取所有工作日
        all_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
        missing_days = all_days.difference(data.index)
        
        # 识别连续的缺失区间
        if len(missing_days) > 0:
            gap_start = missing_days[0]
            gap_end = gap_start
            
            for i in range(1, len(missing_days)):
                if missing_days[i] - missing_days[i-1] == timedelta(days=1):
                    gap_end = missing_days[i]
                else:
                    gaps.append({
                        'start': gap_start.strftime('%Y-%m-%d'),
                        'end': gap_end.strftime('%Y-%m-%d'),
                        'days': (gap_end - gap_start).days + 1
                    })
                    gap_start = missing_days[i]
                    gap_end = gap_start
            
            # 添加最后一个区间
            gaps.append({
                'start': gap_start.strftime('%Y-%m-%d'),
                'end': gap_end.strftime('%Y-%m-%d'),
                'days': (gap_end - gap_start).days + 1
            })
        
        return gaps
    
    @staticmethod
    def _handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 对于adj_close列，如果缺失则使用close列的值
        if 'adj_close' in data.columns and data['adj_close'].isnull().any():
            # 创建adj_close的副本
            adj_close = data['adj_close'].copy()
            # 用close的值填充adj_close的空值
            adj_close[adj_close.isnull()] = data.loc[adj_close.isnull(), 'close']
            data['adj_close'] = adj_close
            
        # 使用前向填充和后向填充替换其他缺失值
        # 先确保数据类型正确
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 分别处理数值列和非数值列
        numeric_data = data[numeric_columns].ffill().bfill()
        non_numeric_data = data.drop(columns=numeric_columns, errors='ignore')
        
        # 合并结果
        result = pd.concat([numeric_data, non_numeric_data], axis=1)
        # 恢复原始列的顺序
        result = result[data.columns]
        
        return result
    
    @staticmethod
    def _detect_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        检测异常值，使用更稳健的方法
        
        Args:
            series: 数据序列
            threshold: 异常值阈值（以四分位距离的倍数表示）
            
        Returns:
            异常值序列
        """
        # 计算四分位数
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值的界限
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # 找出异常值
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers
    
    @staticmethod
    def _handle_outliers(data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        处理异常值，使用更稳健的方法
        
        Args:
            data: 数据DataFrame
            column: 要处理的列名
            
        Returns:
            处理后的DataFrame
        """
        # 获取异常值
        outliers = DataValidator._detect_outliers(data[column])
        
        if not outliers.empty:
            # 对于每个异常值，使用前后5个非异常值的中位数替换
            for idx in outliers.index:
                # 获取异常值前后5天的数据
                start_idx = max(0, data.index.get_loc(idx) - 5)
                end_idx = min(len(data), data.index.get_loc(idx) + 6)
                window_data = data[column].iloc[start_idx:end_idx]
                
                # 排除窗口内的其他异常值
                window_outliers = DataValidator._detect_outliers(window_data)
                valid_data = window_data[~window_data.index.isin(window_outliers.index)]
                
                if not valid_data.empty:
                    # 使用非异常值的中位数替换异常值
                    data.loc[idx, column] = valid_data.median()
                else:
                    # 如果没有有效的非异常值，使用原始窗口数据的中位数
                    data.loc[idx, column] = window_data.median()
        
        return data
    
    @staticmethod
    def _add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        if data.empty:
            return data
            
        # 计算收益率
        data['returns'] = data['close'].pct_change()
        
        # 计算波动率
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # 计算成交量指标
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_std'] = data['volume'].rolling(window=20).std()
        
        # 计算价格动量
        data['momentum'] = data['close'].pct_change(periods=10)
        
        # 计算价格趋势
        data['trend'] = data['close'].rolling(window=20).mean() / data['close'] - 1
        
        # 计算ATR
        data['tr'] = pd.DataFrame({
            'hl': data['high'] - data['low'],
            'hc': abs(data['high'] - data['close'].shift()),
            'lc': abs(data['low'] - data['close'].shift())
        }).max(axis=1)
        data['atr'] = data['tr'].rolling(window=14).mean()
        
        return data
    
    @staticmethod
    def validate_data_completeness(data: pd.DataFrame) -> bool:
        """
        验证数据完整性
        
        Args:
            data: 数据DataFrame
            
        Returns:
            数据是否完整
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 检查必需列是否存在
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"缺少必需列: {missing_columns}")
            return False
        
        # 检查数据是否为空
        if data.empty:
            logger.error("数据为空")
            return False
        
        # 检查是否有足够的数据点
        if len(data) < 10:  # 至少需要10个数据点
            logger.error(f"数据点不足: {len(data)} < 10")
            return False
        
        return True 