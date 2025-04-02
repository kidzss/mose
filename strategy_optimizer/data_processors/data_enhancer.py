import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

class DataEnhancer:
    """数据质量增强器"""
    def __init__(self):
        self.imputer = SimpleImputer(strategy='linear')
        self.outlier_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
    def enhance_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        增强数据质量
        
        参数:
            data: 输入数据框
            
        返回:
            处理后的数据框
        """
        try:
            if data.empty:
                logger.warning("输入数据为空")
                return data
                
            # 处理缺失值
            data = self.handle_missing_data(data)
            
            # 检测和处理异常值
            data = self.handle_outliers(data)
            
            # 添加数据质量分数
            data['data_quality_score'] = self.calculate_quality_score(data)
            
            return data
            
        except Exception as e:
            logger.error(f"增强数据质量时出错: {e}")
            return data
        
    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        参数:
            data: 输入数据框
            
        返回:
            处理后的数据框
        """
        try:
            # 对不同类型的数据使用不同的填充策略
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(exclude=[np.number]).columns
            
            # 数值型数据使用线性插值
            if not numeric_cols.empty:
                data[numeric_cols] = data[numeric_cols].interpolate(
                    method='linear',
                    limit_direction='both',
                    axis=0
                )
            
            # 分类型数据使用前值填充
            if not categorical_cols.empty:
                data[categorical_cols] = data[categorical_cols].fillna(method='ffill')
                data[categorical_cols] = data[categorical_cols].fillna(method='bfill')
            
            return data
            
        except Exception as e:
            logger.error(f"处理缺失值时出错: {e}")
            return data
            
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        检测和处理异常值
        
        参数:
            data: 输入数据框
            
        返回:
            处理后的数据框
        """
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if numeric_cols.empty:
                return data
                
            # 使用IsolationForest检测异常值
            outlier_labels = self.outlier_detector.fit_predict(data[numeric_cols])
            
            # 标记异常值
            data['is_outlier'] = outlier_labels == -1
            
            # 对异常值进行处理（这里使用中位数填充）
            for col in numeric_cols:
                median_value = data[col].median()
                data.loc[data['is_outlier'], col] = median_value
            
            return data
            
        except Exception as e:
            logger.error(f"处理异常值时出错: {e}")
            return data
            
    def calculate_quality_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算数据质量分数
        
        参数:
            data: 输入数据框
            
        返回:
            数据质量分数序列
        """
        try:
            # 计算各个维度的质量分数
            completeness_score = 1 - data.isnull().mean()
            outlier_score = 1 - data['is_outlier'].mean()
            
            # 计算时间连续性分数
            if 'date' in data.columns:
                date_gaps = data['date'].diff().dt.days
                continuity_score = 1 - (date_gaps > 1).mean()
            else:
                continuity_score = 1.0
            
            # 综合质量分数
            quality_score = (
                completeness_score.mean() * 0.4 +
                outlier_score * 0.3 +
                continuity_score * 0.3
            )
            
            return pd.Series(quality_score, index=data.index)
            
        except Exception as e:
            logger.error(f"计算质量分数时出错: {e}")
            return pd.Series(0.5, index=data.index)  # 返回默认分数 