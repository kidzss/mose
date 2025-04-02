# strategy_optimizer/models/conditional_xgboost.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from ..models.xgboost_model import XGBoostSignalCombiner

class ConditionalXGBoostCombiner:
    """
    条件XGBoost信号组合器
    
    为不同的市场状态训练专门的模型
    """
    
    def __init__(self, market_states: List[int] = [1, 2, 3, 4, 5]):
        """
        初始化条件XGBoost信号组合器
        
        参数:
            market_states: 市场状态列表，默认为5种状态
        """
        self.market_states = market_states
        self.models = {state: XGBoostSignalCombiner() for state in market_states}
        self.global_model = XGBoostSignalCombiner()
        self.feature_names = None
        
    def fit(self, 
           X: pd.DataFrame, 
           y: pd.Series, 
           market_state: pd.Series,
           eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series, pd.Series]]] = None,
           min_samples: int = 30):
        """
        训练模型
        
        参数:
            X: 特征数据
            y: 目标变量
            market_state: 市场状态
            eval_set: 评估集
            min_samples: 每个市场状态的最小样本数
        """
        self.feature_names = X.columns.tolist()
        
        # 训练全局模型
        self.global_model.fit(X, y)
        
        # 为每个市场状态训练特定模型
        for state in self.market_states:
            mask = market_state == state
            if mask.sum() >= min_samples:
                X_state = X[mask]
                y_state = y[mask]
                self.models[state].fit(X_state, y_state)
            else:
                # 如果样本不足，使用全局模型
                self.models[state] = self.global_model
                
        return self
    
    def predict(self, X: pd.DataFrame, market_state: pd.Series) -> pd.Series:
        """
        预测
        
        参数:
            X: 特征数据
            market_state: 市场状态
            
        返回:
            预测结果
        """
        predictions = pd.Series(index=X.index, dtype=float)
        
        # 对每个市场状态使用相应的模型进行预测
        for state in self.market_states:
            mask = market_state == state
            if mask.any():
                X_state = X[mask]
                if state in self.models:
                    pred = self.models[state].predict(X_state)
                    predictions[mask] = pred
                else:
                    # 如果没有该状态的模型，使用全局模型
                    pred = self.global_model.predict(X_state)
                    predictions[mask] = pred
                    
        return predictions
    
    def get_feature_importance(self, plot: bool = True) -> Dict[int, pd.DataFrame]:
        """
        获取特征重要性
        
        参数:
            plot: 是否绘制特征重要性图，传递给XGBoostSignalCombiner
            
        返回:
            每个市场状态的特征重要性
        """
        importance_dict = {}
        
        # 全局模型的特征重要性
        importance_dict[0] = self.global_model.get_feature_importance(plot=plot)
        
        # 每个市场状态的特征重要性
        for state in self.market_states:
            if state in self.models and self.models[state] != self.global_model:
                importance_dict[state] = self.models[state].get_feature_importance(plot=plot)
                
        return importance_dict