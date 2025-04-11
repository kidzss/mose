import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    
    使用XGBoost/LGBM对特征进行重要性评估，找出对预测最有贡献的特征。
    支持回归任务和分类任务。
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None, random_state: int = 42):
        """
        初始化特征重要性分析器
        
        参数:
            feature_names: 特征名称列表，默认为None，将使用数据中的列名
            random_state: 随机种子，用于复现结果
        """
        self.feature_names = feature_names
        self.random_state = random_state
        self.models = {}
        self.importances = {}
        self.feature_scores = {}
        self.logger = logging.getLogger(__name__)
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray],
        task: str = 'regression',
        model_type: str = 'xgboost',
        cv: int = 5,
        test_size: float = 0.2,
        time_series: bool = True,
        **model_params
    ) -> Dict[str, Any]:
        """
        训练模型并计算特征重要性
        
        参数:
            X: 特征DataFrame
            y: 目标变量
            task: 任务类型，'regression'或'classification'
            model_type: 模型类型，'xgboost'或'lightgbm'
            cv: 交叉验证折数
            test_size: 测试集比例
            time_series: 是否使用时间序列交叉验证
            model_params: 模型参数
            
        返回:
            包含分析结果的字典
        """
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame):
            if self.feature_names is not None and len(self.feature_names) == X.shape[1]:
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                X = pd.DataFrame(X)
                
        # 保存特征名称
        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)
        
        # 区分任务类型
        if task == 'regression':
            objective = 'reg:squarederror' if model_type == 'xgboost' else 'regression'
            metric = 'rmse'
            self.is_classification = False
        else:  # 分类任务
            objective = 'binary:logistic' if model_type == 'xgboost' else 'binary'
            metric = 'auc'
            self.is_classification = True
            
        # 设置默认参数
        if model_type == 'xgboost':
            default_params = {
                'objective': objective,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state
            }
        else:  # lightgbm
            default_params = {
                'objective': objective,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbose': -1
            }
            
        # 更新参数
        params = default_params.copy()
        params.update(model_params)
        
        # 初始化结果字典
        results = {}
        feature_importances = pd.DataFrame()
        
        # 使用时间序列交叉验证或标准交叉验证
        if time_series:
            tscv = TimeSeriesSplit(n_splits=cv)
            splits = tscv.split(X)
        else:
            # 简单随机划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            splits = [(np.arange(len(X_train)), np.arange(len(X_train), len(X)))]
        
        # 交叉验证训练
        for i, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"训练折 {i+1}/{cv}")
            
            # 获取训练集和测试集
            if time_series:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # 训练模型
            if model_type == 'xgboost':
                model = xgb.XGBRegressor(**params) if task == 'regression' else xgb.XGBClassifier(**params)
                
                # XGBoost在新版本中使用callbacks进行早停，不直接传递early_stopping_rounds
                eval_set = [(X_test_scaled, y_test)]
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=eval_set,
                    verbose=0
                )
            else:  # lightgbm
                model = lgb.LGBMRegressor(**params) if task == 'regression' else lgb.LGBMClassifier(**params)
                
                # LightGBM的早停设置
                callbacks = []
                if 'early_stopping_rounds' in model_params:
                    early_stopping_rounds = model_params.pop('early_stopping_rounds')
                    callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
                    
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    callbacks=callbacks
                )
            
            # 预测测试集
            y_pred = model.predict(X_test_scaled)
            
            # 计算性能指标
            if task == 'regression':
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                logger.info(f"折 {i+1} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
                
                result = {
                    'rmse': rmse,
                    'r2': r2
                }
            else:  # 分类任务
                accuracy = accuracy_score(y_test, y_pred.round())
                logger.info(f"折 {i+1} - 准确率: {accuracy:.4f}")
                
                result = {
                    'accuracy': accuracy
                }
                
            # 获取特征重要性
            if model_type == 'xgboost':
                importance = model.feature_importances_
            else:  # lightgbm
                importance = model.feature_importances_
                
            # 保存特征重要性
            fold_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
            fold_importance['fold'] = i + 1
            feature_importances = pd.concat([feature_importances, fold_importance], ignore_index=True)
            
            results[f'fold_{i+1}'] = result
            self.models[f'fold_{i+1}'] = model
        
        # 计算平均重要性
        mean_importance = feature_importances.groupby('feature')['importance'].mean().reset_index()
        std_importance = feature_importances.groupby('feature')['importance'].std().reset_index()
        
        # 合并并排序
        feature_importance_summary = pd.merge(mean_importance, std_importance, on='feature', suffixes=('_mean', '_std'))
        feature_importance_summary = feature_importance_summary.sort_values('importance_mean', ascending=False)
        
        # 保存结果
        self.importances['raw'] = feature_importances
        self.importances['summary'] = feature_importance_summary
        
        # 返回结果
        return {
            'cross_validation_results': results,
            'feature_importances': feature_importance_summary
        }
        
    def calculate_feature_correlations(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        计算特征与目标变量的相关性
        
        参数:
            X: 特征DataFrame
            y: 目标变量
            
        返回:
            特征相关性DataFrame
        """
        # 合并特征和目标变量
        data = X.copy()
        data['target'] = y
        
        # 计算各特征与目标的相关性
        correlations = {}
        for feature in X.columns:
            # 计算Spearman相关系数
            correlation, p_value = spearmanr(X[feature], y)
            correlations[feature] = {
                'correlation': correlation,
                'p_value': p_value
            }
            
        # 转换为DataFrame并排序
        correlation_df = pd.DataFrame.from_dict(correlations, orient='index')
        correlation_df = correlation_df.sort_values('correlation', ascending=False)
        
        return correlation_df
        
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制特征重要性图表
        
        参数:
            top_n: 显示前N个重要特征
            figsize: 图表大小
            
        返回:
            图表对象
        """
        if not self.importances:
            raise ValueError("必须先调用fit方法")
            
        # 获取特征重要性汇总
        importance_df = self.importances['summary'].head(top_n)
        
        # 创建图表
        plt.figure(figsize=figsize)
        sns.barplot(
            data=importance_df,
            y='feature',
            x='importance_mean',
            orient='h'
        )
        plt.title(f'Top {top_n} 特征重要性')
        plt.xlabel('重要性分数')
        plt.ylabel('特征名称')
        plt.tight_layout()
        
        return plt.gcf()
        
    def plot_correlation_importance(
        self, 
        correlation_df: pd.DataFrame, 
        top_n: int = 20, 
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        绘制特征相关性图表
        
        参数:
            correlation_df: 相关性DataFrame
            top_n: 显示前N个相关特征
            figsize: 图表大小
            
        返回:
            图表对象
        """
        # 选择前N个特征
        corr_df = correlation_df.head(top_n)
        
        # 创建图表
        plt.figure(figsize=figsize)
        sns.barplot(
            data=corr_df.reset_index(),
            y='index',
            x='correlation',
            orient='h'
        )
        plt.title(f'Top {top_n} 特征相关性')
        plt.xlabel('Spearman相关系数')
        plt.ylabel('特征名称')
        plt.tight_layout()
        
        return plt.gcf()
        
    def get_optimal_feature_subset(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray],
        min_features: int = 5,
        max_features: int = 20,
        step: int = 1,
        task: str = 'regression',
        **model_params
    ) -> Dict[str, Any]:
        """
        寻找最优特征子集
        
        参数:
            X: 特征DataFrame
            y: 目标变量
            min_features: 最小特征数量
            max_features: 最大特征数量
            step: 步长
            task: 任务类型
            model_params: 模型参数
            
        返回:
            最优特征子集结果
        """
        if not self.importances:
            self.fit(X, y, task=task, **model_params)
            
        # 获取按重要性排序的特征列表
        sorted_features = self.importances['summary']['feature'].tolist()
        
        results = []
        
        # 为不同特征数量训练模型
        for n_features in range(min_features, min(max_features+1, len(sorted_features)+1), step):
            # 选择前n_features个特征
            selected_features = sorted_features[:n_features]
            X_selected = X[selected_features]
            
            # 训练模型
            cv_result = self.fit(
                X_selected, y, task=task, 
                time_series=True, cv=5,
                **model_params
            )
            
            # 计算平均性能
            if task == 'regression':
                mean_rmse = np.mean([result['rmse'] for result in cv_result['cross_validation_results'].values()])
                mean_r2 = np.mean([result['r2'] for result in cv_result['cross_validation_results'].values()])
                
                results.append({
                    'n_features': n_features,
                    'features': selected_features,
                    'mean_rmse': mean_rmse,
                    'mean_r2': mean_r2
                })
                
                logger.info(f"特征数量: {n_features}, 平均RMSE: {mean_rmse:.4f}, 平均R²: {mean_r2:.4f}")
            else:
                mean_accuracy = np.mean([result['accuracy'] for result in cv_result['cross_validation_results'].values()])
                
                results.append({
                    'n_features': n_features,
                    'features': selected_features,
                    'mean_accuracy': mean_accuracy
                })
                
                logger.info(f"特征数量: {n_features}, 平均准确率: {mean_accuracy:.4f}")
                
        # 找到最优特征子集
        results_df = pd.DataFrame(results)
        if task == 'regression':
            best_idx = results_df['mean_r2'].idxmax()
        else:
            best_idx = results_df['mean_accuracy'].idxmax()
            
        best_subset = results_df.iloc[best_idx].to_dict()
        
        return {
            'all_results': results_df,
            'best_subset': best_subset
        }
        
    def plot_feature_selection_curve(self, results_df: pd.DataFrame, task: str = 'regression', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制特征选择曲线
        
        参数:
            results_df: 特征选择结果DataFrame
            task: 任务类型
            figsize: 图表大小
            
        返回:
            图表对象
        """
        plt.figure(figsize=figsize)
        
        if task == 'regression':
            # 绘制RMSE曲线
            plt.subplot(1, 2, 1)
            plt.plot(results_df['n_features'], results_df['mean_rmse'], 'o-')
            plt.xlabel('特征数量')
            plt.ylabel('平均RMSE')
            plt.title('特征数量 vs. RMSE')
            plt.grid(True)
            
            # 绘制R²曲线
            plt.subplot(1, 2, 2)
            plt.plot(results_df['n_features'], results_df['mean_r2'], 'o-')
            plt.xlabel('特征数量')
            plt.ylabel('平均R²')
            plt.title('特征数量 vs. R²')
            plt.grid(True)
        else:
            # 绘制准确率曲线
            plt.plot(results_df['n_features'], results_df['mean_accuracy'], 'o-')
            plt.xlabel('特征数量')
            plt.ylabel('平均准确率')
            plt.title('特征数量 vs. 准确率')
            plt.grid(True)
            
        plt.tight_layout()
        return plt.gcf() 