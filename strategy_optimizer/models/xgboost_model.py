#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost信号组合模型

使用XGBoost机器学习算法组合多个交易信号，识别最有效的特征组合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

class XGBoostSignalCombiner:
    """
    XGBoost信号组合模型
    
    使用XGBoost算法组合多个交易信号，并进行未来收益率预测
    """
    
    def __init__(self, 
                 objective: str = 'reg:squarederror', 
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42):
        """
        初始化XGBoost信号组合模型
        
        参数:
            objective: XGBoost目标函数
            learning_rate: 学习率
            n_estimators: 树的数量
            max_depth: 树的最大深度
            subsample: 样本子采样比例
            colsample_bytree: 特征列子采样比例
            random_state: 随机种子
        """
        self.model = xgb.XGBRegressor(
            objective=objective,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )
        self.feature_names = None
        self.feature_importance = None
        self.shap_values = None
        self.train_performance = {}
        self.test_performance = {}
        self.cv_performance = {}
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
            early_stopping_rounds: Optional[int] = None,
            verbose: bool = False) -> None:
        """
        训练模型
        
        参数:
            X: 特征数据，信号组合
            y: 目标变量，如未来收益率
            eval_set: 评估集列表, 格式为[(X_train, y_train), (X_val, y_val)]
            early_stopping_rounds: 早停轮数
            verbose: 是否打印训练过程
        """
        self.feature_names = X.columns.tolist()
        
        # 训练模型
        # 注意：较新版本的XGBoost不再接受early_stopping_rounds作为直接参数
        # 而是通过callbacks来实现早停
        callbacks = None
        if early_stopping_rounds is not None:
            try:
                # 尝试导入和使用早停回调
                from xgboost.callback import EarlyStopping
                callbacks = [EarlyStopping(rounds=early_stopping_rounds)]
            except ImportError:
                # 如果不支持，则忽略早停
                pass
        
        fit_params = {
            'eval_set': eval_set,
            'verbose': verbose
        }
        
        if callbacks is not None:
            fit_params['callbacks'] = callbacks
        
        self.model.fit(X, y, **fit_params)
        
        # 保存特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 计算训练集性能
        y_pred_train = self.model.predict(X)
        self.train_performance = self._calculate_performance(y, y_pred_train)
        
        # 如果有评估集，计算测试集性能
        if eval_set and len(eval_set) > 1:
            X_test, y_test = eval_set[1]
            y_pred_test = self.model.predict(X_test)
            self.test_performance = self._calculate_performance(y_test, y_pred_test)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        使用训练好的模型进行预测
        
        参数:
            X: 特征数据
        
        返回:
            预测结果
        """
        X_pred = X[self.feature_names] if self.feature_names else X
        return pd.Series(self.model.predict(X_pred), index=X.index)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估模型在给定数据上的性能
        
        参数:
            X: 特征数据
            y: 真实标签
            
        返回:
            包含性能指标的字典
        """
        # 预测
        y_pred = self.model.predict(X)
        
        # 计算性能指标
        performance = self._calculate_performance(y, y_pred)
        
        return performance
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, 
                       metrics: List[str] = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']) -> Dict[str, Tuple[float, float]]:
        """
        使用时间序列交叉验证评估模型
        
        参数:
            X: 特征数据
            y: 目标变量
            n_splits: 折数
            metrics: 要计算的指标列表
            
        返回:
            包含各指标平均值和标准差的字典
        """
        # 创建时间序列分割器
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 对每个指标进行交叉验证
        cv_results = {}
        for metric in metrics:
            scores = cross_val_score(self.model, X, y, cv=tscv, scoring=metric)
            cv_results[metric] = (scores.mean(), scores.std())
        
        self.cv_performance = cv_results
        return cv_results
    
    def get_feature_importance(self, plot: bool = False, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性
        
        参数:
            plot: 是否绘制特征重要性图
            top_n: 显示的顶部特征数量
            
        返回:
            特征重要性DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")
        
        # 获取顶部特征
        top_features = self.feature_importance.head(top_n)
        
        # 如果需要绘图
        if plot:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'XGBoost - 顶部 {top_n} 特征重要性')
            plt.tight_layout()
            plt.show()
        
        return top_features
    
    def calculate_shap_values(self, X: pd.DataFrame, plot_summary: bool = False) -> np.ndarray:
        """
        计算SHAP值以解释模型
        
        参数:
            X: 要解释的特征数据
            plot_summary: 是否绘制SHAP摘要图
            
        返回:
            SHAP值数组
        """
        # 创建解释器
        explainer = shap.TreeExplainer(self.model)
        
        # 计算SHAP值
        self.shap_values = explainer.shap_values(X)
        
        # 绘制摘要图
        if plot_summary:
            shap.summary_plot(self.shap_values, X, feature_names=self.feature_names)
        
        return self.shap_values
    
    def plot_shap_force(self, X: pd.DataFrame, index: int) -> None:
        """
        为单个预测绘制SHAP力图
        
        参数:
            X: 特征数据
            index: 要解释的样本索引
        """
        if self.shap_values is None:
            self.calculate_shap_values(X, plot_summary=False)
        
        # 绘制力图
        shap.force_plot(
            base_value=self.model.intercept_[0] if hasattr(self.model, 'intercept_') else 0,
            shap_values=self.shap_values[index, :],
            features=X.iloc[index, :],
            feature_names=self.feature_names,
            matplotlib=True,
            show=True
        )
    
    def plot_performance(self, X_test: pd.DataFrame, y_test: pd.Series, plot_type: str = 'predictions') -> None:
        """
        绘制模型性能图表
        
        参数:
            X_test: 测试特征数据
            y_test: 测试目标变量
            plot_type: 图表类型，'predictions'或'cumulative_returns'
        """
        # 获取预测值
        y_pred = self.model.predict(X_test)
        
        if plot_type == 'predictions':
            # 绘制预测vs实际值
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.index, y_test.values, label='实际值', color='blue')
            plt.plot(y_test.index, y_pred, label='预测值', color='red', alpha=0.7)
            plt.title('XGBoost模型 - 预测vs实际值')
            plt.xlabel('日期')
            plt.ylabel('收益率')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        elif plot_type == 'cumulative_returns':
            # 绘制累积收益对比
            cum_actual = (1 + y_test).cumprod() - 1
            
            # 计算模型信号产生的累积收益
            # 假设我们基于预测值>0执行多头策略
            position = np.sign(y_pred)
            strategy_returns = position * y_test
            cum_strategy = (1 + strategy_returns).cumprod() - 1
            
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.index, cum_actual, label='Buy & Hold', color='blue')
            plt.plot(y_test.index, cum_strategy, label='模型策略', color='green')
            plt.title('XGBoost模型 - 累积收益对比')
            plt.xlabel('日期')
            plt.ylabel('累积收益率')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        else:
            raise ValueError(f"不支持的图表类型: {plot_type}")
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型到文件
        
        参数:
            filepath: 文件路径
        """
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        从文件加载模型
        
        参数:
            filepath: 文件路径
        """
        self.model.load_model(filepath)
    
    def _calculate_performance(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算预测性能指标
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            
        返回:
            包含各性能指标的字典
        """
        # 计算基础回归指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算符号准确率 (方向性预测准确率)
        sign_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        # 计算金融特定指标
        # 收益率
        position = np.sign(y_pred)  # 基于预测符号的持仓
        strategy_returns = position * y_true  # 策略收益率
        
        # 假设我们有252个交易日
        ann_factor = 252 / len(y_true)
        
        # 累积收益
        cum_returns = (1 + strategy_returns).cumprod() - 1
        total_return = cum_returns.iloc[-1] if isinstance(cum_returns, pd.Series) else cum_returns[-1]
        
        # 年化收益率
        annual_return = (1 + strategy_returns.mean()) ** 252 - 1
        
        # 夏普比率
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # 最大回撤
        cum_returns_series = pd.Series(cum_returns) if not isinstance(cum_returns, pd.Series) else cum_returns
        running_max = cum_returns_series.cummax()
        drawdown = (cum_returns_series / running_max) - 1
        max_drawdown = drawdown.min()
        
        # 整合所有指标
        performance = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'sign_accuracy': sign_accuracy,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return performance
    
    def backtest(self, X: pd.DataFrame, y: pd.Series, transaction_cost: float = 0.0) -> dict:
        """
        使用模型进行回测
        
        参数:
            X: 特征数据
            y: 实际收益率
            transaction_cost: 交易成本
            
        返回:
            回测结果指标字典
        """
        # 预测收益
        pred_returns = self.predict(X)
        
        # 生成交易信号（1表示做多，-1表示做空，0表示不操作）
        signals = np.sign(pred_returns)
        
        # 计算实际收益（考虑交易成本）
        # 注：这里简化处理，实际应考虑信号变化时才收取成本
        strategy_returns = signals * y - np.abs(signals.diff().fillna(0)) * transaction_cost
        
        # 计算累积收益
        cum_returns = (1 + strategy_returns).cumprod() - 1
        
        # 计算买入持有累积收益
        buy_hold_cum_returns = (1 + y).cumprod() - 1
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(cum_returns)
        
        # 计算年化收益和夏普比率
        days = len(y)
        annual_factor = 252 / days  # 假设252个交易日/年
        annual_return = (1 + cum_returns.iloc[-1]) ** annual_factor - 1
        sharpe_ratio = np.sqrt(annual_factor) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        
        # 计算交易正确率
        correct_direction = (np.sign(y) == np.sign(pred_returns))
        sign_accuracy = correct_direction.mean()
        
        # 计算每月收益
        if isinstance(y.index, pd.DatetimeIndex):
            monthly_returns = strategy_returns.resample('ME').sum()
        else:
            monthly_returns = None
        
        return {
            'signals': signals,
            'strategy_returns': strategy_returns,
            'cumulative_returns': cum_returns,
            'buy_hold_returns': buy_hold_cum_returns,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sign_accuracy': sign_accuracy,
            'monthly_returns': monthly_returns,
            'total_trades': np.abs(signals.diff().fillna(0)).sum() / 2,
            'win_rate': correct_direction[signals != 0].mean() if (signals != 0).any() else 0
        }
    
    def _calculate_max_drawdown(self, returns_or_cum_returns: pd.Series) -> float:
        """计算最大回撤"""
        # 如果输入是收益率，先转换成累积收益
        if not (returns_or_cum_returns == returns_or_cum_returns.cummax()).all():
            cum_returns = returns_or_cum_returns
        else:
            cum_returns = (1 + returns_or_cum_returns).cumprod() - 1
            
        # 计算历史最高点
        running_max = cum_returns.cummax()
        
        # 计算回撤
        drawdown = (cum_returns - running_max) / (1 + running_max)
        
        # 返回最大回撤（负值）
        return drawdown.min()


# 使用示例
if __name__ == "__main__":
    # 导入必要的库
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 假设我们有从SignalExtractor提取的信号数据
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    np.random.seed(42)
    
    # 创建一些虚拟特征
    n_features = 20
    feature_data = {}
    
    for i in range(n_features):
        if i < 5:  # 前5个特征与目标有较强相关性
            feature_data[f'signal_{i}'] = np.random.randn(len(dates)) * 0.1 + np.sin(np.arange(len(dates)) / 20) * 0.2
        else:
            feature_data[f'signal_{i}'] = np.random.randn(len(dates)) * 0.1
    
    # 创建特征DataFrame
    X = pd.DataFrame(feature_data, index=dates)
    
    # 创建目标变量（1天后的收益率）
    # 一些特征的线性组合加上噪声
    y_true = (X['signal_0'] * 0.5 + X['signal_1'] * 0.3 + X['signal_2'] * 0.2) + np.random.randn(len(dates)) * 0.01
    y = pd.Series(y_true, index=dates)
    
    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # 创建并训练XGBoost模型
    model = XGBoostSignalCombiner(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=20,
        verbose=True
    )
    
    # 查看模型性能
    print("\n训练集性能:")
    for metric, value in model.train_performance.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n测试集性能:")
    for metric, value in model.test_performance.items():
        print(f"{metric}: {value:.4f}")
    
    # 显示特征重要性
    print("\n特征重要性:")
    importance = model.get_feature_importance(plot=False)
    print(importance.head(10))
    
    # 交叉验证
    print("\n交叉验证结果:")
    cv_results = model.cross_validate(X, y, n_splits=5)
    for metric, (mean, std) in cv_results.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")
    
    # 绘制性能图表
    model.plot_performance(X_test, y_test, plot_type='predictions')
    model.plot_performance(X_test, y_test, plot_type='cumulative_returns') 