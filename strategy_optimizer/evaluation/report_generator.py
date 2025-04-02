# strategy_optimizer/evaluation/report_generator.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import os
from datetime import datetime

class StrategyReportGenerator:
    """
    策略报告生成器
    
    生成全面的策略评估报告
    """
    
    def __init__(self, output_dir: str = "strategy_optimizer/outputs"):
        """
        初始化报告生成器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建时间戳子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(output_dir, f"report_{timestamp}")
        os.makedirs(self.report_dir, exist_ok=True)
        
    def generate_model_performance_report(self, 
                                        model, 
                                        X_train: pd.DataFrame, 
                                        y_train: pd.Series,
                                        X_test: pd.DataFrame, 
                                        y_test: pd.Series,
                                        market_state_train: Optional[pd.Series] = None,
                                        market_state_test: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        生成模型性能报告
        
        参数:
            model: 训练好的模型
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            market_state_train: 训练集市场状态
            market_state_test: 测试集市场状态
            
        返回:
            包含性能指标的字典
        """
        # 预测
        if hasattr(model, 'predict') and callable(model.predict):
            if market_state_test is not None and hasattr(model, 'market_states'):
                # 条件模型
                y_pred = model.predict(X_test, market_state_test)
            else:
                # 标准模型
                y_pred = model.predict(X_test)
                
            # 计算性能指标
            performance = {}
            
            # 基本指标
            performance["mse"] = ((y_test - y_pred) ** 2).mean()
            performance["rmse"] = np.sqrt(performance["mse"])
            performance["mae"] = np.abs(y_test - y_pred).mean()
            
            # 方向准确率
            direction_accuracy = ((y_test > 0) == (y_pred > 0)).mean()
            performance["direction_accuracy"] = direction_accuracy
            
            # 按市场状态划分性能
            if market_state_test is not None:
                performance["market_state"] = {}
                for state in market_state_test.unique():
                    mask = market_state_test == state
                    if mask.sum() > 0:
                        y_test_state = y_test[mask]
                        y_pred_state = y_pred[mask]
                        
                        perf = {}
                        perf["mse"] = ((y_test_state - y_pred_state) ** 2).mean()
                        perf["rmse"] = np.sqrt(perf["mse"])
                        perf["mae"] = np.abs(y_test_state - y_pred_state).mean()
                        perf["direction_accuracy"] = ((y_test_state > 0) == (y_pred_state > 0)).mean()
                        perf["count"] = mask.sum()
                        
                        performance["market_state"][int(state)] = perf
            
            # 绘制预测与实际值对比图
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.values, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title('Model Predictions vs Actual Values')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.report_dir, "predictions_vs_actual.png"))
            plt.close()
            
            # 绘制预测误差分布图
            plt.figure(figsize=(12, 6))
            sns.histplot(y_test - y_pred, kde=True)
            plt.title('Prediction Error Distribution')
            plt.grid(True)
            plt.savefig(os.path.join(self.report_dir, "error_distribution.png"))
            plt.close()
            
            return performance
        
        return {"error": "Model does not have predict method"}
    
    def generate_feature_importance_report(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """
        生成特征重要性报告
        
        参数:
            model: 训练好的模型
            X: 特征数据
            
        返回:
            特征重要性DataFrame
        """
        if hasattr(model, 'get_feature_importance') and callable(model.get_feature_importance):
            # 获取特征重要性
            importance = model.get_feature_importance(plot=False)
            
            # 处理不同模型返回的不同格式
            if isinstance(importance, dict):
                # ConditionalXGBoostCombiner返回按市场状态分组的特征重要性
                # 使用全局模型的特征重要性绘图
                global_importance = importance.get(0, pd.Series())
                
                # 绘制全局特征重要性图
                if not global_importance.empty:
                    plt.figure(figsize=(12, len(global_importance) * 0.3 + 2))
                    
                    # 处理不同格式的重要性
                    if isinstance(global_importance, pd.Series):
                        # 如果是Series，直接绘制
                        global_importance.sort_values().plot(kind='barh')
                    elif isinstance(global_importance, pd.DataFrame):
                        # 如果是DataFrame，检查格式
                        if 'feature' in global_importance.columns and 'importance' in global_importance.columns:
                            # 标准XGBoost格式
                            sorted_df = global_importance.sort_values('importance')
                            plt.barh(y=sorted_df['feature'], width=sorted_df['importance'])
                        elif 'weight' in global_importance.columns:
                            # 另一种可能的格式
                            sorted_df = global_importance.sort_values('weight')
                            plt.barh(y=sorted_df.index, width=sorted_df['weight'])
                        else:
                            # 未知格式，使用第一列
                            col = global_importance.columns[0]
                            sorted_df = global_importance.sort_values(col)
                            plt.barh(y=range(len(sorted_df)), width=sorted_df[col])
                            plt.yticks(range(len(sorted_df)), sorted_df.index)
                    
                    plt.title('Global Feature Importance')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.report_dir, "global_feature_importance.png"))
                    plt.close()
                    
                # 为每个市场状态生成单独的特征重要性图
                for state, state_importance in importance.items():
                    if state == 0 or state_importance.empty:
                        continue
                        
                    plt.figure(figsize=(12, len(state_importance) * 0.3 + 2))
                    
                    # 处理不同格式的重要性
                    if isinstance(state_importance, pd.Series):
                        # 如果是Series，直接绘制
                        state_importance.sort_values().plot(kind='barh')
                    elif isinstance(state_importance, pd.DataFrame):
                        # 如果是DataFrame，检查格式
                        if 'feature' in state_importance.columns and 'importance' in state_importance.columns:
                            # 标准XGBoost格式
                            sorted_df = state_importance.sort_values('importance')
                            plt.barh(y=sorted_df['feature'], width=sorted_df['importance'])
                        elif 'weight' in state_importance.columns:
                            # 另一种可能的格式
                            sorted_df = state_importance.sort_values('weight')
                            plt.barh(y=sorted_df.index, width=sorted_df['weight'])
                        else:
                            # 未知格式，使用第一列
                            col = state_importance.columns[0]
                            sorted_df = state_importance.sort_values(col)
                            plt.barh(y=range(len(sorted_df)), width=sorted_df[col])
                            plt.yticks(range(len(sorted_df)), sorted_df.index)
                    
                    plt.title(f'Feature Importance for Market State {state}')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.report_dir, f"feature_importance_state_{state}.png"))
                    plt.close()
                    
                # 返回全局特征重要性
                return global_importance
            else:
                # 标准模型返回单一特征重要性
                # 绘制特征重要性图
                plt.figure(figsize=(12, len(importance) * 0.3 + 2))
                
                # 处理不同格式的重要性
                if isinstance(importance, pd.Series):
                    # 如果是Series，直接绘制
                    importance.sort_values().plot(kind='barh')
                elif isinstance(importance, pd.DataFrame):
                    # 如果是DataFrame，检查格式
                    if 'feature' in importance.columns and 'importance' in importance.columns:
                        # 标准XGBoost格式
                        sorted_df = importance.sort_values('importance')
                        plt.barh(y=sorted_df['feature'], width=sorted_df['importance'])
                    elif 'weight' in importance.columns:
                        # 另一种可能的格式
                        sorted_df = importance.sort_values('weight')
                        plt.barh(y=sorted_df.index, width=sorted_df['weight'])
                    else:
                        # 未知格式，使用第一列
                        col = importance.columns[0]
                        sorted_df = importance.sort_values(col)
                        plt.barh(y=range(len(sorted_df)), width=sorted_df[col])
                        plt.yticks(range(len(sorted_df)), sorted_df.index)
                
                plt.title('Feature Importance')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.report_dir, "feature_importance.png"))
                plt.close()
                
                return importance
        
        return pd.DataFrame()
    
    def generate_strategy_comparison_report(self, 
                                          strategies: List[Dict[str, Any]],
                                          X: pd.DataFrame,
                                          y: pd.Series,
                                          market_state: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        生成策略比较报告
        
        参数:
            strategies: 策略列表，每个策略是一个字典，包含名称、模型等
            X: 特征数据
            y: 目标变量
            market_state: 可选，市场状态序列，用于条件模型
            
        返回:
            策略比较DataFrame
        """
        results = []
        
        for strategy in strategies:
            name = strategy.get("name", "Unknown")
            model = strategy.get("model")
            
            if model is None:
                continue
                
            # 预测
            if hasattr(model, 'predict') and callable(model.predict):
                # 处理不同类型的模型预测调用
                if hasattr(model, 'market_states') and market_state is not None:
                    # 条件模型
                    y_pred = model.predict(X, market_state)
                else:
                    # 标准模型
                    y_pred = model.predict(X)
                
                # 计算性能指标
                mse = ((y - y_pred) ** 2).mean()
                rmse = np.sqrt(mse)
                mae = np.abs(y - y_pred).mean()
                direction_accuracy = ((y > 0) == (y_pred > 0)).mean()
                
                # 回测结果
                if hasattr(model, 'backtest') and callable(model.backtest):
                    backtest_results = model.backtest(X, y)
                    sharpe = backtest_results.get("sharpe_ratio", 0)
                    max_drawdown = backtest_results.get("max_drawdown", 0)
                    total_return = backtest_results.get("total_return", 0)
                else:
                    sharpe = 0
                    max_drawdown = 0
                    total_return = 0
                
                results.append({
                    "Strategy": name,
                    "RMSE": rmse,
                    "MAE": mae,
                    "Direction Accuracy": direction_accuracy,
                    "Sharpe Ratio": sharpe,
                    "Max Drawdown": max_drawdown,
                    "Total Return": total_return
                })
        
        # 创建比较DataFrame
        if results:
            comparison_df = pd.DataFrame(results)
            
            # 绘制比较图
            plt.figure(figsize=(12, 8))
            
            # 性能指标比较
            plt.subplot(2, 1, 1)
            sns.barplot(x="Strategy", y="Direction Accuracy", data=comparison_df)
            plt.title('Direction Accuracy by Strategy')
            plt.ylim(0, 1)
            plt.grid(True)
            
            # 回测结果比较
            plt.subplot(2, 1, 2)
            sns.barplot(x="Strategy", y="Sharpe Ratio", data=comparison_df)
            plt.title('Sharpe Ratio by Strategy')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.report_dir, "strategy_comparison.png"))
            plt.close()
            
            # 保存比较结果
            comparison_df.to_csv(os.path.join(self.report_dir, "strategy_comparison.csv"), index=False)
            
            return comparison_df
        
        return pd.DataFrame()
    
    def generate_market_state_analysis(self, 
                                      market_state: pd.Series,
                                      returns: pd.Series) -> Dict[str, Any]:
        """
        生成市场状态分析
        
        参数:
            market_state: 市场状态序列
            returns: 收益率序列
            
        返回:
            市场状态分析结果
        """
        results = {}
        
        # 计算每种市场状态的频率
        state_counts = market_state.value_counts()
        state_pcts = state_counts / len(market_state)
        
        results["state_counts"] = state_counts
        results["state_percentages"] = state_pcts
        
        # 计算每种市场状态下的平均收益率
        state_returns = {}
        for state in market_state.unique():
            mask = market_state == state
            if mask.sum() > 0:
                state_returns[state] = returns[mask].mean()
        
        results["state_returns"] = state_returns
        
        # 绘制市场状态分布
        plt.figure(figsize=(12, 6))
        state_pcts.plot(kind='bar')
        plt.title('Market State Distribution')
        plt.xlabel('Market State')
        plt.ylabel('Percentage')
        plt.grid(True)
        plt.savefig(os.path.join(self.report_dir, "market_state_distribution.png"))
        plt.close()
        
        # 绘制每种市场状态下的平均收益率
        plt.figure(figsize=(12, 6))
        pd.Series(state_returns).plot(kind='bar')
        plt.title('Average Returns by Market State')
        plt.xlabel('Market State')
        plt.ylabel('Average Return')
        plt.grid(True)
        plt.savefig(os.path.join(self.report_dir, "returns_by_market_state.png"))
        plt.close()
        
        return results