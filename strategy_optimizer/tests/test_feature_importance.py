import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from strategy_optimizer.data_processors import FeatureImportanceAnalyzer

class TestFeatureImportanceAnalyzer(unittest.TestCase):
    """
    特征重要性分析器测试类
    """
    
    def setUp(self):
        """
        测试前准备数据
        """
        # 生成回归数据集
        np.random.seed(42)
        X, y = make_regression(
            n_samples=200, 
            n_features=10, 
            n_informative=5,
            random_state=42
        )
        
        # 特征名称
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 创建DataFrame
        self.X_df = pd.DataFrame(X, columns=feature_names)
        self.y_series = pd.Series(y)
        
        # 创建特征重要性分析器
        self.analyzer = FeatureImportanceAnalyzer()
        
    def test_fit_regression(self):
        """
        测试回归任务的拟合
        """
        # 拟合模型
        result = self.analyzer.fit(
            self.X_df, 
            self.y_series,
            task='regression',
            model_type='xgboost',
            cv=3
        )
        
        # 验证结果格式
        self.assertIn('cross_validation_results', result)
        self.assertIn('feature_importances', result)
        
        # 验证特征重要性
        feature_imp = result['feature_importances']
        self.assertIsInstance(feature_imp, pd.DataFrame)
        self.assertEqual(len(feature_imp), self.X_df.shape[1])
        self.assertIn('importance_mean', feature_imp.columns)
        self.assertIn('importance_std', feature_imp.columns)
        
        # 验证cross_validation_results
        cv_results = result['cross_validation_results']
        self.assertEqual(len(cv_results), 3)  # 3折交叉验证
        
        # 验证保存的重要性
        self.assertIn('raw', self.analyzer.importances)
        self.assertIn('summary', self.analyzer.importances)
        
    def test_fit_classification(self):
        """
        测试分类任务的拟合
        """
        # 将目标变量二值化用于分类
        y_binary = (self.y_series > self.y_series.median()).astype(int)
        
        # 拟合模型
        result = self.analyzer.fit(
            self.X_df, 
            y_binary,
            task='classification',
            model_type='lightgbm',
            cv=2
        )
        
        # 验证结果格式
        self.assertIn('cross_validation_results', result)
        self.assertIn('feature_importances', result)
        
        # 验证特征重要性
        feature_imp = result['feature_importances']
        self.assertIsInstance(feature_imp, pd.DataFrame)
        self.assertEqual(len(feature_imp), self.X_df.shape[1])
        
    def test_calculate_feature_correlations(self):
        """
        测试特征相关性计算
        """
        # 计算特征相关性
        corr_df = self.analyzer.calculate_feature_correlations(self.X_df, self.y_series)
        
        # 验证结果
        self.assertIsInstance(corr_df, pd.DataFrame)
        self.assertEqual(len(corr_df), self.X_df.shape[1])
        self.assertIn('correlation', corr_df.columns)
        self.assertIn('p_value', corr_df.columns)
        
    def test_plot_feature_importance(self):
        """
        测试绘制特征重要性图表
        """
        # 首先拟合模型
        self.analyzer.fit(self.X_df, self.y_series, cv=2)
        
        # 绘制图表
        fig = self.analyzer.plot_feature_importance(top_n=5)
        
        # 验证返回的图表
        self.assertIsInstance(fig, plt.Figure)
        
        # 关闭图表以避免内存泄漏
        plt.close(fig)
        
    def test_plot_correlation_importance(self):
        """
        测试绘制相关性重要性图表
        """
        # 计算相关性
        corr_df = self.analyzer.calculate_feature_correlations(self.X_df, self.y_series)
        
        # 绘制图表
        fig = self.analyzer.plot_correlation_importance(corr_df, top_n=5)
        
        # 验证返回的图表
        self.assertIsInstance(fig, plt.Figure)
        
        # 关闭图表以避免内存泄漏
        plt.close(fig)
        
    def test_get_optimal_feature_subset(self):
        """
        测试获取最优特征子集
        """
        # 不调用fit，让函数内部自己调用
        result = self.analyzer.get_optimal_feature_subset(
            self.X_df, 
            self.y_series,
            min_features=2,
            max_features=5,
            step=1
        )
        
        # 验证结果
        self.assertIn('all_results', result)
        self.assertIn('best_subset', result)
        
        # 验证all_results
        all_results = result['all_results']
        self.assertIsInstance(all_results, pd.DataFrame)
        self.assertGreaterEqual(len(all_results), 4)  # 2到5特征，步长1
        
        # 验证best_subset
        best_subset = result['best_subset']
        self.assertIsInstance(best_subset, dict)
        self.assertIn('n_features', best_subset)
        self.assertIn('features', best_subset)
        
    def test_plot_feature_selection_curve(self):
        """
        测试绘制特征选择曲线
        """
        # 获取特征子集结果
        result = self.analyzer.get_optimal_feature_subset(
            self.X_df, 
            self.y_series,
            min_features=2,
            max_features=5,
            step=1
        )
        
        # 绘制曲线
        fig = self.analyzer.plot_feature_selection_curve(
            result['all_results'],
            task='regression'
        )
        
        # 验证返回的图表
        self.assertIsInstance(fig, plt.Figure)
        
        # 关闭图表以避免内存泄漏
        plt.close(fig)
        
if __name__ == '__main__':
    unittest.main() 