#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试XGBoostSignalCombiner模型
"""

import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import os
import sys
import tempfile
from datetime import datetime, timedelta

# 导入被测试模块
from strategy_optimizer.models.xgboost_model import XGBoostSignalCombiner
from strategy_optimizer.utils import DataGenerator


class TestXGBoostSignalCombiner(unittest.TestCase):
    """XGBoostSignalCombiner模型的测试类"""

    def setUp(self):
        """测试前的设置"""
        # 设置随机种子以确保结果可复现
        np.random.seed(42)
        
        # 生成测试数据
        self.generator = DataGenerator(seed=42)
        self.signals, self.returns = self.generator.generate_synthetic_data(
            n_samples=300,
            n_signals=10,
            signal_strength={0: 0.7, 1: 0.5, 2: 0.3},
            noise_level=0.2,
            start_date="2022-01-01"
        )
        
        # 划分训练集和测试集
        self.train_size = 200
        self.X_train = self.signals.iloc[:self.train_size]
        self.y_train = self.returns.iloc[:self.train_size]
        self.X_test = self.signals.iloc[self.train_size:]
        self.y_test = self.returns.iloc[self.train_size:]
        
        # 初始化模型
        self.model = XGBoostSignalCombiner(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )

    def test_initialization(self):
        """测试模型初始化"""
        # 验证默认参数设置
        default_model = XGBoostSignalCombiner()
        self.assertEqual(default_model.n_estimators, 100)
        self.assertEqual(default_model.learning_rate, 0.1)
        self.assertEqual(default_model.max_depth, 3)
        
        # 验证自定义参数设置
        custom_model = XGBoostSignalCombiner(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2
        )
        self.assertEqual(custom_model.n_estimators, 200)
        self.assertEqual(custom_model.learning_rate, 0.05)
        self.assertEqual(custom_model.max_depth, 4)
        self.assertEqual(custom_model.min_child_weight, 2)

    def test_fit_and_predict(self):
        """测试模型拟合和预测功能"""
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 验证模型已训练
        self.assertTrue(hasattr(self.model, 'model'))
        
        # 测试预测功能
        predictions = self.model.predict(self.X_test)
        
        # 验证预测结果
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # 验证预测结果的格式和范围
        # (XGBoost回归预测的值通常不限于特定范围)
        self.assertTrue(np.isfinite(predictions).all())

    def test_early_stopping(self):
        """测试早停功能"""
        # 使用早停训练模型
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=5,
            verbose=False
        )
        
        # 验证早停生效
        self.assertLessEqual(self.model.model.best_iteration, 100)
        
        # 验证最佳迭代轮数被保存
        self.assertTrue(hasattr(self.model.model, 'best_iteration'))

    def test_performance_metrics(self):
        """测试性能指标计算"""
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 计算训练集和测试集的性能指标
        train_perf = self.model.get_performance_metrics(self.X_train, self.y_train)
        test_perf = self.model.get_performance_metrics(self.X_test, self.y_test)
        
        # 验证性能指标对象
        for perf in [train_perf, test_perf]:
            self.assertIsInstance(perf, dict)
            
            # 验证包含关键指标
            for metric in ['mse', 'r2', 'sign_accuracy', 'sharpe_ratio', 'annual_return', 'max_drawdown']:
                self.assertIn(metric, perf)
                self.assertIsInstance(perf[metric], float)
        
        # 验证指标计算的合理性
        self.assertGreaterEqual(train_perf['sign_accuracy'], 0.0)
        self.assertLessEqual(train_perf['sign_accuracy'], 1.0)
        self.assertGreaterEqual(test_perf['sign_accuracy'], 0.0)
        self.assertLessEqual(test_perf['sign_accuracy'], 1.0)

    def test_feature_importance(self):
        """测试特征重要性功能"""
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 获取特征重要性
        importance = self.model.get_feature_importance(plot=False)
        
        # 验证特征重要性DataFrame
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertEqual(len(importance), len(self.X_train.columns))
        
        # 验证特征重要性排序
        self.assertTrue(importance['importance'].is_monotonic_decreasing)
        
        # 检查是否包含所有特征
        for feature in self.X_train.columns:
            self.assertIn(feature, importance['feature'].values)

    def test_save_and_load(self):
        """测试模型保存和加载功能"""
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 获取原始预测结果作为基准
        original_predictions = self.model.predict(self.X_test)
        
        # 创建临时文件保存模型
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            model_path = tmp.name
            
            # 保存模型
            self.model.save_model(model_path)
            
            # 确认文件存在且大小大于0
            self.assertTrue(os.path.exists(model_path))
            self.assertGreater(os.path.getsize(model_path), 0)
            
            # 创建新模型实例
            new_model = XGBoostSignalCombiner()
            
            # 加载模型
            new_model.load_model(model_path)
            
            # 用加载的模型进行预测
            loaded_predictions = new_model.predict(self.X_test)
            
            # 验证加载后的模型预测与原始预测一致
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

    def test_plot_performance(self):
        """测试性能可视化功能"""
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 测试不同类型的性能图
        for plot_type in ['cumulative_returns', 'monthly_returns', 'drawdown']:
            # 在没有保存的情况下测试绘图功能
            fig = self.model.plot_performance(
                self.X_test, self.y_test, 
                plot_type=plot_type,
                show=False,
                return_fig=True
            )
            self.assertIsNotNone(fig)
            
            # 测试保存图表
            with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                save_path = tmp.name
                self.model.plot_performance(
                    self.X_test, self.y_test,
                    plot_type=plot_type,
                    show=False,
                    save_path=save_path
                )
                self.assertTrue(os.path.exists(save_path))
                self.assertGreater(os.path.getsize(save_path), 0)

    def test_cross_validation(self):
        """测试交叉验证功能"""
        # 执行时间序列交叉验证
        cv_results = self.model.time_series_cv(
            self.signals, self.returns,
            n_splits=3,
            test_size=0.2
        )
        
        # 验证交叉验证结果
        self.assertIsInstance(cv_results, dict)
        
        # 检查是否包含所有必要的结果项
        self.assertIn('all_metrics', cv_results)
        self.assertIn('mean_metrics', cv_results)
        self.assertIn('std_metrics', cv_results)
        
        # 验证指标计算
        for metric in ['sign_accuracy', 'sharpe_ratio', 'annual_return', 'max_drawdown']:
            self.assertIn(metric, cv_results['mean_metrics'])
            self.assertIn(metric, cv_results['std_metrics'])
        
        # 验证交叉验证结果格式
        self.assertEqual(len(cv_results['all_metrics']), 3)  # 3个分割
        
        # 测试含有额外参数的交叉验证
        cv_results_with_params = self.model.time_series_cv(
            self.signals, self.returns,
            n_splits=3,
            test_size=0.2,
            xgb_params={'max_depth': 2, 'learning_rate': 0.05}
        )
        
        # 验证额外参数已经应用
        self.assertIsInstance(cv_results_with_params, dict)
        self.assertEqual(len(cv_results_with_params['all_metrics']), 3)
        
        # 验证包含train和test结果
        for fold_result in cv_results['all_metrics']:
            self.assertIn('train_metrics', fold_result)
            self.assertIn('test_metrics', fold_result)

    def test_backtesting(self):
        """测试回测功能"""
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 执行回测
        backtest_result = self.model.backtest(
            self.X_test, self.y_test,
            transaction_cost=0.001
        )
        
        # 验证回测结果
        self.assertIsInstance(backtest_result, dict)
        
        # 检查是否包含所有必要的回测结果
        backtest_keys = [
            'returns', 'cumulative_returns', 'positions',
            'sharpe_ratio', 'annual_return', 'max_drawdown',
            'win_rate', 'profit_factor', 'recovery_factor'
        ]
        for key in backtest_keys:
            self.assertIn(key, backtest_result)
        
        # 验证回测数据的长度一致
        self.assertEqual(len(backtest_result['returns']), len(self.X_test))
        self.assertEqual(len(backtest_result['positions']), len(self.X_test))
        
        # 测试不同交易成本下的回测
        backtest_high_cost = self.model.backtest(
            self.X_test, self.y_test,
            transaction_cost=0.01  # 高交易成本
        )
        
        # 验证高交易成本下回测结果应该更差
        self.assertLessEqual(
            backtest_high_cost['annual_return'],
            backtest_result['annual_return']
        )


if __name__ == '__main__':
    unittest.main() 