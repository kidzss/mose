import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from strategy_optimizer.data_processors import SignalExtractor

class TestSignalExtractor(unittest.TestCase):
    """
    信号提取器测试类
    """
    
    def setUp(self):
        """
        测试前准备数据
        """
        # 创建测试数据
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # 生成价格数据
        np.random.seed(42)  # 设置随机种子，确保结果可复现
        prices = 100 * (1 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        
        # 创建几个特征
        # 特征1：模拟动量指标
        momentum = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
        
        # 特征2：模拟均值回归指标
        mean_reversion = -momentum + np.random.normal(0, 0.5, len(dates))
        
        # 特征3：模拟趋势指标
        trend = pd.Series(np.cumsum(np.random.normal(0, 0.1, len(dates))), index=dates)
        
        # 特征4：模拟周期性指标
        seasonal = pd.Series(
            [np.sin(2 * np.pi * i / 20) for i in range(len(dates))], 
            index=dates
        )
        
        # 特征5：模拟噪声特征（无预测能力）
        noise = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
        
        # 将所有特征放入DataFrame
        self.feature_data = pd.DataFrame({
            'price': prices,
            'momentum': momentum,
            'mean_reversion': mean_reversion,
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise
        })
        
        # 计算未来收益率作为目标变量
        self.feature_data['future_return'] = self.feature_data['price'].pct_change(5).shift(-5)
        
        # 丢弃缺失值
        self.feature_data = self.feature_data.dropna()
        
        # 创建信号提取器实例
        self.signal_extractor = SignalExtractor(
            standardize=True,
            smooth_window=3,
            threshold_method="percentile",
            threshold_params={"upper": 75, "lower": 25}
        )
        
    def test_extract_from_feature(self):
        """
        测试从单个特征提取信号
        """
        # 测试从动量特征提取信号
        signal, _ = self.signal_extractor.extract_from_feature(
            self.feature_data, 'momentum', return_processed=False
        )
        
        # 验证信号类型和长度
        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), len(self.feature_data))
        
        # 验证信号值范围（应该是-1, 0, 1）
        self.assertTrue(signal.isin([-1, 0, 1]).all())
        
        # 验证返回处理后的特征
        signal, processed = self.signal_extractor.extract_from_feature(
            self.feature_data, 'momentum', return_processed=True
        )
        
        self.assertIsNotNone(processed)
        self.assertIsInstance(processed, pd.Series)
        self.assertEqual(len(processed), len(self.feature_data))
        
    def test_extract_from_multiple_features(self):
        """
        测试从多个特征提取信号
        """
        # 测试平均方法
        feature_names = ['momentum', 'mean_reversion', 'trend']
        signal = self.signal_extractor.extract_from_multiple_features(
            self.feature_data, feature_names, method="average"
        )
        
        # 验证信号
        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), len(self.feature_data))
        
        # 测试加权方法
        weights = [0.5, 0.3, 0.2]
        signal = self.signal_extractor.extract_from_multiple_features(
            self.feature_data, feature_names, method="weighted", weights=weights
        )
        
        # 验证信号
        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), len(self.feature_data))
        
        # 测试投票方法
        signal = self.signal_extractor.extract_from_multiple_features(
            self.feature_data, feature_names, method="vote"
        )
        
        # 验证信号
        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), len(self.feature_data))
        self.assertTrue(signal.isin([-1, 0, 1]).all())
        
        # 测试top_n方法
        signal = self.signal_extractor.extract_from_multiple_features(
            self.feature_data, feature_names, method="top_n", weights=[0.5, 0.3, 0.2]
        )
        
        # 验证信号
        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), len(self.feature_data))
        
    def test_optimize_signal_weights(self):
        """
        测试信号权重优化
        """
        feature_names = ['momentum', 'mean_reversion', 'trend']
        target = self.feature_data['future_return']
        
        # 测试相关性方法
        weights = self.signal_extractor.optimize_signal_weights(
            self.feature_data, feature_names, target, method="correlation"
        )
        
        # 验证权重
        self.assertEqual(len(weights), len(feature_names))
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        
        # 所有权重应该为非负数
        for w in weights:
            self.assertGreaterEqual(w, 0)
        
    def test_analyze_signal_performance(self):
        """
        测试信号性能分析
        """
        # 创建一个简单的信号序列
        signal = pd.Series(1, index=self.feature_data.index)
        signal.iloc[::3] = -1  # 每3个样本设置为-1
        signal.iloc[::2] = 0   # 每2个样本设置为0
        
        # 分析信号性能
        performance = self.signal_extractor.analyze_signal_performance(
            signal, 
            self.feature_data['price'],
            position_mode="discrete",
            transaction_cost=0.001,
            plot=False
        )
        
        # 验证返回的指标
        self.assertIn('annual_return', performance)
        self.assertIn('max_drawdown', performance)
        self.assertIn('sharpe_ratio', performance)
        self.assertIn('win_rate', performance)
        self.assertIn('turnover', performance)
        
        # 测试连续持仓模式
        performance_continuous = self.signal_extractor.analyze_signal_performance(
            signal, 
            self.feature_data['price'],
            position_mode="continuous",
            transaction_cost=0.001,
            plot=False
        )
        
        # 验证返回的指标
        self.assertIn('annual_return', performance_continuous)
        
    def test_generate_signal_by_threshold(self):
        """
        测试根据阈值生成信号
        """
        # 创建一个简单的特征序列
        feature = pd.Series(np.random.normal(0, 1, len(self.feature_data)), 
                          index=self.feature_data.index)
        
        # 分位数阈值方法
        self.signal_extractor.threshold_method = "percentile"
        self.signal_extractor.threshold_params = {"upper": 75, "lower": 25}
        
        signal_pct = self.signal_extractor._generate_signal_by_threshold(feature)
        
        # 验证信号
        self.assertIsInstance(signal_pct, pd.Series)
        self.assertTrue(signal_pct.isin([-1, 0, 1]).all())
        
        # 标准差阈值方法
        self.signal_extractor.threshold_method = "std"
        self.signal_extractor.threshold_params = {"n_std": 1.0}
        
        signal_std = self.signal_extractor._generate_signal_by_threshold(feature)
        
        # 验证信号
        self.assertIsInstance(signal_std, pd.Series)
        self.assertTrue(signal_std.isin([-1, 0, 1]).all())
        
        # 固定阈值方法
        self.signal_extractor.threshold_method = "fixed"
        self.signal_extractor.threshold_params = {"upper": 0.5, "lower": -0.5}
        
        signal_fixed = self.signal_extractor._generate_signal_by_threshold(feature)
        
        # 验证信号
        self.assertIsInstance(signal_fixed, pd.Series)
        self.assertTrue(signal_fixed.isin([-1, 0, 1]).all())
        
if __name__ == '__main__':
    unittest.main() 