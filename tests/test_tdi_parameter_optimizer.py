import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy.custom_tdi_strategy import CustomTDIStrategy
from strategy_optimizer.parameter_optimizer import TDIParameterOptimizer

def generate_sample_data():
    """生成样本数据"""
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    n = len(dates)
    
    # 生成随机价格数据
    np.random.seed(42)
    prices = 100 * (1 + np.random.normal(0, 0.02, n).cumsum())
    volumes = np.random.lognormal(10, 1, n)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, n)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n)),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return df

class TestTDIParameterOptimizer(unittest.TestCase):
    """测试TDI参数优化器"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = CustomTDIStrategy()
        self.optimizer = TDIParameterOptimizer(self.strategy, self.data)
        
    def test_prepare_features(self):
        """测试特征准备"""
        features, labels = self.optimizer.prepare_features()
        
        # 检查特征维度
        self.assertEqual(len(features), len(self.data) - 1)
        self.assertEqual(len(labels), len(self.data) - 1)
        
        # 检查特征列
        expected_columns = [
            'rsi', 'adx', 'macd', 'macd_signal', 'volume_ratio',
            'trend_strength', 'momentum_strength', 'volume_strength',
            'volatility', 'atr'
        ]
        for col in expected_columns:
            self.assertIn(col, features.columns)
            
    def test_train_xgboost(self):
        """测试XGBoost模型训练"""
        model = self.optimizer.train_xgboost()
        self.assertIsNotNone(model)
        
        # 测试模型预测
        features, _ = self.optimizer.prepare_features()
        predictions = model.predict(features)
        self.assertEqual(len(predictions), len(features))
        
    def test_trading_env(self):
        """测试交易环境"""
        env = self.optimizer.create_trading_env()
        
        # 测试环境重置
        obs = env.reset()
        self.assertEqual(len(obs), 10)
        
        # 测试环境步进
        obs, reward, done, info = env.step(0)
        self.assertEqual(len(obs), 10)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        
    def test_parameter_optimization(self):
        """测试参数优化"""
        # 使用较小的搜索次数进行测试
        self.optimizer.param_space = {
            'rsi_period': (5, 30),
            'adx_threshold': (15, 35),
            'volume_threshold': (1.0, 3.0)
        }
        
        best_params = self.optimizer.optimize_parameters()
        
        # 检查参数范围
        for param, value in best_params.items():
            low, high = self.optimizer.param_space[param]
            self.assertGreaterEqual(value, low)
            self.assertLessEqual(value, high)
            
    def test_performance_metrics(self):
        """测试性能指标计算"""
        returns = np.random.normal(0.001, 0.02, 100)
        
        # 测试夏普比率
        sharpe = self.optimizer._calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)
        
        # 测试索提诺比率
        sortino = self.optimizer._calculate_sortino_ratio(returns)
        self.assertIsInstance(sortino, float)
        
        # 测试总收益率
        total_return = self.optimizer._calculate_total_return(returns)
        self.assertIsInstance(total_return, float)
        
if __name__ == '__main__':
    unittest.main() 