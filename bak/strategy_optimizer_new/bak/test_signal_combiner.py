import unittest
import numpy as np
import pandas as pd
import torch
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from strategy_optimizer.models.signal_combiner import (
    MarketRegime,
    CombinerConfig,
    MarketStateClassifier,
    AdaptiveWeightModel,
    SignalCombiner,
    SignalCombinerModel
)

class TestMarketStateClassifier(unittest.TestCase):
    """测试市场状态分类器"""
    
    def setUp(self):
        self.config = CombinerConfig(
            hidden_dim=64,
            n_layers=2,
            dropout=0.2,
            market_feature_dim=15
        )
        self.classifier = MarketStateClassifier(self.config)
        self.batch_size = 4
        self.seq_len = 20
        self.feature_dim = 15
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.classifier.config.hidden_dim, 64)
        self.assertEqual(self.classifier.config.n_market_states, 5)
        
    def test_forward(self):
        """测试前向传播"""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        output = self.classifier(x)
        
        # 检查输出维度
        self.assertEqual(output.shape, (self.batch_size, self.config.n_market_states))
        
        # 检查输出是概率分布
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.ones(self.batch_size)))
        self.assertTrue((output >= 0).all().item())
        self.assertTrue((output <= 1).all().item())

class TestAdaptiveWeightModel(unittest.TestCase):
    """测试自适应权重模型"""
    
    def setUp(self):
        self.n_strategies = 6
        self.config = CombinerConfig(
            hidden_dim=64,
            n_layers=2,
            dropout=0.2,
            market_feature_dim=15,
            use_market_state=True,
            time_varying_weights=True
        )
        self.model = AdaptiveWeightModel(self.n_strategies, self.config)
        self.batch_size = 4
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.model.n_strategies, 6)
        self.assertTrue(hasattr(self.model, 'base_weights'))
        self.assertTrue(hasattr(self.model, 'state_weight_matrix'))
        self.assertTrue(hasattr(self.model, 'time_varying_layer'))
        
    def test_forward_with_market_state(self):
        """测试带市场状态的前向传播"""
        market_states = torch.softmax(torch.randn(self.batch_size, self.config.n_market_states), dim=1)
        hidden_state = torch.randn(self.batch_size, self.config.hidden_dim)
        
        weights = self.model(market_states, hidden_state)
        
        # 检查输出维度
        self.assertEqual(weights.shape, (self.batch_size, self.n_strategies))
        
        # 检查权重是概率分布
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(self.batch_size), atol=1e-6))
        self.assertTrue((weights >= 0).all().item())
        self.assertTrue((weights <= 1).all().item())
        
    def test_forward_without_market_state(self):
        """测试不带市场状态的前向传播"""
        # 创建不使用市场状态的模型
        config = CombinerConfig(
            hidden_dim=64,
            use_market_state=False,
            time_varying_weights=True
        )
        model = AdaptiveWeightModel(self.n_strategies, config)
        
        hidden_state = torch.randn(self.batch_size, config.hidden_dim)
        weights = model(None, hidden_state)
        
        # 检查输出维度
        self.assertEqual(weights.shape, (self.batch_size, self.n_strategies))
        
        # 检查权重是概率分布
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(self.batch_size), atol=1e-6))

class TestSignalCombiner(unittest.TestCase):
    """测试信号组合器"""
    
    def setUp(self):
        self.n_strategies = 6
        self.input_dim = 15
        self.config = CombinerConfig(
            hidden_dim=64,
            n_layers=2,
            dropout=0.2,
            market_feature_dim=15,
            use_market_state=True,
            time_varying_weights=True
        )
        self.model = SignalCombiner(self.n_strategies, self.input_dim, self.config)
        self.batch_size = 4
        self.seq_len = 20
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.model.n_strategies, 6)
        self.assertTrue(hasattr(self.model, 'feature_extractor'))
        self.assertTrue(hasattr(self.model, 'lstm'))
        self.assertTrue(hasattr(self.model, 'market_classifier'))
        self.assertTrue(hasattr(self.model, 'weight_model'))
        
    def test_forward(self):
        """测试前向传播"""
        features = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        strategy_signals = torch.randn(self.batch_size, self.n_strategies)
        
        combined_signal, weights, market_states = self.model(features, strategy_signals)
        
        # 检查输出维度
        self.assertEqual(combined_signal.shape, (self.batch_size, 1))
        self.assertEqual(weights.shape, (self.batch_size, self.n_strategies))
        self.assertEqual(market_states.shape, (self.batch_size, self.config.n_market_states))
        
        # 检查权重是概率分布
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(self.batch_size), atol=1e-6))
        self.assertTrue((weights >= 0).all().item())
        
        # 检查市场状态是概率分布
        self.assertTrue(torch.allclose(market_states.sum(dim=1), torch.ones(self.batch_size), atol=1e-6))
        self.assertTrue((market_states >= 0).all().item())

class TestSignalCombinerModel(unittest.TestCase):
    """测试信号组合模型"""
    
    def setUp(self):
        self.n_strategies = 6
        self.input_dim = 15
        self.config = CombinerConfig(
            hidden_dim=64,
            n_layers=2,
            dropout=0.2,
            sequence_length=20,
            batch_size=4,
            epochs=2,
            market_feature_dim=15,
            use_market_state=True,
            time_varying_weights=True
        )
        self.model = SignalCombinerModel(
            n_strategies=self.n_strategies,
            input_dim=self.input_dim,
            config=self.config
        )
        
        # 创建测试数据
        self.n_samples = 100
        self.seq_len = 20
        self.features = np.random.randn(self.n_samples, self.seq_len, self.input_dim)
        self.signals = np.random.randn(self.n_samples, self.n_strategies)
        self.targets = np.random.randn(self.n_samples)
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.model.model.n_strategies, 6)
        self.assertTrue(hasattr(self.model, 'optimizer'))
        self.assertTrue(hasattr(self.model, 'criterion'))
        self.assertTrue(hasattr(self.model, 'feature_scaler'))
        self.assertTrue(hasattr(self.model, 'target_scaler'))
        
    def test_prepare_data(self):
        """测试数据准备"""
        train_loader, val_loader, test_loader = self.model.prepare_data(
            self.features, self.signals, self.targets
        )
        
        # 检查数据加载器
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)
        
        # 检查批次大小
        for batch in train_loader:
            self.assertEqual(len(batch), 3)  # features, signals, targets
            features, signals, targets = batch
            self.assertEqual(features.shape[0], self.config.batch_size)
            break
            
    def test_train_and_evaluate(self):
        """测试训练和评估"""
        # 准备数据
        train_loader, val_loader, test_loader = self.model.prepare_data(
            self.features, self.signals, self.targets
        )
        
        # 训练模型
        history = self.model.train(train_loader, val_loader, verbose=False)
        
        # 检查历史记录
        self.assertTrue('train_loss' in history)
        self.assertTrue('val_loss' in history)
        self.assertEqual(len(history['train_loss']), 2)  # epochs=2
        
        # 评估模型
        test_loss, _ = self.model.evaluate(test_loader)
        self.assertIsInstance(test_loss, float)
        
    def test_predict(self):
        """测试预测"""
        # 准备数据
        train_loader, val_loader, test_loader = self.model.prepare_data(
            self.features, self.signals, self.targets
        )
        
        # 预测
        combined_signals, weights, market_states = self.model.predict(
            self.features, self.signals
        )
        
        # 检查输出维度
        self.assertEqual(combined_signals.shape, (self.n_samples, 1))
        self.assertEqual(weights.shape, (self.n_samples, self.n_strategies))
        self.assertEqual(market_states.shape, (self.n_samples, self.config.n_market_states))
        
        # 检查权重是概率分布
        weight_sums = np.sum(weights, axis=1)
        self.assertTrue(np.allclose(weight_sums, np.ones(self.n_samples), atol=1e-6))
        self.assertTrue(np.all(weights >= 0))
        
    def test_save_and_load(self):
        """测试保存和加载模型"""
        # 创建临时目录
        import tempfile
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "model.pt")
        
        # 保存模型
        self.model.save(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # 创建新模型并加载
        new_model = SignalCombinerModel(
            n_strategies=self.n_strategies,
            input_dim=self.input_dim,
            config=self.config
        )
        new_model.load(model_path)
        
        # 验证加载后的模型
        # 对比两个模型的参数是否相同
        for p1, p2 in zip(self.model.model.parameters(), new_model.model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
            
        # 删除临时文件
        os.remove(model_path)
        os.rmdir(temp_dir)
        
    def test_plot_weights(self):
        """测试绘制权重分布"""
        # 创建随机权重
        weights = np.random.rand(self.n_samples, self.n_strategies)
        weights = weights / weights.sum(axis=1, keepdims=True)  # 归一化
        
        # 创建策略名称
        strategy_names = [f"Strategy_{i}" for i in range(self.n_strategies)]
        
        # 绘制权重分布
        fig = self.model.plot_weights(weights, strategy_names)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
    def test_plot_market_states(self):
        """测试绘制市场状态分布"""
        # 创建随机市场状态
        market_states = np.random.rand(self.n_samples, 5)  # 5个市场状态
        market_states = market_states / market_states.sum(axis=1, keepdims=True)  # 归一化
        
        # 绘制市场状态分布
        fig = self.model.plot_market_states(market_states)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

class TestEndToEndWorkflow(unittest.TestCase):
    """测试端到端工作流程"""
    
    def test_end_to_end(self):
        """测试完整工作流程"""
        # 此测试仅演示工作流程，不验证结果
        try:
            # 创建简单的模拟数据
            n_samples = 100
            n_strategies = 3
            input_dim = 10
            seq_len = 20
            
            # 特征数据
            features = np.random.randn(n_samples, seq_len, input_dim)
            
            # 策略信号 - 每个策略的信号可以假设在不同市场状态表现不同
            signals = np.zeros((n_samples, n_strategies))
            # 策略1在上升趋势表现好
            signals[:, 0] = np.sin(np.linspace(0, 4*np.pi, n_samples))
            # 策略2在下跌趋势表现好
            signals[:, 1] = -np.sin(np.linspace(0, 4*np.pi, n_samples))
            # 策略3在震荡期表现好
            signals[:, 2] = np.cos(np.linspace(0, 8*np.pi, n_samples))
            
            # 目标变量 - 未来收益
            # 假设是策略的加权组合加上噪声
            weights = np.array([0.5, 0.3, 0.2])
            targets = signals @ weights + 0.1 * np.random.randn(n_samples)
            
            # 创建模型配置
            config = CombinerConfig(
                hidden_dim=32,
                n_layers=1,
                dropout=0.1,
                sequence_length=seq_len,
                batch_size=8,
                epochs=5,
                market_feature_dim=input_dim,
                use_market_state=True,
                time_varying_weights=True
            )
            
            # 创建模型
            model = SignalCombinerModel(
                n_strategies=n_strategies,
                input_dim=input_dim,
                config=config
            )
            
            # 准备数据
            train_loader, val_loader, test_loader = model.prepare_data(
                features, signals, targets
            )
            
            # 训练模型
            history = model.train(train_loader, val_loader, verbose=False)
            
            # 评估模型
            test_loss, _ = model.evaluate(test_loader)
            
            # 预测
            predicted_signals, weights, market_states = model.predict(
                features, signals
            )
            
            # 简单验证
            self.assertEqual(predicted_signals.shape, (n_samples, 1))
            self.assertEqual(weights.shape, (n_samples, n_strategies))
            
            # 检查训练历史是否有改善
            self.assertLess(history['train_loss'][-1], history['train_loss'][0])
            
        except Exception as e:
            self.fail(f"端到端测试失败: {str(e)}")

if __name__ == "__main__":
    unittest.main() 