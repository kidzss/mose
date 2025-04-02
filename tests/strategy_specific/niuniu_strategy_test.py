import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import NiuNiuStrategy
from tests.strategy_test import generate_sample_data

class TestNiuNiuStrategyDetailed(unittest.TestCase):
    """NiuNiu策略详细测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.data = generate_sample_data(days=200)
        self.strategy = NiuNiuStrategy()
        
        # 确保输出目录存在
        os.makedirs('tests/output', exist_ok=True)
    
    def test_indicator_calculation(self):
        """测试指标计算"""
        # 计算指标
        result = self.strategy.calculate_indicators(self.data)
        
        # 验证指标计算结果
        self.assertIn('MID', result.columns)
        self.assertIn('Bull_Line', result.columns)
        self.assertIn('Trade_Line', result.columns)
        
        # 验证计算值不含NaN（排除预热期）
        weights_window = self.strategy.parameters['weights_window']
        self.assertFalse(result['Bull_Line'].iloc[weights_window:].isna().any())
        self.assertFalse(result['Trade_Line'].iloc[weights_window+1:].isna().any())
        
        # 验证计算逻辑
        mid_sample = (3 * result['close'] + result['low'] + result['open'] + result['high']) / 6
        mid_sample.name = 'MID'  # 设置name属性以匹配strategy中的对应设置
        pd.testing.assert_series_equal(result['MID'], mid_sample)
        
        # 绘制指标可视化
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(result.index, result['close'], label='Close')
        plt.title('价格走势')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(result.index, result['Bull_Line'], label='牛线', color='red')
        plt.plot(result.index, result['Trade_Line'], label='交易线', color='blue')
        plt.title('NiuNiu策略指标')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('tests/output/niuniu_indicators.png')
        plt.close()
    
    def test_signal_generation(self):
        """测试信号生成"""
        # 生成信号
        result = self.strategy.generate_signals(self.data)
        
        # 验证信号列存在
        self.assertIn('signal', result.columns)
        
        # 计算信号统计
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"NiuNiu策略生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 绘制信号可视化
        plt.figure(figsize=(12, 8))
        plt.plot(result.index, result['close'], label='Close', alpha=0.5)
        plt.plot(result.index, result['Bull_Line'], label='牛线', color='blue', alpha=0.7)
        plt.plot(result.index, result['Trade_Line'], label='交易线', color='orange', alpha=0.7)
        
        # 标记买入和卖出点
        buy_points = result[result['signal'] == 1]
        sell_points = result[result['signal'] == -1]
        
        plt.scatter(buy_points.index, buy_points['close'], marker='^', color='green', s=100, label='买入信号')
        plt.scatter(sell_points.index, sell_points['close'], marker='v', color='red', s=100, label='卖出信号')
        
        plt.title('NiuNiu策略信号')
        plt.legend()
        plt.savefig('tests/output/niuniu_signals.png')
        plt.close()
    
    def test_market_regime(self):
        """测试市场环境判断"""
        # 计算指标
        result = self.strategy.calculate_indicators(self.data)
        
        # 获取市场环境
        regime = self.strategy.get_market_regime(result)
        
        # 验证市场环境返回值
        self.assertIn(regime, ["bullish", "bearish", "sideways", "volatile", "normal", "unknown"])
        
        print(f"当前市场环境: {regime}")
    
    def test_position_sizing(self):
        """测试仓位大小计算"""
        # 计算指标
        result = self.strategy.calculate_indicators(self.data)
        
        # 测试不同信号的仓位大小
        buy_size = self.strategy.get_position_size(result, 1)
        sell_size = self.strategy.get_position_size(result, -1)
        no_signal_size = self.strategy.get_position_size(result, 0)
        
        # 验证仓位范围
        self.assertGreaterEqual(buy_size, 0)
        self.assertLessEqual(buy_size, 1)
        self.assertGreaterEqual(sell_size, 0)
        self.assertLessEqual(sell_size, 1)
        self.assertEqual(no_signal_size, 0)
        
        print(f"买入信号仓位大小: {buy_size}")
        print(f"卖出信号仓位大小: {sell_size}")
    
    def test_stop_loss_and_take_profit(self):
        """测试止损和止盈计算"""
        # 测试样本数据
        entry_price = 100.0
        
        # 计算多头止损止盈
        long_stop_loss = self.strategy.get_stop_loss(self.data, entry_price, 1)
        long_take_profit = self.strategy.get_take_profit(self.data, entry_price, 1)
        
        # 计算空头止损止盈
        short_stop_loss = self.strategy.get_stop_loss(self.data, entry_price, -1)
        short_take_profit = self.strategy.get_take_profit(self.data, entry_price, -1)
        
        # 验证止损和止盈值
        self.assertLess(long_stop_loss, entry_price)
        self.assertGreater(long_take_profit, entry_price)
        self.assertGreater(short_stop_loss, entry_price)
        self.assertLess(short_take_profit, entry_price)
        
        print(f"多头止损价格: {long_stop_loss}")
        print(f"多头止盈价格: {long_take_profit}")
        print(f"空头止损价格: {short_stop_loss}")
        print(f"空头止盈价格: {short_take_profit}")
    
    def test_stop_loss_adjustment(self):
        """测试止损调整"""
        # 计算指标
        result = self.strategy.calculate_indicators(self.data)
        
        # 测试多头止损调整
        current_price = 110.0
        stop_loss = 95.0
        adjusted_stop_loss = self.strategy.should_adjust_stop_loss(result, current_price, stop_loss, 1)
        
        # 验证止损调整
        self.assertGreaterEqual(adjusted_stop_loss, stop_loss)
        
        print(f"原始止损价格: {stop_loss}")
        print(f"调整后止损价格: {adjusted_stop_loss}")

if __name__ == '__main__':
    unittest.main() 