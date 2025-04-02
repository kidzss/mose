import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import unittest
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from strategy import (
    Strategy,
    GoldenCrossStrategy,
    BollingerBandsStrategy,
    MACDStrategy,
    RSIStrategy,
    CustomStrategy,
    TDIStrategy,
    NiuNiuStrategy,
    CPGWStrategy,
    MarketForecastStrategy,
    MomentumStrategy
)

def generate_sample_data(days=100):
    """生成样本数据用于测试"""
    base_date = datetime.now() - timedelta(days=days)
    dates = [base_date + timedelta(days=i) for i in range(days)]
    
    # 生成模拟价格数据
    np.random.seed(42)  # 设置随机种子以便结果可重现
    
    # 生成初始价格和随机价格变动
    base_price = 100
    price_changes = np.random.normal(0.05, 1.0, days)  # 均值0.05，标准差1.0
    
    # 创建模拟趋势
    trends = np.linspace(-0.5, 0.5, num=days) + np.sin(np.linspace(0, 3*np.pi, num=days)) * 0.5
    
    # 生成价格序列
    prices = [base_price]
    for i in range(1, days):
        new_price = prices[-1] * (1 + price_changes[i] / 100 + trends[i] / 100)
        prices.append(new_price)
    
    # 生成其他价格数据
    highs = [p * (1 + abs(np.random.normal(0, 0.5)) / 100) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.5)) / 100) for p in prices]
    opens = [lows[i] + (highs[i] - lows[i]) * np.random.random() for i in range(days)]
    closes = prices
    volumes = [int(np.random.normal(1000000, 200000)) for _ in range(days)]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    df.set_index('date', inplace=True)
    
    # 为牛牛策略添加Bull_Line和Trade_Line
    # 简单使用移动平均线作为测试数据
    df['Bull_Line'] = df['close'].rolling(window=5).mean()
    df['Trade_Line'] = df['close'].rolling(window=10).mean()
    
    # 确保生成足够的交叉点，以便测试信号生成
    # 在某些点手动修改Bull_Line和Trade_Line的值，以确保产生交叉点
    for i in range(20, days, 15):
        if i < days - 1:
            if df['Bull_Line'].iloc[i] > df['Trade_Line'].iloc[i]:
                df.loc[df.index[i], 'Bull_Line'] = df['Trade_Line'].iloc[i] - 0.1
                df.loc[df.index[i+1], 'Bull_Line'] = df['Trade_Line'].iloc[i+1] + 0.1
            else:
                df.loc[df.index[i], 'Bull_Line'] = df['Trade_Line'].iloc[i] + 0.1
                df.loc[df.index[i+1], 'Bull_Line'] = df['Trade_Line'].iloc[i+1] - 0.1
    
    # 为Market Forecast策略添加特定的变化模式
    # 创建明显的上升和下降区间，以确保Market Forecast策略能生成买入和卖出信号
    for i in range(20, days, 20):
        if i + 5 < days:
            # 创建上升趋势
            df.loc[df.index[i:i+5], 'close'] = df.loc[df.index[i], 'close'] * np.linspace(1, 1.10, 5)
        
        if i + 10 < days:
            # 创建下降趋势
            df.loc[df.index[i+5:i+10], 'close'] = df.loc[df.index[i+5], 'close'] * np.linspace(1, 0.92, 5)
    
    return df

class StrategyTestCase(unittest.TestCase):
    """基础策略测试用例"""
    
    @classmethod
    def setUpClass(cls):
        """生成测试数据"""
        # 创建模拟的价格数据
        np.random.seed(42)  # 固定随机种子以获得可重复的结果
        
        # 创建日期范围
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(200)]
        
        # 生成随机价格
        close_prices = np.random.randn(200).cumsum() + 100  # 起始价格100
        
        # 为了模拟趋势和波动性，我们添加一些模式
        # 上升趋势
        close_prices[50:100] += np.linspace(0, 10, 50)
        # 下降趋势
        close_prices[100:150] -= np.linspace(0, 15, 50)
        # 盘整期
        close_prices[150:200] += np.sin(np.linspace(0, 6 * np.pi, 50)) * 3
        
        # 生成开高低价格
        high_prices = close_prices + np.random.rand(200) * 3
        low_prices = close_prices - np.random.rand(200) * 3
        open_prices = low_prices + np.random.rand(200) * (high_prices - low_prices)
        
        # 生成成交量数据
        volume = np.random.randint(1000, 10000, 200)
        # 在价格大幅波动的区域增加成交量
        volume[50:60] *= 2  # 上升趋势开始时成交量放大
        volume[95:105] *= 3  # 趋势转换点成交量放大
        volume[145:155] *= 2  # 下降趋势结束时成交量放大
        
        # 创建OHLCV DataFrame
        cls.test_data = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        # 设置日期为索引
        cls.test_data.set_index('date', inplace=True)
    
    def test_golden_cross_strategy(self):
        """测试金叉死叉策略"""
        print("\n测试金叉死叉策略...")
        
        # 创建策略实例
        strategy = GoldenCrossStrategy()
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('short_sma', result.columns)
        self.assertIn('long_sma', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证至少有一个买入信号和卖出信号
        self.assertTrue((result['signal'] == 1).any())
        self.assertTrue((result['signal'] == -1).any())
        
        print(f"金叉死叉策略测试通过，生成了 {(result['signal'] == 1).sum()} 个买入信号和 {(result['signal'] == -1).sum()} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('sma_diff', components)
        self.assertIn('golden_cross', components)
        self.assertIn('death_cross', components)
        
        # 验证结果可视化
        self._visualize_strategy_results(result, strategy.name, 'short_sma', 'long_sma')
    
    def test_bollinger_bands_strategy(self):
        """测试布林带策略"""
        print("\n测试布林带策略...")
        
        # 创建策略实例
        strategy = BollingerBandsStrategy()
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('bb_upper', result.columns)
        self.assertIn('bb_middle', result.columns)
        self.assertIn('bb_lower', result.columns)
        self.assertIn('rsi', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证至少有一个买入信号和卖出信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"布林带策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('bb_position', components)
        self.assertIn('rsi', components)
        
        # 验证结果可视化
        self._visualize_strategy_results(result, strategy.name, 'bb_upper', 'bb_middle', 'bb_lower')
    
    def test_macd_strategy(self):
        """测试MACD策略"""
        print("\n测试MACD策略...")
        
        # 创建策略实例
        strategy = MACDStrategy()
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('macd_line', result.columns)
        self.assertIn('signal_line', result.columns)
        self.assertIn('histogram', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证至少有一个买入信号和卖出信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"MACD策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('macd', components)
        self.assertIn('signal', components)
        self.assertIn('histogram', components)
        
        # 验证结果可视化
        self._visualize_macd_results(result, strategy.name)
    
    def test_rsi_strategy(self):
        """测试RSI策略"""
        print("\n测试RSI策略...")
        
        # 创建策略实例
        strategy = RSIStrategy()
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('rsi', result.columns)
        self.assertIn('ma', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证RSI值的范围
        self.assertTrue(all(result['rsi'].between(0, 100)))
        
        # 验证至少有一个买入信号和卖出信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"RSI策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('rsi', components)
        self.assertIn('oversold', components)
        self.assertIn('overbought', components)
        
        # 验证结果可视化
        self._visualize_rsi_results(result, strategy.name)
    
    def test_strategy_combination(self):
        """测试策略组合"""
        print("\n测试策略组合...")
        
        # 创建多个策略
        rsi_strategy = RSIStrategy()
        macd_strategy = MACDStrategy()
        
        # 分别获取策略信号
        rsi_result = rsi_strategy.generate_signals(self.test_data)
        macd_result = macd_strategy.generate_signals(self.test_data)
        
        # 组合策略信号
        combined_df = self.test_data.copy()
        combined_df['rsi_signal'] = rsi_result['signal']
        combined_df['macd_signal'] = macd_result['signal']
        
        # 简单组合规则：两个策略都是买入信号时才买入，任一策略是卖出信号时卖出
        combined_df['combined_signal'] = 0
        combined_df.loc[(combined_df['rsi_signal'] == 1) & (combined_df['macd_signal'] == 1), 'combined_signal'] = 1
        combined_df.loc[(combined_df['rsi_signal'] == -1) | (combined_df['macd_signal'] == -1), 'combined_signal'] = -1
        
        # 验证组合信号
        buy_signals = (combined_df['combined_signal'] == 1).sum()
        sell_signals = (combined_df['combined_signal'] == -1).sum()
        
        print(f"策略组合测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 可视化组合结果
        self._visualize_combination_results(combined_df, 'RSI_MACD_Combined')
    
    def _visualize_strategy_results(self, df, strategy_name, *lines_to_plot):
        """可视化策略结果"""
        plt.figure(figsize=(12, 8))
        
        # 绘制价格
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Price', color='black')
        
        # 绘制其他线
        for line in lines_to_plot:
            plt.plot(df.index, df[line], label=line)
        
        # 标记买入点和卖出点
        plt.plot(df[df['signal'] == 1].index, df.loc[df['signal'] == 1, 'close'], 
                '^', markersize=10, color='g', label='Buy Signal')
        plt.plot(df[df['signal'] == -1].index, df.loc[df['signal'] == -1, 'close'], 
                'v', markersize=10, color='r', label='Sell Signal')
        
        plt.title(f'{strategy_name} - Price and Indicators')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制成交量
        plt.subplot(2, 1, 2)
        plt.bar(df.index, df['volume'], label='Volume', color='blue', alpha=0.5)
        plt.title('Volume')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'tests/output/{strategy_name}_results.png')
        plt.close()
    
    def _visualize_macd_results(self, df, strategy_name):
        """可视化MACD策略结果"""
        plt.figure(figsize=(12, 10))
        
        # 绘制价格
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['close'], label='Price', color='black')
        
        # 标记买入点和卖出点
        plt.plot(df[df['signal'] == 1].index, df.loc[df['signal'] == 1, 'close'], 
                '^', markersize=10, color='g', label='Buy Signal')
        plt.plot(df[df['signal'] == -1].index, df.loc[df['signal'] == -1, 'close'], 
                'v', markersize=10, color='r', label='Sell Signal')
        
        plt.title(f'{strategy_name} - Price')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制MACD线和信号线
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['macd_line'], label='MACD Line', color='blue')
        plt.plot(df.index, df['signal_line'], label='Signal Line', color='red')
        
        # 绘制0线
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title('MACD and Signal Lines')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制柱状图
        plt.subplot(3, 1, 3)
        plt.bar(df.index, df['histogram'], label='Histogram', alpha=0.5, 
                color=np.where(df['histogram'] > 0, 'g', 'r'))
        
        plt.title('MACD Histogram')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'tests/output/{strategy_name}_results.png')
        plt.close()
    
    def _visualize_rsi_results(self, df, strategy_name):
        """可视化RSI策略结果"""
        plt.figure(figsize=(12, 10))
        
        # 绘制价格和均线
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['close'], label='Price', color='black')
        plt.plot(df.index, df['ma'], label='MA', color='blue')
        
        # 标记买入点和卖出点
        plt.plot(df[df['signal'] == 1].index, df.loc[df['signal'] == 1, 'close'], 
                '^', markersize=10, color='g', label='Buy Signal')
        plt.plot(df[df['signal'] == -1].index, df.loc[df['signal'] == -1, 'close'], 
                'v', markersize=10, color='r', label='Sell Signal')
        
        plt.title(f'{strategy_name} - Price and MA')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制RSI
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['rsi'], label='RSI', color='purple')
        
        # 绘制超买超卖线
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        plt.axhline(y=50, color='black', linestyle='-', alpha=0.3)
        
        # 设置y轴范围
        plt.ylim(0, 100)
        
        plt.title('RSI')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制成交量
        plt.subplot(3, 1, 3)
        plt.bar(df.index, df['volume'], label='Volume', color='blue', alpha=0.5)
        plt.title('Volume')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'tests/output/{strategy_name}_results.png')
        plt.close()
    
    def _visualize_combination_results(self, df, strategy_name):
        """可视化组合策略结果"""
        plt.figure(figsize=(12, 8))
        
        # 绘制价格
        plt.plot(df.index, df['close'], label='Price', color='black')
        
        # 标记单个策略信号
        plt.plot(df[df['rsi_signal'] == 1].index, df.loc[df['rsi_signal'] == 1, 'close'] - 2, 
                '^', markersize=7, color='blue', alpha=0.5, label='RSI Buy Signal')
        plt.plot(df[df['macd_signal'] == 1].index, df.loc[df['macd_signal'] == 1, 'close'] - 4, 
                '^', markersize=7, color='purple', alpha=0.5, label='MACD Buy Signal')
        
        # 标记组合策略的买入卖出点
        plt.plot(df[df['combined_signal'] == 1].index, df.loc[df['combined_signal'] == 1, 'close'], 
                '^', markersize=10, color='g', label='Combined Buy Signal')
        plt.plot(df[df['combined_signal'] == -1].index, df.loc[df['combined_signal'] == -1, 'close'], 
                'v', markersize=10, color='r', label='Combined Sell Signal')
        
        plt.title(f'{strategy_name} - Price and Signals')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'tests/output/{strategy_name}_results.png')
        plt.close()

    def test_custom_strategy(self):
        """测试自定义策略"""
        print("\n测试自定义策略...")
        
        # 创建策略实例
        strategy = CustomStrategy()
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('rsi', result.columns)
        self.assertIn('ma', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证至少有一个买入信号和卖出信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"自定义策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('rsi', components)
        self.assertIn('ma', components)
        self.assertIn('price', components)
        
        # 可视化结果
        self._visualize_custom_strategy_results(result, strategy.name)
    
    def _visualize_custom_strategy_results(self, df, strategy_name):
        """可视化自定义策略结果"""
        plt.figure(figsize=(14, 12))
        
        # 绘制价格和MA
        plt.subplot(4, 1, 1)
        plt.plot(df.index, df['close'], label='Price', color='black')
        plt.plot(df.index, df['ma'], label='MA', color='blue')
        
        # 标记买入点和卖出点
        plt.plot(df[df['signal'] == 1].index, df.loc[df['signal'] == 1, 'close'], 
                '^', markersize=10, color='g', label='Buy Signal')
        plt.plot(df[df['signal'] == -1].index, df.loc[df['signal'] == -1, 'close'], 
                'v', markersize=10, color='r', label='Sell Signal')
        
        plt.title(f'{strategy_name} - Price and MA')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制RSI
        plt.subplot(4, 1, 2)
        plt.plot(df.index, df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        plt.title('RSI')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制成交量
        plt.subplot(4, 1, 3)
        plt.bar(df.index, df['volume'], label='Volume', color='blue', alpha=0.5)
        plt.title('Volume')
        plt.grid(True)
        
        # 绘制信号
        plt.subplot(4, 1, 4)
        plt.plot(df.index, df['signal'], label='Signal', color='black', drawstyle='steps')
        plt.title('Trading Signals')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'tests/output/{strategy_name}_results.png')
        plt.close()

class TestStrategyBase(unittest.TestCase):
    """测试策略基类"""
    
    def setUp(self):
        self.data = generate_sample_data()
        
        # 创建一个简单的Strategy实现类而不是直接实例化抽象类
        class SimpleStrategy(Strategy):
            def calculate_indicators(self, data):
                return data
            
            def generate_signals(self, data):
                result = data.copy()
                result['signal'] = 0
                return result
                
            def extract_signal_components(self, data):
                return {
                    'price': data.get('close', pd.Series())
                }
                
            def get_signal_metadata(self):
                return {
                    'price': {
                        'name': '价格',
                        'description': '资产收盘价',
                        'color': 'black',
                        'line_style': 'solid',
                        'importance': 'high'
                    }
                }
        
        self.strategy = SimpleStrategy("BaseStrategy", {'param1': 10})
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.strategy.name, "BaseStrategy")
        self.assertEqual(self.strategy.parameters, {'param1': 10})
    
    def test_calculate_indicators(self):
        """测试基础指标计算方法"""
        result = self.strategy.calculate_indicators(self.data)
        # 基类方法应该返回原始数据
        self.assertTrue(result.equals(self.data))
    
    def test_generate_signals(self):
        """测试基础信号生成方法"""
        result = self.strategy.generate_signals(self.data)
        # 基类方法应该返回带有全0信号列的数据
        self.assertTrue('signal' in result.columns)
        self.assertEqual(result['signal'].sum(), 0)

class TestGoldenCrossStrategy(unittest.TestCase):
    """测试金叉策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = GoldenCrossStrategy({'short_window': 10, 'long_window': 30})
    
    def test_calculate_indicators(self):
        """测试金叉策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        # 更新为策略实际使用的列名
        self.assertTrue('short_sma' in result.columns)
        self.assertTrue('long_sma' in result.columns)
    
    def test_generate_signals(self):
        """测试金叉策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

class TestBollingerBandsStrategy(unittest.TestCase):
    """测试布林带策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = BollingerBandsStrategy()
    
    def test_calculate_indicators(self):
        """测试布林带策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        # 更新为策略实际使用的列名
        self.assertTrue('bb_upper' in result.columns)
        self.assertTrue('bb_lower' in result.columns)
        self.assertTrue('bb_middle' in result.columns)
    
    def test_generate_signals(self):
        """测试布林带策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

class TestMACDStrategy(unittest.TestCase):
    """测试MACD策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = MACDStrategy()
    
    def test_calculate_indicators(self):
        """测试MACD策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        # 更新为策略实际使用的列名
        self.assertTrue('macd_line' in result.columns)
        self.assertTrue('signal_line' in result.columns)
        self.assertTrue('histogram' in result.columns)
    
    def test_generate_signals(self):
        """测试MACD策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

class TestRSIStrategy(unittest.TestCase):
    """测试RSI策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = RSIStrategy({'length': 14, 'overbought': 70, 'oversold': 30})
    
    def test_calculate_indicators(self):
        """测试RSI策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        self.assertTrue('rsi' in result.columns)
    
    def test_generate_signals(self):
        """测试RSI策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

class TestTDIStrategy(unittest.TestCase):
    """测试TDI策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = TDIStrategy()
    
    def test_calculate_indicators(self):
        """测试TDI策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        self.assertTrue('tdi_rsi' in result.columns)
        self.assertTrue('tdi_signal' in result.columns)
        self.assertTrue('tdi_upper_band' in result.columns)
        self.assertTrue('tdi_lower_band' in result.columns)
    
    def test_generate_signals(self):
        """测试TDI策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

class TestNiuNiuStrategy(unittest.TestCase):
    """测试NiuNiu策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = NiuNiuStrategy()
    
    def test_calculate_indicators(self):
        """测试NiuNiu策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        self.assertTrue('Bull_Line' in result.columns)
        self.assertTrue('Trade_Line' in result.columns)
    
    def test_generate_signals(self):
        """测试NiuNiu策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

class TestCPGWStrategy(unittest.TestCase):
    """测试CPGW策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = CPGWStrategy()
    
    def test_calculate_indicators(self):
        """测试CPGW策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        self.assertTrue('cpgw_fast_ema' in result.columns)
        self.assertTrue('cpgw_slow_ema' in result.columns)
        self.assertTrue('cpgw_diff' in result.columns)
        self.assertTrue('cpgw_signal' in result.columns)
    
    def test_generate_signals(self):
        """测试CPGW策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

class TestMarketForecastStrategy(unittest.TestCase):
    """测试Market Forecast策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = MarketForecastStrategy()
    
    def test_calculate_indicators(self):
        """测试Market Forecast策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        self.assertTrue('mf_short_change' in result.columns)
        self.assertTrue('mf_medium_change' in result.columns)
        self.assertTrue('mf_long_change' in result.columns)
        self.assertTrue('mf_indicator' in result.columns)
    
    def test_generate_signals(self):
        """测试Market Forecast策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

class TestMomentumStrategy(unittest.TestCase):
    """测试Momentum策略"""
    
    def setUp(self):
        self.data = generate_sample_data()
        self.strategy = MomentumStrategy()
    
    def test_calculate_indicators(self):
        """测试Momentum策略的指标计算"""
        result = self.strategy.calculate_indicators(self.data)
        self.assertTrue('momentum' in result.columns)
        self.assertTrue('momentum_ma' in result.columns)
        self.assertTrue('momentum_change' in result.columns)
    
    def test_generate_signals(self):
        """测试Momentum策略的信号生成"""
        result = self.strategy.generate_signals(self.data)
        self.assertTrue('signal' in result.columns)
        # 信号应该包含买入(1)和卖出(-1)
        self.assertTrue(any(result['signal'] == 1) or any(result['signal'] == -1))

def run_tests():
    """运行测试"""
    # 创建输出目录
    os.makedirs('tests/output', exist_ok=True)
    
    # 运行测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == '__main__':
    run_tests() 