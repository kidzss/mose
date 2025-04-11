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

from strategy import IntegratedStrategy

class IntegratedStrategyTestCase(unittest.TestCase):
    """集成策略测试用例"""
    
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
    
    def test_step1_niuniu_strategy(self):
        """测试第一步：NiuNiu策略集成"""
        print("\n测试步骤1：NiuNiu策略集成...")
        
        # 创建集成策略实例，只激活NiuNiu策略
        strategy = IntegratedStrategy()
        self.assertTrue(strategy.get_active_strategies()['niuniu'])
        self.assertFalse(strategy.get_active_strategies()['cpgw'])
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('niuniu_indicator', result.columns)
        self.assertIn('niuniu_ma', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证是否生成了信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"步骤1 NiuNiu策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('niuniu_indicator', components)
        
        # 可视化结果
        self._visualize_strategy_results(result, "Step1_NiuNiu", ['niuniu_indicator', 'niuniu_ma'])
    
    def test_step2_cpgw_strategy(self):
        """测试第二步：CPGW策略集成"""
        print("\n测试步骤2：CPGW策略集成...")
        
        # 创建集成策略实例，激活NiuNiu和CPGW策略
        strategy = IntegratedStrategy()
        strategy.activate_strategy('cpgw', True)
        
        self.assertTrue(strategy.get_active_strategies()['niuniu'])
        self.assertTrue(strategy.get_active_strategies()['cpgw'])
        self.assertFalse(strategy.get_active_strategies()['market_forecast'])
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('cpgw_fast_ema', result.columns)
        self.assertIn('cpgw_slow_ema', result.columns)
        self.assertIn('cpgw_diff', result.columns)
        self.assertIn('cpgw_signal', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证是否生成了信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"步骤2 CPGW策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('cpgw_diff', components)
        self.assertIn('cpgw_signal', components)
        
        # 可视化结果
        self._visualize_strategy_results(result, "Step2_CPGW", ['cpgw_fast_ema', 'cpgw_slow_ema'])
    
    def test_step3_market_forecast_strategy(self):
        """测试第三步：Market Forecast策略集成"""
        print("\n测试步骤3：Market Forecast策略集成...")
        
        # 创建集成策略实例，激活NiuNiu、CPGW和Market Forecast策略
        strategy = IntegratedStrategy()
        strategy.activate_strategy('cpgw', True)
        strategy.activate_strategy('market_forecast', True)
        
        self.assertTrue(strategy.get_active_strategies()['niuniu'])
        self.assertTrue(strategy.get_active_strategies()['cpgw'])
        self.assertTrue(strategy.get_active_strategies()['market_forecast'])
        self.assertFalse(strategy.get_active_strategies()['momentum'])
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('mf_indicator', result.columns)
        self.assertIn('mf_short_norm', result.columns)
        self.assertIn('mf_medium_norm', result.columns)
        self.assertIn('mf_long_norm', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证是否生成了信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"步骤3 Market Forecast策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('mf_indicator', components)
        
        # 可视化结果
        self._visualize_market_forecast_results(result, "Step3_MarketForecast")
    
    def test_step4_momentum_strategy(self):
        """测试第四步：Momentum策略集成"""
        print("\n测试步骤4：Momentum策略集成...")
        
        # 创建集成策略实例，激活所有前面的策略和Momentum策略
        strategy = IntegratedStrategy()
        strategy.activate_strategy('cpgw', True)
        strategy.activate_strategy('market_forecast', True)
        strategy.activate_strategy('momentum', True)
        
        self.assertTrue(strategy.get_active_strategies()['niuniu'])
        self.assertTrue(strategy.get_active_strategies()['cpgw'])
        self.assertTrue(strategy.get_active_strategies()['market_forecast'])
        self.assertTrue(strategy.get_active_strategies()['momentum'])
        self.assertFalse(strategy.get_active_strategies()['tdi'])
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('momentum', result.columns)
        self.assertIn('momentum_ma', result.columns)
        self.assertIn('momentum_change', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证是否生成了信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"步骤4 Momentum策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('momentum', components)
        self.assertIn('momentum_ma', components)
        
        # 可视化结果
        self._visualize_strategy_results(result, "Step4_Momentum", ['momentum', 'momentum_ma'])
    
    def test_step5_tdi_strategy(self):
        """测试第五步：TDI策略集成"""
        print("\n测试步骤5：TDI策略集成...")
        
        # 创建集成策略实例，激活所有策略
        strategy = IntegratedStrategy()
        strategy.activate_strategy('cpgw', True)
        strategy.activate_strategy('market_forecast', True)
        strategy.activate_strategy('momentum', True)
        strategy.activate_strategy('tdi', True)
        
        self.assertTrue(strategy.get_active_strategies()['niuniu'])
        self.assertTrue(strategy.get_active_strategies()['cpgw'])
        self.assertTrue(strategy.get_active_strategies()['market_forecast'])
        self.assertTrue(strategy.get_active_strategies()['momentum'])
        self.assertTrue(strategy.get_active_strategies()['tdi'])
        
        # 运行策略
        result = strategy.generate_signals(self.test_data)
        
        # 验证结果
        self.assertIn('signal', result.columns)
        self.assertIn('tdi_rsi', result.columns)
        self.assertIn('tdi_signal_line', result.columns)
        self.assertIn('tdi_bb_middle', result.columns)
        self.assertIn('tdi_bb_upper', result.columns)
        self.assertIn('tdi_bb_lower', result.columns)
        
        # 验证信号值是否合法
        self.assertTrue(all(result['signal'].isin([-1, 0, 1])))
        
        # 验证是否生成了信号
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        
        print(f"步骤5 TDI策略测试通过，生成了 {buy_signals} 个买入信号和 {sell_signals} 个卖出信号")
        
        # 提取信号组件
        components = strategy.extract_signal_components(result)
        self.assertIn('tdi_rsi', components)
        self.assertIn('tdi_signal_line', components)
        
        # 可视化结果
        self._visualize_tdi_results(result, "Step5_TDI")
    
    def _visualize_strategy_results(self, df, strategy_name, lines_to_plot):
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
        os.makedirs('tests/output', exist_ok=True)
        plt.savefig(f'tests/output/{strategy_name}_results.png')
        plt.close()
    
    def _visualize_market_forecast_results(self, df, strategy_name):
        """可视化Market Forecast策略结果"""
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
        
        # 绘制Market Forecast指标
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['mf_indicator'], label='Market Forecast Indicator', color='blue')
        
        # 绘制阈值线
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        
        plt.title('Market Forecast Indicator')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制短中长期规范化指标
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['mf_short_norm'], label='Short-term', color='green')
        plt.plot(df.index, df['mf_medium_norm'], label='Medium-term', color='blue')
        plt.plot(df.index, df['mf_long_norm'], label='Long-term', color='red')
        
        plt.title('Market Forecast Components')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs('tests/output', exist_ok=True)
        plt.savefig(f'tests/output/{strategy_name}_results.png')
        plt.close()
    
    def _visualize_tdi_results(self, df, strategy_name):
        """可视化TDI策略结果"""
        plt.figure(figsize=(12, 10))
        
        # 绘制价格
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Price', color='black')
        
        # 标记买入点和卖出点
        plt.plot(df[df['signal'] == 1].index, df.loc[df['signal'] == 1, 'close'], 
                '^', markersize=10, color='g', label='Buy Signal')
        plt.plot(df[df['signal'] == -1].index, df.loc[df['signal'] == -1, 'close'], 
                'v', markersize=10, color='r', label='Sell Signal')
        
        plt.title(f'{strategy_name} - Price')
        plt.legend(loc='best')
        plt.grid(True)
        
        # 绘制TDI指标
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['tdi_rsi'], label='TDI RSI', color='blue')
        plt.plot(df.index, df['tdi_signal_line'], label='Signal Line', color='red')
        plt.plot(df.index, df['tdi_bb_middle'], label='Middle Band', color='black', linestyle='--')
        plt.plot(df.index, df['tdi_bb_upper'], label='Upper Band', color='green', linestyle='--')
        plt.plot(df.index, df['tdi_bb_lower'], label='Lower Band', color='green', linestyle='--')
        
        # 设置y轴范围
        plt.ylim(0, 100)
        
        plt.title('TDI Indicators')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs('tests/output', exist_ok=True)
        plt.savefig(f'tests/output/{strategy_name}_results.png')
        plt.close()

def run_tests():
    """运行测试"""
    # 创建输出目录
    os.makedirs('tests/output', exist_ok=True)
    
    # 运行测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == '__main__':
    run_tests() 