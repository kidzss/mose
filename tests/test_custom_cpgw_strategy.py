import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy.custom_cpgw_strategy import CustomCPGWStrategy
from strategy.strategy_base import MarketRegime

class TestCustomCPGWStrategy(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.normal(1000, 100, len(dates))
        }, index=dates)
        
        # 确保价格数据的合理性
        self.test_data['high'] = self.test_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 1, len(dates)))
        self.test_data['low'] = self.test_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 1, len(dates)))
        self.test_data['volume'] = abs(self.test_data['volume'])
        
        # 创建策略实例
        self.strategy = CustomCPGWStrategy()
        
    def test_initialization(self):
        """测试策略初始化"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.rsi_period, 14)
        self.assertEqual(self.strategy.rsi_oversold, 30)
        self.assertEqual(self.strategy.rsi_overbought, 70)
        self.assertEqual(self.strategy.macd_fast, 12)
        self.assertEqual(self.strategy.macd_slow, 26)
        self.assertEqual(self.strategy.macd_signal, 9)
        self.assertTrue(self.strategy.market_adaptation)
        
    def test_calculate_indicators(self):
        """测试技术指标计算"""
        df = self.strategy.calculate_indicators(self.test_data)
        
        # 检查是否计算了所有必要的指标
        required_indicators = [
            'cpgw_fast_ema', 'cpgw_slow_ema', 'cpgw_diff', 'cpgw_signal',
            'cpgw_histogram', 'trend_short', 'trend_medium', 'trend_long',
            'trend_score', 'rsi', 'resistance', 'support', 'volume_ma', 'breakout'
        ]
        
        for indicator in required_indicators:
            self.assertIn(indicator, df.columns)
            
        # 检查指标值的合理性
        self.assertTrue(df['rsi'].between(0, 100).all())
        self.assertTrue(df['trend_score'].between(0, 3).all())
        self.assertTrue(df['breakout'].isin([0, 1]).all())
        
    def test_market_regime_adaptation(self):
        """测试市场环境适应性"""
        # 测试不同市场环境下的参数调整
        regimes = [MarketRegime.BULLISH, MarketRegime.BEARISH, 
                  MarketRegime.RANGING, MarketRegime.VOLATILE]
        
        for regime in regimes:
            self.strategy.current_regime = regime
            self.strategy._adjust_parameters_for_market()
            
            # 检查参数是否根据市场环境正确调整
            params = self.strategy.market_params[regime]
            self.assertEqual(self.strategy.rsi_oversold, params['rsi_oversold'])
            self.assertEqual(self.strategy.rsi_overbought, params['rsi_overbought'])
            self.assertEqual(self.strategy.volume_threshold, params['volume_threshold'])
            self.assertEqual(self.strategy.volatility_threshold, params['volatility_threshold'])
            
    def test_signal_generation(self):
        """测试信号生成"""
        df = self.strategy.generate_signals(self.test_data)
        
        # 检查信号列是否存在
        self.assertIn('signal', df.columns)
        
        # 检查信号值的范围
        self.assertTrue(df['signal'].between(-1.5, 1.5).all())
        
        # 检查信号生成的连续性
        signal_changes = df['signal'].diff().abs()
        self.assertTrue(signal_changes.max() <= 1.5)
        
    def test_signal_components(self):
        """测试信号组件提取"""
        components = self.strategy.extract_signal_components(self.test_data)
        
        # 检查是否提取了所有必要的信号组件
        required_components = [
            'cpgw_diff', 'cpgw_signal', 'cpgw_histogram',
            'trend_score', 'rsi', 'breakout'
        ]
        
        for component in required_components:
            self.assertIn(component, components)
            
        # 检查信号组件的元数据
        for component in components.values():
            self.assertIsNotNone(component.metadata)
            self.assertIsNotNone(component.series)
            
    def test_market_regime_adjustment(self):
        """测试市场环境调整"""
        # 生成初始信号
        df = self.strategy.generate_signals(self.test_data)
        
        # 测试不同市场环境下的信号调整
        for regime in [MarketRegime.BULLISH, MarketRegime.BEARISH]:
            self.strategy.current_regime = regime
            adjusted_df = self.strategy.adjust_for_market_regime(self.test_data, df)
            
            # 检查信号是否根据市场环境正确调整
            if regime == MarketRegime.BEARISH:
                # 熊市中应该增强卖出信号
                sell_signals = adjusted_df[adjusted_df['signal'] < 0]['signal']
                self.assertTrue(sell_signals.abs().mean() >= df[df['signal'] < 0]['signal'].abs().mean())
            else:
                # 牛市中应该增强买入信号
                buy_signals = adjusted_df[adjusted_df['signal'] > 0]['signal']
                self.assertTrue(buy_signals.mean() >= df[df['signal'] > 0]['signal'].mean())
                
    def test_breakout_signals(self):
        """测试突破信号生成"""
        df = self.strategy.calculate_indicators(self.test_data)
        
        # 检查突破信号的条件
        breakout_conditions = (
            (df['close'] > df['resistance'].shift(1)) & 
            (df['volume'] > df['volume_ma'] * self.strategy.volume_threshold)
        )
        
        self.assertTrue((df['breakout'] == 1).equals(breakout_conditions))
        
    def test_trend_following(self):
        """测试趋势跟踪功能"""
        df = self.strategy.calculate_indicators(self.test_data)
        
        # 检查趋势得分的计算
        self.assertTrue(df['trend_score'].between(0, 3).all())
        
        # 检查趋势得分与价格趋势的一致性
        price_trend = df['close'].diff()
        trend_score_changes = df['trend_score'].diff()
        
        # 趋势得分变化应该与价格趋势方向一致
        correlation = price_trend.corr(trend_score_changes)
        self.assertGreater(correlation, 0)
        
    def test_rsi_signals(self):
        """测试RSI信号生成"""
        df = self.strategy.calculate_indicators(self.test_data)
        
        # 检查RSI超买超卖信号
        oversold_signals = df[df['rsi'] < self.strategy.rsi_oversold]
        overbought_signals = df[df['rsi'] > self.strategy.rsi_overbought]
        
        # 确保RSI信号在合理范围内
        self.assertTrue(oversold_signals['rsi'].between(0, self.strategy.rsi_oversold).all())
        self.assertTrue(overbought_signals['rsi'].between(self.strategy.rsi_overbought, 100).all())

if __name__ == '__main__':
    unittest.main() 