import unittest
import datetime as dt
import pandas as pd
import numpy as np
from data import DataInterface, DataValidator
from config.data_config import default_data_config

class TestDataInterface(unittest.TestCase):
    """测试数据接口的各种功能，同时作为使用示例"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，创建数据接口实例"""
        cls.data = DataInterface()
        cls.test_symbol = 'AAPL'  # 使用苹果股票作为测试用例
        cls.start_date = dt.datetime.now() - dt.timedelta(days=365)
        cls.end_date = dt.datetime.now()
    
    def test_1_basic_data_retrieval(self):
        """测试基本的数据获取功能"""
        print("\n=== 测试基本数据获取 ===")
        
        # 获取单个股票的历史数据
        df = self.data.get_historical_data(
            symbol=self.test_symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 验证数据结构
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        
        # 验证必要的列是否存在
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        for col in required_columns:
            self.assertIn(col, df.columns)
            
        print(f"成功获取 {self.test_symbol} 的历史数据")
        print(f"数据点数量: {len(df)}")
        print(f"数据列: {list(df.columns)}")
        print(f"数据示例:\n{df.head()}")
    
    def test_2_strategy_data(self):
        """测试策略数据获取功能"""
        print("\n=== 测试策略数据获取 ===")
        
        # 获取带技术指标的策略数据
        df = self.data.get_data_for_strategy(
            symbol=self.test_symbol,
            lookback_days=120
        )
        
        # 验证技术指标是否被正确计算
        technical_indicators = ['returns', 'volatility', 'ma5', 'ma10', 'ma20', 'ma60']
        for indicator in technical_indicators:
            self.assertIn(indicator, df.columns)
            
        print(f"成功获取 {self.test_symbol} 的策略数据")
        print(f"可用的技术指标: {list(df.columns)}")
        print(f"数据示例:\n{df.head()}")
    
    def test_3_multiple_symbols(self):
        """测试多股票数据获取功能"""
        print("\n=== 测试多股票数据获取 ===")
        
        # 测试多个股票
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        data_dict = self.data.get_multiple_symbols_data(
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 验证返回的数据字典
        self.assertIsInstance(data_dict, dict)
        for symbol in symbols:
            self.assertIn(symbol, data_dict)
            self.assertIsInstance(data_dict[symbol], pd.DataFrame)
            
        print(f"成功获取多个股票的数据: {symbols}")
        for symbol, df in data_dict.items():
            print(f"{symbol}: {len(df)} 个数据点")
    
    def test_4_data_validation(self):
        """测试数据验证功能"""
        print("\n=== 测试数据验证 ===")
        
        # 获取数据
        df = self.data.get_historical_data(self.test_symbol, self.start_date, self.end_date)
        
        # 进行数据验证
        validated_data, report = DataValidator.validate_data(df)
        
        # 验证结果
        self.assertTrue('validation_passed' in report)
        self.assertTrue('checks' in report)
        
        print("数据验证结果:")
        print(f"验证通过: {report['validation_passed']}")
        print("详细检查结果:")
        for check, result in report['checks'].items():
            print(f"- {check}: {result}")
    
    def test_5_data_update_and_monitoring(self):
        """测试数据更新和监控功能"""
        print("\n=== 测试数据更新和监控 ===")
        
        # 检查数据状态
        status = self.data.check_data_status(self.test_symbol)
        
        # 验证状态报告的结构
        self.assertIn(self.test_symbol, status)
        self.assertIn('last_update', status[self.test_symbol])
        self.assertIn('validation', status[self.test_symbol])
        
        print(f"数据状态检查结果 ({self.test_symbol}):")
        print(f"最后更新时间: {status[self.test_symbol]['last_update']}")
        print(f"数据状态: {status[self.test_symbol]['status']}")
        print(f"数据点数量: {status[self.test_symbol]['data_points']}")
    
    def test_6_market_status(self):
        """测试市场状态监控功能"""
        print("\n=== 测试市场状态监控 ===")
        
        # 获取市场状态
        market_status = self.data.get_market_status()
        
        # 验证市场状态报告的结构
        self.assertIn('total_symbols', market_status)
        self.assertIn('last_update', market_status)
        self.assertIn('data_quality', market_status)
        
        print("市场状态报告:")
        print(f"总股票数量: {market_status['total_symbols']}")
        print(f"最后更新时间: {market_status['last_update']}")
        print("数据质量统计:")
        for category, count in market_status['data_quality'].items():
            print(f"- {category}: {count}")
    
    def test_7_error_handling(self):
        """测试错误处理功能"""
        print("\n=== 测试错误处理 ===")
        
        # 测试无效的股票代码
        invalid_symbol = 'INVALID_SYMBOL'
        df = self.data.get_historical_data(
            symbol=invalid_symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 验证返回空DataFrame而不是抛出错误
        self.assertTrue(df.empty)
        print(f"成功处理无效的股票代码: {invalid_symbol}")
        
        # 测试无效的日期范围
        future_date = dt.datetime.now() + dt.timedelta(days=365)
        df = self.data.get_historical_data(
            symbol=self.test_symbol,
            start_date=future_date,
            end_date=future_date + dt.timedelta(days=10)
        )
        
        # 验证返回空DataFrame而不是抛出错误
        self.assertTrue(df.empty)
        print("成功处理无效的日期范围")
    
    def test_8_cache_mechanism(self):
        """测试缓存机制"""
        print("\n=== 测试缓存机制 ===")
        
        # 第一次调用
        start_time = dt.datetime.now()
        df1 = self.data.get_historical_data(
            symbol=self.test_symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        first_call_time = (dt.datetime.now() - start_time).total_seconds()
        
        # 第二次调用（应该使用缓存）
        start_time = dt.datetime.now()
        df2 = self.data.get_historical_data(
            symbol=self.test_symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        second_call_time = (dt.datetime.now() - start_time).total_seconds()
        
        # 验证两次调用返回相同的数据
        pd.testing.assert_frame_equal(df1, df2)
        
        print("缓存机制测试结果:")
        print(f"第一次调用时间: {first_call_time:.4f}秒")
        print(f"第二次调用时间: {second_call_time:.4f}秒")
        print(f"性能提升: {(first_call_time/second_call_time):.2f}倍")

if __name__ == '__main__':
    unittest.main(verbosity=2) 