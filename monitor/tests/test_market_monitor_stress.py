import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging
from monitor.market_monitor import MarketMonitor
from concurrent.futures import ThreadPoolExecutor
import os
from data.data_loader import DataLoader

class TestMarketMonitorStress(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.monitor = MarketMonitor()
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        
        # 设置测试时间范围
        self.start_date = '2024-01-01'
        self.end_date = '2025-04-13'
        
    def load_real_data(self):
        """从MySQL加载真实数据"""
        self.logger.info("开始从MySQL加载数据...")
        
        try:
            # 获取所有股票代码
            symbols = self.data_loader.get_all_symbols()
            
            # 获取每个股票的数据
            all_data = []
            for symbol in symbols:
                try:
                    df = self.data_loader.get_stock_data(
                        symbol=symbol,
                        start_date=self.start_date,
                        end_date=self.end_date
                    )
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    self.logger.error(f"加载{symbol}数据时发生错误: {str(e)}")
            
            if not all_data:
                raise ValueError("没有找到符合条件的数据")
                
            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"成功加载 {len(combined_data)} 条数据记录")
            return combined_data
            
        except Exception as e:
            self.logger.error(f"加载数据时发生错误: {str(e)}")
            raise

    def test_large_data_processing(self):
        """测试大量数据处理性能"""
        self.logger.info("开始大量数据处理测试...")
        
        # 加载真实数据
        test_data = self.load_real_data()
        
        # 测试数据处理性能
        start_time = datetime.now()
        
        # 1. 测试数据加载性能
        self.monitor._process_market_data(test_data)
        load_time = datetime.now() - start_time
        self.logger.info(f"数据加载耗时: {load_time}")
        
        # 2. 测试指标计算性能
        start_time = datetime.now()
        self.monitor._calculate_indicators(test_data)
        calc_time = datetime.now() - start_time
        self.logger.info(f"指标计算耗时: {calc_time}")
        
        # 3. 测试信号生成性能
        start_time = datetime.now()
        self.monitor._generate_signals(test_data)
        signal_time = datetime.now() - start_time
        self.logger.info(f"信号生成耗时: {signal_time}")
        
        # 验证结果
        self.assertLess(load_time.total_seconds(), 5, "数据加载时间过长")
        self.assertLess(calc_time.total_seconds(), 10, "指标计算时间过长")
        self.assertLess(signal_time.total_seconds(), 5, "信号生成时间过长")

    def test_concurrent_processing(self):
        """测试高并发处理能力"""
        self.logger.info("开始高并发处理测试...")
        
        # 获取所有股票代码
        symbols = self.data_loader.get_all_symbols()
        
        def process_symbol(symbol):
            try:
                # 获取单个股票的数据
                data = self.data_loader.get_stock_data(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                if not data.empty:
                    # 处理数据
                    self.monitor._process_market_data(data)
                    self.monitor._calculate_indicators(data)
                    self.monitor._generate_signals(data)
                return True
            except Exception as e:
                self.logger.error(f"处理{symbol}时发生错误: {str(e)}")
                return False
        
        # 使用线程池进行并发测试
        with ThreadPoolExecutor(max_workers=10) as executor:
            start_time = datetime.now()
            results = list(executor.map(process_symbol, symbols))
            total_time = datetime.now() - start_time
            
        self.logger.info(f"并发处理总耗时: {total_time}")
        self.assertLess(total_time.total_seconds(), 30, "并发处理时间过长")
        self.assertTrue(all(results), "部分并发处理失败")

    def test_error_recovery(self):
        """测试异常恢复能力"""
        self.logger.info("开始异常恢复测试...")
        
        # 加载真实数据
        test_data = self.load_real_data()
        
        # 1. 测试数据异常恢复
        invalid_data = pd.DataFrame({
            'symbol': ['INVALID'],
            'date': [datetime.now()],
            'open': [np.nan],
            'high': [np.nan],
            'low': [np.nan],
            'close': [np.nan],
            'volume': [np.nan]
        })
        
        try:
            self.monitor._process_market_data(invalid_data)
        except Exception as e:
            self.logger.error(f"处理无效数据时发生错误: {str(e)}")
        
        # 验证系统是否继续运行
        self.monitor._process_market_data(test_data)
        
        # 2. 测试Redis连接异常恢复
        original_redis = self.monitor.redis_client
        self.monitor.redis_client = None
        
        try:
            self.monitor._save_to_redis({'test': 'data'})
        except Exception as e:
            self.logger.error(f"Redis连接异常时发生错误: {str(e)}")
        
        # 恢复Redis连接
        self.monitor.redis_client = original_redis
        self.monitor._save_to_redis({'test': 'data'})
        
        # 3. 测试配置加载异常恢复
        original_config = self.monitor.config
        self.monitor.config = {}
        
        try:
            self.monitor._load_config('invalid_path.json')
        except Exception as e:
            self.logger.error(f"加载无效配置时发生错误: {str(e)}")
        
        # 恢复配置
        self.monitor.config = original_config
        self.monitor._load_config()
        
        # 验证系统状态
        self.assertTrue(self.monitor.is_running, "系统未能从异常中恢复")

    def test_memory_usage(self):
        """测试内存使用情况"""
        self.logger.info("开始内存使用测试...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 加载真实数据
        test_data = self.load_real_data()
        
        # 处理数据
        for _ in range(10):
            self.monitor._process_market_data(test_data)
            self.monitor._calculate_indicators(test_data)
            self.monitor._generate_signals(test_data)
            
            # 强制垃圾回收
            import gc
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        self.logger.info(f"内存使用增加: {memory_increase:.2f}MB")
        self.assertLess(memory_increase, 500, "内存使用增加过多")

if __name__ == '__main__':
    unittest.main() 