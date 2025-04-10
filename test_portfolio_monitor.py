import asyncio
import unittest
from monitor.portfolio_monitor import PortfolioMonitor

class TestPortfolioMonitor(unittest.TestCase):
    def setUp(self):
        # 使用实际的持仓数据
        self.positions = {
            'GOOG': {'shares': 59, 'avg_price': 170.478},
            'TSLA': {'shares': 28, 'avg_price': 289.434},
            'AMD': {'shares': 58, 'avg_price': 123.737},
            'NVDA': {'shares': 40, 'avg_price': 138.843},
            'PFE': {'shares': 80, 'avg_price': 25.899},
            'MSFT': {'shares': 3, 'avg_price': 370.95},
            'TMDX': {'shares': 13, 'avg_price': 101.75}
        }
        
        self.config = {
            'price_alert_threshold': 0.05,  # 5% 价格变动触发警报
            'loss_alert_threshold': 0.1,    # 10% 亏损触发警报
            'profit_target': 0.2,           # 20% 目标收益
            'stop_loss': 0.1,               # 10% 止损
            'check_interval': 5,            # 5分钟检查一次
            'email_notifications': False    # 测试时不发送邮件
        }
        self.monitor = PortfolioMonitor(self.positions, self.config)

    async def test_update_positions(self):
        """测试更新持仓数据"""
        await self.monitor.update_positions()
        self.assertIsNotNone(self.monitor.current_prices)
        self.assertIsNotNone(self.monitor.historical_data)
        self.assertGreater(self.monitor.total_value, 0)

    def test_set_monitoring_parameters(self):
        """测试设置监控参数"""
        new_config = {
            'price_alert_threshold': 0.1,
            'loss_alert_threshold': 0.15,
            'profit_target': 0.25,
            'stop_loss': 0.15,
            'check_interval': 10
        }
        self.monitor.set_monitoring_parameters(new_config)
        self.assertEqual(self.monitor.config['price_alert_threshold'], 0.1)
        self.assertEqual(self.monitor.config['loss_alert_threshold'], 0.15)
        self.assertEqual(self.monitor.config['profit_target'], 0.25)
        self.assertEqual(self.monitor.config['stop_loss'], 0.15)

    async def test_check_alerts(self):
        """测试检查警报"""
        await self.monitor.update_positions()
        alerts = self.monitor.check_alerts()
        self.assertIsInstance(alerts, list)

    async def test_monitoring(self):
        """测试完整监控流程"""
        await self.monitor.monitor_stocks()
        self.assertIsNotNone(self.monitor.current_prices)
        self.assertIsNotNone(self.monitor.historical_data)
        self.assertGreater(self.monitor.total_value, 0)

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试用例
    suite.addTest(TestPortfolioMonitor('test_set_monitoring_parameters'))
    
    # 运行异步测试
    async def run_async_tests():
        test = TestPortfolioMonitor()
        test.setUp()
        await test.test_update_positions()
        await test.test_check_alerts()
        await test.test_monitoring()
    
    # 运行测试
    asyncio.run(run_async_tests())
    unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
    run_tests() 