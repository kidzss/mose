import unittest
import json
from pathlib import Path
from monitor.portfolio_monitor import PortfolioMonitor
from monitor.report_generator import ReportGenerator

class TestReportGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 加载配置文件
        config_path = Path(__file__).parent.parent / "monitor" / "configs" / "portfolio_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # 初始化组件
        cls.portfolio_monitor = PortfolioMonitor(
            positions=config["positions"],
            monitor_config=config["monitor_config"]
        )
        cls.report_generator = ReportGenerator()
        
        # 更新持仓信息
        cls.portfolio_monitor.update_positions()
        
    def test_market_returns_calculation(self):
        """测试市场收益率计算"""
        sp500_return, nasdaq_return = self.report_generator._calculate_market_returns()
        
        self.assertIsInstance(sp500_return, float)
        self.assertIsInstance(nasdaq_return, float)
        
    def test_risk_metrics_calculation(self):
        """测试风险指标计算"""
        risk_metrics = self.report_generator._calculate_risk_metrics(self.portfolio_monitor)
        
        self.assertIn('beta', risk_metrics)
        self.assertIn('sharpe', risk_metrics)
        self.assertIn('max_drawdown', risk_metrics)
        self.assertIn('volatility', risk_metrics)
        self.assertIn('var_95', risk_metrics)
        
        self.assertIsInstance(risk_metrics['beta'], float)
        self.assertIsInstance(risk_metrics['sharpe'], float)
        self.assertIsInstance(risk_metrics['max_drawdown'], float)
        self.assertIsInstance(risk_metrics['volatility'], float)
        self.assertIsInstance(risk_metrics['var_95'], float)
        
    def test_report_generation(self):
        """测试报告生成"""
        report = self.report_generator.generate_daily_report(self.portfolio_monitor)
        
        self.assertIsInstance(report, str)
        self.assertIn("每日投资组合报告", report)
        self.assertIn("投资组合概览", report)
        self.assertIn("市场概览", report)
        self.assertIn("持仓明细", report)
        self.assertIn("风险指标", report)
        
        # 检查是否包含所有持仓
        for symbol in self.portfolio_monitor.positions.keys():
            self.assertIn(symbol, report)
            
if __name__ == '__main__':
    unittest.main() 