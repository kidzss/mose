import unittest
import json
from datetime import datetime
from monitor.portfolio_monitor import PortfolioMonitor

class TestPortfolioMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sample portfolio positions
        cls.positions = {
            "AAPL": {
                "cost_basis": 150.0,
                "weight": 0.4
            },
            "MSFT": {
                "cost_basis": 300.0,
                "weight": 0.6
            }
        }
        
        # Load config
        with open("monitor/configs/portfolio_config.json", "r") as f:
            cls.config = json.load(f)
            
        cls.monitor = PortfolioMonitor(cls.positions, cls.config)
        
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        # Manually set current prices for testing
        self.monitor.current_prices = {
            "AAPL": 160.0,
            "MSFT": 310.0
        }
        
        expected_value = (160.0 * 40) + (310.0 * 60)  # weight * 100 shares
        actual_value = self.monitor.calculate_portfolio_value()
        
        self.assertAlmostEqual(actual_value, expected_value, places=2)
        
    def test_returns_calculation(self):
        """Test returns calculation"""
        self.monitor.current_prices = {
            "AAPL": 165.0,  # 10% return
            "MSFT": 270.0   # -10% return
        }
        
        returns = self.monitor.calculate_returns()
        
        self.assertAlmostEqual(returns["AAPL"], 0.10, places=2)
        self.assertAlmostEqual(returns["MSFT"], -0.10, places=2)
        
    def test_alert_generation(self):
        """Test alert generation"""
        # Set prices to trigger alerts
        self.monitor.current_prices = {
            "AAPL": 120.0,  # -20% return, should trigger stop loss
            "MSFT": 400.0   # +33% return, should trigger profit target
        }
        
        alerts = self.monitor.check_alerts()
        
        self.assertTrue(any("Stop Loss Alert: AAPL" in alert for alert in alerts))
        self.assertTrue(any("Profit Target Alert: MSFT" in alert for alert in alerts))
        
    def test_risk_analysis(self):
        """Test risk analysis"""
        risk_metrics = self.monitor.analyze_portfolio_risk()
        
        self.assertIn("var_95", risk_metrics)
        self.assertIn("volatility", risk_metrics)
        self.assertIn("correlation", risk_metrics)
        self.assertIn("concentration_risk", risk_metrics)
        
    def test_recommendations(self):
        """Test recommendation generation"""
        # Set up a scenario with high concentration and poor performance
        self.monitor.current_prices = {
            "AAPL": 120.0,  # -20% return
            "MSFT": 310.0   # +3.3% return
        }
        
        recommendations = self.monitor.get_portfolio_recommendations()
        
        self.assertTrue(len(recommendations) > 0)
        self.assertTrue(any("concentration" in rec.lower() for rec in recommendations))
        self.assertTrue(any("poor performance" in rec.lower() for rec in recommendations))

if __name__ == "__main__":
    unittest.main() 