import unittest
import os
import json
from unittest.mock import patch, MagicMock
from monitor.notification_system import NotificationSystem
from datetime import datetime

class TestNotificationSystem(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            "smtp_server": "test.smtp.com",
            "smtp_port": 587,
            "sender_email": "test@example.com",
            "sender_password": "test_password",
            "recipient_email": "recipient@example.com"
        }
        self.notification_system = NotificationSystem()
        
    def test_load_config(self):
        # Test loading default config
        config = self.notification_system._load_config()
        self.assertEqual(config["smtp_server"], "smtp.gmail.com")
        
        # Test loading custom config
        test_config_path = "test_config.json"
        with open(test_config_path, "w") as f:
            json.dump(self.test_config, f)
        
        config = self.notification_system._load_config(test_config_path)
        self.assertEqual(config["smtp_server"], "test.smtp.com")
        
        os.remove(test_config_path)
        
    def test_format_alert_message(self):
        alerts = {
            "danger": ["Critical risk detected", "Stop loss triggered"],
            "warning": ["High volatility observed"],
            "info": ["Market is trending sideways"],
            "opportunity": ["Potential buying opportunity"]
        }
        
        formatted_message = self.notification_system._format_alert_message(alerts)
        
        self.assertIn("Critical risk detected", formatted_message)
        self.assertIn("High volatility observed", formatted_message)
        self.assertIn("Market is trending sideways", formatted_message)
        self.assertIn("Potential buying opportunity", formatted_message)
        
    @patch('smtplib.SMTP')
    def test_send_notification(self, mock_smtp):
        # Mock SMTP instance
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value = mock_smtp_instance
        
        alerts = {
            "danger": ["Test alert"],
            "warning": [],
            "info": [],
            "opportunity": []
        }
        
        # Test successful notification
        result = self.notification_system.send_notification(alerts)
        self.assertTrue(result)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once()
        mock_smtp_instance.send_message.assert_called_once()
        
        # Test failed notification
        mock_smtp_instance.send_message.side_effect = Exception("Test error")
        result = self.notification_system.send_notification(alerts)
        self.assertFalse(result)
        
    def test_save_config(self):
        test_config_path = "test_save_config.json"
        
        # Test successful save
        result = self.notification_system.save_config(self.test_config, test_config_path)
        self.assertTrue(result)
        
        with open(test_config_path, "r") as f:
            saved_config = json.load(f)
        self.assertEqual(saved_config, self.test_config)
        
        os.remove(test_config_path)
        
        # Test failed save
        result = self.notification_system.save_config(self.test_config, "/invalid/path/config.json")
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main() 