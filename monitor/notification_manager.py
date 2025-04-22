import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.trading_config import default_config
from .trading_monitor import AlertSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NotificationManager")

class NotificationManager:
    """通知管理器，处理各种提醒功能"""
    
    def __init__(self, config=None):
        """初始化通知管理器"""
        self.config = config or default_config
        self.alert_system = AlertSystem(self.config)
        self.thresholds = self.config['notification_threshold']
        self.notification_history = {}
        self.cooldown_periods = {
            'trade_signal': timedelta(minutes=5),  # 缩短信号冷却时间
            'market_alert': timedelta(hours=4),
            'volatility_alert': timedelta(hours=2),
            'price_alert': timedelta(minutes=30),
            'risk_alert': timedelta(hours=1),
            'regular_update': timedelta(minutes=15)  # 新增常规更新冷却时间
        }
        
    def _should_send_notification(self, notification_type: str, identifier: str) -> bool:
        """
        检查是否应该发送通知（避免过于频繁的通知）
        
        参数:
            notification_type: 通知类型
            identifier: 通知标识符（如股票代码）
        """
        key = f"{notification_type}_{identifier}"
        if key in self.notification_history:
            last_sent = self.notification_history[key]
            cooldown = self.cooldown_periods.get(notification_type, timedelta(hours=1))
            if datetime.now() - last_sent < cooldown:
                return False
        return True
        
    def _update_notification_history(self, notification_type: str, identifier: str):
        """更新通知历史"""
        key = f"{notification_type}_{identifier}"
        self.notification_history[key] = datetime.now()
        
    def _format_html_message(self, title: str, content: Dict[str, Union[str, float, List[str]]]) -> str:
        """
        格式化HTML消息
        
        参数:
            title: 消息标题
            content: 消息内容字典
        """
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 15px; margin-bottom: 20px; }}
                .content {{ padding: 15px; }}
                .footer {{ background-color: #f8f9fa; padding: 10px; margin-top: 20px; font-size: 0.9em; }}
                .alert {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .danger {{ background-color: #ffe6e6; }}
                .warning {{ background-color: #fff3cd; }}
                .info {{ background-color: #e7f5ff; }}
                .success {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{title}</h2>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="content">
        """
        
        for key, value in content.items():
            if isinstance(value, list):
                html += f"<p><strong>{key}:</strong></p><ul>"
                for item in value:
                    html += f"<li>{item}</li>"
                html += "</ul>"
            elif isinstance(value, float):
                html += f"<p><strong>{key}:</strong> {value:.2f}</p>"
            else:
                html += f"<p><strong>{key}:</strong> {value}</p>"
        
        html += """
            </div>
            <div class="footer">
                <p>此邮件由自动交易监控系统生成，请勿直接回复。</p>
            </div>
        </body>
        </html>
        """
        return html
        
    def send_trade_signal(self, stock: str, action: str, price: float, reason: str, confidence: float = None):
        """
        发送交易信号提醒
        
        参数:
            stock: 股票代码
            action: 操作（买入/卖出）
            price: 当前价格
            reason: 交易原因
            confidence: 信号置信度（可选）
        """
        if not self._should_send_notification('trade_signal', stock):
            logger.info(f"跳过发送交易信号提醒 - {stock} (冷却期内)")
            return
            
        subject = f"交易信号提醒 - {stock} {action}"
        content = {
            "股票代码": stock,
            "操作": action,
            "价格": price,
            "原因": reason
        }
        if confidence is not None:
            content["信号置信度"] = f"{confidence:.1%}"
            
        body = self._format_html_message("交易信号提醒", content)
        self.alert_system.send_email(subject, body)
        self._update_notification_history('trade_signal', stock)
        
    def send_market_alert(self, market_status: Dict):
        """
        发送市场状况提醒
        
        参数:
            market_status: 市场状况信息字典
        """
        if not self._should_send_notification('market_alert', 'market'):
            logger.info("跳过发送市场状况提醒 (冷却期内)")
            return
            
        subject = f"市场状况提醒 - {market_status.get('market_condition', 'Unknown')}"
        content = {
            "市场状况": market_status.get('market_condition'),
            "风险等级": market_status.get('risk_level'),
            "机会板块": market_status.get('opportunity_sectors', []),
            "市场趋势": market_status.get('trend', '未知'),
            "建议操作": market_status.get('recommendation', '观望')
        }
        
        body = self._format_html_message("市场状况提醒", content)
        self.alert_system.send_email(subject, body)
        self._update_notification_history('market_alert', 'market')
        
    def send_volatility_alert(self, stock: str, volatility: float, avg_volatility: float, additional_info: Dict = None):
        """
        发送波动性提醒
        
        参数:
            stock: 股票代码
            volatility: 当前波动率
            avg_volatility: 平均波动率
            additional_info: 额外信息（可选）
        """
        if not self._should_send_notification('volatility_alert', stock):
            logger.info(f"跳过发送波动性提醒 - {stock} (冷却期内)")
            return
            
        if volatility > avg_volatility * (1 + self.thresholds.market_volatility):
            subject = f"波动性提醒 - {stock}"
            content = {
                "股票代码": stock,
                "当前波动率": volatility,
                "平均波动率": avg_volatility,
                "波动率变化": f"{(volatility/avg_volatility - 1):.1%}"
            }
            
            if additional_info:
                content.update(additional_info)
                
            body = self._format_html_message("波动性提醒", content)
            self.alert_system.send_email(subject, body)
            self._update_notification_history('volatility_alert', stock)
            
    def send_price_alert(self, stock: str, current_price: float, prev_price: float, volume_change: float = None):
        """
        发送价格变动提醒
        
        参数:
            stock: 股票代码
            current_price: 当前价格
            prev_price: 之前价格
            volume_change: 成交量变化（可选）
        """
        if not self._should_send_notification('price_alert', stock):
            logger.info(f"跳过发送价格变动提醒 - {stock} (冷却期内)")
            return
            
        price_change = (current_price - prev_price) / prev_price
        if abs(price_change) > self.thresholds.price_change:
            subject = f"价格变动提醒 - {stock}"
            content = {
                "股票代码": stock,
                "当前价格": current_price,
                "价格变动": f"{price_change:.2%}",
                "变动幅度": "上涨" if price_change > 0 else "下跌"
            }
            
            if volume_change is not None:
                content["成交量变化"] = f"{volume_change:.1%}"
                
            body = self._format_html_message("价格变动提醒", content)
            self.alert_system.send_email(subject, body)
            self._update_notification_history('price_alert', stock)
            
    def send_risk_alert(self, stock: str, risk_type: str, risk_level: str, details: Dict):
        """
        发送风险提醒
        
        参数:
            stock: 股票代码
            risk_type: 风险类型
            risk_level: 风险等级
            details: 风险详情
        """
        if not self._should_send_notification('risk_alert', stock):
            logger.info(f"跳过发送风险提醒 - {stock} (冷却期内)")
            return
            
        subject = f"风险提醒 - {stock} ({risk_type})"
        content = {
            "股票代码": stock,
            "风险类型": risk_type,
            "风险等级": risk_level,
            "风险详情": details.get('description', ''),
            "建议操作": details.get('recommendation', ''),
            "注意事项": details.get('precautions', [])
        }
        
        body = self._format_html_message("风险提醒", content)
        self.alert_system.send_email(subject, body)
        self._update_notification_history('risk_alert', stock)
        
    def send_batch_alerts(self, alerts: List[Dict]):
        """
        批量发送提醒
        
        参数:
            alerts: 提醒列表
        """
        if not alerts:
            return
            
        grouped_alerts = {}
        for alert in alerts:
            alert_type = alert.get('type', 'other')
            if alert_type not in grouped_alerts:
                grouped_alerts[alert_type] = []
            grouped_alerts[alert_type].append(alert)
            
        for alert_type, alert_list in grouped_alerts.items():
            subject = f"批量提醒 - {alert_type} ({len(alert_list)}条)"
            content = {
                "提醒类型": alert_type,
                "提醒数量": len(alert_list),
                "提醒详情": [alert.get('message', '') for alert in alert_list]
            }
            
            body = self._format_html_message("批量提醒", content)
            self.alert_system.send_email(subject, body)
            
    def update_thresholds(self, new_thresholds: Dict):
        """
        更新提醒阈值
        
        参数:
            new_thresholds: 新的阈值设置
        """
        self.thresholds.update(new_thresholds)
        logger.info("已更新提醒阈值设置")
        
    def send_trading_signal(self, stock: str, signal_type: str, price: float, indicators: Dict, confidence: float = None, time_frame: str = None):
        """
        发送交易信号提醒（买入/卖出点）
        
        参数:
            stock: 股票代码
            signal_type: 信号类型 ('buy' 或 'sell')
            price: 当前价格
            indicators: 技术指标字典
            confidence: 信号置信度（可选）
            time_frame: 时间周期 ('short', 'medium', 'long')
        """
        if not self._should_send_notification('trading_signal', stock):
            logger.info(f"跳过发送交易信号提醒 - {stock} (冷却期内)")
            return
            
        # 根据时间周期设置标题
        time_frame_text = {
            'short': '短期',
            'medium': '中期',
            'long': '长期'
        }.get(time_frame, '')
        
        signal_text = "买入" if signal_type == 'buy' else "卖出"
        subject = f"交易信号提醒 - {stock} {time_frame_text}{signal_text}"
        
        # 格式化技术指标信息
        indicators_html = ""
        for name, value in indicators.items():
            if isinstance(value, float):
                indicators_html += f"<li>{name}: {value:.2f}</li>"
            else:
                indicators_html += f"<li>{name}: {value}</li>"
        
        content = {
            "股票代码": stock,
            "信号类型": f"{time_frame_text}{signal_text}",
            "当前价格": price,
            "技术指标": f"<ul>{indicators_html}</ul>"
        }
        
        if confidence is not None:
            content["信号置信度"] = f"{confidence:.1%}"
            
        # 添加时间周期说明
        time_frame_desc = {
            'short': '基于5日均线的短期交易信号',
            'medium': '基于20日均线的中期交易信号',
            'long': '基于50日均线的长期交易信号'
        }.get(time_frame, '')
        content["时间周期说明"] = time_frame_desc
            
        body = self._format_html_message("交易信号提醒", content)
        self.alert_system.send_email(subject, body)
        self._update_notification_history('trading_signal', stock)
        
    def send_regular_update(self, portfolio_status: Dict, watchlist_status: Dict):
        """
        发送常规状态更新邮件
        
        参数:
            portfolio_status: 持仓状态信息
            watchlist_status: 观察股票状态信息
        """
        if not self._should_send_notification('regular_update', 'status'):
            logger.info("跳过发送常规状态更新 (冷却期内)")
            return
            
        subject = "持仓和观察股票状态更新"
        
        # 格式化持仓信息
        holdings_content = []
        for holding in portfolio_status.get('holdings', []):
            holdings_content.append(
                f"{holding['symbol']}: "
                f"持仓 {holding['shares']} 股, "
                f"成本 {holding['avg_cost']:.2f}, "
                f"现价 {holding['current_price']:.2f}, "
                f"盈亏 {holding['pnl']:.2f} ({holding['pnl_percent']:.2f}%)"
            )
            
        # 格式化观察股票信息
        watchlist_content = []
        for stock in watchlist_status.get('watchlist', []):
            signal_text = "买入" if stock['signal'] > 0 else "卖出" if stock['signal'] < 0 else "观望"
            time_frame_text = {
                'short': '短期',
                'medium': '中期',
                'long': '长期'
            }.get(stock.get('time_frame', 'N/A'), stock.get('time_frame', 'N/A'))
            
            watchlist_content.append(
                f"{stock['symbol']}: "
                f"现价 {stock['current_price']:.2f}, "
                f"信号: {time_frame_text}{signal_text}, "
                f"置信度: {stock.get('confidence', 'N/A')}"
            )
            
        content = {
            "持仓状态": holdings_content,
            "观察股票": watchlist_content,
            "总市值": f"{portfolio_status.get('total_value', 0):.2f}",
            "总成本": f"{portfolio_status.get('total_cost', 0):.2f}",
            "总盈亏": f"{portfolio_status.get('total_pnl', 0):.2f}"
        }
        
        body = self._format_html_message("持仓和观察股票状态更新", content)
        self.alert_system.send_email(subject, body)
        self._update_notification_history('regular_update', 'status') 