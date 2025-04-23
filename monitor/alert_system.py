import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

class AlertSystem:
    """高级预警系统"""
    
    def __init__(self, config=None):
        """初始化警报系统"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.email_config = self.config.get('notification', {}).get('email', {})
        self.telegram_config = self.config.get('notification', {}).get('telegram', {})
        self.slack_config = self.config.get('notification', {}).get('slack', {})
        
        self.technical_indicators = {
            'SMA_20': 20,
            'SMA_50': 50,
            'SMA_200': 200,
            'RSI': 14,
            'MACD': (12, 26, 9)
        }
        
        self.alert_thresholds = {
            'volatility_threshold': 2.0,  # 标准差倍数
            'volume_threshold': 2.0,      # 平均成交量倍数
            'stop_loss': -0.10,          # 止损线
            'take_profit': 0.30,         # 止盈线
            'position_limit': 0.30,      # 单一持仓上限
            'sector_limit': 0.40         # 板块持仓上限
        }
    
    def calculate_technical_indicators(self, symbol: str, data: pd.DataFrame) -> Dict:
        """计算技术指标"""
        try:
            indicators = {}
            
            # 计算移动平均线
            for name, period in self.technical_indicators.items():
                if 'SMA' in name:
                    indicators[name] = data['close'].rolling(window=period).mean().iloc[-1]
            
            # 计算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # 计算MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            indicators['MACD'] = macd.iloc[-1]
            indicators['MACD_Signal'] = signal.iloc[-1]
            
            # 计算波动率
            indicators['Volatility'] = data['close'].pct_change().std() * np.sqrt(252)
            
            # 计算成交量相对值
            indicators['Volume_Ratio'] = data['volume'].iloc[-1] / data['volume'].rolling(window=20).mean().iloc[-1]
            
            return indicators
        except Exception as e:
            print(f"计算技术指标时出错: {e}")
            return {}
    
    def generate_alerts(self, symbol: str, position: Dict, current_data: pd.DataFrame) -> List[Dict]:
        """生成预警信息"""
        try:
            alerts = []
            indicators = self.calculate_technical_indicators(symbol, current_data)
            current_price = current_data['close'].iloc[-1]
            cost_basis = position['cost_basis']
            
            # 技术指标预警
            if current_price < indicators['SMA_200']:
                alerts.append({
                    'type': 'technical',
                    'level': 'warning',
                    'message': f"{symbol}跌破200日均线，建议关注"
                })
            
            if indicators['RSI'] > 70:
                alerts.append({
                    'type': 'technical',
                    'level': 'warning',
                    'message': f"{symbol}RSI超买（{indicators['RSI']:.2f}），可考虑减仓"
                })
            elif indicators['RSI'] < 30:
                alerts.append({
                    'type': 'technical',
                    'level': 'opportunity',
                    'message': f"{symbol}RSI超卖（{indicators['RSI']:.2f}），可考虑加仓"
                })
            
            # MACD信号
            if indicators['MACD'] < indicators['MACD_Signal'] and indicators['MACD'] > 0:
                alerts.append({
                    'type': 'technical',
                    'level': 'warning',
                    'message': f"{symbol}MACD形成死叉，注意风险"
                })
            
            # 波动率预警
            if indicators['Volatility'] > self.alert_thresholds['volatility_threshold']:
                alerts.append({
                    'type': 'risk',
                    'level': 'warning',
                    'message': f"{symbol}波动率异常（{indicators['Volatility']:.2f}），建议设置止损"
                })
            
            # 成交量预警
            if indicators['Volume_Ratio'] > self.alert_thresholds['volume_threshold']:
                alerts.append({
                    'type': 'volume',
                    'level': 'info',
                    'message': f"{symbol}成交量放大，成交量是20日均量{indicators['Volume_Ratio']:.1f}倍"
                })
            
            # 止损止盈预警
            returns = (current_price - cost_basis) / cost_basis
            if returns < self.alert_thresholds['stop_loss']:
                alerts.append({
                    'type': 'risk',
                    'level': 'danger',
                    'message': f"{symbol}触及止损线，建议止损或评估加仓"
                })
            elif returns > self.alert_thresholds['take_profit']:
                alerts.append({
                    'type': 'risk',
                    'level': 'info',
                    'message': f"{symbol}已达止盈目标，建议分批减仓"
                })
            
            # 仓位预警
            if position['weight'] > self.alert_thresholds['position_limit']:
                alerts.append({
                    'type': 'position',
                    'level': 'warning',
                    'message': f"{symbol}仓位过重（{position['weight']*100:.1f}%），建议适度分散"
                })
            
            return alerts
        except Exception as e:
            print(f"生成预警信息时出错: {e}")
            return []
    
    def analyze_market_sentiment(self) -> Dict:
        """分析市场情绪"""
        try:
            vix = yf.download('^VIX', period='5d')
            spy = yf.download('SPY', period='5d')
            
            market_state = {
                'vix_level': vix['Close'].iloc[-1],
                'vix_change': (vix['Close'].iloc[-1] / vix['Close'].iloc[-2] - 1) * 100,
                'market_return': (spy['Close'].iloc[-1] / spy['Close'].iloc[-2] - 1) * 100
            }
            
            if market_state['vix_level'] > 30:
                market_state['sentiment'] = '恐慌'
                market_state['suggestion'] = '市场恐慌情绪较重，可以逢低分批建仓优质标的'
            elif market_state['vix_level'] > 20:
                market_state['sentiment'] = '谨慎'
                market_state['suggestion'] = '市场波动加大，建议控制仓位，关注防御性板块'
            else:
                market_state['sentiment'] = '平稳'
                market_state['suggestion'] = '市场情绪平稳，可以维持现有仓位'
            
            return market_state
        except Exception as e:
            print(f"分析市场情绪时出错: {e}")
            return {
                'sentiment': '未知',
                'suggestion': '无法获取市场数据，建议谨慎操作'
            }
    
    def generate_trading_signals(self, symbol: str, data: pd.DataFrame) -> Dict:
        """生成交易信号"""
        try:
            indicators = self.calculate_technical_indicators(symbol, data)
            current_price = data['Close'].iloc[-1]
            
            signals = {
                'action': 'hold',
                'confidence': 0.5,
                'reasons': []
            }
            
            # 技术面信号
            if current_price > indicators['SMA_20'] and indicators['RSI'] < 60:
                signals['reasons'].append('价格站上20日均线且RSI未达超买')
                signals['confidence'] += 0.1
            
            if indicators['MACD'] > indicators['MACD_Signal']:
                signals['reasons'].append('MACD金叉形成')
                signals['confidence'] += 0.1
            
            # 市场情绪
            market_state = self.analyze_market_sentiment()
            if market_state['sentiment'] == '恐慌' and indicators['RSI'] < 30:
                signals['reasons'].append('市场恐慌且RSI超卖')
                signals['action'] = 'buy'
                signals['confidence'] += 0.2
            elif market_state['sentiment'] == '平稳' and indicators['RSI'] > 70:
                signals['reasons'].append('市场平稳但RSI超买')
                signals['action'] = 'sell'
                signals['confidence'] += 0.2
            
            return signals
        except Exception as e:
            print(f"生成交易信号时出错: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reasons': ['数据异常，建议观望']
            }

    def send_alert(self, stock: str, alert_type: str, message: str, price: float, indicators: Dict):
        """发送警报"""
        try:
            # 确保所有数值都是标量而不是Series
            def get_scalar(value):
                if isinstance(value, pd.Series):
                    return value.iloc[0]
                return value

            # 构建完整的消息
            full_message = f"""
股票代码: {stock}
警报类型: {alert_type}

价格信息:
- 当前价格: {get_scalar(price):.2f}
- 成本价格: {get_scalar(indicators.get('cost_basis', 0)):.2f}
- 价格变化: {get_scalar(indicators.get('price_change', 0)):.2%}

技术指标分析:
- RSI: {self._get_rsi_explanation(get_scalar(indicators.get('RSI', 0)))}
- MACD: {self._get_macd_explanation(get_scalar(indicators.get('MACD', 0)), get_scalar(indicators.get('MACD_Signal', 0)))}
- 成交量: {self._get_volume_explanation(get_scalar(indicators.get('volume', 0)), get_scalar(indicators.get('volume_ma20', 0)))}

风险控制:
- 止损价格: {get_scalar(price) * (1 - get_scalar(indicators.get('stop_loss', 0.15))):.2f} ({get_scalar(indicators.get('stop_loss', 0.15)):.1%})
- 仓位权重: {get_scalar(indicators.get('weight', 0)):.2%}
"""
            
            # 发送邮件通知
            if hasattr(self.config, 'email'):
                subject = f"交易警报 - {stock} - {alert_type}"
                self.send_email(subject, full_message)
            
            print(f"警报已发送: {stock} - {alert_type}")
        
        except Exception as e:
            print(f"发送警报失败: {str(e)}")

    def send_email(self, subject: str, body: str, is_html: bool = False):
        """发送邮件通知"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = subject
            
            # 根据 is_html 参数决定内容类型
            content_type = 'html' if is_html else 'plain'
            msg.attach(MIMEText(body, content_type, 'utf-8'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)
            
            print(f"邮件发送成功: {subject}")
        
        except Exception as e:
            print(f"发送邮件时出错: {str(e)}")

    def _get_rsi_explanation(self, rsi: float) -> str:
        """获取RSI指标的解释"""
        if rsi > 70:
            return f"超买区域 ({rsi:.2f})，可能面临回调风险"
        elif rsi < 30:
            return f"超卖区域 ({rsi:.2f})，可能存在反弹机会"
        else:
            return f"中性区域 ({rsi:.2f})，市场相对平衡"

    def _get_macd_explanation(self, macd: float, signal: float) -> str:
        """获取MACD指标的解释"""
        if macd > signal:
            return f"金叉形态，上涨动能增强"
        elif macd < signal:
            return f"死叉形态，下跌动能增强"
        else:
            return f"趋势不明朗，等待方向确认"

    def _get_volume_explanation(self, volume: float, volume_ma20: float) -> str:
        """获取成交量指标的解释"""
        if volume > volume_ma20 * 2:
            return f"成交量显著放大，可能有重要信息"
        elif volume > volume_ma20 * 1.5:
            return f"成交量温和放大，需要关注"
        elif volume < volume_ma20 * 0.5:
            return f"成交量显著萎缩，交投清淡"
        else:
            return f"成交量正常，市场交投平稳" 