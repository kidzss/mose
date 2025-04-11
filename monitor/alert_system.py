import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yfinance as yf
from datetime import datetime, timedelta

class AlertSystem:
    """高级预警系统"""
    
    def __init__(self):
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
                    indicators[name] = data['Close'].rolling(window=period).mean().iloc[-1]
            
            # 计算RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # 计算MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            indicators['MACD'] = macd.iloc[-1]
            indicators['MACD_Signal'] = signal.iloc[-1]
            
            # 计算波动率
            indicators['Volatility'] = data['Close'].pct_change().std() * np.sqrt(252)
            
            # 计算成交量相对值
            indicators['Volume_Ratio'] = data['Volume'].iloc[-1] / data['Volume'].rolling(window=20).mean().iloc[-1]
            
            return indicators
        except Exception as e:
            print(f"计算技术指标时出错: {e}")
            return {}
    
    def generate_alerts(self, symbol: str, position: Dict, current_data: pd.DataFrame) -> List[Dict]:
        """生成预警信息"""
        try:
            alerts = []
            indicators = self.calculate_technical_indicators(symbol, current_data)
            current_price = current_data['Close'].iloc[-1]
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