import logging
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from monitor.stock_monitor import StockMonitor
from data.data_loader import DataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SemiconductorTradingStrategy:
    """半导体板块交易策略"""
    
    def __init__(self):
        self.stock_monitor = StockMonitor()
        self.semiconductor_stocks = ['AMD', 'NVDA', 'INTC', 'QCOM', 'AVGO']
        # 调整止损位，考虑利好因素
        self.stop_loss_levels = {
            'AMD': 65.0,  # 降低止损位以适应波动
            'NVDA': 75.0,  # 降低止损位以适应波动
            'INTC': 28.0,
            'QCOM': 110.0,
            'AVGO': 480.0
        }
        # 添加目标价
        self.target_prices = {
            'AMD': 95.0,   # 预期上涨空间30%
            'NVDA': 110.0, # 预期上涨空间35%
            'INTC': 38.0,  # 预期上涨空间25%
            'QCOM': 150.0, # 预期上涨空间25%
            'AVGO': 650.0  # 预期上涨空间30%
        }
        # 添加波段分析
        self.wave_bands = {
            'AMD': [
                {'range': (115, 126), 'action': 'breakout', 'target': 126},
                {'range': (101, 111), 'action': 'support_resistance', 'level': 111},
                {'range': (89, 98), 'action': 'support_resistance', 'level': 98},
                {'range': (81, 87), 'action': 'support_resistance', 'level': 87},
                {'range': (72, 79), 'action': 'stop_loss', 'level': 72}
            ],
            'NVDA': [
                {'range': (118, 125), 'action': 'breakout', 'target': 125},
                {'range': (112, 116), 'action': 'support_resistance', 'level': 116},
                {'range': (84, 97), 'action': 'stop_loss', 'level': 84}
            ]
        }
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        
    async def analyze_semiconductor_sector(self) -> Dict:
        """
        分析半导体板块整体情况
        :return: 板块分析结果
        """
        try:
            # 主要半导体股票列表
            semiconductor_stocks = {
                'NVDA': '英伟达',
                'AMD': '超微半导体',
                'INTC': '英特尔',
                'AVGO': '博通',
                'QCOM': '高通',
                'ASML': '阿斯麦',
                'TSM': '台积电',
                'MU': '美光科技'
            }
            
            sector_analysis = {
                'stocks': {},
                'sector_health': 0,
                'trend': 'neutral',
                'recommendations': []
            }
            
            # 分析每只股票
            for symbol, name in semiconductor_stocks.items():
                try:
                    analysis = await self.stock_monitor.get_stock_analysis(symbol)
                    if analysis and 'error' not in analysis:
                        sector_analysis['stocks'][symbol] = {
                            'name': name,
                            'current_price': analysis['current_price'],
                            'price_change': analysis['price_change'],
                            'rsi': analysis['rsi'],
                            'macd_status': analysis['macd_status'],
                            'trend': analysis['trend']
                        }
                        
                        # 更新板块健康度
                        if analysis['trend'].startswith('上升'):
                            sector_analysis['sector_health'] += 1
                        elif analysis['trend'].startswith('下降'):
                            sector_analysis['sector_health'] -= 1
                            
                except Exception as e:
                    self.logger.error(f"分析半导体股票 {symbol} 时出错: {e}")
                    
            # 确定板块趋势
            total_stocks = len(sector_analysis['stocks'])
            if total_stocks > 0:
                health_score = sector_analysis['sector_health'] / total_stocks
                
                if health_score > 0.3:
                    sector_analysis['trend'] = 'bullish'
                    sector_analysis['recommendations'].append("半导体板块整体呈现上升趋势，可以考虑增加配置")
                elif health_score < -0.3:
                    sector_analysis['trend'] = 'bearish'
                    sector_analysis['recommendations'].append("半导体板块整体呈现下降趋势，建议谨慎操作")
                else:
                    sector_analysis['trend'] = 'neutral'
                    sector_analysis['recommendations'].append("半导体板块趋势中性，建议观望")
                    
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"分析半导体板块时出错: {e}")
            return {
                'error': str(e),
                'stocks': {},
                'sector_health': 0,
                'trend': 'neutral',
                'recommendations': []
            }
            
    def _analyze_stock_state(self, data):
        """分析个股状态"""
        try:
            # 计算技术指标
            rsi = self._calculate_rsi(data['close'])
            macd, signal = self._calculate_macd(data['close'])
            sma20 = data['close'].rolling(window=20).mean().iloc[-1]
            
            # 判断趋势
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            trend = "上升" if current_price > prev_price else "下降"
            
            # 判断强度
            if rsi > 70:
                strength = "超买"
            elif rsi < 30:
                strength = "超卖"
            else:
                strength = "中性"
                
            # 判断支撑
            if current_price > sma20:
                support = "20日均线支撑"
            else:
                support = "跌破20日均线"
                
            return {
                '趋势': trend,
                '强度': strength,
                '支撑': support
            }
            
        except Exception as e:
            self.logger.error(f"分析个股状态时出错: {e}")
            return {
                '趋势': '未知',
                '强度': '未知',
                '支撑': '数据不足'
            }
            
    def _analyze_sector_trend(self, amd_state, nvda_state):
        """分析板块趋势"""
        try:
            if amd_state['趋势'] == '上升' and nvda_state['趋势'] == '上升':
                return "强势"
            elif amd_state['趋势'] == '上升' or nvda_state['趋势'] == '上升':
                return "分化"
            else:
                return "弱势"
        except Exception as e:
            self.logger.error(f"分析板块趋势时出错: {e}")
            return "未知"
            
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        try:
            if prices is None or len(prices) < period:
                return None
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # 处理可能的除零情况
            rs = gain / loss.replace(0, float('inf'))
            rsi = 100 - (100 / (1 + rs))
            
            # 处理无穷大和NaN值
            rsi = rsi.replace([np.inf, -np.inf], [100, 0])
            rsi = rsi.fillna(50)  # 用中性值填充NaN
            
            # 确保RSI值在0-100范围内
            rsi = rsi.clip(0, 100)
            
            return rsi.iloc[-1] if not rsi.empty else None
        except Exception as e:
            self.logger.error(f"计算RSI时出错: {e}")
            return None
        
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.iloc[-1], signal_line.iloc[-1]
        
    def _analyze_wave_band(self, symbol: str, current_price: float) -> List[Dict]:
        """分析当前价格所在的波段"""
        if symbol not in self.wave_bands:
            return []
            
        bands = self.wave_bands[symbol]
        current_band = None
        alerts = []
        
        for band in bands:
            low, high = band['range']
            if low <= current_price <= high:
                current_band = band
                break
                
        if current_band:
            # 生成波段提醒
            if current_band['action'] == 'breakout':
                if current_price >= current_band['target']:
                    alerts.append({
                        'level': 'info',
                        'message': f"{symbol} 突破波段目标价 {current_band['target']}，可以考虑止盈"
                    })
                else:
                    alerts.append({
                        'level': 'info',
                        'message': f"{symbol} 当前在波段 {current_band['range'][0]}-{current_band['range'][1]}，目标价 {current_band['target']}"
                    })
                    
            elif current_band['action'] == 'support_resistance':
                if current_price >= current_band['level']:
                    alerts.append({
                        'level': 'warning',
                        'message': f"{symbol} 接近阻力位 {current_band['level']}，注意回调风险"
                    })
                else:
                    alerts.append({
                        'level': 'info',
                        'message': f"{symbol} 当前在支撑位 {current_band['range'][0]} 和阻力位 {current_band['range'][1]} 之间"
                    })
                    
            elif current_band['action'] == 'stop_loss':
                if current_price <= current_band['level']:
                    alerts.append({
                        'level': 'danger',
                        'message': f"{symbol} 跌破止损位 {current_band['level']}，建议清仓"
                    })
                else:
                    alerts.append({
                        'level': 'warning',
                        'message': f"{symbol} 接近止损位 {current_band['level']}，注意风险"
                    })
                    
        return alerts

    def _analyze_stock(self, data: Dict) -> Dict:
        """分析单个半导体股票"""
        analysis = {}
        
        # 获取当前价格
        current_price = data.get('current_price', 0)
        symbol = data.get('symbol', '')
        
        # 计算技术指标
        rsi = data.get('rsi', 50)
        macd = data.get('macd', 0)
        macd_signal = data.get('macd_signal', 0)
        
        # 判断趋势
        price_change = float(data.get('price_change', '0%').strip('%')) / 100
        trend = "上升" if price_change > 0 else "下降"
        
        # 判断强度
        if rsi > 70:
            strength = "超买"
        elif rsi < 30:
            strength = "超卖"
        else:
            strength = "中性"
            
        # 判断MACD
        macd_status = "金叉" if macd > macd_signal else "死叉"
        
        # 生成交易建议
        recommendations = []
        
        # 分析波段
        wave_band_alerts = self._analyze_wave_band(symbol, current_price)
        for alert in wave_band_alerts:
            recommendations.append(alert['message'])
        
        # 关税利好分析
        if symbol in self.semiconductor_stocks:
            target_price = self.target_prices.get(symbol)
            if target_price and current_price < target_price:
                upside = (target_price - current_price) / current_price * 100
                recommendations.append(f"受益于关税政策利好，目标价{target_price:.2f}美元，上涨空间{upside:.1f}%")
        
        # 止损建议（考虑利好因素，更灵活的止损策略）
        stop_loss = self.stop_loss_levels.get(symbol, 0)
        if stop_loss and current_price < stop_loss * 1.1:  # 接近止损位
            if symbol in ['AMD', 'NVDA']:  # 龙头股
                recommendations.append(f"接近支撑位{stop_loss}美元，可以考虑分批建仓")
            else:
                recommendations.append(f"接近支撑位{stop_loss}美元，建议观望")
                
        # 趋势建议
        if trend == "上升":
            if strength == "超买":
                recommendations.append("短期获利了结一部分，保留核心仓位")
            else:
                recommendations.append("上升趋势确认，可以继续持有")
        elif trend == "下降" and strength == "超卖":
            recommendations.append("超卖区域，可以考虑分批买入")
            
        # MACD建议
        if macd_status == "金叉" and trend == "上升":
            recommendations.append("MACD金叉，可以适度加仓")
        elif macd_status == "死叉" and trend == "下降":
            recommendations.append("MACD死叉，注意控制仓位")
            
        analysis.update({
            'current_price': current_price,
            'trend': trend,
            'strength': strength,
            'rsi': rsi,
            'macd_status': macd_status,
            'target_price': self.target_prices.get(symbol),
            'stop_loss': self.stop_loss_levels.get(symbol),
            'recommendations': recommendations,
            'wave_band_alerts': wave_band_alerts
        })
        
        return analysis
        
    def _analyze_sector_trend(self, sector_analysis: Dict) -> Dict:
        """分析半导体板块整体趋势"""
        trend_analysis = {}
        
        # 统计上涨和下跌的股票数量
        up_count = sum(1 for a in sector_analysis.values() if a.get('trend') == '上升')
        down_count = sum(1 for a in sector_analysis.values() if a.get('trend') == '下降')
        
        # 判断板块整体趋势
        if up_count > down_count * 2:
            trend_analysis['trend'] = "强势"
        elif up_count > down_count:
            trend_analysis['trend'] = "偏强"
        elif down_count > up_count * 2:
            trend_analysis['trend'] = "弱势"
        else:
            trend_analysis['trend'] = "偏弱"
            
        # 统计超买超卖情况
        overbought = sum(1 for a in sector_analysis.values() if a.get('strength') == '超买')
        oversold = sum(1 for a in sector_analysis.values() if a.get('strength') == '超卖')
        
        # 判断板块整体强度
        if overbought > oversold * 2:
            trend_analysis['strength'] = "超买"
        elif overbought > oversold:
            trend_analysis['strength'] = "偏强"
        elif oversold > overbought * 2:
            trend_analysis['strength'] = "超卖"
        else:
            trend_analysis['strength'] = "中性"
            
        # 生成板块建议
        recommendations = []
        
        if trend_analysis['trend'] == "强势" and trend_analysis['strength'] != "超买":
            recommendations.append("板块整体强势，可以适当参与")
        elif trend_analysis['trend'] == "弱势" and trend_analysis['strength'] != "超卖":
            recommendations.append("板块整体弱势，建议谨慎")
        elif trend_analysis['strength'] == "超买":
            recommendations.append("板块整体超买，注意风险")
        elif trend_analysis['strength'] == "超卖":
            recommendations.append("板块整体超卖，可以关注反弹机会")
            
        trend_analysis['recommendations'] = recommendations
        
        return trend_analysis
        
    async def get_trading_signals(self) -> List[Dict]:
        """获取交易信号"""
        try:
            signals = []
            
            # 分析半导体板块
            sector_analysis = await self.analyze_semiconductor_sector()
            
            # 生成个股交易信号
            for symbol, analysis in sector_analysis['stocks'].items():
                signal = {
                    'symbol': symbol,
                    'current_price': analysis['current_price'],
                    'trend': analysis['trend'],
                    'strength': analysis['strength'],
                    'recommendations': analysis['recommendations']
                }
                
                signals.append(signal)
                
            # 添加板块整体信号
            if 'trend' in sector_analysis:
                signals.append({
                    'symbol': 'SEMI_SECTOR',
                    'trend': sector_analysis['trend'],
                    'strength': sector_analysis['strength'],
                    'recommendations': sector_analysis['recommendations']
                })
                
            return signals
            
        except Exception as e:
            logger.error(f"获取交易信号时出错: {str(e)}")
            return [] 