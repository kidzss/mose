#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘中实时交易监控系统
专业级股票盘中监控、信号识别与交易决策系统
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_interface import DataInterface
from monitor.stock_type_analyzer import StockTypeAnalyzer
from utils.portfolio_config_loader import get_portfolio_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntradayTradingMonitor")

class TradingSignal(Enum):
    """交易信号枚举"""
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"
    WAIT = "观望"

class BreakthroughType(Enum):
    """突破类型枚举"""
    RESISTANCE_BREAK = "阻力突破"
    SUPPORT_BREAK = "支撑突破"
    VOLUME_SPIKE = "成交量激增"
    RSI_DIVERGENCE = "RSI背离"
    TREND_REVERSAL = "趋势反转"
    CONSOLIDATION = "盘整突破"

@dataclass
class IntradaySignal:
    """盘中信号数据类"""
    symbol: str
    timestamp: datetime
    signal_type: TradingSignal
    breakthrough_type: Optional[BreakthroughType]
    confidence: float  # 信心度 0-1
    price: float
    volume: int
    rsi: float
    macd: Tuple[float, float, float]  # (MACD, Signal, Histogram)
    reason: str
    suggested_action: str
    risk_level: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

class IntradayTradingMonitor:
    """盘中实时交易监控系统"""
    
    def __init__(self, config_file: str = "intraday_config.json"):
        """
        初始化盘中监控系统
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.data_interface = DataInterface()
        self.stock_analyzer = StockTypeAnalyzer()
        self.portfolio_config = get_portfolio_config()
        
        # 监控状态
        self.is_monitoring = False
        self.last_check_time = {}
        self.signal_history = []
        self.price_alerts = {}
        
        logger.info("盘中交易监控系统初始化完成")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        default_config = {
            "hot_stocks": ["AMD", "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL"],
            "monitoring_interval": 300,  # 5分钟检查一次
            "market_hours": {
                "open": "09:30",
                "close": "16:00",
                "timezone": "US/Eastern"
            },
            "signal_thresholds": {
                "volume_spike_ratio": 1.5,  # 成交量激增阈值
                "price_change_threshold": 0.02,  # 价格变动阈值2%
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "rsi_extreme_overbought": 80,
                "rsi_extreme_oversold": 20
            },
            "risk_management": {
                "max_position_size": 0.10,  # 最大单股仓位10%
                "stop_loss_pct": 0.08,  # 止损8%
                "take_profit_pct": 0.15,  # 止盈15%
                "max_daily_trades": 5  # 每日最大交易次数
            },
            "notifications": {
                "enable_alerts": True,
                "alert_channels": ["console", "file"],
                "urgent_threshold": 0.8  # 紧急信号阈值
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            else:
                # 创建默认配置文件
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                logger.info(f"创建默认配置文件: {self.config_file}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
        
        return default_config
    
    def is_market_open(self) -> bool:
        """检查市场是否开盘"""
        now = datetime.now()
        # 简化判断，假设工作日9:30-16:00为交易时间
        if now.weekday() >= 5:  # 周末
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_realtime_data(self, symbol: str) -> Optional[Dict]:
        """获取实时股票数据"""
        try:
            # 获取最近的数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            data = self.data_interface.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='daily'
            )
            
            if data is None or data.empty:
                return None
            
            # 获取最新数据
            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else latest
            
            # 计算技术指标
            data = self._add_technical_indicators(data)
            latest_with_indicators = data.iloc[-1]
            
            return {
                'symbol': symbol,
                'price': float(latest['close']),
                'volume': int(latest['volume']),
                'change': float(latest['close'] - prev['close']),
                'change_pct': float((latest['close'] - prev['close']) / prev['close'] * 100),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'rsi': float(latest_with_indicators.get('rsi', 50)),
                'macd': float(latest_with_indicators.get('macd', 0)),
                'macd_signal': float(latest_with_indicators.get('macd_signal', 0)),
                'macd_hist': float(latest_with_indicators.get('macd_hist', 0)),
                'sma_20': float(latest_with_indicators.get('sma_20', latest['close'])),
                'sma_50': float(latest_with_indicators.get('sma_50', latest['close'])),
                'volume_avg': float(data['volume'].tail(20).mean()),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"获取{symbol}实时数据失败: {e}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        try:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # 移动平均线
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            
            return data
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return data
    
    def analyze_amd_breakthrough(self, current_data: Dict) -> Dict:
        """专门分析AMD的突破情况"""
        analysis = {
            'is_breakthrough': False,
            'breakthrough_type': None,
            'confidence': 0.0,
            'key_levels': {},
            'risk_warnings': [],
            'opportunities': []
        }
        
        try:
            price = current_data['price']
            rsi = current_data['rsi']
            volume_ratio = current_data['volume'] / current_data['volume_avg']
            change_pct = current_data['change_pct']
            
            # 关键价位分析（基于您提供的数据）
            target_price = 130.34  # 分析师目标价
            resistance_levels = [130.0, 135.0, 140.0]
            support_levels = [125.0, 120.0, 115.0]
            
            analysis['key_levels'] = {
                'target_price': target_price,
                'distance_to_target': (target_price - price) / price * 100,
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'rsi_level': rsi,
                'volume_ratio': volume_ratio
            }
            
            # 突破分析
            if price >= target_price * 0.995:  # 接近目标价0.5%以内
                analysis['is_breakthrough'] = True
                analysis['breakthrough_type'] = 'TARGET_PRICE_TEST'
                analysis['confidence'] = 0.8
                analysis['opportunities'].append("接近分析师目标价，突破后上涨空间打开")
                
            if rsi >= 70:
                analysis['risk_warnings'].append(f"RSI超买({rsi:.1f})，短期回调风险增加")
                
            if volume_ratio >= 1.5:
                analysis['opportunities'].append(f"成交量放大{volume_ratio:.1f}倍，有资金关注")
                analysis['confidence'] += 0.2
                
            if change_pct >= 1:
                analysis['opportunities'].append(f"价格上涨{change_pct:.1f}%，趋势向好")
                
            # 综合判断
            if analysis['is_breakthrough']:
                if rsi > 75:
                    analysis['risk_warnings'].append("超买严重，建议等待回调")
                elif rsi > 70:
                    analysis['risk_warnings'].append("技术面过热，谨慎追高")
                    
        except Exception as e:
            logger.error(f"分析AMD突破失败: {e}")
            
        return analysis
    
    def monitor_single_stock(self, symbol: str) -> Optional[IntradaySignal]:
        """监控单只股票"""
        try:
            # 获取实时数据
            current_data = self.get_realtime_data(symbol)
            if not current_data:
                return None
            
            # 特殊处理AMD
            if symbol == "AMD":
                amd_analysis = self.analyze_amd_breakthrough(current_data)
                return self._generate_amd_signal(current_data, amd_analysis)
            
            # 其他股票的通用处理
            return self._generate_general_signal(symbol, current_data)
            
        except Exception as e:
            logger.error(f"监控{symbol}失败: {e}")
            return None
    
    def _generate_amd_signal(self, data: Dict, analysis: Dict) -> Optional[IntradaySignal]:
        """生成AMD专用信号"""
        try:
            confidence = analysis['confidence']
            price = data['price']
            rsi = data['rsi']
            
            # 基于分析结果确定信号
            if analysis['is_breakthrough'] and confidence >= 0.6:
                if len(analysis['risk_warnings']) >= 2:
                    signal_type = TradingSignal.HOLD
                    action = "目标价附近，RSI超买，建议等待回调至$125-127再加仓"
                elif len(analysis['risk_warnings']) == 1:
                    signal_type = TradingSignal.BUY
                    action = "接近目标价，可小幅加仓，严格止损"
                else:
                    signal_type = TradingSignal.STRONG_BUY
                    action = "突破目标价，可适量加仓"
            else:
                signal_type = TradingSignal.HOLD
                action = "维持现有仓位，观察是否突破关键位"
            
            # 组合原因
            reasons = []
            reasons.extend(analysis['opportunities'])
            reasons.extend([f"⚠️ {w}" for w in analysis['risk_warnings']])
            
            reason = " | ".join(reasons) if reasons else "常规监控"
            
            return IntradaySignal(
                symbol="AMD",
                timestamp=data['timestamp'],
                signal_type=signal_type,
                breakthrough_type=BreakthroughType.RESISTANCE_BREAK if analysis['is_breakthrough'] else None,
                confidence=confidence,
                price=price,
                volume=data['volume'],
                rsi=rsi,
                macd=(data['macd'], data['macd_signal'], data['macd_hist']),
                reason=reason,
                suggested_action=action,
                risk_level="MEDIUM",
                target_price=135.0 if signal_type in [TradingSignal.BUY, TradingSignal.STRONG_BUY] else None,
                stop_loss=price * 0.92 if signal_type in [TradingSignal.BUY, TradingSignal.STRONG_BUY] else None
            )
            
        except Exception as e:
            logger.error(f"生成AMD信号失败: {e}")
            return None
    
    def _generate_general_signal(self, symbol: str, data: Dict) -> Optional[IntradaySignal]:
        """生成通用股票信号"""
        # 简化版本，主要用于其他股票
        price = data['price']
        change_pct = data['change_pct']
        rsi = data['rsi']
        
        if change_pct > 3 and rsi < 70:
            signal_type = TradingSignal.BUY
        elif change_pct > 1:
            signal_type = TradingSignal.HOLD
        elif change_pct < -3:
            signal_type = TradingSignal.SELL
        else:
            return None  # 无明确信号
        
        return IntradaySignal(
            symbol=symbol,
            timestamp=data['timestamp'],
            signal_type=signal_type,
            breakthrough_type=None,
            confidence=0.6,
            price=price,
            volume=data['volume'],
            rsi=rsi,
            macd=(data['macd'], data['macd_signal'], data['macd_hist']),
            reason=f"价格变动{change_pct:.1f}%",
            suggested_action=f"{signal_type.value}信号",
            risk_level="MEDIUM"
        )
    
    def start_monitoring(self):
        """开始监控"""
        logger.info("开始盘中监控...")
        self.is_monitoring = True
        
        while self.is_monitoring:
            try:
                if not self.is_market_open():
                    logger.info("市场未开盘，等待...")
                    time.sleep(300)  # 5分钟后再检查
                    continue
                
                logger.info(f"开始监控热门股票: {self.config['hot_stocks']}")
                
                for symbol in self.config['hot_stocks']:
                    signal = self.monitor_single_stock(symbol)
                    if signal:
                        self._handle_signal(signal)
                
                time.sleep(self.config['monitoring_interval'])
                
            except KeyboardInterrupt:
                logger.info("用户停止监控")
                break
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(60)
    
    def _handle_signal(self, signal: IntradaySignal):
        """处理交易信号"""
        try:
            # 记录信号
            logger.info(f"🎯 {signal.symbol} 交易信号: {signal.signal_type.value}")
            logger.info(f"   价格: ${signal.price:.2f} | 信心度: {signal.confidence:.1%}")
            logger.info(f"   原因: {signal.reason}")
            logger.info(f"   建议: {signal.suggested_action}")
            
            # 紧急信号处理
            if signal.confidence >= self.config['notifications']['urgent_threshold']:
                self._send_urgent_alert(signal)
            
            # 保存到文件
            self._save_signal_to_file(signal)
            
        except Exception as e:
            logger.error(f"处理信号失败: {e}")
    
    def _send_urgent_alert(self, signal: IntradaySignal):
        """发送紧急警报"""
        alert_msg = f"🚨 紧急交易信号 🚨\n"
        alert_msg += f"股票: {signal.symbol}\n"
        alert_msg += f"信号: {signal.signal_type.value}\n"
        alert_msg += f"价格: ${signal.price:.2f}\n"
        alert_msg += f"信心度: {signal.confidence:.1%}\n"
        alert_msg += f"原因: {signal.reason}\n"
        alert_msg += f"建议: {signal.suggested_action}"
        
        print("\n" + "="*50)
        print(alert_msg)
        print("="*50 + "\n")
    
    def _save_signal_to_file(self, signal: IntradaySignal):
        """保存信号到文件"""
        try:
            signal_data = {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'price': signal.price,
                'reason': signal.reason,
                'suggested_action': signal.suggested_action
            }
            
            filename = f"intraday_signals_{datetime.now().strftime('%Y%m%d')}.json"
            
            signals = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    signals = json.load(f)
            
            signals.append(signal_data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(signals, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存信号失败: {e}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        logger.info("监控已停止")
    
    def get_today_signals(self) -> List[Dict]:
        """获取今日信号"""
        try:
            filename = f"intraday_signals_{datetime.now().strftime('%Y%m%d')}.json"
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"获取今日信号失败: {e}")
            return []

# 使用示例
if __name__ == "__main__":
    monitor = IntradayTradingMonitor()
    
    # 测试单次监控
    print("=== AMD 实时突破分析 ===")
    signal = monitor.monitor_single_stock("AMD")
    if signal:
        print(f"🎯 信号类型: {signal.signal_type.value}")
        print(f"📊 信心度: {signal.confidence:.1%}")
        print(f"💰 当前价格: ${signal.price:.2f}")
        print(f"📈 RSI: {signal.rsi:.1f}")
        print(f"🔍 分析原因: {signal.reason}")
        print(f"💡 操作建议: {signal.suggested_action}")
        if signal.target_price:
            print(f"🎯 目标价: ${signal.target_price:.2f}")
        if signal.stop_loss:
            print(f"🛡️ 止损价: ${signal.stop_loss:.2f}")
    else:
        print("❌ 暂无明确交易信号")
    
    # 如果需要持续监控，取消下面的注释
    # monitor.start_monitoring() 