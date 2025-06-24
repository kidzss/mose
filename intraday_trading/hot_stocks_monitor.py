#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热门股票实时监控核心模块
专业级盘中交易信号监控系统
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import yfinance as yf
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HotStocksMonitor")

@dataclass
class TradingSignal:
    """交易信号数据类"""
    symbol: str
    timestamp: datetime
    signal_type: str  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    confidence: float  # 0-1
    price: float
    price_change_pct: float
    volume_ratio: float
    rsi: float
    breakthrough_type: Optional[str]
    reason: str
    suggested_action: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

class HotStocksMonitor:
    """热门股票实时监控器"""
    
    def __init__(self, config_file: str = "hot_stocks_config.json"):
        """初始化监控器"""
        self.config_file = config_file
        self.config = self._load_config()
        self.is_monitoring = False
        self.signal_history = []
        self.last_alerts = {}  # 防止重复警报
        
        logger.info("热门股票监控器初始化完成")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        default_config = {
            "hot_stocks": {
                "AMD": {
                    "type": "growth_stock",
                    "priority": "HIGH",
                    "target_price": 130.34,
                    "resistance_levels": [130.0, 135.0, 140.0],
                    "support_levels": [125.0, 120.0, 115.0],
                    "analysis_weights": {"technical": 0.35, "fundamental": 0.55, "volume": 0.10},
                    "alert_thresholds": {
                        "price_change": 0.03,
                        "volume_spike": 1.5,
                        "rsi_extreme": [20, 80],
                        "breakthrough_confidence": 0.7
                    }
                },
                "TSLA": {
                    "type": "wave_trading_stock", 
                    "priority": "HIGH",
                    "resistance_levels": [350.0, 400.0, 450.0],
                    "support_levels": [300.0, 250.0, 200.0],
                    "analysis_weights": {"technical": 0.50, "fundamental": 0.40, "sentiment": 0.10},
                    "alert_thresholds": {
                        "price_change": 0.05,
                        "volume_spike": 2.0,
                        "rsi_extreme": [15, 85]
                    }
                },
                "NVDA": {
                    "type": "growth_stock",
                    "priority": "HIGH", 
                    "resistance_levels": [150.0, 160.0, 170.0],
                    "support_levels": [140.0, 130.0, 120.0],
                    "analysis_weights": {"technical": 0.40, "fundamental": 0.50, "volume": 0.10}
                }
            },
            "monitoring": {
                "interval_seconds": 300,  # 5分钟检查一次
                "market_hours": {"open": "09:30", "close": "16:00"},
                "min_confidence": 0.6,  # 最低信心度阈值
                "max_alerts_per_hour": 3  # 每小时最大警报数
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并配置
                    for key, value in user_config.items():
                        if key in default_config:
                            if isinstance(value, dict):
                                default_config[key].update(value)
                            else:
                                default_config[key] = value
            else:
                # 创建默认配置文件
                os.makedirs(os.path.dirname(self.config_file) if os.path.dirname(self.config_file) else ".", exist_ok=True)
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                logger.info(f"创建默认配置文件: {self.config_file}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
        
        return default_config
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """获取股票实时数据"""
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取最近数据
            data = ticker.history(period="5d", interval="1d")
            if data.empty:
                return None
            
            # 计算技术指标
            data = self._calculate_indicators(data)
            
            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else latest
            
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'volume': int(latest['Volume']),
                'change': float(latest['Close'] - prev['Close']),
                'change_pct': float((latest['Close'] - prev['Close']) / prev['Close'] * 100),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'rsi': float(latest.get('RSI', 50)),
                'macd': float(latest.get('MACD', 0)),
                'volume_avg': float(data['Volume'].tail(20).mean()),
                'sma_20': float(latest.get('SMA_20', latest['Close'])),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"获取{symbol}数据失败: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            
            # 移动平均线
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            
            return data
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return data
    
    def analyze_stock(self, symbol: str, stock_data: Dict) -> Optional[TradingSignal]:
        """分析单只股票并生成信号"""
        try:
            stock_config = self.config['hot_stocks'].get(symbol, {})
            if not stock_config:
                return None
            
            price = stock_data['price']
            change_pct = stock_data['change_pct']
            volume_ratio = stock_data['volume'] / stock_data['volume_avg'] if stock_data['volume_avg'] > 0 else 1
            rsi = stock_data['rsi']
            
            # 分析结果
            analysis = {
                'opportunities': [],
                'risks': [],
                'confidence': 0.0,
                'signal_type': 'HOLD',
                'breakthrough_type': None
            }
            
            # 1. 价格突破分析
            resistance_levels = stock_config.get('resistance_levels', [])
            target_price = stock_config.get('target_price')
            
            if target_price and price >= target_price * 0.995:
                analysis['breakthrough_type'] = 'TARGET_APPROACH'
                analysis['opportunities'].append(f"接近目标价${target_price:.2f}")
                analysis['confidence'] += 0.3
                
                if price > target_price:
                    analysis['breakthrough_type'] = 'TARGET_BREAK'
                    analysis['opportunities'].append("突破目标价！")
                    analysis['confidence'] += 0.4
            
            # 检查阻力位突破
            for resistance in resistance_levels:
                if price > resistance * 1.005:  # 突破阻力位0.5%以上
                    analysis['breakthrough_type'] = 'RESISTANCE_BREAK'
                    analysis['opportunities'].append(f"突破阻力位${resistance:.2f}")
                    analysis['confidence'] += 0.2
                    break
            
            # 2. RSI分析
            rsi_extreme = stock_config.get('alert_thresholds', {}).get('rsi_extreme', [20, 80])
            if rsi >= rsi_extreme[1]:
                analysis['risks'].append(f"RSI超买({rsi:.1f})")
                if rsi >= 75:
                    analysis['confidence'] -= 0.2
            elif rsi <= rsi_extreme[0]:
                analysis['opportunities'].append(f"RSI超卖({rsi:.1f})")
                analysis['confidence'] += 0.3
            
            # 3. 成交量分析
            volume_threshold = stock_config.get('alert_thresholds', {}).get('volume_spike', 1.5)
            if volume_ratio >= volume_threshold:
                analysis['opportunities'].append(f"成交量放大{volume_ratio:.1f}倍")
                analysis['confidence'] += 0.2
            
            # 4. 价格变动分析
            price_threshold = stock_config.get('alert_thresholds', {}).get('price_change', 0.03)
            if abs(change_pct) >= price_threshold * 100:
                if change_pct > 0:
                    analysis['opportunities'].append(f"价格上涨{change_pct:.1f}%")
                    analysis['confidence'] += 0.1
                else:
                    analysis['risks'].append(f"价格下跌{change_pct:.1f}%")
            
            # 5. 确定交易信号
            risk_count = len(analysis['risks'])
            opportunity_count = len(analysis['opportunities'])
            
            if analysis['confidence'] >= 0.8 and opportunity_count > risk_count:
                analysis['signal_type'] = 'STRONG_BUY'
            elif analysis['confidence'] >= 0.6 and opportunity_count > 0:
                analysis['signal_type'] = 'BUY'
            elif analysis['confidence'] >= 0.6 and risk_count > opportunity_count:
                analysis['signal_type'] = 'SELL'
            elif risk_count >= 2:
                analysis['signal_type'] = 'REDUCE'
            else:
                analysis['signal_type'] = 'HOLD'
            
            # 生成操作建议
            suggested_action = self._generate_action_suggestion(symbol, analysis, stock_config, price)
            
            # 组合原因
            all_points = analysis['opportunities'] + [f"⚠️ {r}" for r in analysis['risks']]
            reason = " | ".join(all_points) if all_points else "常规监控"
            
            # 只返回高置信度信号
            min_confidence = self.config['monitoring']['min_confidence']
            if analysis['confidence'] >= min_confidence:
                return TradingSignal(
                    symbol=symbol,
                    timestamp=stock_data['timestamp'],
                    signal_type=analysis['signal_type'],
                    confidence=analysis['confidence'],
                    price=price,
                    price_change_pct=change_pct,
                    volume_ratio=volume_ratio,
                    rsi=rsi,
                    breakthrough_type=analysis['breakthrough_type'],
                    reason=reason,
                    suggested_action=suggested_action,
                    target_price=self._calculate_target_price(symbol, price, stock_config),
                    stop_loss=self._calculate_stop_loss(symbol, price, stock_config)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"分析{symbol}失败: {e}")
            return None
    
    def _generate_action_suggestion(self, symbol: str, analysis: Dict, stock_config: Dict, price: float) -> str:
        """生成操作建议"""
        suggestions = []
        signal_type = analysis['signal_type']
        stock_type = stock_config.get('type', 'growth_stock')
        
        if signal_type == 'STRONG_BUY':
            suggestions.append("强烈建议买入")
            if stock_type == 'wave_trading_stock':
                suggestions.append("适合短线操作，严格止损")
            else:
                suggestions.append("可加仓2-4股")
        elif signal_type == 'BUY':
            suggestions.append("可以买入")
            suggestions.append("建议小幅加仓1-2股")
        elif signal_type == 'HOLD':
            suggestions.append("维持现有仓位")
            suggestions.append("继续观察市场变化")
        elif signal_type in ['SELL', 'REDUCE']:
            suggestions.append("考虑减仓")
            if len(analysis['risks']) >= 2:
                suggestions.append("风险较高，谨慎操作")
        
        return " | ".join(suggestions)
    
    def _calculate_target_price(self, symbol: str, current_price: float, stock_config: Dict) -> Optional[float]:
        """计算目标价"""
        resistance_levels = stock_config.get('resistance_levels', [])
        for resistance in resistance_levels:
            if resistance > current_price:
                return resistance
        return None
    
    def _calculate_stop_loss(self, symbol: str, current_price: float, stock_config: Dict) -> Optional[float]:
        """计算止损价"""
        stock_type = stock_config.get('type', 'growth_stock')
        if stock_type == 'wave_trading_stock':
            return current_price * 0.95  # 5%止损
        else:
            return current_price * 0.92  # 8%止损
    
    def monitor_once(self) -> List[TradingSignal]:
        """执行一次监控"""
        signals = []
        
        for symbol in self.config['hot_stocks'].keys():
            try:
                # 获取数据
                stock_data = self.get_stock_data(symbol)
                if not stock_data:
                    continue
                
                # 分析信号
                signal = self.analyze_stock(symbol, stock_data)
                if signal:
                    # 检查是否需要警报（避免重复）
                    if self._should_alert(signal):
                        signals.append(signal)
                        self.signal_history.append(signal)
                        self._send_alert(signal)
                        
            except Exception as e:
                logger.error(f"监控{symbol}失败: {e}")
        
        return signals
    
    def _should_alert(self, signal: TradingSignal) -> bool:
        """判断是否应该发送警报"""
        now = datetime.now()
        symbol = signal.symbol
        
        # 检查是否在最近1小时内已经发送过类似警报
        if symbol in self.last_alerts:
            last_time = self.last_alerts[symbol]['time']
            if (now - last_time).seconds < 3600:  # 1小时内
                return False
        
        # 检查每小时警报限制
        hour_signals = [s for s in self.signal_history 
                       if (now - s.timestamp).seconds < 3600]
        max_alerts = self.config['monitoring']['max_alerts_per_hour']
        
        if len(hour_signals) >= max_alerts:
            return False
        
        return True
    
    def _send_alert(self, signal: TradingSignal):
        """发送警报"""
        try:
            # 更新最后警报时间
            self.last_alerts[signal.symbol] = {
                'time': signal.timestamp,
                'signal_type': signal.signal_type
            }
            
            # 控制台输出
            self._print_alert(signal)
            
            # 保存到文件
            self._save_signal(signal)
            
        except Exception as e:
            logger.error(f"发送警报失败: {e}")
    
    def _print_alert(self, signal: TradingSignal):
        """打印警报到控制台"""
        print("\n" + "="*60)
        print(f"🚨 {signal.symbol} 交易信号警报 🚨")
        print("="*60)
        print(f"📊 当前价格: ${signal.price:.2f} ({signal.price_change_pct:+.2f}%)")
        print(f"🎯 信号类型: {signal.signal_type}")
        print(f"📈 信心度: {signal.confidence:.1%}")
        print(f"📊 RSI: {signal.rsi:.1f}")
        print(f"📈 成交量: {signal.volume_ratio:.1f}x")
        
        if signal.breakthrough_type:
            print(f"⚡ 突破类型: {signal.breakthrough_type}")
        
        print(f"🔍 分析原因: {signal.reason}")
        print(f"💡 操作建议: {signal.suggested_action}")
        
        if signal.target_price:
            print(f"🎯 目标价: ${signal.target_price:.2f}")
        if signal.stop_loss:
            print(f"🛡️ 止损价: ${signal.stop_loss:.2f}")
        
        print(f"⏰ 时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
    
    def _save_signal(self, signal: TradingSignal):
        """保存信号到文件"""
        try:
            filename = f"signals_{datetime.now().strftime('%Y%m%d')}.json"
            
            signal_data = {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'price': signal.price,
                'price_change_pct': signal.price_change_pct,
                'volume_ratio': signal.volume_ratio,
                'rsi': signal.rsi,
                'breakthrough_type': signal.breakthrough_type,
                'reason': signal.reason,
                'suggested_action': signal.suggested_action,
                'target_price': signal.target_price,
                'stop_loss': signal.stop_loss
            }
            
            # 读取现有信号
            signals = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    signals = json.load(f)
            
            signals.append(signal_data)
            
            # 保存
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(signals, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存信号失败: {e}")
    
    def start_monitoring(self):
        """开始持续监控"""
        logger.info("开始热门股票实时监控...")
        self.is_monitoring = True
        
        try:
            while self.is_monitoring:
                logger.info("执行监控检查...")
                signals = self.monitor_once()
                
                if signals:
                    logger.info(f"检测到{len(signals)}个交易信号")
                else:
                    logger.info("未检测到交易信号")
                
                # 等待下次检查
                interval = self.config['monitoring']['interval_seconds']
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("用户停止监控")
        except Exception as e:
            logger.error(f"监控过程出错: {e}")
        finally:
            self.is_monitoring = False
            logger.info("监控已停止")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False

# 使用示例
if __name__ == "__main__":
    # 创建监控器
    monitor = HotStocksMonitor()
    
    print("🎯 热门股票实时监控系统")
    print("支持股票: AMD, TSLA, NVDA")
    print("=" * 50)
    
    # 执行一次监控测试
    print("执行一次监控测试...")
    signals = monitor.monitor_once()
    
    if not signals:
        print("✅ 当前无交易信号")
    
    # 询问是否开始持续监控
    start_continuous = input("\n是否开始持续监控? (y/N): ").lower().strip()
    if start_continuous == 'y':
        print("开始持续监控，按 Ctrl+C 停止...")
        monitor.start_monitoring()
    else:
        print("监控结束") 