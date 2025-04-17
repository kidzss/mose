import pandas as pd
import numpy as np
import datetime as dt
import time
import logging
from typing import List, Dict, Optional, Union, Tuple
import threading
from pathlib import Path
import json
import os
from sqlalchemy import create_engine
import mysql.connector
import yfinance as yf

from data.data_interface import YahooFinanceRealTimeSource
from monitor.market_monitor import MarketMonitor
from monitor.notification_manager import NotificationManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_time_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealTimeMonitor")

class RealTimeMonitor:
    """实时监控类"""
    def __init__(self, symbols: List[str], config: Dict = None):
        """
        初始化实时监控器
        
        Args:
            symbols: 要监控的股票代码列表
            config: 监控配置
        """
        self.symbols = symbols
        self.config = config or {}
        
        # 初始化组件
        self.data_source = YahooFinanceRealTimeSource()
        self.market_monitor = MarketMonitor()
        self.notification_manager = NotificationManager()
        
        # 初始化数据存储
        self.current_data = {}
        self.historical_data = {}
        self.signals = {}
        self.last_update = None
        
        # 设置默认阈值
        self.alert_thresholds = {
            'price_change': 0.02,      # 价格变动阈值
            'volume_ratio': 2.0,       # 成交量比率阈值
            'volatility_ratio': 1.5,   # 波动率比率阈值
            'rsi_overbought': 70,      # RSI超买阈值
            'rsi_oversold': 30,        # RSI超卖阈值
            'signal_threshold': 0.7     # 信号强度阈值
        }
        
        if config and 'alert_thresholds' in config:
            self.alert_thresholds.update(config['alert_thresholds'])
            
        # 初始化数据
        self._initialize_data()
        
        # 监控状态
        self.is_running = False
        self.monitor_thread = None
        
        # 信号缓存
        self._signal_cache = {}
        
        logger.info("RealTimeMonitor初始化完成")
        
    def start_monitoring(self) -> None:
        """启动监控"""
        if self.is_running:
            logger.warning("监控已经在运行中")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("开始实时监控")
        
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self.is_running:
            logger.warning("监控未在运行")
            return
            
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        logger.info("停止实时监控")
        
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_running:
            try:
                # 获取监控的股票列表
                stocks = self.symbols
                if not stocks:
                    logger.warning("没有需要监控的股票")
                    time.sleep(60)  # 出错后等待一分钟再重试
                    continue
                    
                logger.info(f"检查 {len(stocks)} 只股票的实时数据")
                
                # 获取实时数据并分析
                self._analyze_real_time_data(stocks)
                
                # 等待下一次检查
                time.sleep(60)  # 出错后等待一分钟再重试
                
            except Exception as e:
                logger.error(f"监控循环中出错: {e}")
                time.sleep(60)  # 出错后等待一分钟再重试
                
    def _analyze_real_time_data(self, stocks: List[str]) -> None:
        """分析实时数据"""
        try:
            for symbol in stocks:
                data = self.current_data.get(symbol)
                if data is None or data.empty:
                    continue
                    
                # 标准化数据
                data = self.standardize_data(data)
                if data is None:
                    logger.warning(f"股票 {symbol} 的数据标准化失败")
                    continue
                
                # 计算技术指标
                data = self.calculate_indicators(data)
                if data is None:
                    logger.warning(f"股票 {symbol} 的技术指标计算失败")
                    continue
                
                # 运行策略
                signals = self._run_strategies(data, symbol)
                if signals:
                    self._process_signals(symbol, data, signals)
                    
        except Exception as e:
            logger.error(f"分析实时数据时出错: {e}")
            
    def _run_strategies(self, data: pd.DataFrame, symbol: str) -> Dict:
        """运行策略"""
        try:
            if data is None or data.empty:
                logger.warning(f"股票 {symbol} 的数据为空，无法运行策略")
                return {}
            
            results = {}
            for strategy in self.strategies:
                try:
                    # 验证策略所需的列是否存在
                    required_columns = strategy.get_required_columns()
                    if not all(col in data.columns for col in required_columns):
                        missing = [col for col in required_columns if col not in data.columns]
                        logger.warning(f"策略 {strategy.name} 缺少所需列: {missing}")
                        continue
                        
                    # 生成信号
                    signals = strategy.generate_signals(data.copy())
                    if signals is None or signals.empty:
                        logger.warning(f"策略 {strategy.name} 没有生成信号")
                        continue
                        
                    # 获取最新信号
                    latest_signal = signals['signal'].iloc[-1]
                    
                    results[strategy.name] = {
                        "signal": latest_signal,
                        "data": signals
                    }
                    
                except Exception as e:
                    logger.error(f"运行策略 {strategy.name} 于股票 {symbol} 时出错: {e}")
                    continue
                
            return results
        except Exception as e:
            logger.error(f"运行策略时出错: {e}")
            return {}
            
    def _should_generate_alert(self, signals: Dict, symbol: str) -> bool:
        """判断是否应该生成警报"""
        try:
            # 如果没有信号，返回False
            if not signals:
                return False
                
            # 计算有效信号数量
            valid_signals = sum(1 for result in signals.values() 
                              if abs(result.get("signal", 0)) > 0)
            
            # 检查是否达到信号确认阈值
            if valid_signals >= self.alert_thresholds["signal_threshold"]:
                # 检查是否与缓存中的信号不同
                if symbol not in self._signal_cache:
                    self._signal_cache[symbol] = signals
                    return True
                else:
                    # 比较新旧信号
                    old_signals = self._signal_cache[symbol]
                    has_changes = False
                    
                    for strategy_name, result in signals.items():
                        old_signal = old_signals.get(strategy_name, {}).get("signal", 0)
                        new_signal = result.get("signal", 0)
                        if old_signal != new_signal:
                            has_changes = True
                            break
                            
                    if has_changes:
                        self._signal_cache[symbol] = signals
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"判断是否生成警报时出错: {e}")
            return False
            
    def _generate_alert(self, symbol: str, data: pd.DataFrame, signals: Dict) -> None:
        """生成警报和交易建议"""
        try:
            # 获取最新数据
            latest_data = data.iloc[-1]
            prev_data = data.iloc[-2]
            
            # 计算关键指标
            price_change = (latest_data['Close'] - prev_data['Close']) / prev_data['Close']
            volume_ratio = latest_data['Volume'] / data['Volume'].iloc[-20:].mean()
            
            # 计算波动率
            returns = data['Close'].pct_change()
            current_vol = returns.iloc[-5:].std() * np.sqrt(252)
            historical_vol = returns.iloc[-20:].std() * np.sqrt(252)
            volatility_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            # 计算技术指标
            ma5 = data['Close'].rolling(window=5).mean().iloc[-1]
            ma10 = data['Close'].rolling(window=10).mean().iloc[-1]
            ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
            
            # 计算RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # 确定信号方向和强度
            signal_direction = 0
            signal_strength = 0
            for result in signals.values():
                sig = result.get("signal", 0)
                signal_direction += sig
                signal_strength += abs(sig)
            
            # 生成交易建议
            trade_action = "观望"
            confidence = "低"
            risk_level = "中等"
            
            # 买入条件
            buy_signals = [
                price_change > 0,
                volume_ratio > 1.5,
                ma5 > ma10,
                rsi < 70,
                signal_direction > 0
            ]
            
            # 卖出条件
            sell_signals = [
                price_change < 0,
                volume_ratio > 2.0,
                ma5 < ma10,
                rsi > 30,
                signal_direction < 0
            ]
            
            # 计算信号强度
            buy_strength = sum(buy_signals) / len(buy_signals)
            sell_strength = sum(sell_signals) / len(sell_signals)
            
            # 确定交易建议
            if buy_strength > 0.7 and signal_direction > 0:
                trade_action = "买入"
                confidence = "高" if buy_strength > 0.8 else "中"
            elif sell_strength > 0.7 and signal_direction < 0:
                trade_action = "卖出"
                confidence = "高" if sell_strength > 0.8 else "中"
            
            # 确定风险等级
            if volatility_ratio > 1.5 or abs(price_change) > 0.03:
                risk_level = "高"
            elif volatility_ratio < 0.8 and abs(price_change) < 0.01:
                risk_level = "低"
            
            # 生成警报消息
            message = f"股票 {symbol} 分析报告:\n"
            message += f"交易建议: {trade_action} (置信度: {confidence})\n"
            message += f"风险等级: {risk_level}\n"
            message += f"价格变动: {price_change:.2%}\n"
            message += f"成交量比率: {volume_ratio:.2f}\n"
            message += f"波动率比率: {volatility_ratio:.2f}\n"
            message += f"RSI指标: {rsi:.1f}\n"
            message += "\n技术指标信号:\n"
            
            for strategy_name, result in signals.items():
                signal = result.get("signal", 0)
                if signal != 0:
                    message += f"- {strategy_name}: {'看多' if signal > 0 else '看空'}\n"
            
            # 添加风险提示
            if risk_level == "高":
                message += "\n⚠️ 风险提示：\n"
                if volatility_ratio > 1.5:
                    message += "- 市场波动性显著高于历史水平\n"
                if abs(price_change) > 0.03:
                    message += "- 价格变动幅度较大，建议谨慎操作\n"
                if volume_ratio > 2.0:
                    message += "- 成交量异常放大，可能存在重大信息\n"
            
            # 确定警报级别
            level = "info"
            if risk_level == "高" or abs(signal_direction) >= 2:
                level = "warning"
            
            # 创建警报
            alert = {
                "symbol": symbol,
                "alert_type": "strategy_signal",
                "message": message,
                "level": level,
                "data": {
                    "trade_action": trade_action,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "price_change": price_change,
                    "volume_ratio": volume_ratio,
                    "volatility_ratio": volatility_ratio,
                    "rsi": rsi,
                    "signals": signals
                }
            }
            
            # 添加到市场监控器
            self.market_monitor.add_alert(alert)
            
            logger.info(f"为股票 {symbol} 生成分析报告和交易建议")
            
        except Exception as e:
            logger.error(f"生成警报时出错: {e}")
            
    def update_alert_thresholds(self, new_thresholds: Dict) -> None:
        """更新警报阈值"""
        try:
            self.alert_thresholds.update(new_thresholds)
            logger.info("更新警报阈值成功")
        except Exception as e:
            logger.error(f"更新警报阈值时出错: {e}")
            
    def add_strategy(self, strategy: MonitorStrategy) -> None:
        """添加策略"""
        try:
            self.strategies.append(strategy)
            logger.info(f"添加策略: {strategy.name}")
        except Exception as e:
            logger.error(f"添加策略时出错: {e}")
            
    def remove_strategy(self, strategy_name: str) -> None:
        """移除策略"""
        try:
            self.strategies = [s for s in self.strategies if s.name != strategy_name]
            logger.info(f"移除策略: {strategy_name}")
        except Exception as e:
            logger.error(f"移除策略时出错: {e}")
            
    def get_monitoring_status(self) -> Dict:
        """获取监控状态"""
        try:
            return {
                "is_running": self.is_running,
                "monitored_stocks": len(self.symbols),
                "active_strategies": [s.name for s in self.strategies],
                "alert_thresholds": self.alert_thresholds,
                "last_check_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"获取监控状态时出错: {e}")
            return {}
            
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """验证数据是否包含所需的列"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"数据中缺少以下列: {missing_columns}")
            return False
        return True

    def _get_db_connection(self):
        """获取数据库连接"""
        try:
            # 使用 SQLAlchemy
            engine = create_engine('mysql+pymysql://root@localhost/mose')
            return engine
        except Exception as e:
            logger.error(f"创建数据库连接时出错: {e}")
            raise

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史数据"""
        query = """
        SELECT 
            Date as date,
            Open as open,
            High as high,
            Low as low,
            Close as close,
            Volume as volume,
            AdjClose as adj_close
        FROM stock_code_time
        WHERE Code = %s
        AND Date BETWEEN %s AND %s
        ORDER BY Date ASC
        """
        try:
            df = pd.read_sql_query(query, self._get_db_connection(), params=(symbol, start_date, end_date))
            if df.empty:
                logger.warning(f"获取股票 {symbol} 的数据为空")
                return pd.DataFrame()
            
            # 确保所有列名小写
            df.columns = df.columns.str.lower()
            
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"获取历史数据时出错: {e}")
            return pd.DataFrame()

    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据列名和格式"""
        try:
            if df is None or df.empty:
                return None
            
            # 复制数据框以避免修改原始数据
            df = df.copy()
            
            # 确保列名小写
            df.columns = df.columns.str.lower()
            
            # 标准化必需的列名
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"数据缺少必需的列: {missing_columns}")
                return None
            
            # 确保数值类型正确
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 处理缺失值
            df.dropna(subset=['close'], inplace=True)
            
            if df.empty:
                logger.warning("数据标准化后为空")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"标准化数据时出错: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            if df is None or df.empty:
                return None
            
            # 复制数据框以避免修改原始数据
            df = df.copy()
            
            # 首先处理基础数据中的空值
            df['close'] = df['close'].ffill().bfill()
            df['open'] = df['open'].fillna(df['close'])
            df['high'] = df['high'].fillna(df['close'])
            df['low'] = df['low'].fillna(df['close'])
            df['volume'] = df['volume'].fillna(0)
            
            # 基础移动平均线
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['ma100'] = df['close'].rolling(window=100, min_periods=1).mean()
            
            # 波动率突破策略指标
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
            df['ma'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['std'] = df['close'].rolling(window=20, min_periods=1).std()
            df['upper_band'] = df['ma'] + 2 * df['std']
            df['lower_band'] = df['ma'] - 2 * df['std']
            df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            
            # 价格突破策略指标
            df['high_window'] = df['high'].rolling(window=20, min_periods=1).max()
            df['low_window'] = df['low'].rolling(window=20, min_periods=1).min()
            df['price_change'] = df['close'].pct_change()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 牛牛策略指标
            df['bull_line'] = df['close'].rolling(window=13, min_periods=1).mean()
            
            # TDI策略指标
            df['ma1'] = df['close'].ewm(span=13, min_periods=1).mean()
            
            # 动量指标
            df['momentum'] = df['close'].pct_change(periods=10)
            
            # 市场预测策略指标
            df['momentum_bottom_reversal'] = (
                (df['close'] > df['ma20']) & 
                (df['close'].shift(1) < df['ma20'].shift(1))
            ).astype(int)
            
            # CPGW策略指标
            df['long_line'] = df['close'].rolling(window=50, min_periods=1).mean()
            
            # 最后检查是否还有空值
            null_columns = df.columns[df.isnull().any()].tolist()
            if null_columns:
                logger.warning(f"以下列仍存在空值: {null_columns}")
                # 使用前向填充和后向填充处理剩余的空值
                df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            return None

    def get_latest_data(self, symbols: List[str], days: int = 250) -> Dict[str, pd.DataFrame]:
        """获取最新数据，默认获取250天的数据以确保有足够的历史数据计算指标"""
        try:
            end_date = dt.datetime.now().strftime('%Y-%m-%d')
            start_date = (dt.datetime.now() - dt.timedelta(days=days)).strftime('%Y-%m-%d')
            
            result = {}
            for symbol in symbols:
                try:
                    # 1. 获取原始数据
                    data = self.get_historical_data(symbol, start_date, end_date)
                    if data is None or data.empty:
                        logger.warning(f"获取股票 {symbol} 的数据为空")
                        continue
                    
                    # 2. 标准化数据
                    data = self.standardize_data(data)
                    if data is None:
                        logger.warning(f"股票 {symbol} 的数据标准化失败")
                        continue
                    
                    # 3. 计算技术指标
                    data = self.calculate_indicators(data)
                    if data is None:
                        logger.warning(f"股票 {symbol} 的技术指标计算失败")
                        continue
                    
                    result[symbol] = data
                    
                except Exception as e:
                    logger.error(f"处理股票 {symbol} 的数据时出错: {e}")
                    continue
                
            return result
            
        except Exception as e:
            logger.error(f"获取最新数据时出错: {e}")
            return {}

    def test_data_pipeline(self, symbol: str) -> bool:
        """测试数据处理管道"""
        try:
            # 1. 测试数据获取
            end_date = dt.datetime.now().strftime('%Y-%m-%d')
            start_date = (dt.datetime.now() - dt.timedelta(days=30)).strftime('%Y-%m-%d')
            
            raw_data = self.get_historical_data(symbol, start_date, end_date)
            if raw_data is None or raw_data.empty:
                logger.error("数据获取测试失败")
                return False
            
            logger.info(f"原始数据列: {raw_data.columns.tolist()}")
            
            # 2. 测试数据标准化
            std_data = self.standardize_data(raw_data)
            if std_data is None:
                logger.error("数据标准化测试失败")
                return False
            
            logger.info(f"标准化后的列: {std_data.columns.tolist()}")
            
            # 3. 测试技术指标计算
            tech_data = self.calculate_indicators(std_data)
            if tech_data is None:
                logger.error("技术指标计算测试失败")
                return False
            
            logger.info(f"技术指标列: {tech_data.columns.tolist()}")
            
            # 4. 测试策略运行
            signals = self._run_strategies(tech_data, symbol)
            if not signals:
                logger.error("策略运行测试失败")
                return False
            
            logger.info(f"生成的信号: {signals}")
            
            return True
            
        except Exception as e:
            logger.error(f"数据管道测试失败: {e}")
            return False

    def _process_signals(self, symbol: str, data: pd.DataFrame, signals: Dict) -> None:
        """处理策略信号"""
        try:
            # 检查是否应该生成警报
            if self._should_generate_alert(signals, symbol):
                self._generate_alert(symbol, data, signals)
                
        except Exception as e:
            logger.error(f"处理信号时出错: {e}")

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """计算技术指标"""
        try:
            # 计算移动平均
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # 判断趋势
            current_price = data['Close'].iloc[-1]
            ma20 = data['MA20'].iloc[-1]
            ma50 = data['MA50'].iloc[-1]
            
            if current_price > ma20 > ma50:
                trend = "uptrend"
            elif current_price < ma20 < ma50:
                trend = "downtrend"
            else:
                trend = "sideways"
            
            # 计算支撑位和阻力位
            price_range = data['Close'].iloc[-20:]
            support = price_range.min()
            resistance = price_range.max()
            
            # 计算波动率
            returns = data['Close'].pct_change()
            annualization_factor = np.full(len(data), np.sqrt(252))
            data['volatility'] = returns.rolling(window=20).std().mul(annualization_factor)
            
            return {
                "trend": trend,
                "support_resistance": {
                    "support": support,
                    "resistance": resistance
                },
                "volatility": data['volatility'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            return {"trend": "unknown", "support_resistance": {}, "volatility": None}

    def _initialize_data(self):
        """初始化数据"""
        try:
            # 获取最新数据
            self.current_data = self.get_latest_data(self.symbols)
            self.historical_data = {symbol: self.get_historical_data(symbol, '2020-01-01', '2023-04-30') for symbol in self.symbols}
            self.signals = {symbol: self._run_strategies(data, symbol) for symbol, data in self.current_data.items()}
            self.last_update = dt.datetime.now()
            
            logger.info("数据初始化完成")
        except Exception as e:
            logger.error(f"初始化数据时出错: {e}")