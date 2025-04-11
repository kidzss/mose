import logging
import pandas as pd
import talib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any
from strategy_optimizer.models.strategy_optimizer import StrategyOptimizer
from strategy_optimizer.data_processors.data_processor import DataProcessor
from strategy_optimizer.utils.config_loader import load_config

logger = logging.getLogger(__name__)

class AlertSystem:
    """简单的邮件提醒系统"""
    def __init__(self, email_config: Dict[str, Any]):
        self.email_config = email_config
        
    def send_email(self, subject: str, body: str):
        """发送邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['receiver_email']
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(
                    self.email_config['sender_email'],
                    self.email_config['sender_password']
                )
                server.send_message(msg)

            logger.info(f"成功发送邮件: {subject}")
        except Exception as e:
            logger.error(f"发送邮件时出错: {str(e)}")

class StrategyOptimizedAlert:
    """策略优化预警类"""
    
    def __init__(self, model_path: str, config_path: str, email_config: Dict[str, Any]):
        """
        初始化预警系统
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
            email_config: 邮件配置
        """
        self.config = load_config(config_path)
        self.model = StrategyOptimizer(self.config)
        self.model.load_model(model_path)
        self.data_processor = DataProcessor()
        self.alert_system = AlertSystem(email_config)
        
    def _calculate_composite_signal(self, strategy_weights: dict, strategy_signals: dict) -> tuple:
        """计算综合信号
        
        Args:
            strategy_weights: 策略权重字典
            strategy_signals: 策略信号字典
        
        Returns:
            tuple: (综合信号值, 建议操作)
        """
        # 计算综合信号
        composite_signal = 0.0
        for strategy, weight_str in strategy_weights.items():
            # 从百分比字符串中提取数值
            weight = float(weight_str.strip('%')) / 100
            
            # 获取对应的信号值
            signal_key = strategy.replace('策略', '')
            signal_value = strategy_signals.get(signal_key.lower(), 0)
            
            # 累加加权信号
            composite_signal += weight * signal_value
        
        # 根据综合信号确定建议
        if composite_signal >= 0.5:
            action = "强烈买入"
            signal_class = "strong-buy"
        elif composite_signal >= 0.2:
            action = "建议买入"
            signal_class = "buy"
        elif composite_signal >= 0.1:
            action = "观望偏多"
            signal_class = "watch-buy"
        elif composite_signal >= -0.1:
            action = "观望"
            signal_class = "neutral"
        elif composite_signal >= -0.2:
            action = "观望偏空"
            signal_class = "watch-sell"
        elif composite_signal >= -0.5:
            action = "建议卖出"
            signal_class = "sell"
        else:
            action = "强烈卖出"
            signal_class = "strong-sell"
        
        return composite_signal, action, signal_class

    def generate_alert(self, symbol: str) -> None:
        """生成预警"""
        try:
            # 获取最新数据
            latest_data = self.data_processor.get_stock_data(
                symbol,
                (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            if latest_data.empty:
                logger.warning(f"无法获取股票 {symbol} 的数据")
                return
            
            # 准备特征
            features, feature_columns = self.data_processor.prepare_features(latest_data)
            
            if len(features) == 0:
                logger.warning(f"无法为股票 {symbol} 准备特征")
                return
            
            # 预测策略权重
            weights = self.model.predict(features)
            
            if weights is None or len(weights) == 0:
                logger.warning(f"无法为股票 {symbol} 预测策略权重")
                return
            
            # 生成综合信号
            signals = self._calculate_signals(latest_data, weights[-1])
            
            if signals is None:
                logger.warning(f"无法为股票 {symbol} 计算信号")
                return
            
            # 生成HTML格式的邮件内容
            latest_price = latest_data.iloc[-1]['close']
            prev_price = latest_data.iloc[-2]['close']
            price_change = (latest_price - prev_price) / prev_price * 100
            
            email_body = f"""
            <html>
            <head>
                <style>
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin-bottom: 20px;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    .positive {{
                        color: green;
                    }}
                    .negative {{
                        color: red;
                    }}
                    .neutral {{
                        color: gray;
                    }}
                    .strong-buy {{
                        color: darkgreen;
                        font-weight: bold;
                    }}
                    .buy {{
                        color: green;
                    }}
                    .watch-buy {{
                        color: lightgreen;
                    }}
                    .watch-sell {{
                        color: pink;
                    }}
                    .sell {{
                        color: red;
                    }}
                    .strong-sell {{
                        color: darkred;
                        font-weight: bold;
                    }}
                    .signal-summary {{
                        font-size: 1.2em;
                        margin: 20px 0;
                        padding: 10px;
                        border: 1px solid #ddd;
                        background-color: #f9f9f9;
                    }}
                </style>
            </head>
            <body>
                <h2>交易策略预警 - {symbol}</h2>
                <p>最新价格: {latest_price:.2f} ({'+' if price_change >= 0 else ''}{price_change:.2f}%)</p>
                
                <div class="signal-summary">
                    <h3>综合信号分析</h3>
                    {self._calculate_composite_signal(signals['strategy_weights'], signals)[1]}
                    (信号强度: {self._calculate_composite_signal(signals['strategy_weights'], signals)[0]:.3f})
                </div>

                <h3>策略权重分布</h3>
                <table>
                    <tr>
                        <th>策略</th>
                        <th>权重</th>
                    </tr>
            """
            
            # 添加策略权重
            for strategy, weight in signals['strategy_weights'].items():
                email_body += f"""
                    <tr>
                        <td>{strategy}</td>
                        <td>{weight}</td>
                    </tr>
                """
            
            email_body += """
                </table>
                
                <h3>策略信号</h3>
                <table>
                    <tr>
                        <th>策略</th>
                        <th>信号</th>
                    </tr>
            """
            
            # 添加策略信号
            strategy_signals = {
                'GoldTriangle策略': signals['gold_triangle'],
                'Momentum策略': signals['momentum'],
                'Niuniu策略': signals['niuniu'],
                'TDI策略': signals['tdi'],
                'MarketForecast策略': signals['market_forecast'],
                'CPGW策略': signals['cpgw']
            }
            
            for strategy, signal in strategy_signals.items():
                signal_class = 'positive' if signal > 0.3 else ('negative' if signal < -0.3 else 'neutral')
                signal_text = '买入' if signal > 0.3 else ('卖出' if signal < -0.3 else '观望')
                email_body += f"""
                    <tr>
                        <td>{strategy}</td>
                        <td class="{signal_class}">{signal_text} ({signal:.2f})</td>
                    </tr>
                """
            
            email_body += """
                </table>
                
                <h3>技术指标</h3>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>值</th>
                    </tr>
            """
            
            # 添加技术指标
            indicators = {
                'RSI': signals['rsi'],
                'MACD': signals['macd'],
                'SMA趋势': '上升' if signals['sma_trend'] > 0 else '下降',
                '布林带位置': '上轨' if signals['bb_position'] > 0 else ('下轨' if signals['bb_position'] < 0 else '中轨')
            }
            
            for indicator, value in indicators.items():
                email_body += f"""
                    <tr>
                        <td>{indicator}</td>
                        <td>{value if isinstance(value, str) else f'{value:.2f}'}</td>
                    </tr>
                """
            
            email_body += """
                </table>
            </body>
            </html>
            """
            
            # 发送邮件
            self.alert_system.send_email(
                subject=f"交易策略预警 - {symbol}",
                body=email_body
            )
            
            logger.info(f"已发送股票 {symbol} 的预警邮件")
            
        except Exception as e:
            logger.error(f"生成预警时出错: {str(e)}")
    
    def _calculate_signals(self, data: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
        """计算交易信号"""
        try:
            latest = data.iloc[-1]
            
            # 计算各种技术指标
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # === 基础指标计算 ===
            # RSI
            rsi = talib.RSI(close)
            if rsi is None or len(rsi) == 0 or pd.isna(rsi[-1]):
                return None
            rsi = rsi[-1]
            
            # MACD
            macd, signal, hist = talib.MACD(close)
            if hist is None or len(hist) == 0 or pd.isna(hist[-1]):
                return None
            
            # 均线
            sma_short = talib.SMA(close, timeperiod=20)
            sma_long = talib.SMA(close, timeperiod=50)
            
            # 布林带
            upper, middle, lower = talib.BBANDS(close)
            
            # === 高级策略信号计算 ===
            # 黄金三角策略
            gold_triangle_signal = self._calculate_gold_triangle_signal(data)
            
            # 动量策略
            momentum_signal = self._calculate_momentum_signal(data)
            
            # 牛牛策略
            niuniu_signal = self._calculate_niuniu_signal(data)
            
            # TDI策略
            tdi_signal = self._calculate_tdi_signal(data)
            
            # 市场预测策略
            market_forecast_signal = self._calculate_market_forecast_signal(data)
            
            # CPGW策略
            cpgw_signal = self._calculate_cpgw_signal(data)
            
            # 综合信号计算
            signals = {
                'gold_triangle': gold_triangle_signal,
                'momentum': momentum_signal,
                'niuniu': niuniu_signal,
                'tdi': tdi_signal,
                'market_forecast': market_forecast_signal,
                'cpgw': cpgw_signal,
                'rsi': rsi,
                'macd': hist[-1],
                'sma_trend': 1 if sma_short[-1] > sma_long[-1] else -1,
                'bb_position': 1 if latest['close'] > upper[-1] else (-1 if latest['close'] < lower[-1] else 0)
            }
            
            # 生成HTML格式的策略分析报告
            strategy_weights = {
                'GoldTriangle策略': f"{weights[0]*100:.1f}%",
                'Momentum策略': f"{weights[1]*100:.1f}%",
                'Niuniu策略': f"{weights[2]*100:.1f}%",
                'TDI策略': f"{weights[3]*100:.1f}%",
                'MarketForecast策略': f"{weights[4]*100:.1f}%",
                'CPGW策略': f"{weights[5]*100:.1f}%"
            }
            
            signals['strategy_weights'] = strategy_weights
            return signals
            
        except Exception as e:
            logger.error(f"计算信号时出错: {str(e)}")
            return None
            
    def _calculate_gold_triangle_signal(self, data: pd.DataFrame) -> float:
        """计算黄金三角策略信号"""
        try:
            close = data['close'].values
            ma5 = talib.SMA(close, timeperiod=5)
            ma10 = talib.SMA(close, timeperiod=10)
            ma20 = talib.SMA(close, timeperiod=20)
            
            # 判断三线关系
            if ma5[-1] > ma10[-1] > ma20[-1]:
                return 1.0  # 强烈买入信号
            elif ma5[-1] < ma10[-1] < ma20[-1]:
                return -1.0  # 强烈卖出信号
            else:
                return 0.0  # 中性信号
        except Exception as e:
            logger.error(f"计算黄金三角信号时出错: {str(e)}")
            return 0.0
            
    def _calculate_momentum_signal(self, data: pd.DataFrame) -> float:
        """计算动量策略信号"""
        try:
            close = data['close'].values
            momentum = talib.MOM(close, timeperiod=10)
            roc = talib.ROC(close, timeperiod=10)
            
            # 综合动量指标
            signal = (momentum[-1] / close[-1] + roc[-1]) / 2
            return np.clip(signal, -1.0, 1.0)
        except Exception as e:
            logger.error(f"计算动量信号时出错: {str(e)}")
            return 0.0
            
    def _calculate_niuniu_signal(self, data: pd.DataFrame) -> float:
        """计算牛牛策略信号"""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # 计算趋势强度
            atr = talib.ATR(high, low, close, timeperiod=14)
            trend = (close[-1] - close[-5]) / (atr[-1] * 5)
            
            # 计算突破信号
            upper_break = close[-1] > max(high[-20:-1])
            lower_break = close[-1] < min(low[-20:-1])
            
            if upper_break:
                return 1.0
            elif lower_break:
                return -1.0
            else:
                return np.clip(trend, -1.0, 1.0)
        except Exception as e:
            logger.error(f"计算牛牛信号时出错: {str(e)}")
            return 0.0
            
    def _calculate_tdi_signal(self, data: pd.DataFrame) -> float:
        """计算TDI策略信号"""
        try:
            close = data['close'].values
            rsi = talib.RSI(close, timeperiod=13)
            rsi_ma = talib.SMA(rsi, timeperiod=7)
            rsi_smooth = talib.SMA(rsi_ma, timeperiod=2)
            
            # 计算信号线
            signal_line = talib.SMA(rsi_smooth, timeperiod=7)
            
            # 计算市场趋势
            trend = 1 if rsi_smooth[-1] > signal_line[-1] else -1
            
            # 计算超买超卖
            if rsi[-1] > 70:
                trend *= 0.5  # 减弱买入信号
            elif rsi[-1] < 30:
                trend *= 0.5  # 减弱卖出信号
                
            return trend
        except Exception as e:
            logger.error(f"计算TDI信号时出错: {str(e)}")
            return 0.0
            
    def _calculate_market_forecast_signal(self, data: pd.DataFrame) -> float:
        """计算市场预测策略信号"""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # 计算预测指标
            ema = talib.EMA(close, timeperiod=9)
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            
            # 趋势预测
            trend = (ema[-1] - ema[-2]) / ema[-2]
            
            # 动量预测
            momentum = stoch_k[-1] - stoch_d[-1]
            
            # 综合信号
            signal = (trend * 100 + momentum) / 2
            return np.clip(signal, -1.0, 1.0)
        except Exception as e:
            logger.error(f"计算市场预测信号时出错: {str(e)}")
            return 0.0
            
    def _calculate_cpgw_signal(self, data: pd.DataFrame) -> float:
        """计算CPGW策略信号"""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # 计算通道
            upper = talib.MAX(high, timeperiod=20)
            lower = talib.MIN(low, timeperiod=20)
            
            # 计算位置
            position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])
            
            # 计算趋势
            ma20 = talib.SMA(close, timeperiod=20)
            trend = (ma20[-1] - ma20[-5]) / ma20[-5]
            
            # 综合信号
            signal = (position - 0.5) + trend
            return np.clip(signal, -1.0, 1.0)
        except Exception as e:
            logger.error(f"计算CPGW信号时出错: {str(e)}")
            return 0.0
    
    def _should_alert(self, signals: Dict[str, Any]) -> bool:
        """
        判断是否需要发送预警
        
        Args:
            signals: 信号字典
        
        Returns:
            是否需要预警
        """
        try:
            # 信号强度阈值
            strength_threshold = 0.7
            
            # 波动率阈值
            volatility_threshold = 0.3
            
            # 成交量比阈值
            volume_ratio_threshold = 2.0
            
            # 检查条件
            strength_condition = abs(signals.get('strength', 0)) > strength_threshold
            volatility_condition = signals.get('volatility', 0) > volatility_threshold
            volume_condition = signals.get('volume_ratio', 0) > volume_ratio_threshold
            
            # 任一条件满足即触发预警
            return strength_condition or volatility_condition or volume_condition
            
        except Exception as e:
            logger.error(f"判断是否预警时出错: {str(e)}")
            return False
    
    def _create_alert_message(self, symbol: str, signals: Dict[str, Any], weights: np.ndarray) -> str:
        """
        创建预警邮件内容
        
        Args:
            symbol: 股票代码
            signals: 信号字典
            weights: 策略权重
            
        Returns:
            格式化的HTML消息
        """
        try:
            if not signals:
                return f"<p>无法为股票 {symbol} 生成有效的信号</p>"
            
            if weights is None or len(weights) != 6:  # 现在我们有6个策略
                logger.warning(f"策略权重无效或长度不正确: {weights}")
                weights = np.ones(6) / 6  # 使用均匀权重作为后备方案
            
            # 策略名称列表
            strategy_names = [
                "价格趋势策略",
                "成交量策略",
                "RSI策略",
                "MACD策略",
                "均线策略",
                "布林带策略"
            ]
            
            # 创建策略权重HTML列表
            strategy_weights_html = "<ul>"
            for name, weight in zip(strategy_names, weights):
                strategy_weights_html += f"<li>{name}: {weight*100:.1f}%</li>"
            strategy_weights_html += "</ul>"
            
            # 检查必要的信号值是否存在
            required_signals = ['strength', 'action', 'confidence', 'risk_level', 'stop_loss']
            if not all(signal in signals for signal in required_signals):
                return f"<p>股票 {symbol} 的信号数据不完整</p>"
            
            # 格式化消息
            message = f"""
            <h2>交易策略预警 - {symbol}</h2>
            <hr>
            <h3>策略权重分布</h3>
            {strategy_weights_html}
            <hr>
            <h3>交易信号</h3>
            <p><strong>建议操作:</strong> {signals['action']}</p>
            <p><strong>信号强度:</strong> {signals['strength']:.2f}</p>
            <p><strong>置信度:</strong> {signals['confidence']*100:.1f}%</p>
            <hr>
            <h3>风险评估</h3>
            <p><strong>风险等级:</strong> {signals['risk_level']}</p>
            <p><strong>波动率:</strong> {signals['volatility']*100:.1f}%</p>
            <p><strong>建议止损价:</strong> {signals['stop_loss']:.2f}</p>
            <hr>
            <h3>技术指标</h3>
            <ul>
                <li>RSI: {signals['rsi']:.1f}</li>
                <li>MACD柱状值: {signals['macd']:.3f}</li>
                <li>成交量比: {signals['volume_ratio']:.1f}</li>
                <li>均线趋势: {'上升' if signals['sma_trend'] > 0 else '下降'}</li>
                <li>布林带位置: {'上轨以上' if signals['bb_position'] > 0 else '下轨以下' if signals['bb_position'] < 0 else '区间内'}</li>
            </ul>
            <hr>
            <p><small>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            """
            
            return message
            
        except Exception as e:
            logger.error(f"创建预警消息时出错: {str(e)}")
            return f"<p>生成预警消息时发生错误: {str(e)}</p>" 