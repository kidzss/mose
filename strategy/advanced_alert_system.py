import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from enum import Enum
from datetime import datetime, timedelta
import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 添加项目根目录到sys.path以便导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .market_environment_classifier import MarketEnvironment, MarketEnvironmentClassifier
from .dynamic_strategy_selector import DynamicStrategySelector
from .signal_quality_evaluator import SignalQualityEvaluator, SignalStrength

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alert_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlertSystem")

class AlertLevel(Enum):
    """提醒级别枚举类型"""
    CRITICAL = "紧急"      # 高优先级警报，需要立即关注的市场事件或交易信号
    HIGH = "重要"          # 重要的市场变化或强交易信号
    MEDIUM = "一般"        # 普通的交易机会或市场观察
    LOW = "提示"           # 低优先级信息，仅供参考


class AlertCategory(Enum):
    """提醒类别枚举类型"""
    TREND_CHANGE = "趋势转变"        # 市场趋势发生重要变化
    BREAKOUT = "突破"                # 价格突破重要支撑阻力位
    TRADE_SIGNAL = "交易信号"        # 买入或卖出信号
    RISK_WARNING = "风险警告"        # 止损触发或风险提示
    MARKET_CONDITION = "市场环境"     # 市场环境状态变化
    PORTFOLIO = "投资组合"           # 投资组合相关提醒
    TECHNICAL = "技术指标"           # 技术指标相关提醒


class AdvancedAlertSystem:
    """
    高级提醒系统
    
    基于市场环境、策略选择和信号质量，提供分级提醒机制。
    整合了现有的monitor模块，提供全面的分析和提醒功能。
    """
    
    def __init__(self, config=None):
        """
        初始化高级提醒系统
        
        参数:
        config (dict, optional): 配置参数
        """
        logger.info("初始化高级提醒系统")
        
        # 默认配置
        self.default_config = {
            'min_quality_score': 0.6,  # 最低质量分数以产生提醒
            'enable_notification_manager': True,  # 是否启用通知管理器
            'risk_management_alerts': True,      # 是否启用风险管理提醒
            'max_alerts_per_day': 10,           # 每天最大提醒数量
            'alert_cooldown_minutes': 60,       # 提醒冷却时间（分钟）
            'signal_quality_thresholds': {       # 信号质量阈值对应的提醒级别
                AlertLevel.CRITICAL: 0.85,       # 紧急提醒阈值
                AlertLevel.HIGH: 0.75,           # 重要提醒阈值
                AlertLevel.MEDIUM: 0.65,         # 一般提醒阈值
                AlertLevel.LOW: 0.0              # 提示提醒阈值
            }
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # 初始化组件
        self.market_classifier = MarketEnvironmentClassifier()
        self.strategy_selector = DynamicStrategySelector()
        self.signal_evaluator = SignalQualityEvaluator()
        
        # 初始化状态变量
        self.last_environment = None
        self.environment_history = []  # 记录环境历史
        self.signal_history = []  # 记录信号历史
        self.alert_history = []  # 记录提醒历史
        self.last_alert_time = {}  # 记录每个股票最后提醒时间
              
        logger.info(f"高级提醒系统配置: {self.config}")
        
    def process_market_data(self, symbol, data):
        """
        处理市场数据，寻找需要提醒的条件
        
        参数:
        symbol (str): 股票代码
        data (DataFrame): 市场数据
        
        返回:
        list: 提醒列表
        """
        logger.info(f"开始处理 {symbol} 的市场数据")
        
        try:
            alerts = []
            
            # 1. 分析市场环境
            env_result = self.market_classifier.classify_environment(data)
            environment = env_result['environment']
            confidence = env_result['confidence']
            
            logger.info(f"当前市场环境: {environment.value}, 置信度: {confidence:.2f}")
            
            # 2. 检测市场环境变化
            if self._check_environment_change(environment, data):
                alert = self._create_market_condition_alert(symbol, environment, confidence)
                alerts.append(alert)
                logger.info(f"检测到市场环境变化，创建提醒: {alert['title']}")
            
            # 3. 分析技术指标突破
            tech_alerts = self._analyze_technical_indicators(symbol, data)
            if tech_alerts:
                alerts.extend(tech_alerts)
                logger.info(f"检测到 {len(tech_alerts)} 个技术指标突破提醒")
            
            # 4. 风险管理提醒
            if self.config['risk_management_alerts']:
                risk_alerts = self._check_risk_conditions(symbol, data, environment)
                if risk_alerts:
                    alerts.extend(risk_alerts)
                    logger.info(f"生成 {len(risk_alerts)} 个风险管理提醒")
            
            # 应用提醒过滤器
            filtered_alerts = self._filter_alerts(alerts)
            logger.info(f"过滤后剩余 {len(filtered_alerts)} 个提醒")
            
            # 保存提醒历史
            self._update_alert_history(filtered_alerts)
            
            # 发送提醒
            if filtered_alerts:
                self._send_alerts(filtered_alerts)
                
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"处理市场数据时出错: {str(e)}", exc_info=True)
            return []
            
    def process_trading_signal(
            self, 
            symbol: str,
            signal_data: Dict[str, Any],
            market_data: pd.DataFrame,
            additional_data: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
        """
        处理交易信号，评估质量并生成提醒
        
        参数:
            symbol: 股票代码
            signal_data: 信号数据，包含方向、入场价、止损价、目标价等
            market_data: 市场OHLCV数据
            additional_data: 额外数据，如多时间框架数据等
            
        返回:
            处理结果，包含是否通过评估、提醒等信息
        """
        logger.info(f"处理 {symbol} 的交易信号")
        logger.debug(f"信号数据: {signal_data}")
        
        try:
            # 1. 分析市场环境
            environment_result = self.market_classifier.classify_environment(market_data)
            current_environment = environment_result['environment']
            
            # 2. 评估信号质量
            evaluation_result = self.signal_evaluator.evaluate_signal(
                signal_data, market_data, current_environment, additional_data
            )
            
            # 3. 判断是否生成提醒
            alert = None
            if evaluation_result['passed_threshold']:
                # 根据信号强度确定提醒级别
                alert_level = self._determine_alert_level(evaluation_result['quality_score'])
                
                # 创建提醒
                alert = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'category': AlertCategory.TRADE_SIGNAL,
                    'level': alert_level,
                    'title': f"{alert_level.value}交易信号: {symbol} {'买入' if signal_data.get('direction', 0) > 0 else '卖出'}",
                    'message': self._format_signal_alert_message(symbol, signal_data, evaluation_result),
                    'signal_data': signal_data,
                    'evaluation': evaluation_result,
                    'environment': current_environment
                }
                
                # 记录信号历史
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'signal': signal_data,
                    'evaluation': evaluation_result,
                    'environment': current_environment
                })
                # 保留最近100条记录
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
                
                # 发送提醒
                if alert:
                    self._send_alerts([alert])
            
            result = {
                'passed': evaluation_result['passed_threshold'],
                'quality_score': evaluation_result['quality_score'],
                'strength': evaluation_result['strength'],
                'recommendations': evaluation_result['recommendations'],
                'alert_generated': alert is not None,
                'alert': alert
            }
            
            return result
            
        except Exception as e:
            logger.error(f"处理交易信号时出错: {str(e)}", exc_info=True)
            return {
                'passed': False,
                'quality_score': 0.0,
                'strength': SignalStrength.VERY_WEAK,
                'recommendations': [f"处理出错: {str(e)}"],
                'alert_generated': False,
                'alert': None
            }
            
    def _check_environment_change(self, current_environment, data):
        """检查市场环境是否发生变化"""
        try:
            # 简单版本：如果当前数据少于30行，无法可靠判断环境变化
            if len(data) < 30:
                logger.debug("数据不足，无法判断环境变化")
                return False
                
            # 实际应用中应该维护历史环境记录，这里简化处理
            # 检测当前K线的移动平均线交叉或反转形态
            
            # 示例：检查20日和50日均线交叉
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                crosses_above = (data['sma_20'].iloc[-2] <= data['sma_50'].iloc[-2] and 
                                data['sma_20'].iloc[-1] > data['sma_50'].iloc[-1])
                                
                crosses_below = (data['sma_20'].iloc[-2] >= data['sma_50'].iloc[-2] and 
                                data['sma_20'].iloc[-1] < data['sma_50'].iloc[-1])
                                
                if crosses_above or crosses_below:
                    logger.info(f"检测到移动平均线交叉: {'上穿' if crosses_above else '下穿'}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查环境变化时出错: {str(e)}", exc_info=True)
            return False
    
    def _detect_breakouts(self, symbol, market_data):
        """检测价格突破支撑/阻力位"""
        try:
            # 简化版突破检测
            alerts = []
            
            # 实际实现应该检测关键支撑阻力位的突破
            # 这里仅做简单示例
            
            return alerts
            
        except Exception as e:
            logger.error(f"检测突破时出错: {str(e)}", exc_info=True)
            return []
    
    def _detect_risk_warnings(self, symbol: str, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测风险警告条件"""
        alerts = []
        
        try:
            # 确保有足够的数据
            if len(market_data) < 30:
                return alerts
                
            # 1. 检查波动率异常
            returns = market_data['close'].pct_change().dropna()
            current_volatility = returns.iloc[-10:].std() * (252 ** 0.5)  # 年化最近10天波动率
            historical_volatility = returns.iloc[:-10].std() * (252 ** 0.5)  # 历史波动率
            
            if current_volatility > historical_volatility * 1.5 and current_volatility > self.config['risk_warning']['volatility_threshold']:
                # 创建波动率警告
                alert = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'category': AlertCategory.RISK_WARNING,
                    'level': AlertLevel.HIGH,
                    'title': f"波动率异常: {symbol}",
                    'message': f"{symbol}波动率激增至{current_volatility:.2%}，"
                            f"是历史波动率{historical_volatility:.2%}的{current_volatility/historical_volatility:.1f}倍，"
                            f"建议调整止损位置或减小仓位",
                    'current_volatility': current_volatility,
                    'historical_volatility': historical_volatility
                }
                
                alerts.append(alert)
            
            # 2. 检查回撤
            peak = market_data['close'].iloc[:-20].max()
            current_price = market_data['close'].iloc[-1]
            drawdown = (peak - current_price) / peak
            
            if drawdown > self.config['risk_warning']['drawdown_threshold']:
                # 创建回撤警告
                alert = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'category': AlertCategory.RISK_WARNING,
                    'level': AlertLevel.HIGH if drawdown > self.config['risk_warning']['drawdown_threshold'] * 1.5 else AlertLevel.MEDIUM,
                    'title': f"显著回撤: {symbol}",
                    'message': f"{symbol}从高点{peak:.2f}回撤{drawdown:.2%}，"
                            f"已超过警戒阈值{self.config['risk_warning']['drawdown_threshold']:.2%}，"
                            f"请评估止损策略",
                    'peak': peak,
                    'current_price': current_price,
                    'drawdown': drawdown
                }
                
                alerts.append(alert)
            
            # 3. 检查成交量异常
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].iloc[-20:].mean()
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > self.config['risk_warning']['volume_surge_threshold']:
                # 判断价格变动方向
                price_change = market_data['close'].iloc[-1] - market_data['close'].iloc[-2]
                direction = "上涨" if price_change > 0 else "下跌"
                
                # 创建成交量异常警告
                alert = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'category': AlertCategory.RISK_WARNING,
                    'level': AlertLevel.MEDIUM,
                    'title': f"成交量激增: {symbol}",
                    'message': f"{symbol}成交量为近期均值的{volume_ratio:.1f}倍，"
                            f"同时价格{direction}，可能是趋势转变信号，请密切关注",
                    'volume_ratio': volume_ratio,
                    'price_change': price_change
                }
                
                alerts.append(alert)
        
        except Exception as e:
            logger.error(f"检测风险警告时出错: {str(e)}", exc_info=True)
            
        return alerts
    
    def _determine_alert_level(self, quality_score: float) -> AlertLevel:
        """根据信号质量确定提醒级别"""
        thresholds = self.config['signal_quality_thresholds']
        
        if quality_score >= thresholds[AlertLevel.CRITICAL]:
            return AlertLevel.CRITICAL
        elif quality_score >= thresholds[AlertLevel.HIGH]:
            return AlertLevel.HIGH
        elif quality_score >= thresholds[AlertLevel.MEDIUM]:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def _format_signal_alert_message(
            self, 
            symbol: str, 
            signal_data: Dict[str, Any],
            evaluation_result: Dict[str, Any]
        ) -> str:
        """格式化交易信号提醒消息"""
        direction = "买入" if signal_data.get('direction', 0) > 0 else "卖出"
        entry_price = signal_data.get('entry_price', 'N/A')
        stop_loss = signal_data.get('stop_loss', 'N/A')
        target_price = signal_data.get('target_price', 'N/A')
        
        quality_score = evaluation_result['quality_score']
        strength = evaluation_result['strength'].value
        
        message = f"【{symbol} {direction}信号】\n\n"
        message += f"信号强度: {strength} ({quality_score:.2f})\n"
        message += f"入场价: {entry_price}\n"
        
        if stop_loss != 'N/A':
            message += f"止损价: {stop_loss}\n"
            
        if target_price != 'N/A':
            message += f"目标价: {target_price}\n"
            
        if 'risk_reward' in evaluation_result['dimension_scores']:
            rr_score = evaluation_result['dimension_scores']['risk_reward']
            if rr_score > 0.7:
                message += f"风险回报比: 优 ({rr_score:.2f})\n"
            elif rr_score > 0.4:
                message += f"风险回报比: 中 ({rr_score:.2f})\n"
            else:
                message += f"风险回报比: 差 ({rr_score:.2f})\n"
                
        # 添加支持此信号的主要因素
        if evaluation_result['details'].get('technical_consistency', {}).get('consistent_indicators', 0) > 0:
            message += "\n主要支持因素:\n"
            tech_details = evaluation_result['details']['technical_consistency']
            for group, consistency in tech_details.get('group_consistency', {}).items():
                if consistency > 0.6:
                    message += f"- {group}指标支持 ({consistency:.2f})\n"
                    
        # 添加警告和建议
        if evaluation_result['recommendations']:
            message += "\n注意事项:\n"
            for rec in evaluation_result['recommendations'][:3]:  # 最多显示3条建议
                message += f"- {rec}\n"
                
        return message
    
    def _format_environment_change_message(
            self, 
            symbol: str, 
            previous_env: MarketEnvironment, 
            current_env: MarketEnvironment,
            details: Dict[str, Any]
        ) -> str:
        """格式化环境变化提醒消息"""
        message = f"【市场环境变化】\n\n"
        message += f"股票: {symbol}\n"
        message += f"环境变化: {previous_env.value} -> {current_env.value}\n\n"
        
        # 添加详细原因
        if 'reasons' in details and details['reasons']:
            message += "变化原因:\n"
            for reason in details['reasons'][:5]:  # 最多显示5条原因
                message += f"- {reason}\n"
                
        # 添加适合的策略
        if 'suitable_strategies' in details and details['suitable_strategies']:
            message += "\n适合的策略:\n"
            for strategy in details['suitable_strategies']:
                message += f"- {strategy}\n"
                
        # 添加行动建议
        message += "\n建议操作:\n"
        
        if current_env in [MarketEnvironment.STRONG_UPTREND, MarketEnvironment.WEAK_UPTREND]:
            message += "- 保持多头头寸，考虑顺势交易\n"
            message += "- 设置追踪止损保护利润\n"
            if current_env == MarketEnvironment.STRONG_UPTREND:
                message += "- 可增加仓位或持有时间\n"
            else:
                message += "- 保持适中仓位，注意市场变化\n"
        elif current_env in [MarketEnvironment.STRONG_DOWNTREND, MarketEnvironment.WEAK_DOWNTREND]:
            message += "- 减少多头头寸或考虑对冲\n"
            message += "- 设置更紧的止损\n"
            if current_env == MarketEnvironment.STRONG_DOWNTREND:
                message += "- 考虑观望或逆势短线操作\n"
            else:
                message += "- 寻找潜在的底部反转信号\n"
        elif current_env == MarketEnvironment.RANGE_BOUND:
            message += "- 考虑区间交易策略\n"
            message += "- 在支撑位买入，阻力位卖出\n"
            message += "- 避免追涨杀跌，设置合理止损\n"
        elif current_env == MarketEnvironment.CHOPPY:
            message += "- 减少交易频率，降低仓位\n"
            message += "- 等待市场方向明确\n"
            message += "- 只交易高质量信号\n"
            
        return message
    
    def _send_alerts(self, alerts):
        """发送提醒"""
        try:
            now = datetime.now()
            for alert in alerts:
                # 检查冷却期
                if not self._check_alert_cooldown(alert):
                    logger.debug(f"跳过发送提醒（冷却期内）: {alert['title']}")
                    continue
                
                # 记录日志
                alert_level = alert.get('level', AlertLevel.LOW)
                log_level = logging.INFO
                if alert_level == AlertLevel.CRITICAL:
                    log_level = logging.CRITICAL
                elif alert_level == AlertLevel.HIGH:
                    log_level = logging.WARNING
                    
                logger.log(log_level, f"发送提醒: {alert['title']} - {alert['message'][:100]}...")
                
                # 记录提醒时间
                self.last_alert_time[alert['symbol']] = now
                
        except Exception as e:
            logger.error(f"发送提醒时出错: {str(e)}", exc_info=True)
    
    def get_market_summary(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成市场摘要信息
        
        参数:
            symbol: 股票代码
            market_data: 市场OHLCV数据
            
        返回:
            市场摘要信息
        """
        logger.info(f"生成 {symbol} 的市场摘要")
        
        try:
            # 分析市场环境
            environment_result = self.market_classifier.classify_environment(market_data)
            current_environment = environment_result['environment']
            
            # 获取适合的策略
            strategy_result = self.strategy_selector.get_best_strategy(market_data)
            
            # 计算基本技术指标
            current_price = market_data['close'].iloc[-1]
            price_change = (current_price / market_data['close'].iloc[-2] - 1) * 100
            
            # 计算移动平均线位置
            ma_status = {}
            if 'sma_20' in market_data.columns:
                ma_status['sma_20'] = "上方" if current_price > market_data['sma_20'].iloc[-1] else "下方"
            if 'sma_50' in market_data.columns:
                ma_status['sma_50'] = "上方" if current_price > market_data['sma_50'].iloc[-1] else "下方"
            if 'sma_200' in market_data.columns:
                ma_status['sma_200'] = "上方" if current_price > market_data['sma_200'].iloc[-1] else "下方"
                
            # 生成摘要
            summary = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'price_change': price_change,
                'environment': current_environment,
                'environment_confidence': environment_result['confidence'],
                'ma_status': ma_status,
                'suitable_strategies': strategy_result['strategy_weights'],
                'primary_strategy': strategy_result['primary_strategy']
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"生成市场摘要时出错: {str(e)}", exc_info=True)
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'error': str(e),
                'current_price': market_data['close'].iloc[-1] if len(market_data) > 0 else None
            }
    
    def _filter_alerts(self, alerts):
        """过滤提醒，防止过多提醒"""
        try:
            if not alerts:
                return []
                
            # 1. 按优先级排序
            priority_map = {
                AlertLevel.CRITICAL: 4,
                AlertLevel.HIGH: 3,
                AlertLevel.MEDIUM: 2,
                AlertLevel.LOW: 1
            }
            
            sorted_alerts = sorted(
                alerts, 
                key=lambda x: (priority_map.get(x['level'], 0), x.get('quality_score', 0)), 
                reverse=True
            )
            
            # 2. 检查每日提醒配额
            max_alerts = self.config['max_alerts_per_day']
            today_alerts = [a for a in self.alert_history 
                          if a['timestamp'].split('T')[0] == datetime.now().date().isoformat()]
                          
            remaining_quota = max_alerts - len(today_alerts)
            
            if remaining_quota <= 0:
                logger.warning(f"今日提醒配额已用尽，不再生成新提醒")
                return []
                
            # 只保留配额允许的提醒数
            allowed_alerts = sorted_alerts[:remaining_quota]
            
            logger.info(f"提醒过滤：原始={len(alerts)}个，保留={len(allowed_alerts)}个")
            return allowed_alerts
            
        except Exception as e:
            logger.error(f"过滤提醒时出错: {str(e)}", exc_info=True)
            return alerts[:1] if alerts else []  # 错误时只返回最重要的一个提醒
    
    def _update_alert_history(self, alerts):
        """更新提醒历史"""
        try:
            self.alert_history.extend(alerts)
            
            # 保留最近500条提醒
            if len(self.alert_history) > 500:
                self.alert_history = self.alert_history[-500:]
                
            logger.debug(f"更新提醒历史，当前共有 {len(self.alert_history)} 条提醒记录")
                
        except Exception as e:
            logger.error(f"更新提醒历史时出错: {str(e)}", exc_info=True)
    
    def _get_technical_summary(self, data):
        """获取技术指标摘要"""
        try:
            summary = {}
            
            # RSI
            if 'rsi' in data.columns:
                summary['rsi'] = data['rsi'].iloc[-1]
                
            # MACD
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                summary['macd'] = data['macd'].iloc[-1]
                summary['macd_signal'] = data['macd_signal'].iloc[-1]
                summary['macd_histogram'] = data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]
                
            # 移动平均线
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                summary['ma_20'] = data['sma_20'].iloc[-1]
                summary['ma_50'] = data['sma_50'].iloc[-1]
                summary['ma_cross'] = 'above' if data['sma_20'].iloc[-1] > data['sma_50'].iloc[-1] else 'below'
            
            # 波动率
            if 'atr' in data.columns:
                summary['atr'] = data['atr'].iloc[-1]
                summary['atr_pct'] = data['atr'].iloc[-1] / data['close'].iloc[-1] * 100
                
            return summary
            
        except Exception as e:
            logger.error(f"获取技术指标摘要时出错: {str(e)}", exc_info=True)
            return {}
    
    def _create_market_condition_alert(self, symbol, environment, confidence):
        """创建市场环境变化提醒"""
        try:
            title = f"{symbol} 市场环境变化"
            message = f"{symbol}市场环境变为{environment.value}，置信度为{confidence:.2f}。"
            
            if environment in [MarketEnvironment.STRONG_UPTREND, MarketEnvironment.WEAK_UPTREND]:
                message += "市场呈上升趋势，可关注突破点和顺势交易机会。"
                level = AlertLevel.HIGH if environment == MarketEnvironment.STRONG_UPTREND else AlertLevel.MEDIUM
            elif environment in [MarketEnvironment.STRONG_DOWNTREND, MarketEnvironment.WEAK_DOWNTREND]:
                message += "市场呈下降趋势，注意风险控制，可寻找做空机会。"
                level = AlertLevel.HIGH if environment == MarketEnvironment.STRONG_DOWNTREND else AlertLevel.MEDIUM
            elif environment == MarketEnvironment.RANGE_BOUND:
                message += "市场处于区间震荡，适合区间交易策略。"
                level = AlertLevel.MEDIUM
            elif environment == MarketEnvironment.CHOPPY:
                message += "市场波动性较大，方向不明确，建议谨慎交易或暂时观望。"
                level = AlertLevel.HIGH
            else:
                message += "市场状况不明确，建议谨慎交易。"
                level = AlertLevel.LOW
            
            alert = {
                'title': title,
                'message': message,
                'level': level,
                'category': AlertCategory.MARKET_CONDITION,
                'timestamp': datetime.now().isoformat()
            }
            
            return alert
            
        except Exception as e:
            logger.error(f"创建市场环境提醒时出错: {str(e)}", exc_info=True)
            return {
                'title': f"{symbol} 市场环境更新",
                'message': f"检测到市场环境变化。",
                'level': AlertLevel.LOW,
                'category': AlertCategory.MARKET_CONDITION,
                'timestamp': datetime.now().isoformat()
            }

    def _check_risk_conditions(self, symbol, data, environment):
        """检查风险条件，生成风险管理提醒"""
        try:
            alerts = []
            
            # 检查波动率异常
            if 'atr' in data.columns:
                current_atr = data['atr'].iloc[-1]
                avg_atr = data['atr'].iloc[-20:].mean()
                
                if current_atr > avg_atr * 2:
                    alert = {
                        'title': f"{symbol} 波动率异常提醒",
                        'message': f"{symbol}的波动率显著上升，当前ATR为过去20个周期平均值的{current_atr/avg_atr:.1f}倍，建议调整风险管理策略。",
                        'level': AlertLevel.HIGH,
                        'category': AlertCategory.RISK_MANAGEMENT,
                        'timestamp': datetime.now().isoformat()
                    }
                    alerts.append(alert)
                    logger.info(f"检测到波动率异常: 当前ATR为平均的{current_atr/avg_atr:.1f}倍")
            
            # 检查交易量异常
            if 'volume' in data.columns:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].iloc[-20:].mean()
                
                if current_volume > avg_volume * 3:
                    alert = {
                        'title': f"{symbol} 交易量异常提醒",
                        'message': f"{symbol}的交易量突然放大，当前交易量为过去20个周期平均值的{current_volume/avg_volume:.1f}倍，可能有重要事件发生。",
                        'level': AlertLevel.HIGH,
                        'category': AlertCategory.RISK_MANAGEMENT,
                        'timestamp': datetime.now().isoformat()
                    }
                    alerts.append(alert)
                    logger.info(f"检测到交易量异常: 当前交易量为平均的{current_volume/avg_volume:.1f}倍")
                    
            # 市场环境为混沌时的警告
            if environment == MarketEnvironment.CHOPPY:
                alert = {
                    'title': f"{symbol} 市场混沌提醒",
                    'message': f"{symbol}当前处于混沌市场环境，波动性高且方向不明，建议减少交易规模或暂停交易。",
                    'level': AlertLevel.MEDIUM,
                    'category': AlertCategory.RISK_MANAGEMENT,
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
                logger.info("检测到混沌市场环境，生成风险提醒")
                
            return alerts
            
        except Exception as e:
            logger.error(f"检查风险条件时出错: {str(e)}", exc_info=True)
            return [] 

    def _check_alert_cooldown(self, alert):
        """检查提醒是否在冷却期内"""
        try:
            symbol = alert.get('symbol', '')
            if symbol not in self.last_alert_time:
                return True  # 没有记录，允许发送
                
            # 获取冷却时间（分钟）
            cooldown_minutes = self.config['alert_cooldown_minutes']
            
            # 计算距离上次提醒的时间（分钟）
            elapsed_minutes = (datetime.now() - self.last_alert_time[symbol]).total_seconds() / 60
            
            # 如果超过冷却时间，则允许发送
            return elapsed_minutes >= cooldown_minutes
            
        except Exception as e:
            logger.error(f"检查提醒冷却时间时出错: {str(e)}", exc_info=True)
            return True  # 出错时默认允许发送 

    def _analyze_technical_indicators(self, symbol, data):
        """分析技术指标，寻找突破或信号"""
        try:
            alerts = []
            
            # 检查RSI超买超卖
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                
                if rsi > 70:
                    alert = {
                        'title': f"{symbol} RSI超买提醒",
                        'message': f"{symbol}的RSI达到{rsi:.1f}，处于超买区域，可能出现回调。",
                        'level': AlertLevel.MEDIUM,
                        'category': AlertCategory.TECHNICAL,
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol
                    }
                    alerts.append(alert)
                    logger.info(f"检测到RSI超买: {rsi:.1f}")
                    
                elif rsi < 30:
                    alert = {
                        'title': f"{symbol} RSI超卖提醒",
                        'message': f"{symbol}的RSI达到{rsi:.1f}，处于超卖区域，可能出现反弹。",
                        'level': AlertLevel.MEDIUM,
                        'category': AlertCategory.TECHNICAL,
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol
                    }
                    alerts.append(alert)
                    logger.info(f"检测到RSI超卖: {rsi:.1f}")
            
            # 检查MACD交叉
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd = data['macd'].iloc[-1]
                macd_signal = data['macd_signal'].iloc[-1]
                prev_macd = data['macd'].iloc[-2]
                prev_macd_signal = data['macd_signal'].iloc[-2]
                
                if prev_macd < prev_macd_signal and macd > macd_signal:
                    alert = {
                        'title': f"{symbol} MACD金叉提醒",
                        'message': f"{symbol}出现MACD金叉信号，可能表明上升趋势开始。",
                        'level': AlertLevel.HIGH if macd < 0 else AlertLevel.MEDIUM,
                        'category': AlertCategory.TECHNICAL,
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol
                    }
                    alerts.append(alert)
                    logger.info(f"检测到MACD金叉，MACD值: {macd:.4f}")
                    
                elif prev_macd > prev_macd_signal and macd < macd_signal:
                    alert = {
                        'title': f"{symbol} MACD死叉提醒",
                        'message': f"{symbol}出现MACD死叉信号，可能表明下降趋势开始。",
                        'level': AlertLevel.HIGH if macd > 0 else AlertLevel.MEDIUM,
                        'category': AlertCategory.TECHNICAL,
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol
                    }
                    alerts.append(alert)
                    logger.info(f"检测到MACD死叉，MACD值: {macd:.4f}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"分析技术指标时出错: {str(e)}", exc_info=True)
            return [] 