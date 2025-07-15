#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态权重系统 - 基于真实历史表现调整AI和策略权重
集成到个人投资自动化系统中
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class DynamicWeightSystem:
    """动态权重系统 - 基于真实历史表现调整AI和策略权重"""
    
    def __init__(self, db_path: str = "dynamic_weights.db"):
        self.db_path = db_path
        
        # 权重配置
        self.base_ai_weight = 0.7
        self.base_strategy_weight = 0.3
        
        # 调整参数
        self.max_adjustment = 0.05  # 最大±5%调整（更保守）
        self.min_ai_weight = 0.3   # AI权重下限
        self.max_ai_weight = 0.8   # AI权重上限
        self.min_strategy_weight = 0.2  # 策略权重下限
        self.max_strategy_weight = 0.7  # 策略权重上限
        
        # 学习参数
        self.learning_rate = 0.3   # 每0.1准确性差异对应3%权重调整（更保守）
        self.min_accuracy_diff = 0.03  # 最小调整阈值
        self.smoothing_factor = 0.9  # 平滑因子，避免权重剧烈变化
        
        # 个人化设置
        self.risk_tolerance = 'moderate'  # conservative, moderate, aggressive
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建信号历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    ai_signal TEXT,
                    ai_score REAL,
                    strategy_signals TEXT,
                    strategy_score REAL,
                    current_price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建表现跟踪表（新增）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    predicted_signal TEXT,
                    predicted_score REAL,
                    actual_return REAL,
                    accuracy_score REAL,
                    tracking_days INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建权重调整历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weight_adjustment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    ai_weight REAL,
                    strategy_weight REAL,
                    adjustment_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"动态权重数据库初始化完成: {self.db_path}")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
    
    def record_signal(self, symbol: str, ai_signal: Dict, strategy_signals: Dict, current_price: float) -> bool:
        """记录信号预测"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signal_history 
                (symbol, date, ai_signal, ai_score, strategy_signals, strategy_score, current_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now().strftime('%Y-%m-%d'),
                json.dumps(ai_signal),
                ai_signal.get('score', 0.0),
                json.dumps(strategy_signals),
                strategy_signals.get('weighted_score', 0.0),
                current_price
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"记录信号成功: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"记录信号失败: {e}")
            return False
    
    def track_performance(self, symbol: str, signal_date: str, signal_type: str, 
                         predicted_signal: str, predicted_score: float, 
                         tracking_days: int = 30) -> bool:
        """
        追踪信号的实际表现
        
        Args:
            symbol: 股票代码
            signal_date: 信号日期
            signal_type: 信号类型 ('ai' 或 'strategy')
            predicted_signal: 预测信号
            predicted_score: 预测分数
            tracking_days: 追踪天数
        """
        try:
            # 获取实际价格表现
            actual_return = self._get_actual_return(symbol, signal_date, tracking_days)
            
            # 计算准确性分数
            accuracy_score = self._calculate_accuracy_score(predicted_signal, predicted_score, actual_return)
            
            # 记录到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_tracking 
                (symbol, signal_date, signal_type, predicted_signal, predicted_score, 
                 actual_return, accuracy_score, tracking_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, signal_date, signal_type, predicted_signal, predicted_score,
                actual_return, accuracy_score, tracking_days
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"记录表现追踪: {symbol} {signal_type} 准确性: {accuracy_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"记录表现追踪失败: {e}")
            return False
    
    def _get_actual_return(self, symbol: str, signal_date: str, tracking_days: int) -> float:
        """获取实际收益率"""
        try:
            # 获取历史数据
            ticker = yf.Ticker(symbol)
            end_date = datetime.strptime(signal_date, '%Y-%m-%d') + timedelta(days=tracking_days)
            hist = ticker.history(start=signal_date, end=end_date)
            
            if len(hist) < 2:
                return 0.0
            
            # 计算实际收益率
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            actual_return = (end_price - start_price) / start_price
            
            return actual_return
            
        except Exception as e:
            logger.error(f"获取实际收益率失败: {e}")
            return 0.0
    
    def _calculate_accuracy_score(self, predicted_signal: str, predicted_score: float, actual_return: float) -> float:
        """计算准确性分数"""
        try:
            # 基于预测信号和实际表现计算准确性
            if predicted_signal == 'buy' and actual_return > 0:
                # 买入信号且实际上涨
                accuracy = 0.8 + (actual_return * 2)  # 收益越高，准确性越高
            elif predicted_signal == 'sell' and actual_return < 0:
                # 卖出信号且实际下跌
                accuracy = 0.8 + (abs(actual_return) * 2)
            elif predicted_signal == 'hold' and abs(actual_return) < 0.05:
                # 持有信号且实际波动较小
                accuracy = 0.7
            else:
                # 信号与实际不符
                accuracy = 0.3 + (predicted_score * 0.2)  # 基于预测分数给予部分准确性
            
            # 确保准确性在合理范围内
            accuracy = max(0.1, min(1.0, accuracy))
            
            return accuracy
            
        except Exception as e:
            logger.error(f"计算准确性分数失败: {e}")
            return 0.5
    
    def get_current_weights(self, symbol: str) -> Dict[str, float]:
        """获取当前权重"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ai_weight, strategy_weight 
                FROM weight_adjustment_history 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'ai_weight': result[0],
                    'strategy_weight': result[1]
                }
            else:
                # 返回基础权重
                return {
                    'ai_weight': self.base_ai_weight,
                    'strategy_weight': self.base_strategy_weight
                }
                
        except Exception as e:
            logger.error(f"获取当前权重失败: {e}")
            return {
                'ai_weight': self.base_ai_weight,
                'strategy_weight': self.base_strategy_weight
            }
    
    def calculate_dynamic_weights(self, symbol: str, accuracy_comparison: Dict) -> Dict[str, Any]:
        """
        计算动态权重
        
        Args:
            symbol: 股票代码
            accuracy_comparison: 准确性比较结果
            
        Returns:
            动态权重结果
        """
        try:
            # 获取当前权重
            current_weights = self.get_current_weights(symbol)
            
            # 计算权重调整
            weight_adjustment = self._calculate_weight_adjustment(accuracy_comparison)
            
            # 应用调整
            new_weights = self._apply_weight_adjustment(current_weights, weight_adjustment)
            
            # 记录权重调整
            self._record_weight_adjustment(symbol, new_weights, weight_adjustment)
            
            return {
                'symbol': symbol,
                'current_weights': current_weights,
                'new_weights': new_weights,
                'adjustment': weight_adjustment,
                'accuracy_comparison': accuracy_comparison
            }
            
        except Exception as e:
            logger.error(f"动态权重计算失败: {e}")
            return {'error': f'Dynamic weight calculation failed: {str(e)}'}
    
    def _calculate_weight_adjustment(self, accuracy_comparison: Dict) -> Dict[str, Any]:
        """计算权重调整"""
        ai_accuracy = accuracy_comparison.get('ai_accuracy', 0.0)
        strategy_accuracy = accuracy_comparison.get('strategy_accuracy', 0.0)
        accuracy_diff = accuracy_comparison.get('accuracy_difference', 0.0)
        
        # 根据风险偏好调整学习速度
        learning_rate = self._get_learning_rate()
        
        # 计算调整因子
        adjustment_factor = self._calculate_adjustment_factor(accuracy_diff, learning_rate)
        
        # 生成调整原因
        reason = self._generate_adjustment_reason(accuracy_diff, ai_accuracy, strategy_accuracy)
        
        return {
            'adjustment_factor': adjustment_factor,
            'reason': reason,
            'accuracy_diff': accuracy_diff,
            'ai_accuracy': ai_accuracy,
            'strategy_accuracy': strategy_accuracy,
            'learning_rate': learning_rate
        }
    
    def _get_learning_rate(self) -> float:
        """根据风险偏好获取学习速度"""
        if self.risk_tolerance == 'conservative':
            return self.learning_rate * 0.5  # 保守型：慢速学习
        elif self.risk_tolerance == 'aggressive':
            return self.learning_rate * 1.5  # 积极型：快速学习
        else:
            return self.learning_rate  # 中等风险：标准学习速度
    
    def _calculate_adjustment_factor(self, accuracy_diff: float, learning_rate: float) -> float:
        """计算调整因子"""
        # 如果准确性差异小于阈值，不调整
        if abs(accuracy_diff) < self.min_accuracy_diff:
            return 0.0
        
        # 计算调整因子
        raw_adjustment = accuracy_diff * learning_rate
        
        # 限制调整范围
        adjustment = max(-self.max_adjustment, min(self.max_adjustment, raw_adjustment))
        
        # 应用平滑因子
        smoothed_adjustment = adjustment * self.smoothing_factor
        
        return smoothed_adjustment
    
    def _generate_adjustment_reason(self, accuracy_diff: float, ai_accuracy: float, strategy_accuracy: float) -> str:
        """生成调整原因"""
        if abs(accuracy_diff) < self.min_accuracy_diff:
            return "准确性差异较小，保持当前权重"
        
        if accuracy_diff > 0.1:
            return f"策略准确性显著高于AI({accuracy_diff:.2f})，增加策略权重"
        elif accuracy_diff < -0.1:
            return f"AI准确性显著高于策略({abs(accuracy_diff):.2f})，增加AI权重"
        elif accuracy_diff > 0.05:
            return f"策略准确性略高于AI({accuracy_diff:.2f})，小幅增加策略权重"
        elif accuracy_diff < -0.05:
            return f"AI准确性略高于策略({abs(accuracy_diff):.2f})，小幅增加AI权重"
        else:
            return "准确性差异适中，保持当前权重"
    
    def _apply_weight_adjustment(self, current_weights: Dict, adjustment: Dict) -> Dict[str, float]:
        """应用权重调整"""
        current_ai_weight = current_weights['ai_weight']
        current_strategy_weight = current_weights['strategy_weight']
        adjustment_factor = adjustment['adjustment_factor']
        
        # 计算新权重
        new_ai_weight = current_ai_weight - adjustment_factor
        new_strategy_weight = current_strategy_weight + adjustment_factor
        
        # 确保权重在合理范围内
        new_ai_weight = max(self.min_ai_weight, min(self.max_ai_weight, new_ai_weight))
        new_strategy_weight = max(self.min_strategy_weight, min(self.max_strategy_weight, new_strategy_weight))
        
        # 归一化权重（确保总和为1）
        total_weight = new_ai_weight + new_strategy_weight
        new_ai_weight = new_ai_weight / total_weight
        new_strategy_weight = new_strategy_weight / total_weight
        
        return {
            'ai_weight': new_ai_weight,
            'strategy_weight': new_strategy_weight
        }
    
    def _record_weight_adjustment(self, symbol: str, new_weights: Dict, adjustment: Dict):
        """记录权重调整"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO weight_adjustment_history 
                (symbol, date, ai_weight, strategy_weight, adjustment_reason)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now().strftime('%Y-%m-%d'),
                new_weights['ai_weight'],
                new_weights['strategy_weight'],
                adjustment['reason']
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"记录权重调整: {symbol}")
            
        except Exception as e:
            logger.error(f"记录权重调整失败: {e}")
    
    def calculate_accuracy_comparison(self, symbol: str) -> Dict[str, Any]:
        """计算准确性比较（基于真实历史表现）"""
        try:
            # 获取历史表现记录
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT signal_type, AVG(accuracy_score) as avg_accuracy, COUNT(*) as count
                FROM performance_tracking 
                WHERE symbol = ? 
                GROUP BY signal_type
            ''', (symbol,))
            
            results = cursor.fetchall()
            conn.close()
            
            # 初始化准确性
            ai_accuracy = 0.65  # 基础AI准确性
            strategy_accuracy = 0.75  # 基础策略准确性
            
            # 基于真实历史表现调整准确性
            for signal_type, avg_accuracy, count in results:
                if count >= 2:  # 至少需要2个样本
                    if signal_type == 'ai':
                        ai_accuracy = avg_accuracy
                    elif signal_type == 'strategy':
                        strategy_accuracy = avg_accuracy
            
            # 计算准确性差异
            accuracy_diff = strategy_accuracy - ai_accuracy
            
            return {
                'ai_accuracy': ai_accuracy,
                'strategy_accuracy': strategy_accuracy,
                'accuracy_difference': accuracy_diff,
                'ai_samples': sum(1 for r in results if r[0] == 'ai'),
                'strategy_samples': sum(1 for r in results if r[0] == 'strategy'),
                'total_samples': sum(r[2] for r in results)
            }
            
        except Exception as e:
            logger.error(f"准确性比较计算失败: {e}")
            return {
                'ai_accuracy': 0.65,
                'strategy_accuracy': 0.75,
                'accuracy_difference': 0.1,
                'ai_samples': 0,
                'strategy_samples': 0,
                'total_samples': 0,
                'error': f'准确性计算失败: {e}'
            }
    
    def get_weight_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """获取权重历史"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM weight_adjustment_history 
                WHERE symbol = ? AND date >= date('now', '-{} days')
                ORDER BY date DESC
            '''.format(days), (symbol,))
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                history.append({
                    'id': row[0],
                    'symbol': row[1],
                    'date': row[2],
                    'ai_weight': row[3],
                    'strategy_weight': row[4],
                    'reason': row[5],
                    'created_at': row[6]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"获取权重历史失败: {e}")
            return []
    
    def get_weight_trend(self, symbol: str) -> Dict[str, Any]:
        """获取权重趋势"""
        history = self.get_weight_history(symbol, 30)
        
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'message': '数据不足，无法分析趋势'}
        
        # 计算权重变化趋势
        recent_weights = history[:5]  # 最近5次调整
        ai_weights = [w['ai_weight'] for w in recent_weights]
        strategy_weights = [w['strategy_weight'] for w in recent_weights]
        
        # 计算趋势
        ai_trend = np.polyfit(range(len(ai_weights)), ai_weights, 1)[0]
        strategy_trend = np.polyfit(range(len(strategy_weights)), strategy_weights, 1)[0]
        
        # 判断趋势
        if ai_trend > 0.01:
            ai_trend_desc = "AI权重上升趋势"
        elif ai_trend < -0.01:
            ai_trend_desc = "AI权重下降趋势"
        else:
            ai_trend_desc = "AI权重稳定"
        
        if strategy_trend > 0.01:
            strategy_trend_desc = "策略权重上升趋势"
        elif strategy_trend < -0.01:
            strategy_trend_desc = "策略权重下降趋势"
        else:
            strategy_trend_desc = "策略权重稳定"
        
        return {
            'ai_trend': ai_trend,
            'strategy_trend': strategy_trend,
            'ai_trend_desc': ai_trend_desc,
            'strategy_trend_desc': strategy_trend_desc,
            'recent_adjustments': len(recent_weights)
        }
    
    def set_risk_tolerance(self, risk_tolerance: str):
        """设置风险偏好"""
        if risk_tolerance in ['conservative', 'moderate', 'aggressive']:
            self.risk_tolerance = risk_tolerance
            logger.info(f"风险偏好设置为: {risk_tolerance}")
        else:
            logger.warning(f"无效的风险偏好设置: {risk_tolerance}")
    
    def get_performance_summary(self, symbol: str) -> Dict[str, Any]:
        """获取表现摘要"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取AI和策略的表现统计
            cursor.execute('''
                SELECT signal_type, 
                       AVG(accuracy_score) as avg_accuracy,
                       COUNT(*) as total_signals,
                       AVG(actual_return) as avg_return,
                       MAX(accuracy_score) as max_accuracy,
                       MIN(accuracy_score) as min_accuracy
                FROM performance_tracking 
                WHERE symbol = ?
                GROUP BY signal_type
            ''', (symbol,))
            
            results = cursor.fetchall()
            conn.close()
            
            summary = {
                'symbol': symbol,
                'ai_performance': {},
                'strategy_performance': {},
                'comparison': {}
            }
            
            for signal_type, avg_accuracy, total_signals, avg_return, max_accuracy, min_accuracy in results:
                performance_data = {
                    'avg_accuracy': avg_accuracy,
                    'total_signals': total_signals,
                    'avg_return': avg_return,
                    'max_accuracy': max_accuracy,
                    'min_accuracy': min_accuracy
                }
                
                if signal_type == 'ai':
                    summary['ai_performance'] = performance_data
                elif signal_type == 'strategy':
                    summary['strategy_performance'] = performance_data
            
            # 计算比较
            if summary['ai_performance'] and summary['strategy_performance']:
                ai_accuracy = summary['ai_performance']['avg_accuracy']
                strategy_accuracy = summary['strategy_performance']['avg_accuracy']
                
                summary['comparison'] = {
                    'accuracy_difference': strategy_accuracy - ai_accuracy,
                    'ai_better': ai_accuracy > strategy_accuracy,
                    'strategy_better': strategy_accuracy > ai_accuracy,
                    'recommendation': self._get_recommendation(ai_accuracy, strategy_accuracy)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取表现摘要失败: {e}")
            return {'error': f'获取表现摘要失败: {e}'}
    
    def _get_recommendation(self, ai_accuracy: float, strategy_accuracy: float) -> str:
        """获取权重调整建议"""
        diff = strategy_accuracy - ai_accuracy
        
        if abs(diff) < 0.05:
            return "保持当前权重，AI和策略表现相近"
        elif diff > 0.1:
            return "建议增加策略权重，策略表现显著优于AI"
        elif diff < -0.1:
            return "建议增加AI权重，AI表现显著优于策略"
        elif diff > 0.05:
            return "建议小幅增加策略权重"
        else:
            return "建议小幅增加AI权重" 