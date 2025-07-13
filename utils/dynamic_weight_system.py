#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态权重系统 - 基于准确性差异调整AI和策略权重
集成到个人投资自动化系统中
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DynamicWeightSystem:
    """动态权重系统 - 基于准确性差异调整AI和策略权重"""
    
    def __init__(self, db_path: str = "dynamic_weights.db"):
        self.db_path = db_path
        
        # 权重配置
        self.base_ai_weight = 0.7
        self.base_strategy_weight = 0.3
        
        # 调整参数
        self.max_adjustment = 0.2  # 最大±20%调整
        self.min_ai_weight = 0.3   # AI权重下限
        self.max_ai_weight = 0.8   # AI权重上限
        self.min_strategy_weight = 0.2  # 策略权重下限
        self.max_strategy_weight = 0.7  # 策略权重上限
        
        # 学习参数
        self.learning_rate = 0.5   # 每0.1准确性差异对应5%权重调整
        self.min_accuracy_diff = 0.05  # 最小调整阈值
        self.smoothing_factor = 0.8  # 平滑因子，避免权重剧烈变化
        
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
            
            # 创建表现跟踪表
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
        
        # 计算调整因子
        adjustment_factor = self._calculate_adjustment_factor(accuracy_diff)
        
        # 生成调整原因
        reason = self._generate_adjustment_reason(accuracy_diff, ai_accuracy, strategy_accuracy)
        
        return {
            'adjustment_factor': adjustment_factor,
            'reason': reason,
            'accuracy_diff': accuracy_diff,
            'ai_accuracy': ai_accuracy,
            'strategy_accuracy': strategy_accuracy
        }
    
    def _calculate_adjustment_factor(self, accuracy_diff: float) -> float:
        """计算调整因子"""
        # 如果准确性差异小于阈值，不调整
        if abs(accuracy_diff) < self.min_accuracy_diff:
            return 0.0
        
        # 计算调整因子
        raw_adjustment = accuracy_diff * self.learning_rate
        
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
    
    def calculate_accuracy_comparison(self, symbol: str) -> Dict[str, Any]:
        """计算准确性比较（简化版本）"""
        try:
            # 获取历史信号记录
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM signal_history 
                WHERE symbol = ? 
                ORDER BY date DESC
            ''', (symbol,))
            
            results = cursor.fetchall()
            conn.close()
            
            if len(results) < 2:
                return {'error': '历史信号数据不足，需要更多数据来计算准确性'}
            
            # 简化版本：基于历史数据计算准确性
            # 实际应用中应该基于真实价格表现计算
            ai_accuracy = 0.6  # 模拟AI准确性
            strategy_accuracy = 0.7  # 模拟策略准确性
            
            return {
                'ai_accuracy': ai_accuracy,
                'strategy_accuracy': strategy_accuracy,
                'accuracy_difference': strategy_accuracy - ai_accuracy
            }
            
        except Exception as e:
            return {'error': f'准确性计算失败: {e}'} 