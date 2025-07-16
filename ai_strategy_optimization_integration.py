#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI策略优化集成系统
将策略优化结果持久化存储并与AI分析系统集成
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIStrategyOptimizationIntegration:
    """AI策略优化集成系统"""
    
    def __init__(self, db_path: str = "ai_strategy_optimization.db"):
        """初始化集成系统"""
        self.db_path = db_path
        self.optimization_history = {}
        self.ai_analysis_cache = {}
        
        # 初始化数据库
        self._init_database()
        
        # 加载历史优化数据
        self._load_optimization_history()
        
        print("🚀 AI策略优化集成系统初始化完成")
        print("📊 功能: 策略优化结果持久化 + AI智能分析集成")
        print("=" * 80)
    
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 策略优化历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_optimization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    weight REAL,
                    performance_score REAL,
                    expected_return REAL,
                    expected_sharpe REAL,
                    expected_drawdown REAL,
                    market_condition TEXT,
                    risk_level TEXT,
                    optimization_version TEXT
                )
            ''')
            
            # AI分析结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    ai_signal REAL,
                    ai_confidence REAL,
                    ai_recommendation TEXT,
                    strategy_signals TEXT,
                    combined_signal REAL,
                    market_sentiment TEXT,
                    risk_assessment TEXT,
                    position_recommendation TEXT
                )
            ''')
            
            # 策略表现跟踪表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    actual_return REAL,
                    actual_sharpe REAL,
                    actual_drawdown REAL,
                    signal_accuracy REAL,
                    weight_used REAL,
                    performance_vs_expected REAL
                )
            ''')
            
            # 优化配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    config_name TEXT,
                    config_data TEXT,
                    description TEXT,
                    is_active INTEGER DEFAULT 1
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 数据库初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
    
    def _load_optimization_history(self):
        """加载历史优化数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 加载最新的优化配置
            df_configs = pd.read_sql_query('''
                SELECT * FROM optimization_configs 
                WHERE is_active = 1 
                ORDER BY timestamp DESC
            ''', conn)
            
            if not df_configs.empty:
                latest_config = df_configs.iloc[0]
                self.optimization_history = json.loads(latest_config['config_data'])
                logger.info(f"✅ 加载历史优化配置: {latest_config['config_name']}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 加载历史优化数据失败: {e}")
    
    def save_optimization_results(self, optimization_results: Dict[str, Any], 
                                 config_name: str = None) -> bool:
        """保存优化结果到数据库"""
        try:
            print(f"💾 保存优化结果到数据库...")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            config_name = config_name or f"optimization_{timestamp[:10]}"
            
            # 保存优化配置
            config_data = {
                'optimization_results': optimization_results,
                'timestamp': timestamp,
                'version': '1.0'
            }
            
            cursor.execute('''
                INSERT INTO optimization_configs 
                (timestamp, config_name, config_data, description)
                VALUES (?, ?, ?, ?)
            ''', (
                timestamp, config_name, json.dumps(config_data),
                f"智能策略优化配置 - {len(optimization_results)} 只股票"
            ))
            
            # 保存详细的策略权重历史
            for symbol, result in optimization_results.items():
                weights = result['optimal_weights']
                performance = result['expected_performance']
                
                for strategy, weight in weights.items():
                    cursor.execute('''
                        INSERT INTO strategy_optimization_history 
                        (timestamp, symbol, strategy, weight, performance_score, 
                         expected_return, expected_sharpe, expected_drawdown, 
                         market_condition, risk_level, optimization_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp, symbol, strategy, weight,
                        result['strategy_scores'].get(strategy, 0),
                        performance['total_return'], performance['sharpe_ratio'],
                        performance['max_drawdown'], result['market_condition'],
                        self._calculate_risk_level(performance), '1.0'
                    ))
            
            conn.commit()
            conn.close()
            
            # 更新内存中的历史数据
            self.optimization_history = config_data
            
            print(f"✅ 优化结果已保存: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存优化结果失败: {e}")
            return False
    
    def get_optimized_weights(self, symbol: str) -> Dict[str, float]:
        """获取指定股票的优化权重"""
        try:
            if not self.optimization_history:
                return {}
            
            results = self.optimization_history.get('optimization_results', {})
            if symbol in results:
                return results[symbol]['optimal_weights']
            
            return {}
            
        except Exception as e:
            logger.error(f"获取优化权重失败: {e}")
            return {}
    
    def get_all_optimized_weights(self) -> Dict[str, Dict[str, float]]:
        """获取所有股票的优化权重"""
        try:
            if not self.optimization_history:
                return {}
            
            results = self.optimization_history.get('optimization_results', {})
            return {symbol: result['optimal_weights'] for symbol, result in results.items()}
            
        except Exception as e:
            logger.error(f"获取所有优化权重失败: {e}")
            return {}
    
    def integrate_with_ai_analysis(self, symbol: str, ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """将优化权重与AI分析结果集成"""
        try:
            print(f"🤖 集成AI分析: {symbol}")
            
            # 获取优化权重
            optimized_weights = self.get_optimized_weights(symbol)
            
            if not optimized_weights:
                logger.warning(f"未找到 {symbol} 的优化权重")
                return ai_analysis
            
            # 获取AI信号
            ai_signal = ai_analysis.get('signal', 0)
            ai_confidence = ai_analysis.get('confidence', 0.5)
            
            # 获取策略信号
            strategy_signals = ai_analysis.get('strategy_signals', {})
            
            # 计算加权组合信号
            weighted_signal = 0
            total_weight = 0
            
            for strategy, weight in optimized_weights.items():
                if strategy in strategy_signals:
                    strategy_signal = strategy_signals[strategy].get('signal', 0)
                    weighted_signal += strategy_signal * weight
                    total_weight += weight
            
            # 归一化加权信号
            if total_weight > 0:
                weighted_signal = weighted_signal / total_weight
            
            # AI权重调整（基于AI置信度）
            ai_weight = min(0.4, ai_confidence * 0.4)  # 最大40%权重给AI
            strategy_weight = 1 - ai_weight
            
            # 最终组合信号
            final_signal = ai_signal * ai_weight + weighted_signal * strategy_weight
            
            # 生成集成结果
            integrated_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'ai_analysis': ai_analysis,
                'optimized_weights': optimized_weights,
                'weighted_strategy_signal': weighted_signal,
                'ai_weight': ai_weight,
                'strategy_weight': strategy_weight,
                'final_signal': final_signal,
                'signal_strength': abs(final_signal),
                'recommendation': self._generate_recommendation(final_signal, ai_confidence),
                'risk_assessment': self._assess_risk(optimized_weights, ai_analysis),
                'position_size': self._calculate_position_size(final_signal, ai_confidence)
            }
            
            # 保存到数据库
            self._save_ai_integration_result(integrated_result)
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"AI集成失败: {e}")
            return ai_analysis
    
    def _generate_recommendation(self, signal: float, confidence: float) -> str:
        """生成投资建议"""
        if signal > 0.3 and confidence > 0.7:
            return "强烈买入"
        elif signal > 0.1 and confidence > 0.6:
            return "买入"
        elif signal < -0.3 and confidence > 0.7:
            return "强烈卖出"
        elif signal < -0.1 and confidence > 0.6:
            return "卖出"
        else:
            return "持有"
    
    def _assess_risk(self, weights: Dict[str, float], ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """评估风险"""
        # 计算权重风险
        high_risk_strategies = ['CPGW']
        risk_score = sum(weights.get(s, 0) for s in high_risk_strategies)
        
        # 市场风险
        market_risk = ai_analysis.get('market_risk', 0.5)
        
        # 综合风险评分
        total_risk = (risk_score * 0.6 + market_risk * 0.4)
        
        return {
            'risk_score': total_risk,
            'risk_level': 'high' if total_risk > 0.6 else 'medium' if total_risk > 0.3 else 'low',
            'weight_risk': risk_score,
            'market_risk': market_risk
        }
    
    def _calculate_position_size(self, signal: float, confidence: float) -> float:
        """计算建议仓位大小"""
        base_size = abs(signal) * 0.3  # 基础仓位
        confidence_boost = confidence * 0.2  # 置信度加成
        return min(0.5, base_size + confidence_boost)  # 最大50%仓位
    
    def _save_ai_integration_result(self, result: Dict[str, Any]):
        """保存AI集成结果"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_analysis_results 
                (timestamp, symbol, ai_signal, ai_confidence, ai_recommendation,
                 strategy_signals, combined_signal, market_sentiment, 
                 risk_assessment, position_recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp'], result['symbol'],
                result['ai_analysis'].get('signal', 0),
                result['ai_analysis'].get('confidence', 0.5),
                result['recommendation'],
                json.dumps(result['optimized_weights']),
                result['final_signal'],
                result['ai_analysis'].get('market_sentiment', 'neutral'),
                json.dumps(result['risk_assessment']),
                f"仓位: {result['position_size']:.1%}"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存AI集成结果失败: {e}")
    
    def track_strategy_performance(self, symbol: str, strategy: str, 
                                 actual_return: float, actual_sharpe: float,
                                 actual_drawdown: float, signal_accuracy: float,
                                 weight_used: float):
        """跟踪策略实际表现"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取预期表现
            expected_performance = self._get_expected_performance(symbol, strategy)
            
            # 计算表现差异
            performance_vs_expected = actual_return - expected_performance.get('return', 0)
            
            cursor.execute('''
                INSERT INTO strategy_performance_tracking 
                (timestamp, symbol, strategy, actual_return, actual_sharpe,
                 actual_drawdown, signal_accuracy, weight_used, performance_vs_expected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(), symbol, strategy, actual_return,
                actual_sharpe, actual_drawdown, signal_accuracy, weight_used,
                performance_vs_expected
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"跟踪策略表现失败: {e}")
    
    def _get_expected_performance(self, symbol: str, strategy: str) -> Dict[str, float]:
        """获取预期表现"""
        try:
            if not self.optimization_history:
                return {}
            
            results = self.optimization_history.get('optimization_results', {})
            if symbol in results:
                return results[symbol]['expected_performance']
            
            return {}
            
        except Exception as e:
            logger.error(f"获取预期表现失败: {e}")
            return {}
    
    def _calculate_risk_level(self, performance: Dict[str, float]) -> str:
        """计算风险等级"""
        try:
            sharpe = performance.get('sharpe_ratio', 0)
            drawdown = abs(performance.get('max_drawdown', 0))
            volatility = performance.get('volatility', 0.2)
            
            if sharpe > 0.8 and drawdown < 0.1 and volatility < 0.2:
                return 'low'
            elif sharpe > 0.5 and drawdown < 0.15 and volatility < 0.3:
                return 'medium'
            else:
                return 'high'
        except Exception as e:
            logger.error(f"计算风险等级失败: {e}")
            return 'medium'
    
    def generate_performance_report(self, days: int = 30) -> str:
        """生成性能报告"""
        try:
            print(f"📊 生成性能报告 (最近{days}天)...")
            
            conn = sqlite3.connect(self.db_path)
            
            # 获取最近的AI分析结果
            df_ai = pd.read_sql_query(f'''
                SELECT * FROM ai_analysis_results 
                WHERE timestamp >= datetime('now', '-{days} days')
                ORDER BY timestamp DESC
            ''', conn)
            
            # 获取策略表现跟踪
            df_performance = pd.read_sql_query(f'''
                SELECT * FROM strategy_performance_tracking 
                WHERE timestamp >= datetime('now', '-{days} days')
                ORDER BY timestamp DESC
            ''', conn)
            
            conn.close()
            
            report = []
            report.append("=" * 80)
            report.append("📊 AI策略优化集成系统性能报告")
            report.append("=" * 80)
            report.append(f"📅 报告期间: 最近{days}天")
            report.append(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # AI分析统计
            if not df_ai.empty:
                report.append("🤖 AI分析统计:")
                report.append(f"  总分析次数: {len(df_ai)}")
                report.append(f"  平均AI置信度: {df_ai['ai_confidence'].mean():.2f}")
                report.append(f"  平均信号强度: {df_ai['combined_signal'].abs().mean():.2f}")
                
                # 推荐分布
                recommendations = df_ai['ai_recommendation'].value_counts()
                report.append("  推荐分布:")
                for rec, count in recommendations.items():
                    report.append(f"    {rec}: {count}次 ({count/len(df_ai)*100:.1f}%)")
                report.append("")
            
            # 策略表现统计
            if not df_performance.empty:
                report.append("📈 策略表现统计:")
                report.append(f"  跟踪记录数: {len(df_performance)}")
                report.append(f"  平均实际收益: {df_performance['actual_return'].mean():.2%}")
                report.append(f"  平均夏普比率: {df_performance['actual_sharpe'].mean():.2f}")
                report.append(f"  平均表现差异: {df_performance['performance_vs_expected'].mean():.2%}")
                
                # 按策略统计
                strategy_stats = df_performance.groupby('strategy').agg({
                    'actual_return': ['mean', 'std'],
                    'signal_accuracy': 'mean',
                    'performance_vs_expected': 'mean'
                }).round(4)
                
                report.append("  按策略统计:")
                for strategy in strategy_stats.index:
                    stats = strategy_stats.loc[strategy]
                    report.append(f"    {strategy}:")
                    report.append(f"      平均收益: {stats[('actual_return', 'mean')]:.2%}")
                    report.append(f"      信号准确率: {stats[('signal_accuracy', 'mean')]:.1%}")
                    report.append(f"      表现差异: {stats[('performance_vs_expected', 'mean')]:.2%}")
                report.append("")
            
            # 优化建议
            report.append("💡 优化建议:")
            if not df_performance.empty:
                avg_performance_diff = df_performance['performance_vs_expected'].mean()
                if avg_performance_diff < -0.05:
                    report.append("  ⚠️ 实际表现低于预期，建议重新优化策略权重")
                elif avg_performance_diff > 0.05:
                    report.append("  ✅ 实际表现优于预期，当前权重配置有效")
                else:
                    report.append("  ✅ 实际表现符合预期，权重配置合理")
            
            report.append("  🔄 建议定期更新优化配置")
            report.append("  📊 持续监控策略表现")
            report.append("  🤖 结合AI分析动态调整")
            
            report_text = "\n".join(report)
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"ai_optimization_performance_report_{timestamp}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"📊 性能报告已保存到: {report_filename}")
            return report_text
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return "报告生成失败"
    
    def export_optimization_data(self, export_path: str = None) -> str:
        """导出优化数据"""
        try:
            if not export_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"optimization_data_export_{timestamp}.json"
            
            # 准备导出数据
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'optimization_history': self.optimization_history,
                'database_summary': self._get_database_summary()
            }
            
            # 保存到文件
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"📤 优化数据已导出到: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"导出优化数据失败: {e}")
            return ""
    
    def _get_database_summary(self) -> Dict[str, Any]:
        """获取数据库摘要"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 统计各表记录数
            tables = ['strategy_optimization_history', 'ai_analysis_results', 
                     'strategy_performance_tracking', 'optimization_configs']
            
            summary = {}
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                summary[f"{table}_count"] = count
            
            conn.close()
            return summary
            
        except Exception as e:
            logger.error(f"获取数据库摘要失败: {e}")
            return {}

def main():
    """主函数"""
    print("🚀 AI策略优化集成系统")
    print("=" * 80)
    
    # 创建集成系统
    integration = AIStrategyOptimizationIntegration()
    
    # 示例：加载之前的优化结果
    optimization_file = "optimized_portfolio_config_20250715_102348.json"
    if os.path.exists(optimization_file):
        print(f"📂 加载优化配置: {optimization_file}")
        
        with open(optimization_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 提取优化结果
        optimization_results = {}
        for symbol, data in config_data.get('stocks', {}).items():
            optimization_results[symbol] = {
                'optimal_weights': data['strategy_weights'],
                'expected_performance': data['expected_performance'],
                'strategy_scores': {},  # 需要从原始数据补充
                'market_condition': data['market_condition']
            }
        
        # 保存到数据库
        success = integration.save_optimization_results(optimization_results, "回归测试优化配置")
        
        if success:
            print("✅ 优化配置已成功集成到AI系统")
            
            # 生成性能报告
            report = integration.generate_performance_report()
            print("\n" + report)
            
            # 导出数据
            export_path = integration.export_optimization_data()
            print(f"\n📤 数据导出完成: {export_path}")
        else:
            print("❌ 优化配置集成失败")
    else:
        print(f"⚠️ 未找到优化配置文件: {optimization_file}")
    
    print("\n🎉 AI策略优化集成系统演示完成！")
    print("📋 系统功能:")
    print("   ✅ 策略优化结果持久化存储")
    print("   ✅ AI分析与优化权重智能集成")
    print("   ✅ 策略表现实时跟踪")
    print("   ✅ 性能报告自动生成")
    print("   ✅ 数据导出与备份")
    print("   ✅ 风险评估与仓位建议")

if __name__ == "__main__":
    main() 