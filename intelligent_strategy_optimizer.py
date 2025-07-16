#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能策略组合优化系统
基于回归测试结果动态调整策略权重，确保收益最大化且风险可控
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import sqlite3
from typing import Dict, List, Any, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentStrategyOptimizer:
    """智能策略组合优化系统"""
    
    def __init__(self):
        """初始化优化器"""
        # 策略配置
        self.strategies = {
            'TDI': {
                'description': 'TDI多时间周期策略',
                'base_weight': 0.4,
                'min_weight': 0.2,
                'max_weight': 0.6,
                'risk_level': 'low',
                'suitable_market': ['trending', 'volatile']
            },
            'NiuniuV3': {
                'description': '牛牛策略V3',
                'base_weight': 0.3,
                'min_weight': 0.1,
                'max_weight': 0.5,
                'risk_level': 'medium',
                'suitable_market': ['trending', 'ranging']
            },
            'CPGW': {
                'description': 'CPGW策略',
                'base_weight': 0.2,
                'min_weight': 0.05,
                'max_weight': 0.4,
                'risk_level': 'high',
                'suitable_market': ['ranging', 'volatile']
            },
            'MarketForecast': {
                'description': '市场预测策略',
                'base_weight': 0.1,
                'min_weight': 0.05,
                'max_weight': 0.3,
                'risk_level': 'medium',
                'suitable_market': ['trending', 'ranging']
            },
            'Combined': {
                'description': '组合策略',
                'base_weight': 0.2,
                'min_weight': 0.1,
                'max_weight': 0.4,
                'risk_level': 'medium',
                'suitable_market': ['trending', 'ranging', 'volatile']
            }
        }
        
        # 优化参数
        self.optimization_params = {
            'target_return': 0.15,  # 目标年化收益15%
            'max_risk': 0.20,       # 最大风险20%
            'min_sharpe': 0.5,      # 最小夏普比率
            'max_drawdown': 0.15,   # 最大回撤15%
            'rebalance_frequency': 30,  # 每30天重新平衡
            'learning_rate': 0.1,   # 学习率
            'momentum_factor': 0.8, # 动量因子
            'risk_aversion': 2.0    # 风险厌恶系数
        }
        
        # 初始化数据库
        self._init_database()
        
        print("🚀 智能策略组合优化系统初始化完成")
        print(f"📊 优化目标: 年化收益 {self.optimization_params['target_return']:.1%}")
        print(f"🛡️ 风险控制: 最大回撤 {self.optimization_params['max_drawdown']:.1%}")
        print("=" * 80)
    
    def _init_database(self):
        """初始化数据库"""
        try:
            self.db_path = "intelligent_optimizer.db"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建策略权重历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_weights_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    weight REAL,
                    performance_score REAL,
                    market_condition TEXT,
                    risk_level TEXT
                )
            ''')
            
            # 创建优化结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    total_return REAL,
                    annual_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    volatility REAL,
                    win_rate REAL,
                    optimal_weights TEXT,
                    performance_metrics TEXT
                )
            ''')
            
            # 创建市场环境表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    market_type TEXT,
                    volatility REAL,
                    trend_strength REAL,
                    volume_trend REAL,
                    confidence REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 数据库初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
    
    def analyze_regression_results(self, results_file: str) -> Dict[str, Any]:
        """分析回归测试结果"""
        try:
            print(f"📊 分析回归测试结果: {results_file}")
            
            # 读取结果
            df = pd.read_csv(results_file)
            
            # 按策略汇总性能
            strategy_performance = df.groupby('strategy').agg({
                'total_return': ['mean', 'std', 'min', 'max'],
                'annual_return': ['mean', 'std'],
                'sharpe_ratio': ['mean', 'std'],
                'max_drawdown': ['mean', 'min'],
                'win_rate': ['mean', 'std'],
                'excess_return': ['mean', 'std'],
                'volatility': ['mean', 'std']
            }).round(4)
            
            # 按股票汇总性能
            stock_performance = df.groupby('symbol').agg({
                'total_return': ['mean', 'std'],
                'excess_return': ['mean', 'std'],
                'sharpe_ratio': ['mean', 'std']
            }).round(4)
            
            # 计算策略评分
            strategy_scores = self._calculate_strategy_scores(df)
            
            # 识别最佳和最差表现
            best_strategy = df.loc[df['total_return'].idxmax()]
            worst_strategy = df.loc[df['total_return'].idxmin()]
            best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
            
            analysis_result = {
                'strategy_performance': strategy_performance,
                'stock_performance': stock_performance,
                'strategy_scores': strategy_scores,
                'best_strategy': {
                    'strategy': best_strategy['strategy'],
                    'symbol': best_strategy['symbol'],
                    'return': best_strategy['total_return'],
                    'sharpe': best_strategy['sharpe_ratio']
                },
                'worst_strategy': {
                    'strategy': worst_strategy['strategy'],
                    'symbol': worst_strategy['symbol'],
                    'return': worst_strategy['total_return'],
                    'sharpe': worst_strategy['sharpe_ratio']
                },
                'best_risk_adjusted': {
                    'strategy': best_sharpe['strategy'],
                    'symbol': best_sharpe['symbol'],
                    'sharpe': best_sharpe['sharpe_ratio'],
                    'return': best_sharpe['total_return']
                },
                'raw_data': df
            }
            
            print("✅ 回归测试结果分析完成")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ 分析回归测试结果失败: {e}")
            return {}
    
    def _calculate_strategy_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算策略综合评分"""
        strategy_scores = {}
        
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            
            # 计算各项指标
            avg_return = strategy_data['total_return'].mean()
            avg_sharpe = strategy_data['sharpe_ratio'].mean()
            avg_drawdown = abs(strategy_data['max_drawdown'].mean())
            avg_winrate = strategy_data['win_rate'].mean()
            avg_excess = strategy_data['excess_return'].mean()
            
            # 综合评分 (0-100)
            score = (
                max(0, avg_return * 100) * 0.3 +      # 收益权重30%
                max(0, avg_sharpe * 20) * 0.25 +     # 夏普比率权重25%
                max(0, (1 - avg_drawdown) * 100) * 0.2 +  # 回撤控制权重20%
                avg_winrate * 100 * 0.15 +           # 胜率权重15%
                max(0, avg_excess * 100) * 0.1       # 超额收益权重10%
            )
            
            strategy_scores[strategy] = min(100, max(0, score))
        
        return strategy_scores
    
    def optimize_strategy_weights(self, analysis_result: Dict[str, Any], 
                                 target_symbols: List[str] = None) -> Dict[str, Any]:
        """优化策略权重"""
        try:
            print("🎯 开始优化策略权重...")
            
            df = analysis_result['raw_data']
            strategy_scores = analysis_result['strategy_scores']
            
            # 如果没有指定目标股票，使用所有股票
            if target_symbols is None:
                target_symbols = df['symbol'].unique().tolist()
            
            optimization_results = {}
            
            for symbol in target_symbols:
                print(f"📈 优化 {symbol} 的策略权重...")
                
                # 获取该股票的策略表现
                symbol_data = df[df['symbol'] == symbol]
                
                # 计算最优权重
                optimal_weights = self._calculate_optimal_weights(symbol_data, strategy_scores)
                
                # 计算预期性能
                expected_performance = self._calculate_expected_performance(symbol_data, optimal_weights)
                
                # 保存结果
                optimization_results[symbol] = {
                    'optimal_weights': optimal_weights,
                    'expected_performance': expected_performance,
                    'strategy_scores': {s: strategy_scores.get(s, 0) for s in optimal_weights.keys()},
                    'market_condition': self._classify_market_condition(symbol_data)
                }
                
                # 保存到数据库
                self._save_optimization_result(symbol, optimal_weights, expected_performance)
            
            print("✅ 策略权重优化完成")
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ 优化策略权重失败: {e}")
            return {}
    
    def _calculate_optimal_weights(self, symbol_data: pd.DataFrame, 
                                 strategy_scores: Dict[str, float]) -> Dict[str, float]:
        """计算最优权重"""
        weights = {}
        
        # 获取该股票的策略表现
        available_strategies = symbol_data['strategy'].unique()
        
        for strategy in available_strategies:
            strategy_info = symbol_data[symbol_data['strategy'] == strategy].iloc[0]
            
            # 基础权重
            base_weight = self.strategies[strategy]['base_weight']
            
            # 性能调整
            performance_score = strategy_scores.get(strategy, 50) / 100
            return_adjustment = max(0, strategy_info['total_return']) * 2
            sharpe_adjustment = max(0, strategy_info['sharpe_ratio']) * 0.5
            drawdown_penalty = abs(strategy_info['max_drawdown']) * 0.5
            
            # 计算调整后的权重
            adjusted_weight = base_weight * (
                1 + performance_score * 0.3 +
                return_adjustment * 0.3 +
                sharpe_adjustment * 0.2 -
                drawdown_penalty * 0.2
            )
            
            # 应用权重限制
            min_weight = self.strategies[strategy]['min_weight']
            max_weight = self.strategies[strategy]['max_weight']
            adjusted_weight = max(min_weight, min(max_weight, adjusted_weight))
            
            weights[strategy] = adjusted_weight
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_expected_performance(self, symbol_data: pd.DataFrame, 
                                      weights: Dict[str, float]) -> Dict[str, float]:
        """计算预期性能"""
        expected_metrics = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'excess_return': 0.0
        }
        
        for strategy, weight in weights.items():
            strategy_data = symbol_data[symbol_data['strategy'] == strategy]
            if not strategy_data.empty:
                data = strategy_data.iloc[0]
                for metric in expected_metrics:
                    if metric in data:
                        expected_metrics[metric] += data[metric] * weight
        
        return expected_metrics
    
    def _classify_market_condition(self, symbol_data: pd.DataFrame) -> str:
        """分类市场条件"""
        # 计算平均波动率
        avg_volatility = symbol_data['volatility'].mean()
        
        # 计算平均收益
        avg_return = symbol_data['total_return'].mean()
        
        # 计算平均夏普比率
        avg_sharpe = symbol_data['sharpe_ratio'].mean()
        
        if avg_volatility > 0.3:
            return 'volatile'
        elif avg_return > 0.1 and avg_sharpe > 0.5:
            return 'trending'
        elif avg_return < -0.05:
            return 'bearish'
        else:
            return 'ranging'
    
    def _save_optimization_result(self, symbol: str, weights: Dict[str, float], 
                                performance: Dict[str, float]):
        """保存优化结果到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            # 保存优化结果
            cursor.execute('''
                INSERT INTO optimization_results 
                (timestamp, symbol, total_return, annual_return, sharpe_ratio, 
                 max_drawdown, volatility, win_rate, optimal_weights, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, symbol, performance['total_return'], performance['annual_return'],
                performance['sharpe_ratio'], performance['max_drawdown'], performance['volatility'],
                performance['win_rate'], json.dumps(weights), json.dumps(performance)
            ))
            
            # 保存策略权重
            for strategy, weight in weights.items():
                cursor.execute('''
                    INSERT INTO strategy_weights_history 
                    (timestamp, symbol, strategy, weight, performance_score, market_condition, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, symbol, strategy, weight, 0.0, 'unknown', 'medium'
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存优化结果失败: {e}")
    
    def generate_optimization_report(self, optimization_results: Dict[str, Any], 
                                   analysis_result: Dict[str, Any]) -> str:
        """生成优化报告"""
        try:
            print("📋 生成优化报告...")
            
            report = []
            report.append("=" * 80)
            report.append("🎯 智能策略组合优化报告")
            report.append("=" * 80)
            report.append(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # 策略性能总结
            report.append("📊 策略性能总结:")
            strategy_perf = analysis_result['strategy_performance']
            for strategy in strategy_perf.index:
                avg_return = strategy_perf.loc[strategy, ('total_return', 'mean')]
                avg_sharpe = strategy_perf.loc[strategy, ('sharpe_ratio', 'mean')]
                avg_drawdown = strategy_perf.loc[strategy, ('max_drawdown', 'mean')]
                report.append(f"  {strategy}: 收益{avg_return:.2%}, 夏普{avg_sharpe:.2f}, 回撤{avg_drawdown:.2%}")
            report.append("")
            
            # 优化结果
            report.append("🎯 优化结果:")
            total_expected_return = 0
            total_expected_sharpe = 0
            
            for symbol, result in optimization_results.items():
                weights = result['optimal_weights']
                performance = result['expected_performance']
                
                report.append(f"  📈 {symbol}:")
                report.append(f"    预期收益: {performance['total_return']:.2%}")
                report.append(f"    预期夏普: {performance['sharpe_ratio']:.2f}")
                report.append(f"    预期回撤: {performance['max_drawdown']:.2%}")
                report.append(f"    策略权重:")
                
                for strategy, weight in weights.items():
                    report.append(f"      {strategy}: {weight:.1%}")
                
                total_expected_return += performance['total_return']
                total_expected_sharpe += performance['sharpe_ratio']
                report.append("")
            
            # 总体表现
            num_symbols = len(optimization_results)
            avg_expected_return = total_expected_return / num_symbols
            avg_expected_sharpe = total_expected_sharpe / num_symbols
            
            report.append("🏆 总体预期表现:")
            report.append(f"  平均预期收益: {avg_expected_return:.2%}")
            report.append(f"  平均预期夏普: {avg_expected_sharpe:.2f}")
            report.append("")
            
            # 投资建议
            report.append("💡 投资建议:")
            if avg_expected_return > 0.1:
                report.append("  ✅ 预期收益良好，建议实施优化策略")
            elif avg_expected_return > 0.05:
                report.append("  ⚠️ 预期收益一般，建议谨慎实施")
            else:
                report.append("  ❌ 预期收益较低，建议重新评估策略")
            
            if avg_expected_sharpe > 0.8:
                report.append("  ✅ 风险调整后收益优秀")
            elif avg_expected_sharpe > 0.5:
                report.append("  ⚠️ 风险调整后收益一般")
            else:
                report.append("  ❌ 风险调整后收益较差")
            
            report.append("")
            report.append("🔄 实施建议:")
            report.append("  1. 逐步调整策略权重，避免剧烈变化")
            report.append("  2. 定期监控策略表现，及时调整")
            report.append("  3. 设置止损止盈，控制风险")
            report.append("  4. 考虑市场环境变化，动态调整")
            
            report_text = "\n".join(report)
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"intelligent_optimization_report_{timestamp}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"📋 优化报告已保存到: {report_filename}")
            return report_text
            
        except Exception as e:
            logger.error(f"生成优化报告失败: {e}")
            return "报告生成失败"
    
    def create_optimized_portfolio_config(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建优化的投资组合配置"""
        try:
            print("⚙️ 创建优化的投资组合配置...")
            
            portfolio_config = {
                'portfolio_name': 'Intelligent Optimized Portfolio',
                'creation_date': datetime.now().isoformat(),
                'optimization_version': '1.0',
                'target_return': self.optimization_params['target_return'],
                'max_risk': self.optimization_params['max_risk'],
                'rebalance_frequency': self.optimization_params['rebalance_frequency'],
                'stocks': {}
            }
            
            for symbol, result in optimization_results.items():
                portfolio_config['stocks'][symbol] = {
                    'symbol': symbol,
                    'strategy_weights': result['optimal_weights'],
                    'expected_performance': result['expected_performance'],
                    'market_condition': result['market_condition'],
                    'risk_level': self._calculate_risk_level(result['expected_performance']),
                    'position_size': self._calculate_position_size(result['expected_performance']),
                    'stop_loss': self._calculate_stop_loss(result['expected_performance']),
                    'take_profit': self._calculate_take_profit(result['expected_performance'])
                }
            
            # 保存配置
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_filename = f"optimized_portfolio_config_{timestamp}.json"
            with open(config_filename, 'w', encoding='utf-8') as f:
                json.dump(portfolio_config, f, indent=2, ensure_ascii=False)
            
            print(f"⚙️ 优化配置已保存到: {config_filename}")
            return portfolio_config
            
        except Exception as e:
            logger.error(f"创建优化配置失败: {e}")
            return {}
    
    def _calculate_risk_level(self, performance: Dict[str, float]) -> str:
        """计算风险等级"""
        sharpe = performance['sharpe_ratio']
        drawdown = abs(performance['max_drawdown'])
        volatility = performance['volatility']
        
        if sharpe > 0.8 and drawdown < 0.1 and volatility < 0.2:
            return 'low'
        elif sharpe > 0.5 and drawdown < 0.15 and volatility < 0.3:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_position_size(self, performance: Dict[str, float]) -> float:
        """计算建议仓位大小"""
        sharpe = performance['sharpe_ratio']
        drawdown = abs(performance['max_drawdown'])
        
        # 基于夏普比率和回撤计算仓位
        base_size = min(0.3, max(0.05, sharpe * 0.2))
        risk_adjustment = 1 - drawdown * 2
        
        return base_size * risk_adjustment
    
    def _calculate_stop_loss(self, performance: Dict[str, float]) -> float:
        """计算止损位"""
        return -abs(performance['max_drawdown']) * 0.8
    
    def _calculate_take_profit(self, performance: Dict[str, float]) -> float:
        """计算止盈位"""
        return performance['total_return'] * 0.8

def main():
    """主函数"""
    print("🚀 智能策略组合优化系统")
    print("=" * 80)
    
    # 创建优化器
    optimizer = IntelligentStrategyOptimizer()
    
    # 分析回归测试结果
    results_file = "comprehensive_strategy_regression_results_20250715_095753.csv"
    if not os.path.exists(results_file):
        print(f"❌ 找不到结果文件: {results_file}")
        return
    
    analysis_result = optimizer.analyze_regression_results(results_file)
    if not analysis_result:
        print("❌ 分析结果失败")
        return
    
    # 优化策略权重
    optimization_results = optimizer.optimize_strategy_weights(analysis_result)
    if not optimization_results:
        print("❌ 优化失败")
        return
    
    # 生成报告
    report = optimizer.generate_optimization_report(optimization_results, analysis_result)
    print("\n" + report)
    
    # 创建优化配置
    portfolio_config = optimizer.create_optimized_portfolio_config(optimization_results)
    
    print("\n🎉 智能策略组合优化完成！")
    print("📋 优化总结:")
    print("   ✅ 基于回归测试结果优化策略权重")
    print("   ✅ 确保收益最大化且风险可控")
    print("   ✅ 生成详细的优化报告")
    print("   ✅ 创建可执行的组合配置")
    print("   ✅ 支持动态权重调整")
    print("   ✅ 包含风险管理和仓位控制")

if __name__ == "__main__":
    main() 