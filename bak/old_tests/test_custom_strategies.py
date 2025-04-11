#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('strategy_test')

# 导入策略
from strategy.cpgw_strategy import CPGWStrategy
from strategy.tdi_strategy import TDIStrategy
from strategy.uss_gold_triangle_strategy import GoldTriangleStrategy
from strategy.uss_market_forecast_strategy import MarketForecastStrategy
from strategy.composite_strategy import CompositeStrategy
from strategy.mean_reversion_strategy import MeanReversionStrategy
from strategy.breakout_strategy import BreakoutStrategy
from strategy.enhanced_momentum_strategy import EnhancedMomentumStrategy
from strategy.strategy_manager import StrategyManager
from strategy.custom_cpgw_strategy import CustomCPGWStrategy  # 导入新的CustomCPGWStrategy

# 导入数据源接口
from data.data_interface import DataInterface, MySQLDataSource

# 定义测试函数
def test_custom_strategies():
    """测试自定义策略"""
    logger.info("开始测试自定义策略系统...")
    
    # 股票符号列表
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # 初始化数据接口，使用MySQL数据源
    data_interface = DataInterface(default_source="mysql")
    
    # 获取过去一年的历史数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = {}
    for symbol in symbols:
        try:
            logger.info(f"获取{symbol}的历史数据...")
            df = data_interface.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='daily'
            )
            
            if df is not None and not df.empty:
                data[symbol] = df
                logger.info(f"获取到{symbol}的历史数据: {len(df)}条记录")
            else:
                logger.warning(f"无法获取{symbol}的历史数据，跳过此股票")
        except Exception as e:
            logger.error(f"获取{symbol}数据出错: {str(e)}")
    
    # 如果没有获取到任何数据，退出
    if not data:
        logger.error("没有获取到任何数据，无法进行测试")
        return None
    
    # 确保输出目录存在
    output_dir = 'strategy_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试各个策略
    results = {}
    
    for symbol, df in data.items():
        logger.info(f"\n========== 测试 {symbol} ==========")
        
        # 测试CPGW策略
        results[f"{symbol}_cpgw"] = test_cpgw_strategy(symbol, df, output_dir)
        
        # 测试TDI策略
        results[f"{symbol}_tdi"] = test_tdi_strategy(symbol, df, output_dir)
        
        # 测试金三角策略
        results[f"{symbol}_gold"] = test_gold_triangle_strategy(symbol, df, output_dir)
        
        # 测试市场预测策略
        results[f"{symbol}_mf"] = test_market_forecast_strategy(symbol, df, output_dir)
        
        # 测试复合策略
        results[f"{symbol}_composite"] = test_composite_strategy(symbol, df, output_dir)
    
    # 测试策略管理器
    test_strategy_manager(data, output_dir)
    
    logger.info("自定义策略系统测试完成")
    return results

def test_cpgw_strategy(symbol, data, output_dir):
    """测试CustomCPGWStrategy策略"""
    strategy = CustomCPGWStrategy()
    logger.info(f"开始测试CustomCPGWStrategy策略: {symbol}")
    
    # 生成信号
    result = strategy.generate_signals(data)
    
    # 获取信号组件
    signal_components = strategy.extract_signal_components(data)
    
    # 输出信号统计
    buy_signals = result[result['signal'] > 0]
    sell_signals = result[result['signal'] < 0]
    logger.info(f"{symbol} - CustomCPGWStrategy: 买入信号 {len(buy_signals)}, 卖出信号 {len(sell_signals)}")
    
    if len(buy_signals) > 0:
        first_buy = buy_signals.iloc[0]
        logger.info(f"首个买入信号: {first_buy.name} 价格: {first_buy['close']}")
    
    if len(sell_signals) > 0:
        first_sell = sell_signals.iloc[0]
        logger.info(f"首个卖出信号: {first_sell.name} 价格: {first_sell['close']}")
    
    # 输出市场环境统计
    market_regime_counts = {}
    for i in range(len(data)):
        regime = strategy.get_market_regime(data.iloc[:i+1] if i > 0 else data.iloc[:1])
        if regime in market_regime_counts:
            market_regime_counts[regime] += 1
        else:
            market_regime_counts[regime] = 1
    
    logger.info(f"{symbol} - 市场环境统计: {market_regime_counts}")
    
    # 绘制策略结果
    plot_strategy_results(result, signal_components, strategy.name, symbol, output_dir)
    
    return result

def test_tdi_strategy(symbol, data, output_dir):
    """测试TDI策略"""
    strategy = TDIStrategy()
    logger.info(f"开始测试TDI策略: {symbol}")
    
    # 生成信号
    result = strategy.generate_signals(data)
    
    # 获取信号组件
    signal_components = strategy.extract_signal_components(data)
    
    # 输出信号统计
    buy_signals = result[result['signal'] > 0]
    sell_signals = result[result['signal'] < 0]
    logger.info(f"{symbol} - TDI策略: 买入信号 {len(buy_signals)}, 卖出信号 {len(sell_signals)}")
    
    if len(buy_signals) > 0:
        first_buy = buy_signals.iloc[0]
        logger.info(f"首个买入信号: {first_buy.name} 价格: {first_buy['close']}")
    
    if len(sell_signals) > 0:
        first_sell = sell_signals.iloc[0]
        logger.info(f"首个卖出信号: {first_sell.name} 价格: {first_sell['close']}")
    
    # 绘制策略结果
    plot_strategy_results(result, signal_components, strategy.name, symbol, output_dir)
    
    return result

def test_gold_triangle_strategy(symbol, data, output_dir):
    """测试GoldTriangle策略"""
    strategy = GoldTriangleStrategy()
    logger.info(f"开始测试金三角策略: {symbol}")
    
    # 生成信号
    result = strategy.generate_signals(data)
    
    # 获取信号组件
    signal_components = strategy.extract_signal_components(data)
    
    # 输出信号统计
    buy_signals = result[result['signal'] > 0]
    sell_signals = result[result['signal'] < 0]
    logger.info(f"{symbol} - 金三角策略: 买入信号 {len(buy_signals)}, 卖出信号 {len(sell_signals)}")
    
    if len(buy_signals) > 0:
        first_buy = buy_signals.iloc[0]
        logger.info(f"首个买入信号: {first_buy.name} 价格: {first_buy['close']}")
    
    if len(sell_signals) > 0:
        first_sell = sell_signals.iloc[0]
        logger.info(f"首个卖出信号: {first_sell.name} 价格: {first_sell['close']}")
    
    # 绘制策略结果
    plot_strategy_results(result, signal_components, strategy.name, symbol, output_dir)
    
    return result

def test_market_forecast_strategy(symbol, data, output_dir):
    """测试MarketForecast策略"""
    strategy = MarketForecastStrategy()
    logger.info(f"开始测试市场预测策略: {symbol}")
    
    # 生成信号
    result = strategy.generate_signals(data)
    
    # 获取信号组件
    signal_components = strategy.extract_signal_components(data)
    
    # 输出信号统计
    buy_signals = result[result['signal'] > 0]
    sell_signals = result[result['signal'] < 0]
    logger.info(f"{symbol} - 市场预测策略: 买入信号 {len(buy_signals)}, 卖出信号 {len(sell_signals)}")
    
    if len(buy_signals) > 0:
        first_buy = buy_signals.iloc[0]
        logger.info(f"首个买入信号: {first_buy.name} 价格: {first_buy['close']}")
    
    if len(sell_signals) > 0:
        first_sell = sell_signals.iloc[0]
        logger.info(f"首个卖出信号: {first_sell.name} 价格: {first_sell['close']}")
    
    # 绘制策略结果
    plot_strategy_results(result, signal_components, strategy.name, symbol, output_dir)
    
    return result

def test_composite_strategy(symbol, data, output_dir):
    """测试复合策略"""
    # 创建各个子策略
    tdi_strategy = TDIStrategy()
    gold_triangle_strategy = GoldTriangleStrategy()
    market_forecast_strategy = MarketForecastStrategy()
    cpgw_strategy = CustomCPGWStrategy()
    
    # 创建策略列表
    strategies = [tdi_strategy, gold_triangle_strategy, market_forecast_strategy, cpgw_strategy]
    
    # 创建复合策略参数
    parameters = {
        'combination_method': 'weighted_average',
        'adaptive_weights': True,
        'confirmation_threshold': 0.6,
        'use_market_regime': True
    }
    
    # 创建复合策略
    composite_strategy = CompositeStrategy(parameters=parameters, strategies=strategies)
    logger.info(f"开始测试复合策略: {symbol}")
    
    # 为每个策略设置初始权重
    composite_strategy.add_strategy(tdi_strategy, 1.0)
    composite_strategy.add_strategy(gold_triangle_strategy, 1.0)
    composite_strategy.add_strategy(market_forecast_strategy, 1.0)
    composite_strategy.add_strategy(cpgw_strategy, 1.0)
    
    # 生成信号
    result = composite_strategy.generate_signals(data)
    
    # 输出信号统计
    buy_signals = result[result['signal'] > 0]
    sell_signals = result[result['signal'] < 0]
    logger.info(f"{symbol} - 复合策略: 买入信号 {len(buy_signals)}, 卖出信号 {len(sell_signals)}")
    
    if len(buy_signals) > 0:
        first_buy = buy_signals.iloc[0]
        logger.info(f"首个买入信号: {first_buy.name} 价格: {first_buy['close']}")
    
    if len(sell_signals) > 0:
        first_sell = sell_signals.iloc[0]
        logger.info(f"首个卖出信号: {first_sell.name} 价格: {first_sell['close']}")
    
    # 获取各个策略的权重
    logger.info(f"当前策略权重: {composite_strategy.strategy_weights}")
    
    # 绘制复合策略结果
    plot_strategy_results(result, None, composite_strategy.name, symbol, output_dir)
    
    return result

def test_strategy_manager(data, output_dir):
    """测试策略管理器"""
    logger.info("开始测试策略管理器...")
    
    # 创建策略管理器
    manager = StrategyManager()
    
    # 获取注册的策略类
    strategy_classes = manager.strategy_classes
    logger.info(f"已注册策略类: {len(strategy_classes)}")
    for cls_name in strategy_classes:
        logger.info(f"  - {cls_name}")
    
    # 创建策略实例
    strategy_names = manager.get_all_strategy_names()
    logger.info(f"可创建的策略: {len(strategy_names)}")
    
    # 创建所有策略的实例
    for strategy_name in strategy_names[:3]:  # 只创建前3个策略用于测试
        try:
            strategy = manager.create_strategy(strategy_name)
            logger.info(f"  - 创建策略: {strategy_name} 成功")
        except Exception as e:
            logger.error(f"  - 创建策略: {strategy_name} 失败: {e}")
    
    # 获取活跃策略
    active_strategies = manager.get_active_strategies()
    logger.info(f"活跃策略实例: {len(active_strategies)}")
    
    # 测试策略管理器的信号生成
    for symbol, df in data.items():
        logger.info(f"使用策略管理器为{symbol}生成信号...")
        
        # 如果有活跃策略，尝试生成合并信号
        if active_strategies:
            signal_info = manager.generate_consolidated_signals({symbol: df})
            if symbol in signal_info:
                logger.info(f"{symbol} - 合并信号: {signal_info[symbol]['signal']}, 最强策略: {signal_info[symbol]['strongest_strategy']}")
            else:
                logger.info(f"{symbol} - 没有生成有效信号")
        else:
            logger.info(f"没有活跃策略，跳过信号生成")

def plot_strategy_results(signals, signal_components, strategy_name, symbol, output_dir):
    """绘制策略结果图表"""
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制价格
    ax.plot(signals.index, signals['close'], label='Price', color='blue', alpha=0.7)
    
    # 标记买入和卖出信号
    buy_signals = signals[signals['signal'] > 0]
    sell_signals = signals[signals['signal'] < 0]
    
    ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
    ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
    
    # 设置图表标题和标签
    ax.set_title(f"{symbol} - {strategy_name} Signals")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    
    # 美化日期轴
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    # 保存图表
    plt.savefig(f"{output_dir}/{symbol}_{strategy_name}_signals.png")
    plt.close()

def test_optimized_strategies_evaluation():
    """测试优化后的策略评估"""
    logger.info("开始测试优化后的策略系统...")
    
    # 股票符号列表
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # 初始化数据接口
    data_interface = DataInterface(default_source="mysql")
    
    # 获取过去一年的历史数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = {}
    for symbol in symbols:
        try:
            logger.info(f"获取{symbol}的历史数据...")
            df = data_interface.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='daily'
            )
            
            if df is not None and not df.empty:
                data[symbol] = df
                logger.info(f"获取到{symbol}的历史数据: {len(df)}条记录")
            else:
                logger.warning(f"无法获取{symbol}的历史数据，跳过此股票")
        except Exception as e:
            logger.error(f"获取{symbol}数据出错: {str(e)}")
    
    # 如果没有获取到任何数据，退出
    if not data:
        logger.error("没有获取到任何数据，无法进行测试")
        return None
    
    # 确保输出目录存在
    output_dir = 'strategy_results/optimized'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建策略实例
    cpgw_strategy = CustomCPGWStrategy()
    gold_triangle_strategy = GoldTriangleStrategy()
    tdi_strategy = TDIStrategy()
    market_forecast_strategy = MarketForecastStrategy()
    
    # 创建优化版复合策略
    composite_parameters = {
        'combination_method': 'weighted_average',
        'adaptive_weights': True,
        'confirmation_threshold': 0.6,
        'use_market_regime': True,
        'market_regime_weight_adjust': True,
        'bullish_weight_boost': 1.5,
        'bearish_weight_boost': 1.5, 
        'ranging_weight_boost': 1.5
    }
    
    composite_strategy = CompositeStrategy(parameters=composite_parameters)
    composite_strategy.add_strategy(cpgw_strategy, 0.25)
    composite_strategy.add_strategy(gold_triangle_strategy, 0.25)
    composite_strategy.add_strategy(tdi_strategy, 0.25)
    composite_strategy.add_strategy(market_forecast_strategy, 0.25)
    
    from backtest.strategy_evaluator import StrategyEvaluator
    
    # 测试所有策略并收集绩效指标
    performance_results = {}
    
    for symbol, df in data.items():
        logger.info(f"\n========== 测试优化策略 {symbol} ==========")
        
        symbol_results = {}
        
        # 测试优化版CPGW策略
        cpgw_signals = cpgw_strategy.generate_signals(df)
        evaluator = StrategyEvaluator(df)
        cpgw_metrics = evaluator.analyze_trades(cpgw_signals['signal'])
        symbol_results['CPGW'] = cpgw_metrics.to_dict()
        logger.info(f"{symbol} - 优化CPGW策略绩效指标:")
        logger.info(f"  总交易次数: {cpgw_metrics.total_trades}")
        logger.info(f"  胜率: {cpgw_metrics.win_rate:.2%}")
        logger.info(f"  夏普比率: {cpgw_metrics.sharpe_ratio:.2f}")
        logger.info(f"  索提诺比率: {cpgw_metrics.sortino_ratio:.2f}")
        logger.info(f"  卡尔马比率: {cpgw_metrics.calmar_ratio:.2f}")
        logger.info(f"  最大回撤: {cpgw_metrics.max_drawdown:.2%}")
        logger.info(f"  年化收益: {cpgw_metrics.annual_return:.2f}%")
        
        # 测试优化版金三角策略
        gold_signals = gold_triangle_strategy.generate_signals(df)
        evaluator = StrategyEvaluator(df)
        gold_metrics = evaluator.analyze_trades(gold_signals['signal'])
        symbol_results['GoldTriangle'] = gold_metrics.to_dict()
        logger.info(f"{symbol} - 优化金三角策略绩效指标:")
        logger.info(f"  总交易次数: {gold_metrics.total_trades}")
        logger.info(f"  胜率: {gold_metrics.win_rate:.2%}")
        logger.info(f"  夏普比率: {gold_metrics.sharpe_ratio:.2f}")
        logger.info(f"  索提诺比率: {gold_metrics.sortino_ratio:.2f}")
        logger.info(f"  卡尔马比率: {gold_metrics.calmar_ratio:.2f}")
        logger.info(f"  最大回撤: {gold_metrics.max_drawdown:.2%}")
        logger.info(f"  年化收益: {gold_metrics.annual_return:.2f}%")
        
        # 测试TDI策略
        tdi_signals = tdi_strategy.generate_signals(df)
        evaluator = StrategyEvaluator(df)
        tdi_metrics = evaluator.analyze_trades(tdi_signals['signal'])
        symbol_results['TDI'] = tdi_metrics.to_dict()
        logger.info(f"{symbol} - TDI策略绩效指标:")
        logger.info(f"  总交易次数: {tdi_metrics.total_trades}")
        logger.info(f"  胜率: {tdi_metrics.win_rate:.2%}")
        logger.info(f"  夏普比率: {tdi_metrics.sharpe_ratio:.2f}")
        logger.info(f"  索提诺比率: {tdi_metrics.sortino_ratio:.2f}")
        logger.info(f"  卡尔马比率: {tdi_metrics.calmar_ratio:.2f}")
        logger.info(f"  最大回撤: {tdi_metrics.max_drawdown:.2%}")
        logger.info(f"  年化收益: {tdi_metrics.annual_return:.2f}%")
        
        # 测试市场预测策略
        mf_signals = market_forecast_strategy.generate_signals(df)
        evaluator = StrategyEvaluator(df)
        mf_metrics = evaluator.analyze_trades(mf_signals['signal'])
        symbol_results['MarketForecast'] = mf_metrics.to_dict()
        logger.info(f"{symbol} - 市场预测策略绩效指标:")
        logger.info(f"  总交易次数: {mf_metrics.total_trades}")
        logger.info(f"  胜率: {mf_metrics.win_rate:.2%}")
        logger.info(f"  夏普比率: {mf_metrics.sharpe_ratio:.2f}")
        logger.info(f"  索提诺比率: {mf_metrics.sortino_ratio:.2f}")
        logger.info(f"  卡尔马比率: {mf_metrics.calmar_ratio:.2f}")
        logger.info(f"  最大回撤: {mf_metrics.max_drawdown:.2%}")
        logger.info(f"  年化收益: {mf_metrics.annual_return:.2f}%")
        
        # 测试复合策略
        composite_signals = composite_strategy.generate_signals(df)
        evaluator = StrategyEvaluator(df)
        composite_metrics = evaluator.analyze_trades(composite_signals['signal'])
        symbol_results['Composite'] = composite_metrics.to_dict()
        logger.info(f"{symbol} - 优化复合策略绩效指标:")
        logger.info(f"  总交易次数: {composite_metrics.total_trades}")
        logger.info(f"  胜率: {composite_metrics.win_rate:.2%}")
        logger.info(f"  夏普比率: {composite_metrics.sharpe_ratio:.2f}")
        logger.info(f"  索提诺比率: {composite_metrics.sortino_ratio:.2f}")
        logger.info(f"  卡尔马比率: {composite_metrics.calmar_ratio:.2f}")
        logger.info(f"  最大回撤: {composite_metrics.max_drawdown:.2%}")
        logger.info(f"  年化收益: {composite_metrics.annual_return:.2f}%")
        logger.info(f"  市场环境适应性指标:")
        logger.info(f"    最大连胜次数: {composite_metrics.max_win_streak}")
        logger.info(f"    最大连亏次数: {composite_metrics.max_lose_streak}")
        logger.info(f"    最大回撤持续期: {composite_metrics.max_drawdown_duration}天")
        logger.info(f"    恢复期: {composite_metrics.recovery_periods}天")
        
        # 输出当前策略权重
        logger.info(f"  当前策略权重: {composite_strategy.strategy_weights}")
        
        # 绘制策略结果
        plot_strategy_results(composite_signals, None, "优化复合策略", symbol, output_dir)
        
        # 保存至结果字典
        performance_results[symbol] = symbol_results
    
    # 生成整体绩效报告
    logger.info("\n========== 策略绩效对比 ==========")
    
    # 计算各策略平均表现
    avg_performance = {}
    
    for strategy_name in ['CPGW', 'GoldTriangle', 'TDI', 'MarketForecast', 'Composite']:
        strategy_metrics = {
            'win_rate': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'annual_return': 0,
            'count': 0
        }
        
        for symbol in performance_results:
            if strategy_name in performance_results[symbol]:
                metrics = performance_results[symbol][strategy_name]
                strategy_metrics['win_rate'] += metrics['win_rate']
                strategy_metrics['sharpe_ratio'] += metrics['sharpe_ratio']
                strategy_metrics['sortino_ratio'] += metrics['sortino_ratio'] 
                strategy_metrics['calmar_ratio'] += metrics['calmar_ratio']
                strategy_metrics['max_drawdown'] += metrics['max_drawdown']
                strategy_metrics['annual_return'] += metrics['annual_return']
                strategy_metrics['count'] += 1
        
        if strategy_metrics['count'] > 0:
            avg_performance[strategy_name] = {
                'win_rate': strategy_metrics['win_rate'] / strategy_metrics['count'],
                'sharpe_ratio': strategy_metrics['sharpe_ratio'] / strategy_metrics['count'],
                'sortino_ratio': strategy_metrics['sortino_ratio'] / strategy_metrics['count'],
                'calmar_ratio': strategy_metrics['calmar_ratio'] / strategy_metrics['count'],
                'max_drawdown': strategy_metrics['max_drawdown'] / strategy_metrics['count'],
                'annual_return': strategy_metrics['annual_return'] / strategy_metrics['count']
            }
    
    # 打印平均表现
    logger.info("\n策略平均表现:")
    for strategy_name, metrics in avg_performance.items():
        logger.info(f"{strategy_name}:")
        logger.info(f"  胜率: {metrics['win_rate']:.2%}")
        logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  索提诺比率: {metrics['sortino_ratio']:.2f}")
        logger.info(f"  卡尔马比率: {metrics['calmar_ratio']:.2f}")
        logger.info(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        logger.info(f"  年化收益: {metrics['annual_return']:.2f}%")

    # 保存绩效结果到文件
    try:
        import json
        with open(f"{output_dir}/performance_results.json", 'w') as f:
            # 将结果转换为可序列化的格式
            serializable_results = {}
            for symbol, strategies in performance_results.items():
                serializable_results[symbol] = {}
                for strategy_name, metrics in strategies.items():
                    serializable_results[symbol][strategy_name] = {k: float(v) if isinstance(v, np.float) else v 
                                                              for k, v in metrics.items()}
            
            json.dump(serializable_results, f, indent=2)
        logger.info(f"绩效结果已保存至 {output_dir}/performance_results.json")
    except Exception as e:
        logger.error(f"保存绩效结果时出错: {e}")
        
    logger.info("优化策略评估测试完成")
    return performance_results

if __name__ == "__main__":
    test_custom_strategies()
    
    # 测试优化版策略
    test_optimized_strategies_evaluation() 