#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("strategy_optimizer")

# 导入数据接口和策略
from data.data_interface import DataInterface
from strategy.custom_cpgw_strategy import CustomCPGWStrategy
from strategy.uss_gold_triangle_strategy import GoldTriangleStrategy
from strategy.tdi_strategy import TDIStrategy
from strategy.market_forecast_strategy import MarketForecastStrategy
from strategy.composite_strategy import CompositeStrategy
from backtest.strategy_evaluator import StrategyEvaluator

def plot_strategy_results(result_df, title, symbol, output_dir):
    """绘制策略结果图表"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f"{title} - {symbol}", fontsize=16)
        
        # 在上图中绘制价格
        ax1.plot(result_df.index, result_df['close'], label='价格', color='black')
        
        # 标记买入和卖出信号
        buy_signals = result_df[result_df['signal'] > 0]
        sell_signals = result_df[result_df['signal'] < 0]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, label='买入信号')
        ax1.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, label='卖出信号')
        
        # 添加图例和标签
        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title(f"{symbol} 价格和交易信号", fontsize=14)
        ax1.legend()
        ax1.grid(True)
        
        # 在下图中绘制信号强度
        ax2.fill_between(result_df.index, 0, result_df['signal'], color='blue', alpha=0.3, label='信号强度')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('信号强度', fontsize=12)
        ax2.set_title('信号强度', fontsize=14)
        ax2.grid(True)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图表
        plt.savefig(f"{output_dir}/{symbol}_{title.replace(' ', '_')}_signals.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图表已保存至: {output_dir}/{symbol}_{title.replace(' ', '_')}_signals.png")
        
    except Exception as e:
        logger.error(f"绘制图表时出错: {e}")

def test_optimized_strategies_evaluation():
    """测试优化后的策略评估"""
    logger.info("开始测试优化后的策略系统...")
    
    # 股票符号列表
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # 初始化数据接口
    try:
        data_interface = DataInterface(default_source="mysql")
    except Exception as e:
        logger.error(f"数据接口初始化失败: {e}")
        logger.info("尝试使用默认配置初始化...")
        data_interface = DataInterface()
    
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
    
    # 创建CompositeStrategy的派生类，实现必要的抽象方法
    class EnhancedCompositeStrategy(CompositeStrategy):
        def extract_signal_components(self, data: pd.DataFrame) -> dict:
            """实现抽象方法：提取信号组件"""
            components = {}
            if 'signal' in data.columns:
                components['combined_signal'] = data['signal']
            return components
            
        def get_signal_metadata(self) -> dict:
            """实现抽象方法：获取信号元数据"""
            return {
                'combined_signal': {
                    'name': '组合信号',
                    'description': '多个策略的加权组合信号',
                    'type': 'continuous',
                    'range': [-1, 1]
                }
            }
    
    # 使用增强版复合策略
    composite_strategy = EnhancedCompositeStrategy(parameters=composite_parameters)
    composite_strategy.add_strategy(cpgw_strategy, 0.25)
    composite_strategy.add_strategy(gold_triangle_strategy, 0.25)
    composite_strategy.add_strategy(tdi_strategy, 0.25)
    composite_strategy.add_strategy(market_forecast_strategy, 0.25)
    
    # 测试所有策略并收集绩效指标
    performance_results = {}
    
    for symbol, df in data.items():
        logger.info(f"\n========== 测试优化策略 {symbol} ==========")
        
        symbol_results = {}
        
        # 测试优化版CPGW策略
        logger.info(f"测试优化版CPGW策略: {symbol}")
        cpgw_signals = cpgw_strategy.generate_signals(df)
        # 确保列名一致
        if 'signal' in cpgw_signals.columns and 'Close' in cpgw_signals.columns and 'close' not in cpgw_signals.columns:
            cpgw_signals['close'] = cpgw_signals['Close']
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
        plot_strategy_results(cpgw_signals, "优化CPGW策略", symbol, output_dir)
        
        # 测试优化版金三角策略
        logger.info(f"测试优化版金三角策略: {symbol}")
        gold_signals = gold_triangle_strategy.generate_signals(df)
        # 确保列名一致
        if 'signal' in gold_signals.columns and 'Close' in gold_signals.columns and 'close' not in gold_signals.columns:
            gold_signals['close'] = gold_signals['Close']
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
        plot_strategy_results(gold_signals, "优化金三角策略", symbol, output_dir)
        
        # 测试TDI策略
        logger.info(f"测试TDI策略: {symbol}")
        tdi_signals = tdi_strategy.generate_signals(df)
        # 确保列名一致
        if 'signal' in tdi_signals.columns and 'Close' in tdi_signals.columns and 'close' not in tdi_signals.columns:
            tdi_signals['close'] = tdi_signals['Close']
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
        plot_strategy_results(tdi_signals, "TDI策略", symbol, output_dir)
        
        # 测试市场预测策略
        logger.info(f"测试市场预测策略: {symbol}")
        mf_signals = market_forecast_strategy.generate_signals(df)
        # 确保列名一致
        if 'signal' in mf_signals.columns and 'Close' in mf_signals.columns and 'close' not in mf_signals.columns:
            mf_signals['close'] = mf_signals['Close']
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
        plot_strategy_results(mf_signals, "市场预测策略", symbol, output_dir)
        
        # 测试复合策略
        logger.info(f"测试优化复合策略: {symbol}")
        composite_signals = composite_strategy.generate_signals(df)
        # 确保列名一致
        if 'signal' in composite_signals.columns and 'Close' in composite_signals.columns and 'close' not in composite_signals.columns:
            composite_signals['close'] = composite_signals['Close']
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
        plot_strategy_results(composite_signals, "优化复合策略", symbol, output_dir)
        
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
                    serializable_results[symbol][strategy_name] = {k: float(v) if isinstance(v, (np.float_, np.float32, np.float64)) else v 
                                                              for k, v in metrics.items()}
            
            json.dump(serializable_results, f, indent=2)
        logger.info(f"绩效结果已保存至 {output_dir}/performance_results.json")
    except Exception as e:
        logger.error(f"保存绩效结果时出错: {e}")
        
    logger.info("优化策略评估测试完成")
    return performance_results

if __name__ == "__main__":
    # 测试优化版策略
    test_optimized_strategies_evaluation() 