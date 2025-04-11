import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_strategy_system')

# 导入数据接口和策略系统
from data.data_interface import DataInterface
from strategy.strategy_manager import StrategyManager
from strategy.enhanced_momentum_strategy import EnhancedMomentumStrategy
from strategy.mean_reversion_strategy import MeanReversionStrategy
from strategy.breakout_strategy import BreakoutStrategy
from strategy.composite_strategy import CompositeStrategy


def load_test_data(symbols, start_date, end_date=None):
    """
    加载测试数据
    
    参数:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        
    返回:
        股票数据字典
    """
    # 初始化数据接口
    data_interface = DataInterface()
    
    # 准备日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 获取数据
    data_dict = {}
    for symbol in symbols:
        logger.info(f"获取 {symbol} 的历史数据")
        try:
            df = data_interface.get_historical_data(symbol, start_date, end_date)
            if not df.empty:
                data_dict[symbol] = df
                logger.info(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
            else:
                logger.warning(f"未能获取到 {symbol} 的数据")
        except Exception as e:
            logger.error(f"获取 {symbol} 数据时出错: {e}")
    
    return data_dict


def visualize_signals(symbol, data, strategy_name):
    """
    可视化策略信号
    
    参数:
        symbol: 股票代码
        data: 带信号的股票数据
        strategy_name: 策略名称
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制价格和信号
    ax1.set_title(f"{strategy_name} - {symbol}")
    ax1.plot(data.index, data['close'], label='Close Price')
    
    # 买入点和卖出点
    buy_signals = data[data['signal'] == 1]
    sell_signals = data[data['signal'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    # 绘制技术指标 - 根据策略类型选择不同的指标
    if 'rsi' in data.columns and 'Momentum' in strategy_name:
        ax2.set_title('RSI')
        ax2.plot(data.index, data['rsi'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
    
    elif 'momentum' in data.columns and 'Momentum' in strategy_name:
        ax2.set_title('Momentum')
        ax2.plot(data.index, data['momentum'], label='Momentum')
        ax2.axhline(y=0, color='k', linestyle='--')
    
    elif 'price_deviation_pct' in data.columns and 'MeanReversion' in strategy_name:
        ax2.set_title('Price Deviation (%)')
        ax2.plot(data.index, data['price_deviation_pct'], label='Price Deviation')
        ax2.axhline(y=0, color='k', linestyle='--')
        ax2.axhline(y=5, color='r', linestyle='--')
        ax2.axhline(y=-5, color='g', linestyle='--')
    
    elif 'volatility_breakout' in data.columns and 'Breakout' in strategy_name:
        ax2.set_title('Breakout Indicators')
        if 'price_breakout' in data.columns:
            ax2.plot(data.index, data['price_breakout'], label='Price Breakout', alpha=0.7)
        if 'volume_confirm' in data.columns:
            ax2.plot(data.index, data['volume_confirm'], label='Volume Confirm', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--')
    
    elif 'weighted_signal' in data.columns and 'Composite' in strategy_name:
        ax2.set_title('组合信号')
        ax2.plot(data.index, data['weighted_signal'], label='Weighted Signal')
        ax2.axhline(y=0, color='k', linestyle='--')
    
    # 格式化图形
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    
    # 保存图片
    save_dir = 'strategy_results'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{symbol}_{strategy_name}_signals.png")
    logger.info(f"保存图表: {save_dir}/{symbol}_{strategy_name}_signals.png")
    plt.close()


def run_enhanced_momentum_test():
    """测试增强版动量策略"""
    logger.info("开始测试增强版动量策略")
    
    # 创建策略实例
    strategy = EnhancedMomentumStrategy()
    
    # 定义测试股票
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # 加载数据
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    data_dict = load_test_data(symbols, start_date)
    
    if not data_dict:
        logger.error("未能获取到有效的测试数据")
        return
    
    # 运行策略分析
    logger.info("运行增强动量策略分析")
    for symbol, data in data_dict.items():
        # 生成信号
        data_with_signals = strategy.generate_signals(data)
        
        # 提取信号组件
        signal_components = strategy.extract_signal_components(data_with_signals)
        
        # 显示基本统计信息
        logger.info(f"\n{symbol} 分析结果 - 增强动量策略:")
        
        buy_signals = data_with_signals[data_with_signals['signal'] == 1]
        sell_signals = data_with_signals[data_with_signals['signal'] == -1]
        
        logger.info(f"生成 {len(buy_signals)} 个买入信号, {len(sell_signals)} 个卖出信号")
        
        if not buy_signals.empty:
            logger.info(f"第一个买入信号: {buy_signals.index[0]}, 价格: {buy_signals['close'].iloc[0]:.2f}")
        
        if not sell_signals.empty:
            logger.info(f"第一个卖出信号: {sell_signals.index[0]}, 价格: {sell_signals['close'].iloc[0]:.2f}")
        
        # 统计市场环境
        market_regimes = [strategy.get_market_regime(data_with_signals.iloc[i:i+100]) 
                         for i in range(0, len(data_with_signals), 100)]
        regime_counts = pd.Series(market_regimes).value_counts()
        logger.info(f"市场环境统计: {regime_counts}")
        
        # 可视化结果
        visualize_signals(symbol, data_with_signals, strategy.name)


def run_mean_reversion_test():
    """测试均值回归策略"""
    logger.info("开始测试均值回归策略")
    
    # 创建策略实例
    strategy = MeanReversionStrategy()
    
    # 定义测试股票
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # 加载数据
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    data_dict = load_test_data(symbols, start_date)
    
    if not data_dict:
        logger.error("未能获取到有效的测试数据")
        return
    
    # 运行策略分析
    logger.info("运行均值回归策略分析")
    for symbol, data in data_dict.items():
        # 生成信号
        data_with_signals = strategy.generate_signals(data)
        
        # 提取信号组件
        signal_components = strategy.extract_signal_components(data_with_signals)
        
        # 显示基本统计信息
        logger.info(f"\n{symbol} 分析结果 - 均值回归策略:")
        
        buy_signals = data_with_signals[data_with_signals['signal'] == 1]
        sell_signals = data_with_signals[data_with_signals['signal'] == -1]
        
        logger.info(f"生成 {len(buy_signals)} 个买入信号, {len(sell_signals)} 个卖出信号")
        
        if not buy_signals.empty:
            logger.info(f"第一个买入信号: {buy_signals.index[0]}, 价格: {buy_signals['close'].iloc[0]:.2f}")
        
        if not sell_signals.empty:
            logger.info(f"第一个卖出信号: {sell_signals.index[0]}, 价格: {sell_signals['close'].iloc[0]:.2f}")
        
        # 可视化结果
        visualize_signals(symbol, data_with_signals, strategy.name)


def run_breakout_test():
    """测试突破策略"""
    logger.info("开始测试突破策略")
    
    # 创建策略实例
    strategy = BreakoutStrategy()
    
    # 定义测试股票
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # 加载数据
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    data_dict = load_test_data(symbols, start_date)
    
    if not data_dict:
        logger.error("未能获取到有效的测试数据")
        return
    
    # 运行策略分析
    logger.info("运行突破策略分析")
    for symbol, data in data_dict.items():
        # 生成信号
        data_with_signals = strategy.generate_signals(data)
        
        # 提取信号组件
        signal_components = strategy.extract_signal_components(data_with_signals)
        
        # 显示基本统计信息
        logger.info(f"\n{symbol} 分析结果 - 突破策略:")
        
        buy_signals = data_with_signals[data_with_signals['signal'] == 1]
        sell_signals = data_with_signals[data_with_signals['signal'] == -1]
        
        logger.info(f"生成 {len(buy_signals)} 个买入信号, {len(sell_signals)} 个卖出信号")
        
        if not buy_signals.empty:
            logger.info(f"第一个买入信号: {buy_signals.index[0]}, 价格: {buy_signals['close'].iloc[0]:.2f}")
        
        if not sell_signals.empty:
            logger.info(f"第一个卖出信号: {sell_signals.index[0]}, 价格: {sell_signals['close'].iloc[0]:.2f}")
        
        # 可视化结果
        visualize_signals(symbol, data_with_signals, strategy.name)


def run_composite_test():
    """测试组合策略"""
    logger.info("开始测试组合策略")
    
    # 创建单独的策略实例
    momentum_strategy = EnhancedMomentumStrategy()
    mean_reversion_strategy = MeanReversionStrategy()
    breakout_strategy = BreakoutStrategy()
    
    # 创建组合策略实例
    composite_strategy = CompositeStrategy()
    
    # 添加子策略
    composite_strategy.add_strategy(momentum_strategy, weight=0.4)
    composite_strategy.add_strategy(mean_reversion_strategy, weight=0.3)
    composite_strategy.add_strategy(breakout_strategy, weight=0.3)
    
    # 定义测试股票
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # 加载数据
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    data_dict = load_test_data(symbols, start_date)
    
    if not data_dict:
        logger.error("未能获取到有效的测试数据")
        return
    
    # 运行策略分析
    logger.info("运行组合策略分析")
    for symbol, data in data_dict.items():
        # 生成信号
        data_with_signals = composite_strategy.generate_signals(data)
        
        # 显示基本统计信息
        logger.info(f"\n{symbol} 分析结果 - 组合策略:")
        
        buy_signals = data_with_signals[data_with_signals['signal'] == 1]
        sell_signals = data_with_signals[data_with_signals['signal'] == -1]
        
        logger.info(f"生成 {len(buy_signals)} 个买入信号, {len(sell_signals)} 个卖出信号")
        
        if not buy_signals.empty:
            logger.info(f"第一个买入信号: {buy_signals.index[0]}, 价格: {buy_signals['close'].iloc[0]:.2f}")
        
        if not sell_signals.empty:
            logger.info(f"第一个卖出信号: {sell_signals.index[0]}, 价格: {sell_signals['close'].iloc[0]:.2f}")
        
        # 显示策略权重
        logger.info(f"策略权重: {composite_strategy.strategy_weights}")
        
        # 可视化结果
        visualize_signals(symbol, data_with_signals, composite_strategy.name)


def run_strategy_manager_test():
    """测试策略管理器"""
    logger.info("开始测试策略管理器")
    
    # 初始化策略管理器
    manager = StrategyManager()
    
    # 查看发现的策略
    strategy_names = manager.get_all_strategy_names()
    logger.info(f"自动发现的策略: {strategy_names}")
    
    # 如果未自动发现，手动注册策略
    if 'EnhancedMomentumStrategy' not in strategy_names:
        logger.info("未自动发现增强动量策略，手动注册")
        manager.register_strategy('EnhancedMomentumStrategy', EnhancedMomentumStrategy)
    
    if 'MeanReversionStrategy' not in strategy_names:
        logger.info("未自动发现均值回归策略，手动注册")
        manager.register_strategy('MeanReversionStrategy', MeanReversionStrategy)
    
    if 'BreakoutStrategy' not in strategy_names:
        logger.info("未自动发现突破策略，手动注册")
        manager.register_strategy('BreakoutStrategy', BreakoutStrategy)
    
    if 'CompositeStrategy' not in strategy_names:
        logger.info("未自动发现组合策略，手动注册")
        manager.register_strategy('CompositeStrategy', CompositeStrategy)
    
    # 创建策略实例
    momentum_strategy = manager.create_strategy('EnhancedMomentumStrategy')
    mean_reversion_strategy = manager.create_strategy('MeanReversionStrategy')
    breakout_strategy = manager.create_strategy('BreakoutStrategy')
    
    # 创建组合策略，需要单独处理因为它依赖于其他策略实例
    composite_params = {'adaptive_weights': True, 'minimum_consensus': 0.4}
    composite_strategy = CompositeStrategy(composite_params)
    composite_strategy.add_strategy(momentum_strategy, weight=0.4)
    composite_strategy.add_strategy(mean_reversion_strategy, weight=0.3)
    composite_strategy.add_strategy(breakout_strategy, weight=0.3)
    
    # 手动注册组合策略
    manager.strategy_instances[composite_strategy.name] = composite_strategy
    
    # 定义测试股票
    symbols = ['AAPL', 'MSFT']
    
    # 加载数据
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    data_dict = load_test_data(symbols, start_date)
    
    if not data_dict:
        logger.error("未能获取到有效的测试数据")
        return
    
    # 定义市场状态
    market_state = {
        'trend': 'bullish',
        'volatility': 'moderate',
        'risk_level': 'medium'
    }
    
    # 运行所有策略
    logger.info("运行所有策略分析")
    all_results = manager.run_all_strategies(data_dict, market_state)
    
    # 显示结果
    for strategy_name, results in all_results.items():
        logger.info(f"\n{strategy_name} 结果:")
        for symbol, result in results.items():
            logger.info(f"  {symbol}:")
            logger.info(f"    信号: {result['signal']}")
            logger.info(f"    价格: {result['price']:.2f}")
            logger.info(f"    市场环境: {result['market_regime']}")
    
    # 运行组合信号生成
    logger.info("\n生成合并信号")
    composite_results = manager.generate_consolidated_signals(data_dict, market_state)
    
    # 显示组合结果
    for symbol, result in composite_results.items():
        logger.info(f"\n{symbol} 合并信号:")
        logger.info(f"  平均信号: {result['signal']:.2f}")
        logger.info(f"  最强信号: {result['strongest_signal']}")
        logger.info(f"  来自策略: {result['strongest_strategy']}")
        logger.info(f"  策略数量: {result['total_strategies']}")
        logger.info(f"  市场环境: {result['market_regime']}")


if __name__ == "__main__":
    logger.info("策略系统测试开始")
    
    # 测试各个策略
    run_enhanced_momentum_test()
    run_mean_reversion_test()
    run_breakout_test()
    run_composite_test()
    
    # 测试策略管理器
    run_strategy_manager_test()
    
    logger.info("测试完成") 