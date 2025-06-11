<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.advanced_multi_factor_strategy import AdvancedMultiFactorStrategy

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data():
    """加载测试数据，这里使用随机生成的数据，实际使用时应替换为真实数据"""
    # 生成日期索引
    date_rng = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
    
    # 生成随机价格数据
    np.random.seed(42)  # 固定随机种子以便复现
    close = np.random.normal(100, 10, size=len(date_rng))
    close = np.cumsum(np.random.normal(0, 1, size=len(date_rng))) + 100
    
    # 确保价格为正
    close = np.maximum(close, 1)
    
    # 生成OHLCV数据
    high = close * np.random.uniform(1, 1.05, size=len(date_rng))
    low = close * np.random.uniform(0.95, 1, size=len(date_rng))
    open_price = low + np.random.uniform(0, 1, size=len(date_rng)) * (high - low)
    volume = np.random.uniform(1000000, 5000000, size=len(date_rng))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_rng)
    
    return df

def test_strategy():
    """测试高级多因子策略"""
    logger.info("开始测试高级多因子策略")
    
    # 加载测试数据
    data = load_test_data()
    logger.info(f"加载测试数据，共{len(data)}条记录")
    
    # 初始化策略
    strategy = AdvancedMultiFactorStrategy()
    logger.info(f"初始化策略，版本: {strategy.version}")
    
    # 计算指标并生成信号
    result = strategy.generate_signals(data)
    logger.info(f"生成信号完成，信号统计: 买入: {sum(result['signal'] == 1)}, 卖出: {sum(result['signal'] == -1)}, 持有: {sum(result['signal'] == 0)}")
    
    # 提取信号组件
    components = strategy.extract_signal_components(data)
    logger.info(f"提取信号组件完成，组件包括: {list(components.keys())}")
    
    # 绘制结果
    plot_results(result, components)
    
    return result, components

def plot_results(result_df, components):
    """绘制策略结果和各个因子"""
    plt.figure(figsize=(16, 12))
    
    # 绘制价格和信号
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(result_df.index, result_df['close'], label='价格')
    ax1.plot(result_df.index[result_df['signal'] == 1], 
             result_df['close'][result_df['signal'] == 1], 
             '^', markersize=10, color='g', label='买入信号')
    ax1.plot(result_df.index[result_df['signal'] == -1], 
             result_df['close'][result_df['signal'] == -1], 
             'v', markersize=10, color='r', label='卖出信号')
    ax1.set_title('价格和交易信号')
    ax1.set_ylabel('价格')
    ax1.legend()
    
    # 绘制综合因子
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(result_df.index, components['composite'], label='综合因子')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.axhline(y=0.3, color='g', linestyle='--', label='买入阈值')
    ax2.axhline(y=-0.3, color='r', linestyle='--', label='卖出阈值')
    ax2.set_title('综合因子')
    ax2.set_ylabel('因子值')
    ax2.legend()
    
    # 绘制趋势和动量因子
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(result_df.index, components['trend_factor'], label='趋势因子')
    ax3.plot(result_df.index, components['momentum_factor'], label='动量因子')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title('趋势和动量因子')
    ax3.set_ylabel('因子值')
    ax3.legend()
    
    # 绘制波动率、成交量和支撑阻力因子
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(result_df.index, components['volatility_factor'], label='波动率因子')
    ax4.plot(result_df.index, components['volume_factor'], label='成交量因子')
    ax4.plot(result_df.index, components['sr_factor'], label='支撑阻力因子')
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title('波动率、成交量和支撑阻力因子')
    ax4.set_ylabel('因子值')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('multi_factor_strategy_test.png')
    logger.info("结果图表已保存到 multi_factor_strategy_test.png")
    plt.close()

if __name__ == "__main__":
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.advanced_multi_factor_strategy import AdvancedMultiFactorStrategy

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data():
    """加载测试数据，这里使用随机生成的数据，实际使用时应替换为真实数据"""
    # 生成日期索引
    date_rng = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
    
    # 生成随机价格数据
    np.random.seed(42)  # 固定随机种子以便复现
    close = np.random.normal(100, 10, size=len(date_rng))
    close = np.cumsum(np.random.normal(0, 1, size=len(date_rng))) + 100
    
    # 确保价格为正
    close = np.maximum(close, 1)
    
    # 生成OHLCV数据
    high = close * np.random.uniform(1, 1.05, size=len(date_rng))
    low = close * np.random.uniform(0.95, 1, size=len(date_rng))
    open_price = low + np.random.uniform(0, 1, size=len(date_rng)) * (high - low)
    volume = np.random.uniform(1000000, 5000000, size=len(date_rng))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_rng)
    
    return df

def test_strategy():
    """测试高级多因子策略"""
    logger.info("开始测试高级多因子策略")
    
    # 加载测试数据
    data = load_test_data()
    logger.info(f"加载测试数据，共{len(data)}条记录")
    
    # 初始化策略
    strategy = AdvancedMultiFactorStrategy()
    logger.info(f"初始化策略，版本: {strategy.version}")
    
    # 计算指标并生成信号
    result = strategy.generate_signals(data)
    logger.info(f"生成信号完成，信号统计: 买入: {sum(result['signal'] == 1)}, 卖出: {sum(result['signal'] == -1)}, 持有: {sum(result['signal'] == 0)}")
    
    # 提取信号组件
    components = strategy.extract_signal_components(data)
    logger.info(f"提取信号组件完成，组件包括: {list(components.keys())}")
    
    # 绘制结果
    plot_results(result, components)
    
    return result, components

def plot_results(result_df, components):
    """绘制策略结果和各个因子"""
    plt.figure(figsize=(16, 12))
    
    # 绘制价格和信号
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(result_df.index, result_df['close'], label='价格')
    ax1.plot(result_df.index[result_df['signal'] == 1], 
             result_df['close'][result_df['signal'] == 1], 
             '^', markersize=10, color='g', label='买入信号')
    ax1.plot(result_df.index[result_df['signal'] == -1], 
             result_df['close'][result_df['signal'] == -1], 
             'v', markersize=10, color='r', label='卖出信号')
    ax1.set_title('价格和交易信号')
    ax1.set_ylabel('价格')
    ax1.legend()
    
    # 绘制综合因子
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(result_df.index, components['composite'], label='综合因子')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.axhline(y=0.3, color='g', linestyle='--', label='买入阈值')
    ax2.axhline(y=-0.3, color='r', linestyle='--', label='卖出阈值')
    ax2.set_title('综合因子')
    ax2.set_ylabel('因子值')
    ax2.legend()
    
    # 绘制趋势和动量因子
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(result_df.index, components['trend_factor'], label='趋势因子')
    ax3.plot(result_df.index, components['momentum_factor'], label='动量因子')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title('趋势和动量因子')
    ax3.set_ylabel('因子值')
    ax3.legend()
    
    # 绘制波动率、成交量和支撑阻力因子
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(result_df.index, components['volatility_factor'], label='波动率因子')
    ax4.plot(result_df.index, components['volume_factor'], label='成交量因子')
    ax4.plot(result_df.index, components['sr_factor'], label='支撑阻力因子')
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title('波动率、成交量和支撑阻力因子')
    ax4.set_ylabel('因子值')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('multi_factor_strategy_test.png')
    logger.info("结果图表已保存到 multi_factor_strategy_test.png")
    plt.close()

if __name__ == "__main__":
>>>>>>> 3d7330be7ea0ecb409ac485e1c8391bc6d56a2de
    result, components = test_strategy() 