import sys
import os
import pandas as pd
import json
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# 设置当前目录路径，确保能够正确导入模块
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, parent_path)

from strategy.OpenBB.openbb_market_strategy import OpenBBMarketStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_path, 'openbb_strategy.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_results(results, output_dir=None):
    """保存策略结果到JSON文件"""
    if output_dir is None:
        output_dir = current_path
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 格式化datetime对象为字符串
    results_copy = results.copy()
    results_copy['timestamp'] = results_copy['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    
    if 'economic_indicators' in results_copy and 'last_update' in results_copy['economic_indicators']:
        if isinstance(results_copy['economic_indicators']['last_update'], datetime):
            results_copy['economic_indicators']['last_update'] = results_copy['economic_indicators']['last_update'].strftime('%Y-%m-%d %H:%M:%S')
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f'strategy_results_{timestamp}.json')
    
    # 保存到JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results_copy, f, indent=4, ensure_ascii=False)
    
    logger.info(f"结果已保存到：{file_path}")
    return file_path

def plot_market_analysis(strategy, results, output_dir=None):
    """绘制市场分析图表"""
    if output_dir is None:
        output_dir = current_path
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取市场指数数据
    market_index = strategy.parameters['market_index']
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=strategy.parameters['lookback_days'])
    
    # 使用OpenBB获取历史数据
    from openbb import obb
    market_data = obb.equity.price.historical(
        symbol=market_index,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    ).to_df()
    
    # 计算技术指标
    market_data['MA20'] = market_data['close'].rolling(window=20).mean()
    market_data['MA50'] = market_data['close'].rolling(window=50).mean()
    market_data['MA200'] = market_data['close'].rolling(window=200).mean()
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制价格和均线
    plt.subplot(2, 1, 1)
    plt.plot(market_data.index, market_data['close'], label=f'{market_index} 收盘价')
    plt.plot(market_data.index, market_data['MA20'], label='20日均线')
    plt.plot(market_data.index, market_data['MA50'], label='50日均线')
    plt.plot(market_data.index, market_data['MA200'], label='200日均线')
    
    plt.title(f'市场分析 - {market_index} - {results["market_regime"]}')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    # 绘制成交量
    plt.subplot(2, 1, 2)
    plt.bar(market_data.index, market_data['volume'], width=1, label='成交量')
    
    # 添加移动平均成交量
    market_data['volume_ma'] = market_data['volume'].rolling(window=20).mean()
    plt.plot(market_data.index, market_data['volume_ma'], color='red', label='成交量20日均线')
    
    plt.xlabel('日期')
    plt.ylabel('成交量')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f'market_analysis_{timestamp}.png')
    plt.savefig(file_path)
    plt.close()
    
    logger.info(f"市场分析图表已保存到：{file_path}")
    return file_path

def print_portfolio_recommendations(results):
    """打印投资组合建议"""
    recommendations = results.get('portfolio_recommendation', [])
    
    if not recommendations:
        print("没有投资组合建议")
        return
    
    print("\n==== 投资组合建议 ====")
    print(f"市场环境: {results['market_regime']}")
    print(f"推荐总数: {len(recommendations)}")
    print("\n推荐操作:")
    
    # 创建表格格式
    fmt = "{:<8} {:<6} {:<8} {:<10} {:<10} {:<12} {:<12}"
    print(fmt.format("股票", "操作", "强度", "价格", "头寸比例", "止损", "止盈"))
    print("-" * 70)
    
    for rec in recommendations:
        if rec['action'] == 'BUY':
            print(fmt.format(
                rec['symbol'], 
                rec['action'], 
                f"{rec['strength']:.2f}", 
                f"${rec['price']:.2f}", 
                f"{rec.get('position_size', 0):.2%}" if 'position_size' in rec else "N/A",
                f"${rec.get('stop_loss', 0):.2f}" if 'stop_loss' in rec else "N/A",
                f"${rec.get('take_profit', 0):.2f}" if 'take_profit' in rec else "N/A"
            ))
        else:
            print(fmt.format(
                rec['symbol'], 
                rec['action'], 
                f"{rec['strength']:.2f}", 
                f"${rec['price']:.2f}",
                "N/A", "N/A", "N/A"
            ))

def main():
    """运行OpenBB市场策略"""
    try:
        logger.info("开始运行OpenBB市场策略")
        
        # 创建策略实例
        custom_params = {
            'market_index': 'SPY',
            'lookback_days': 120,
            'screening_criteria': {
                'market_cap_min': 5000000000,  # 50亿美元
                'price_min': 20,
                'volume_min': 1000000,
                'beta_min': 0.8,
                'beta_max': 1.5,
            },
            'max_positions': 10,
        }
        
        strategy = OpenBBMarketStrategy(custom_params)
        
        # 运行策略
        logger.info("分析市场环境...")
        market_regime = strategy.analyze_market_regime()
        logger.info(f"当前市场环境: {market_regime.value}")
        
        # 筛选股票
        logger.info("筛选股票...")
        stocks = strategy.screen_stocks()
        logger.info(f"筛选到 {len(stocks)} 只股票")
        
        if not stocks:
            logger.warning("没有找到符合条件的股票，使用默认股票列表")
            stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'PG']
        
        # 使用部分股票进行测试，加速处理
        test_stocks = stocks[:10]
        logger.info(f"使用 {len(test_stocks)} 只股票进行策略运行")
        
        # 运行完整策略
        logger.info("运行完整策略分析...")
        results = strategy.run_strategy(test_stocks)
        
        # 保存结果
        save_results(results)
        
        # 绘制市场分析图表
        plot_market_analysis(strategy, results)
        
        # 打印投资组合建议
        print_portfolio_recommendations(results)
        
        logger.info("策略运行完成")
        
    except Exception as e:
        logger.error(f"运行策略时出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 