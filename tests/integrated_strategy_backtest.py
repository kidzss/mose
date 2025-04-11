import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import IntegratedStrategy
from tests.strategy_backtest import StrategyBacktest, generate_sample_data

def run_integrated_strategy_backtest():
    """运行集成策略分步回测"""
    # 创建输出目录
    os.makedirs('tests/output', exist_ok=True)
    
    # 生成样本数据
    data = generate_sample_data(days=500)
    
    # 创建回测环境
    backtest = StrategyBacktest(data)
    
    # 第一步：测试仅激活NiuNiu策略
    strategy_step1 = IntegratedStrategy()
    print("\n步骤1：测试NiuNiu策略")
    result_step1 = backtest.run_backtest(strategy_step1)
    backtest.plot_backtest_results(result_step1, "Step1_NiuNiu", "tests/output")
    
    # 第二步：测试激活NiuNiu和CPGW策略
    strategy_step2 = IntegratedStrategy()
    strategy_step2.activate_strategy('cpgw', True)
    print("\n步骤2：测试NiuNiu+CPGW策略")
    result_step2 = backtest.run_backtest(strategy_step2)
    backtest.plot_backtest_results(result_step2, "Step2_CPGW", "tests/output")
    
    # 第三步：测试激活NiuNiu、CPGW和Market Forecast策略
    strategy_step3 = IntegratedStrategy()
    strategy_step3.activate_strategy('cpgw', True)
    strategy_step3.activate_strategy('market_forecast', True)
    print("\n步骤3：测试NiuNiu+CPGW+Market Forecast策略")
    result_step3 = backtest.run_backtest(strategy_step3)
    backtest.plot_backtest_results(result_step3, "Step3_MarketForecast", "tests/output")
    
    # 第四步：测试激活NiuNiu、CPGW、Market Forecast和Momentum策略
    strategy_step4 = IntegratedStrategy()
    strategy_step4.activate_strategy('cpgw', True)
    strategy_step4.activate_strategy('market_forecast', True)
    strategy_step4.activate_strategy('momentum', True)
    print("\n步骤4：测试NiuNiu+CPGW+Market Forecast+Momentum策略")
    result_step4 = backtest.run_backtest(strategy_step4)
    backtest.plot_backtest_results(result_step4, "Step4_Momentum", "tests/output")
    
    # 第五步：测试激活所有策略
    strategy_step5 = IntegratedStrategy()
    strategy_step5.activate_strategy('cpgw', True)
    strategy_step5.activate_strategy('market_forecast', True)
    strategy_step5.activate_strategy('momentum', True)
    strategy_step5.activate_strategy('tdi', True)
    print("\n步骤5：测试所有策略集成")
    result_step5 = backtest.run_backtest(strategy_step5)
    backtest.plot_backtest_results(result_step5, "Step5_AllIntegrated", "tests/output")
    
    # 比较各策略步骤的性能
    compare_strategy_steps(
        [result_step1, result_step2, result_step3, result_step4, result_step5],
        ["Step1_NiuNiu", "Step2_CPGW", "Step3_MarketForecast", "Step4_Momentum", "Step5_AllIntegrated"]
    )

def compare_strategy_steps(results_list, names_list):
    """
    比较不同策略步骤的性能
    
    参数:
        results_list: 回测结果列表
        names_list: 策略名称列表
    """
    # 比较累积收益率
    plt.figure(figsize=(14, 8))
    for i, result in enumerate(results_list):
        name = names_list[i]
        metrics = result.attrs['metrics']
        plt.plot(result.index, result['cumulative_return'] * 100, 
                label=f"{name} ({metrics['total_return']:.2%})")
    
    plt.title('集成策略各步骤累积收益率比较')
    plt.xlabel('日期')
    plt.ylabel('累积收益率(%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('tests/output/integrated_strategy_steps_comparison.png')
    plt.close()
    
    # 比较性能指标
    metrics_summary = pd.DataFrame({
        names_list[i]: {
            '总收益率': f"{result.attrs['metrics']['total_return']:.2%}",
            '年化收益率': f"{result.attrs['metrics']['annualized_return']:.2%}",
            '最大回撤': f"{result.attrs['metrics']['max_drawdown']:.2%}",
            '交易次数': result.attrs['metrics']['num_trades'],
            '胜率': f"{result.attrs['metrics']['win_rate']:.2%}",
            '盈亏比': f"{result.attrs['metrics']['profit_loss_ratio']:.2f}",
            '夏普比率': f"{result.attrs['metrics']['sharpe_ratio']:.2f}",
            '索提诺比率': f"{result.attrs['metrics']['sortino_ratio']:.2f}",
            '卡玛比率': f"{result.attrs['metrics']['calmar_ratio']:.2f}"
        } for i, result in enumerate(results_list)
    })
    
    # 保存性能指标对比到CSV文件
    metrics_summary.to_csv('tests/output/integrated_strategy_steps_metrics.csv')
    
    print("\n集成策略各步骤性能对比:")
    print(metrics_summary)

if __name__ == '__main__':
    run_integrated_strategy_backtest() 