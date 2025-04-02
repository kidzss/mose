#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略优化模型评估与回测一体化脚本

该脚本提供了一个自动化工具，用于运行一系列评估和回测任务，以全面评估训练好的策略优化模型的性能。
它按顺序执行多个步骤，包括模型加载、性能评估、回测和结果可视化，生成完整的评估报告。

主要功能:
1. 加载训练好的策略优化模型
2. 在独立测试集上评估模型性能
3. 执行历史回测，模拟真实交易环境
4. 生成权重分析和市场状态分析图表
5. 计算关键绩效指标 (KPIs)，如夏普比率、最大回撤等
6. 生成综合评估报告

使用方法:
```
python -m strategy_optimizer.run_strategy_evaluation \
    --model_path outputs/models/transformer_model_20230101.pth \
    --config configs/evaluation_config.json \
    --output_dir outputs/evaluation \
    --test_start "2022-01-01" \
    --test_end "2023-01-01"
```

配置参数:
- model_path: 训练好的模型路径
- config: 评估配置文件路径
- output_dir: 评估结果输出目录
- test_start/test_end: 测试数据的起止时间
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """运行命令并记录结果"""
    logger.info(f"执行{description}...")
    logger.info(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"{description}成功完成，退出代码: {result.returncode}")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description}失败，退出代码: {e.returncode}")
        logger.error(f"错误信息: {e.stderr}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='执行策略优化模型评估与回测流程')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'MSFT', 'TSLA'], 
                        help='要回测的股票代码列表')
    parser.add_argument('--days', type=int, default=365, help='回测天数')
    parser.add_argument('--diversity', type=float, default=0.5, help='策略多样性因子(0-1)')
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"{args.output_dir}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件路径
    weights_file = output_dir / 'strategy_weights.csv'
    visualization_file = output_dir / 'strategy_weights_viz.png'
    
    # 步骤1: 评估模型
    eval_cmd = [
        sys.executable,
        'strategy_optimizer/evaluate_model.py',
        '--model_path', args.model_path,
        '--config', args.config,
        '--output', str(weights_file),
        '--diversity', str(args.diversity)
    ]
    
    if not run_command(eval_cmd, "模型评估"):
        logger.error("模型评估失败，终止流程")
        return 1
    
    # 步骤2: 可视化权重
    vis_cmd = [
        sys.executable,
        'strategy_optimizer/visualize_weights.py',
        '--input', str(weights_file),
        '--output', str(visualization_file)
    ]
    
    if not run_command(vis_cmd, "权重可视化"):
        logger.warning("权重可视化失败，但继续执行后续步骤")
    
    # 步骤3: 对每个股票进行回测
    backtest_results = []
    
    for symbol in args.symbols:
        backtest_file = output_dir / f'{symbol}_backtest.png'
        
        backtest_cmd = [
            sys.executable,
            'strategy_optimizer/backtest_weights.py',
            '--weights', str(weights_file),
            '--symbol', symbol,
            '--days', str(args.days),
            '--output', str(backtest_file)
        ]
        
        if run_command(backtest_cmd, f"回测股票 {symbol}"):
            backtest_results.append((symbol, True))
        else:
            logger.warning(f"股票 {symbol} 回测失败")
            backtest_results.append((symbol, False))
    
    # 总结
    logger.info(f"\n==== 策略优化评估与回测完成 ====")
    logger.info(f"结果保存在目录: {output_dir}")
    logger.info(f"策略权重文件: {weights_file}")
    logger.info(f"权重可视化: {visualization_file}")
    
    logger.info("\n回测结果:")
    for symbol, success in backtest_results:
        status = "成功" if success else "失败"
        logger.info(f"{symbol}: {status}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 