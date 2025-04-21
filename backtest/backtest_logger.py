import logging
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

class BacktestLogger:
    """回测记录器类"""
    
    def __init__(self, log_dir: str = 'backtest/logs'):
        """
        初始化回测记录器
        
        Args:
            log_dir: 日志目录
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(log_dir, f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # 配置日志
        self.logger = logging.getLogger('backtest')
        self.logger.setLevel(logging.INFO)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 初始化结果存储
        self.results = {}
        self.summary = {}
    
    def log_stock_result(self, symbol: str, result: Dict[str, Any]):
        """
        记录单个股票的回测结果
        
        Args:
            symbol: 股票代码
            result: 回测结果
        """
        if result is None:
            self.logger.error(f"股票 {symbol} 的回测结果为空")
            return
        
        # 记录结果
        self.results[symbol] = result
        
        # 记录摘要信息
        self.summary[symbol] = {
            'start_date': result['start_date'],
            'end_date': result['end_date'],
            'total_return': result['total_return'],
            'annual_return': result['annual_return'],
            'max_drawdown': result['max_drawdown'],
            'sharpe_ratio': result['sharpe_ratio'],
            'sortino_ratio': result['sortino_ratio'],
            'win_rate': result['win_rate'],
            'total_trades': result['total_trades'],
            'avg_holding_days': result['avg_holding_days']
        }
        
        # 记录日志
        self.logger.info(f"股票 {symbol} 回测结果:")
        self.logger.info(f"  总收益率: {result['total_return']:.2%}")
        self.logger.info(f"  年化收益率: {result['annual_return']:.2%}")
        self.logger.info(f"  最大回撤: {result['max_drawdown']:.2%}")
        self.logger.info(f"  夏普比率: {result['sharpe_ratio']:.2f}")
        self.logger.info(f"  索提诺比率: {result['sortino_ratio']:.2f}")
        self.logger.info(f"  胜率: {result['win_rate']:.2%}")
        self.logger.info(f"  总交易次数: {result['total_trades']}")
        self.logger.info(f"  平均持仓天数: {result['avg_holding_days']:.1f}")
    
    def save_results(self, output_dir: str = 'backtest/results'):
        """
        保存回测结果
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'backtest_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        
        # 保存摘要
        summary_file = os.path.join(output_dir, f'backtest_summary_{timestamp}.csv')
        summary_df = pd.DataFrame.from_dict(self.summary, orient='index')
        summary_df.to_csv(summary_file)
        
        # 保存权益曲线
        for symbol, result in self.results.items():
            equity_file = os.path.join(output_dir, f'equity_curve_{symbol}_{timestamp}.csv')
            equity_df = pd.DataFrame(result['equity_curve'])
            equity_df.to_csv(equity_file, index=False)
        
        self.logger.info(f"回测结果已保存到 {output_dir}")
    
    def get_summary(self) -> pd.DataFrame:
        """
        获取回测摘要
        
        Returns:
            回测摘要DataFrame
        """
        return pd.DataFrame.from_dict(self.summary, orient='index')
    
    def get_equity_curve(self, symbol: str) -> pd.DataFrame:
        """
        获取权益曲线
        
        Args:
            symbol: 股票代码
            
        Returns:
            权益曲线DataFrame
        """
        if symbol in self.results:
            return pd.DataFrame(self.results[symbol]['equity_curve'])
        return pd.DataFrame()
    
    def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """
        获取交易记录
        
        Args:
            symbol: 股票代码
            
        Returns:
            交易记录列表
        """
        if symbol in self.results:
            return self.results[symbol]['trades']
        return []
    
    def get_drawdown(self, symbol: str) -> pd.DataFrame:
        """
        获取回撤数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            回撤数据DataFrame
        """
        if symbol in self.results:
            return pd.DataFrame(self.results[symbol]['drawdown'])
        return pd.DataFrame()
    
    def get_monthly_returns(self, symbol: str) -> pd.DataFrame:
        """
        获取月度收益
        
        Args:
            symbol: 股票代码
            
        Returns:
            月度收益DataFrame
        """
        if symbol in self.results:
            return pd.DataFrame(self.results[symbol]['monthly_returns'])
        return pd.DataFrame() 