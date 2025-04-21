import pandas as pd
from typing import Dict, Any
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.strategy_base import Strategy

def run_backtest(data: pd.DataFrame, strategy: Strategy, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    运行回测
    
    参数:
        data: OHLCV数据
        strategy: 策略实例
        initial_capital: 初始资金
        
    返回:
        回测结果字典
    """
    try:
        # 1. 计算技术指标
        data = strategy.calculate_indicators(data)
        
        # 2. 生成交易信号
        data = strategy.generate_signals(data)
        
        # 3. 初始化回测变量
        position = 0
        capital = initial_capital
        trades = []
        current_trade = None
        
        # 4. 遍历数据执行回测
        for i in range(1, len(data)):
            current_data = data.iloc[i]
            prev_data = data.iloc[i-1]
            
            # 获取当前信号
            signal = current_data['signal']
            
            # 计算当前价格
            current_price = current_data['close']
            
            # 获取市场环境
            market_regime = strategy.get_market_regime(data.iloc[:i+1]) if hasattr(strategy, 'get_market_regime') else 'unknown'
            
            # 计算波动率
            volatility = current_data['atr'] / current_price if 'atr' in current_data else 0.02
            
            # 计算仓位大小
            position_size = strategy.calculate_position_size(data.iloc[:i+1], volatility > 0.02)
            
            # 根据市场环境调整仓位大小
            if market_regime == 'volatile':
                position_size *= 0.8  # 高波动环境下降低仓位
            elif market_regime == 'bullish':
                position_size *= 1.1  # 牛市环境下适当增加仓位
            elif market_regime == 'bearish':
                position_size *= 0.9  # 熊市环境下适当降低仓位
            
            # 处理开仓信号
            if signal > 0 and position == 0:
                # 计算开仓数量
                shares = (capital * position_size) / current_price
                position = shares
                capital -= shares * current_price
                
                # 计算动态止损止盈
                stop_loss = strategy.get_stop_loss(data.iloc[:i+1], current_price, 1) if hasattr(strategy, 'get_stop_loss') else current_price * (1 - 2 * volatility)
                take_profit = strategy.get_take_profit(data.iloc[:i+1], current_price, 1) if hasattr(strategy, 'get_take_profit') else current_price * (1 + 3 * volatility)
                
                # 记录交易
                current_trade = {
                    'entry_time': current_data.name,
                    'entry_price': current_price,
                    'shares': shares,
                    'direction': 'long',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'market_regime': market_regime,
                    'volatility': volatility
                }
                
            # 处理平仓信号
            elif signal < 0 and position > 0:
                # 计算平仓收益
                pnl = position * (current_price - current_trade['entry_price'])
                capital += position * current_price
                
                # 记录交易结果
                current_trade.update({
                    'exit_time': current_data.name,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'return_pct': pnl / (current_trade['entry_price'] * current_trade['shares']) * 100,
                    'exit_reason': 'signal'
                })
                trades.append(current_trade)
                
                # 重置仓位
                position = 0
                current_trade = None
            
            # 检查止损止盈
            elif position > 0 and current_trade:
                # 检查止损
                if current_price <= current_trade['stop_loss']:
                    pnl = position * (current_price - current_trade['entry_price'])
                    capital += position * current_price
                    
                    current_trade.update({
                        'exit_time': current_data.name,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return_pct': pnl / (current_trade['entry_price'] * current_trade['shares']) * 100,
                        'exit_reason': 'stop_loss'
                    })
                    trades.append(current_trade)
                    
                    position = 0
                    current_trade = None
                
                # 检查止盈
                elif current_price >= current_trade['take_profit']:
                    pnl = position * (current_price - current_trade['entry_price'])
                    capital += position * current_price
                    
                    current_trade.update({
                        'exit_time': current_data.name,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return_pct': pnl / (current_trade['entry_price'] * current_trade['shares']) * 100,
                        'exit_reason': 'take_profit'
                    })
                    trades.append(current_trade)
                    
                    position = 0
                    current_trade = None
        
        # 5. 计算回测结果
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # 计算关键指标
            total_return = (capital - initial_capital) / initial_capital * 100
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
            max_drawdown = calculate_max_drawdown(trades_df['pnl'].cumsum())
            
            # 计算年化收益率
            days = (data.index[-1] - data.index[0]).days
            annual_return = (1 + total_return/100) ** (365/days) - 1 if days > 0 else 0
            
            # 计算夏普比率
            returns = trades_df['return_pct'] / 100
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
            
            # 计算交易统计
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            # 计算平均持仓时间
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                trades_df['holding_period'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.days
                avg_holding_period = trades_df['holding_period'].mean()
            else:
                avg_holding_period = 0
            
            # 计算最大连续盈利和亏损
            trades_df['win_streak'] = (trades_df['pnl'] > 0).astype(int)
            trades_df['loss_streak'] = (trades_df['pnl'] < 0).astype(int)
            
            max_win_streak = trades_df['win_streak'].groupby((trades_df['win_streak'] != trades_df['win_streak'].shift()).cumsum()).sum().max()
            max_loss_streak = trades_df['loss_streak'].groupby((trades_df['loss_streak'] != trades_df['loss_streak'].shift()).cumsum()).sum().max()
            
            # 计算盈亏比
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            
            # 计算市场环境分析
            market_regime = strategy.get_market_regime(data) if hasattr(strategy, 'get_market_regime') else 'unknown'
            
            # 计算趋势分析
            trend_analysis = strategy._analyze_trend(data) if hasattr(strategy, '_analyze_trend') else 'unknown'
            
            # 计算波动率分析
            volatility_analysis = strategy._analyze_volatility(data) if hasattr(strategy, '_analyze_volatility') else 'unknown'
            
            # 计算不同市场环境下的表现
            market_regime_performance = {}
            if 'market_regime' in trades_df.columns:
                for regime in trades_df['market_regime'].unique():
                    regime_trades = trades_df[trades_df['market_regime'] == regime]
                    if len(regime_trades) > 0:
                        regime_return = regime_trades['pnl'].sum() / initial_capital * 100
                        regime_win_rate = len(regime_trades[regime_trades['pnl'] > 0]) / len(regime_trades) * 100
                        market_regime_performance[regime] = {
                            'return': regime_return,
                            'win_rate': regime_win_rate,
                            'trades': len(regime_trades)
                        }
            
            # 返回回测结果
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'avg_holding_period': avg_holding_period,
                'market_regime': market_regime,
                'trend_analysis': trend_analysis,
                'volatility_analysis': volatility_analysis,
                'market_regime_performance': market_regime_performance,
                'trades': trades,
                'final_capital': capital
            }
        else:
            return {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'max_win_streak': 0,
                'max_loss_streak': 0,
                'avg_holding_period': 0,
                'market_regime': 'unknown',
                'trend_analysis': 'unknown',
                'volatility_analysis': 'unknown',
                'market_regime_performance': {},
                'trades': [],
                'final_capital': initial_capital
            }
            
    except Exception as e:
        logger.error(f"回测过程中发生错误: {str(e)}")
        raise 