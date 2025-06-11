import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pymysql
from strategy.niuniu_strategy_v3 import NiuniuStrategyV3
from strategy.tdi_strategy import TDIStrategy
from data.market_sentiment import MarketSentimentData
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, strategy, initial_capital=100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.positions = []
        self.trades = []
        self.current_position = None
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.market_sentiment = MarketSentimentData()
        
    def run(self, data):
        logger.info(f"开始回测策略: {self.strategy.__class__.__name__}")
        
        # 计算技术指标
        data = self.strategy.calculate_indicators(data)
        
        for i in range(len(data)):
            current_data = data.iloc[i]
            date = current_data.name
            price = current_data['close']
            
            # 获取市场情绪数据
            vix = self.market_sentiment.get_vix(date)
            put_call_ratio = self.market_sentiment.get_put_call_ratio(date)
            sentiment = self.market_sentiment.get_sentiment(date)
            
            # 更新数据中的市场情绪指标
            data.loc[date, 'vix'] = vix
            data.loc[date, 'put_call_ratio'] = put_call_ratio
            data.loc[date, 'sentiment'] = sentiment
            
            # 生成信号
            if isinstance(self.strategy, TDIStrategy):
                signal = self.strategy.generate_signals(data.iloc[:i+1])['signal'].iloc[-1]
            else:
                signal = self.strategy.generate_signal(data.iloc[:i+1])
            
            # 处理信号
            if signal == 1 and not self.current_position:  # 买入信号
                self._open_position(date, price, 'long', data.iloc[:i+1])
            elif signal == -1 and self.current_position:  # 卖出信号
                self._close_position(date, price)
            
            # 检查止损和止盈
            if self.current_position:
                # 检查止损
                if self.current_position['direction'] == 'long' and price <= self.current_position['stop_loss']:
                    logger.info(f"触发止损: {date}, 价格: {price:.2f}, 止损价: {self.current_position['stop_loss']:.2f}")
                    self._close_position(date, price)
                elif self.current_position['direction'] == 'short' and price >= self.current_position['stop_loss']:
                    logger.info(f"触发止损: {date}, 价格: {price:.2f}, 止损价: {self.current_position['stop_loss']:.2f}")
                    self._close_position(date, price)
                
                # 检查止盈
                if self.current_position['direction'] == 'long' and price >= self.current_position['take_profit']:
                    logger.info(f"触发止盈: {date}, 价格: {price:.2f}, 止盈价: {self.current_position['take_profit']:.2f}")
                    self._close_position(date, price)
                elif self.current_position['direction'] == 'short' and price <= self.current_position['take_profit']:
                    logger.info(f"触发止盈: {date}, 价格: {price:.2f}, 止盈价: {self.current_position['take_profit']:.2f}")
                    self._close_position(date, price)
                
                # 更新止损价格（追踪止损）
                if self.current_position:
                    new_stop_loss = self.strategy.should_adjust_stop_loss(
                        data.iloc[:i+1],
                        price,
                        self.current_position['stop_loss'],
                        1 if self.current_position['direction'] == 'long' else -1
                    )
                    if new_stop_loss != self.current_position['stop_loss']:
                        logger.info(f"更新止损价: {date}, 旧止损: {self.current_position['stop_loss']:.2f}, 新止损: {new_stop_loss:.2f}")
                        self.current_position['stop_loss'] = new_stop_loss
            
            # 更新投资组合价值
            if self.current_position:
                self.portfolio_value = self.cash + self.current_position['shares'] * price
            else:
                self.portfolio_value = self.cash
        
        logger.info(f"回测完成，最终投资组合价值: {self.portfolio_value:.2f}")
        return self._calculate_metrics(data)
    
    def _open_position(self, date, price, direction, data):
        """
        开仓
        
        参数:
            date: 交易日期
            price: 交易价格
            direction: 交易方向 ('long' 或 'short')
            data: 当前可用的数据
        """
        # 计算仓位大小
        position_size = self.strategy.get_position_size(data, 1 if direction == 'long' else -1)
        shares = int(self.cash * position_size / price)
        
        if shares > 0:
            self.current_position = {
                'date': date,
                'price': price,
                'shares': shares,
                'direction': direction,
                'position_size': position_size,
                'stop_loss': self.strategy.get_stop_loss(data, price, 1 if direction == 'long' else -1),
                'take_profit': self.strategy.get_take_profit(data, price, 1 if direction == 'long' else -1)
            }
            
            # 更新现金和交易记录
            self.cash -= shares * price
            self.trades.append({
                'date': date,
                'type': 'buy',
                'price': price,
                'shares': shares,
                'value': shares * price,
                'position_size': position_size
            })
            
            logger.info(f"开仓: {date}, 价格: {price:.2f}, 股数: {shares}, 仓位大小: {position_size:.2%}")
    
    def _close_position(self, date, price):
        if self.current_position:
            # 计算盈亏
            profit = (price - self.current_position['price']) * self.current_position['shares']
            profit_loss = profit if self.current_position['direction'] == 'long' else -profit
            
            # 更新现金和交易记录
            self.cash += self.current_position['shares'] * price
            self.trades.append({
                'date': date,
                'type': 'sell',
                'price': price,
                'shares': self.current_position['shares'],
                'value': self.current_position['shares'] * price,
                'profit': profit
            })
            
            # 更新策略的交易统计
            trade_result = {
                'profit_loss': profit_loss,
                'entry_price': self.current_position['price'],
                'exit_price': price,
                'position_size': self.current_position['shares'] * self.current_position['price'] / self.initial_capital,
                'entry_time': self.current_position['date'],
                'exit_time': date,
                'direction': self.current_position['direction']
            }
            self.strategy.update_trade_stats(trade_result)
            
            logger.info(f"平仓: {date}, 价格: {price:.2f}, 盈利: {profit:.2f}")
            self.current_position = None
    
    def _calculate_metrics(self, data):
        trades_df = pd.DataFrame(self.trades)
        if trades_df.empty:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0
            }
            
        # 计算总收益率
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # 计算夏普比率
        returns = data['close'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        # 计算最大回撤
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # 计算胜率
        win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df) if 'profit' in trades_df.columns else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades_df)
        }

def load_data(stock_code, start_date, end_date):
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='mose'
        )
        
        query = f"""
        SELECT Date, Open, High, Low, Close, Volume 
        FROM stock_code_time 
        WHERE Code = '{stock_code}' 
        AND Date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY Date
        """
        
        data = pd.read_sql(query, conn)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # 确保列名一致性
        data.columns = data.columns.str.lower()
        
        conn.close()
        return data
        
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise

def plot_results(data, trades, metrics, strategy_name):
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot price and trades
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['close'], label='Price', color='blue', linewidth=1)
        
        # Plot buy and sell points
        buy_points = [trade for trade in trades if trade['type'] == 'buy']
        sell_points = [trade for trade in trades if trade['type'] == 'sell']
        
        if buy_points:
            buy_dates = [trade['date'] for trade in buy_points]
            buy_prices = [trade['price'] for trade in buy_points]
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
            
        if sell_points:
            sell_dates = [trade['date'] for trade in sell_points]
            sell_prices = [trade['price'] for trade in sell_points]
            plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')
        
        plt.title(f'{strategy_name} Strategy - Price and Trades\n{data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Plot portfolio value
        plt.subplot(2, 1, 2)
        portfolio_values = []
        current_value = 100000  # Initial portfolio value
        current_shares = 0
        
        for i in range(len(data)):
            current_date = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Update portfolio value based on trades
            trades_on_date = [t for t in trades if t['date'] == current_date]
            for trade in trades_on_date:
                if trade['type'] == 'buy':
                    current_shares = trade['shares']
                else:
                    current_shares = 0
                    
            # Calculate current portfolio value
            current_value = (current_shares * current_price) if current_shares > 0 else current_value
            portfolio_values.append(current_value)
        
        plt.plot(data.index, portfolio_values, label='Portfolio Value', color='purple', linewidth=1)
        
        # Add metrics text box
        metrics_text = f"Total Return: {metrics['total_return']:.2%}\n"
        metrics_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        metrics_text += f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        metrics_text += f"Win Rate: {metrics['win_rate']:.2%}\n"
        metrics_text += f"Number of Trades: {metrics['num_trades']}"
        
        plt.text(0.02, 0.95, metrics_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top',
                fontsize=10)
        
        plt.title(f'{strategy_name} Strategy - Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting results for {strategy_name}: {str(e)}")
        raise

def main():
    # 设置回测参数
    stock_code = 'GOOG'
    start_date = '2023-01-01'
    end_date = '2024-04-10'
    
    # 加载数据
    data = load_data(stock_code, start_date, end_date)
    
    # 测试NiuniuV3策略
    niuniu_strategy = NiuniuStrategyV3()
    niuniu_engine = BacktestEngine(niuniu_strategy)
    niuniu_metrics = niuniu_engine.run(data)
    plot_results(data, niuniu_engine.trades, niuniu_metrics, 'NiuniuV3')
    
    # 测试TDI策略
    tdi_strategy = TDIStrategy()
    tdi_engine = BacktestEngine(tdi_strategy)
    tdi_metrics = tdi_engine.run(data)
    plot_results(data, tdi_engine.trades, tdi_metrics, 'TDI')
    
    # 打印比较结果
    logger.info("\n策略比较结果:")
    logger.info(f"{'指标':<15} {'NiuniuV3':<15} {'TDI':<15}")
    for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'num_trades']:
        logger.info(f"{metric:<15} {niuniu_metrics[metric]:<15.2f} {tdi_metrics[metric]:<15.2f}")

def calculate_indicators(self, data):
    # 基础指标
    data['rsi'] = self._calculate_rsi(data['close'])
    data['fast_ma'] = data['close'].rolling(window=self.fast_window).mean()
    data['slow_ma'] = data['close'].rolling(window=self.slow_window).mean()
    data['ma_diff'] = data['fast_ma'] - data['slow_ma']
    
    # 波动率指标
    data['atr'] = self._calculate_atr(data)
    data['volatility_ratio'] = data['atr'] / data['close'].rolling(window=20).mean()
    
    # 市场情绪指标
    data['vix'] = self._get_vix_data(data.index)  # 需要实现VIX数据获取
    data['put_call_ratio'] = self._get_put_call_ratio(data.index)  # 需要实现Put/Call Ratio获取
    
    # 多时间框架分析
    data['daily_trend'] = self._calculate_daily_trend(data)
    data['4h_trend'] = self._calculate_4h_trend(data)
    data['1h_trend'] = self._calculate_1h_trend(data)
    
    return data

def generate_signal(self, data):
    current_data = data.iloc[-1]
    
    # 波动率自适应
    volatility_level = self._get_volatility_level(current_data['volatility_ratio'])
    
    # 市场情绪判断
    sentiment = self._analyze_market_sentiment(
        current_data['vix'],
        current_data['put_call_ratio']
    )
    
    # 多时间框架趋势判断
    trend_alignment = self._check_trend_alignment(
        current_data['daily_trend'],
        current_data['4h_trend'],
        current_data['1h_trend']
    )
    
    # 综合信号生成
    if (trend_alignment > 0 and 
        sentiment == 'bullish' and 
        volatility_level == 'normal'):
        return 1  # 买入信号
    elif (trend_alignment < 0 and 
          sentiment == 'bearish' and 
          volatility_level == 'normal'):
        return -1  # 卖出信号
    
    return 0  # 无信号

def _get_volatility_level(self, volatility_ratio):
    if volatility_ratio > 0.03:  # 高波动
        return 'high'
    elif volatility_ratio < 0.01:  # 低波动
        return 'low'
    else:
        return 'normal'

def _analyze_market_sentiment(self, vix, put_call_ratio):
    if vix > 30 and put_call_ratio > 1.2:
        return 'bearish'
    elif vix < 20 and put_call_ratio < 0.8:
        return 'bullish'
    else:
        return 'neutral'

def _check_trend_alignment(self, daily_trend, h4_trend, h1_trend):
    # 计算趋势一致性得分
    score = 0
    if daily_trend == h4_trend:
        score += daily_trend
    if h4_trend == h1_trend:
        score += h4_trend
    return score

def _calculate_atr(self, data, period=14):
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

if __name__ == "__main__":
    main() 