import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from strategy.tdi_strategy import TDIStrategy
from data.market_sentiment import MarketSentimentData
import logging
import mysql.connector
from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtest.backtest_engine import BacktestEngine

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_mysql(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load stock data from MySQL database"""
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='mose'
        )
        
        # Query data
        query = f"""
        SELECT * FROM stock_code_time 
        WHERE Code = '{symbol}' 
        AND Date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY Date
        """
        
        # Read data into DataFrame
        df = pd.read_sql(query, conn)
        
        # Close connection
        conn.close()
        
        # Convert column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from MySQL: {str(e)}")
        return pd.DataFrame()

def run_backtest(strategy: TDIStrategy, data: pd.DataFrame, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """运行回测"""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    
    for i in range(len(data)):
        current_data = data.iloc[:i+1]
        
        # 计算指标
        current_data = strategy.calculate_indicators(current_data)
        
        # 生成信号
        signal = strategy.generate_signal(current_data)
        
        # 处理交易
        if signal['action'] == 'buy' and position == 0:
            position = signal['position_size']
            entry_price = signal['price']
            trades.append({
                'date': current_data.index[-1],
                'action': 'buy',
                'price': entry_price,
                'position': position,
                'capital': capital
            })
            
        elif signal['action'] == 'sell' and position > 0:
            exit_price = signal['price']
            pnl = (exit_price - entry_price) * position * capital
            capital += pnl
            position = 0
            trades.append({
                'date': current_data.index[-1],
                'action': 'sell',
                'price': exit_price,
                'position': 0,
                'pnl': pnl,
                'capital': capital
            })
            
    # 计算性能指标
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        returns = trades_df['capital'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        max_drawdown = (trades_df['capital'].cummax() - trades_df['capital']).max() / trades_df['capital'].cummax().max()
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) if 'pnl' in trades_df else 0
    else:
        sharpe_ratio = 0
        max_drawdown = 0
        win_rate = 0
        
    return {
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades': trades_df
    }

def plot_results(engine: BacktestEngine, symbol: str):
    """绘制回测结果"""
    try:
        # 创建图表
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                '价格和交易信号',
                '技术指标',
                '资金曲线',
                '回撤分析'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. 价格和交易信号
        fig.add_trace(
            go.Candlestick(
                x=engine.data.index,
                open=engine.data['open'],
                high=engine.data['high'],
                low=engine.data['low'],
                close=engine.data['close'],
                name='价格',
                increasing_line_color='red',
                decreasing_line_color='green'
            ),
            row=1, col=1
        )
        
        # 添加移动平均线
        if 'ma20' in engine.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=engine.data.index,
                    y=engine.data['ma20'],
                    name='MA20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        if 'ma50' in engine.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=engine.data.index,
                    y=engine.data['ma50'],
                    name='MA50',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        # 添加交易点
        trades_df = pd.DataFrame(engine.trades)
        if len(trades_df) > 0:
            # 买入点
            buy_trades = trades_df[trades_df['action'] == 'buy']
            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades['date'],
                        y=buy_trades['price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='red',
                            line=dict(width=2, color='white')
                        ),
                        name='买入信号'
                    ),
                    row=1, col=1
                )
            
            # 卖出点
            sell_trades = trades_df[trades_df['action'] == 'sell']
            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades['date'],
                        y=sell_trades['price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='green',
                            line=dict(width=2, color='white')
                        ),
                        name='卖出信号'
                    ),
                    row=1, col=1
                )
        
        # 2. 技术指标
        if 'rsi' in engine.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=engine.data.index,
                    y=engine.data['rsi'],
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
            
            # 添加RSI的超买超卖线
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        if 'adx' in engine.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=engine.data.index,
                    y=engine.data['adx'],
                    name='ADX',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
        
        # 3. 资金曲线
        fig.add_trace(
            go.Scatter(
                x=engine.data.index,
                y=engine.data['portfolio_value'],
                name='资金曲线',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        # 4. 回撤分析
        # 计算回撤
        portfolio_value = engine.data['portfolio_value']
        rolling_max = portfolio_value.expanding().max()
        drawdown = (rolling_max - portfolio_value) / rolling_max
        
        fig.add_trace(
            go.Scatter(
                x=engine.data.index,
                y=drawdown,
                name='回撤',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=4, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f'TDI策略回测结果 - {symbol}',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            xaxis_rangeslider_visible=False,
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 添加性能指标注释
        metrics_text = (
            f"总收益率: {engine.metrics['total_return']:.2%}<br>"
            f"夏普比率: {engine.metrics['sharpe_ratio']:.2f}<br>"
            f"最大回撤: {engine.metrics['max_drawdown']:.2%}<br>"
            f"胜率: {engine.metrics['win_rate']:.2%}<br>"
            f"盈亏比: {engine.metrics['profit_factor']:.2f}<br>"
            f"总交易次数: {engine.metrics['total_trades']}"
        )
        
        fig.add_annotation(
            text=metrics_text,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            bgcolor='white',
            font=dict(size=12)
        )
        
        # 保存图表
        fig.write_html(f'tdi_strategy_results_{symbol}.html')
        logger.info(f"回测结果图表已保存至 tdi_strategy_results_{symbol}.html")
        
    except Exception as e:
        logger.error(f"绘制回测结果时出错: {str(e)}")

def analyze_backtest_results(engine: BacktestEngine, symbol: str) -> Dict[str, Any]:
    """分析回测结果"""
    try:
        metrics = engine.metrics
        
        # 打印详细分析结果
        logger.info("\n=== 详细回测分析结果 ===")
        logger.info(f"\n1. 基础性能指标:")
        logger.info(f"总收益率: {metrics['total_return']:.2%}")
        logger.info(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        logger.info(f"总交易次数: {metrics['total_trades']}")
        logger.info(f"胜率: {metrics['win_rate']:.2%}")
        logger.info(f"盈亏比: {metrics['profit_factor']:.2f}")
        logger.info(f"最终资金: ${metrics['final_portfolio_value']:,.2f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"分析回测结果时出错: {str(e)}")
        return {}

def main():
    """主函数"""
    try:
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 设置参数
        symbol = 'GOOG'
        start_date = '2023-01-01'
        end_date = '2024-04-10'
        initial_capital = 100000
        
        # 加载数据
        data = load_data_from_mysql(symbol, start_date, end_date)
        if data.empty:
            logger.error("未能加载数据")
            return
            
        # 初始化回测引擎
        engine = BacktestEngine(
            data=data,
            initial_capital=initial_capital,
            commission=0.001
        )
        
        # 运行回测
        engine.run()
        
        # 分析结果
        analyze_backtest_results(engine, symbol)
        
        # 绘制结果
        plot_results(engine, symbol)
        
    except Exception as e:
        logger.error(f"主函数执行出错: {str(e)}")

if __name__ == "__main__":
    main() 