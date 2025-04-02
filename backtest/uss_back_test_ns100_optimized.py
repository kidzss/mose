import pandas as pd
import numpy as np
import datetime as dt
from sqlalchemy import create_engine
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import traceback
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "mose"
}

# 策略配置
@dataclass
class StrategyConfig:
    sma_periods: List[int] = None
    rsi_period: int = 14
    med_len: int = 31
    mom_len: int = 5
    near_len: int = 3
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 100]

class DatabaseConnection:
    def __init__(self, config: Dict[str, Union[str, int]]):
        self.config = config
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(
                f"mysql+pymysql://{self.config['user']}:{self.config['password']}@"
                f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
        return self._engine

    def get_stock_data(self, stock: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """从数据库获取股票数据"""
        query = """
        SELECT 
            Date AS `date`, 
            Open AS `Open`, 
            High AS `High`, 
            Low AS `Low`, 
            Close AS `Close`, 
            Volume AS `Volume`, 
            AdjClose AS `Adj Close`
        FROM stock_code_time
        WHERE Code = %s
        AND Date BETWEEN %s AND %s
        ORDER BY Date ASC;
        """
        try:
            data = pd.read_sql_query(
                query, 
                self.engine, 
                params=(stock, start_date, end_date)
            )
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            return data
        except Exception as e:
            logger.error(f"获取股票 {stock} 数据失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

class TechnicalIndicators:
    """技术指标计算类"""
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, periods: List[int], price_col: str = 'Adj Close') -> pd.DataFrame:
        """计算简单移动平均线"""
        if df.empty:
            return df
        try:
            for period in periods:
                df[f"SMA_{period}"] = df[price_col].rolling(window=period).mean()
            return df
        except Exception as e:
            logger.error(f"计算SMA失败: {str(e)}")
            return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'Adj Close') -> pd.DataFrame:
        """计算RSI指标"""
        if df.empty:
            return df
        try:
            delta = df[price_col].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / (avg_loss + 1e-10)  # 避免除零错误
            df['RSI'] = 100 - (100 / (1 + rs))
            return df
        except Exception as e:
            logger.error(f"计算RSI失败: {str(e)}")
            return df

    @staticmethod
    def calculate_market_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """计算市场预测指标"""
        if df.empty:
            return df
        try:
            # 计算中期指标
            df['lowest_low_med'] = df['Low'].rolling(window=config.med_len).min()
            df['highest_high_med'] = df['High'].rolling(window=config.med_len).max()
            df['fastK_I'] = (df['Close'] - df['lowest_low_med']) / (df['highest_high_med'] - df['lowest_low_med'] + 1e-10) * 100

            # 计算短期指标
            df['lowest_low_near'] = df['Low'].rolling(window=config.near_len).min()
            df['highest_high_near'] = df['High'].rolling(window=config.near_len).max()
            df['fastK_N'] = (df['Close'] - df['lowest_low_near']) / (df['highest_high_near'] - df['lowest_low_near'] + 1e-10) * 100

            # 计算动量指标
            min1 = df['Low'].rolling(window=4).min()
            max1 = df['High'].rolling(window=4).max()
            df['momentum'] = ((df['Close'] - min1) / (max1 - min1 + 1e-10)) * 100

            # 计算牛熊聚类
            df['bull_cluster'] = (df['momentum'] <= 20) & (df['fastK_I'] <= 20) & (df['fastK_N'] <= 20)
            df['bear_cluster'] = (df['momentum'] >= 80) & (df['fastK_I'] >= 80) & (df['fastK_N'] >= 80)

            return df
        except Exception as e:
            logger.error(f"计算市场指标失败: {str(e)}")
            return df

    @staticmethod
    def calculate_cpgw_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算长庄股王指标"""
        if df.empty:
            return df
        try:
            # 计算A、B、D线
            def safe_percentage(x):
                return -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10)

            df['A'] = df['Close'].rolling(window=34).apply(safe_percentage)
            df['A'] = df['A'].rolling(window=19).mean()

            df['B'] = df['Close'].rolling(window=14).apply(safe_percentage)

            df['D'] = df['Close'].rolling(window=34).apply(safe_percentage)
            df['D'] = df['D'].ewm(span=4).mean()

            # 计算三线
            df['Long Line'] = df['A'] + 100
            df['Hot Money Line'] = df['B'] + 100
            df['Main Force Line'] = df['D'] + 100

            # 买卖信号
            df['Sell Signal'] = (
                (df['Main Force Line'] < df['Main Force Line'].shift(1)) &
                (df['Main Force Line'].shift(1) > 80) &
                ((df['Hot Money Line'].shift(1) > 95) | (df['Hot Money Line'].shift(2) > 95)) &
                (df['Long Line'] > 60) &
                (df['Hot Money Line'] < 83.5) &
                (df['Hot Money Line'] < df['Main Force Line']) &
                (df['Hot Money Line'] < df['Main Force Line'] + 4)
            )

            df['Buy Signal'] = (
                ((df['Long Line'] < 12) &
                (df['Main Force Line'] < 8) &
                ((df['Hot Money Line'] < 7.2) | (df['Main Force Line'].shift(1) < 5)) &
                ((df['Main Force Line'] > df['Main Force Line'].shift(1)) |
                (df['Hot Money Line'] > df['Hot Money Line'].shift(1)))) |
                ((df['Long Line'] < 8) & (df['Main Force Line'] < 7) &
                (df['Hot Money Line'] < 15) & (df['Hot Money Line'] > df['Hot Money Line'].shift(1))) |
                ((df['Long Line'] < 10) & (df['Main Force Line'] < 7) & (df['Hot Money Line'] < 1))
            )

            return df
        except Exception as e:
            logger.error(f"计算CPGW指标失败: {str(e)}")
            return df

@dataclass
class TradeResult:
    """交易结果数据类"""
    entry_date: dt.datetime
    exit_date: dt.datetime
    entry_price: float
    exit_price: float
    percent_change: float
    trade_type: str
    signal_conditions: Dict[str, bool]

class BacktestStrategy:
    """回测策略基类"""
    
    def __init__(self, df: pd.DataFrame, config: StrategyConfig):
        self.df = df
        self.config = config
        self.trades: List[TradeResult] = []
        self.current_position = 0
        self.entry_price = 0
        self.entry_date = None
        
    def _record_trade(self, exit_price: float, exit_date: dt.datetime, conditions: Dict[str, bool]):
        if self.current_position != 0:
            percent_change = ((exit_price / self.entry_price) - 1) * 100
            trade = TradeResult(
                entry_date=self.entry_date,
                exit_date=exit_date,
                entry_price=self.entry_price,
                exit_price=exit_price,
                percent_change=percent_change,
                trade_type="LONG" if self.current_position > 0 else "SHORT",
                signal_conditions=conditions
            )
            self.trades.append(trade)

    def get_results(self) -> Dict:
        """计算回测结果统计"""
        if not self.trades:
            return {
                "Total Trades": 0,
                "Win Rate": 0,
                "Average Gain": 0,
                "Average Loss": 0,
                "Profit Factor": 0,
                "Total Return": 0
            }

        gains = [t.percent_change for t in self.trades if t.percent_change > 0]
        losses = [t.percent_change for t in self.trades if t.percent_change <= 0]
        
        total_trades = len(self.trades)
        win_trades = len(gains)
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        total_gains = sum(gains) if gains else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
        
        total_return = np.prod([(1 + change/100) for change in [t.percent_change for t in self.trades]]) - 1
        
        return {
            "Total Trades": total_trades,
            "Win Rate": win_rate,
            "Average Gain": avg_gain,
            "Average Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Total Return": total_return * 100
        }

class MomentumRiskStrategy(BacktestStrategy):
    """动量风险策略"""
    
    def run_backtest(self) -> List[TradeResult]:
        """执行回测"""
        if self.df.empty:
            logger.warning("数据为空，无法执行回测")
            return []

        for i in range(1, len(self.df) - 1):
            current_bar = self.df.iloc[i]
            
            # 检查是否有足够的数据
            required_cols = ['Adj Close', 'SMA_5', 'SMA_10', 'RSI']
            if not all(col in current_bar.index for col in required_cols):
                continue

            conditions = {
                'SMA5_above_SMA10': current_bar['SMA_5'] > current_bar['SMA_10'],
                'RSI_above_50': current_bar['RSI'] > 50,
                'RSI_above_70': current_bar['RSI'] > 70,
                'RSI_below_30': current_bar['RSI'] < 30
            }

            # 买入信号
            if (conditions['SMA5_above_SMA10'] and 
                conditions['RSI_above_50'] and 
                self.current_position == 0):
                
                self.current_position = 1
                self.entry_price = current_bar['Adj Close']
                self.entry_date = self.df.index[i]
                
            # 卖出信号
            elif (self.current_position == 1 and 
                  (not conditions['SMA5_above_SMA10'] or 
                   conditions['RSI_above_70'] or 
                   conditions['RSI_below_30'])):
                
                self._record_trade(
                    current_bar['Adj Close'],
                    self.df.index[i],
                    conditions
                )
                self.current_position = 0

        # 处理最后一个持仓
        if self.current_position == 1:
            self._record_trade(
                self.df['Adj Close'].iloc[-1],
                self.df.index[-1],
                conditions
            )
            
        return self.trades

class MarketForecastStrategy(BacktestStrategy):
    """市场预测策略"""
    
    def run_backtest(self) -> List[TradeResult]:
        """执行回测"""
        if self.df.empty:
            logger.warning("数据为空，无法执行回测")
            return []

        for i in range(len(self.df) - 1):
            current_bar = self.df.iloc[i]
            next_bar = self.df.iloc[i + 1]
            
            conditions = {
                'bull_cluster': current_bar['bull_cluster'],
                'bear_cluster': current_bar['bear_cluster'],
                'SMA5_above_SMA10': next_bar['SMA_5'] > next_bar['SMA_10']
            }

            # 买入信号
            if (conditions['bull_cluster'] and 
                conditions['SMA5_above_SMA10'] and 
                self.current_position == 0):
                
                self.current_position = 1
                self.entry_price = current_bar['Adj Close']
                self.entry_date = self.df.index[i]
                
            # 卖出信号
            elif (self.current_position == 1 and 
                  (conditions['bear_cluster'] or 
                   not conditions['SMA5_above_SMA10'])):
                
                self._record_trade(
                    current_bar['Adj Close'],
                    self.df.index[i],
                    conditions
                )
                self.current_position = 0

        # 处理最后一个持仓
        if self.current_position == 1:
            self._record_trade(
                self.df['Adj Close'].iloc[-1],
                self.df.index[-1],
                conditions
            )
            
        return self.trades

class CPGWStrategy(BacktestStrategy):
    """长庄股王策略"""
    
    def run_backtest(self) -> List[TradeResult]:
        """执行回测"""
        if self.df.empty:
            logger.warning("数据为空，无法执行回测")
            return []

        for i in range(len(self.df) - 1):
            current_bar = self.df.iloc[i]
            
            conditions = {
                'buy_signal': current_bar['Buy Signal'],
                'sell_signal': current_bar['Sell Signal']
            }

            # 买入信号
            if conditions['buy_signal'] and self.current_position == 0:
                self.current_position = 1
                self.entry_price = current_bar['Adj Close']
                self.entry_date = self.df.index[i]
                
            # 卖出信号
            elif conditions['sell_signal'] and self.current_position == 1:
                self._record_trade(
                    current_bar['Adj Close'],
                    self.df.index[i],
                    conditions
                )
                self.current_position = 0

        # 处理最后一个持仓
        if self.current_position == 1:
            self._record_trade(
                self.df['Adj Close'].iloc[-1],
                self.df.index[-1],
                conditions
            )
            
        return self.trades

class BacktestRunner:
    """回测运行器"""
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.db = DatabaseConnection(DB_CONFIG)
        self.indicators = TechnicalIndicators()
        
    def prepare_data(self, stock: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """准备回测数据"""
        df = self.db.get_stock_data(stock, start_date, end_date)
        if df.empty:
            return df
            
        # 计算技术指标
        df = self.indicators.calculate_sma(df, self.config.sma_periods)
        df = self.indicators.calculate_rsi(df, self.config.rsi_period)
        df = self.indicators.calculate_market_indicators(df, self.config)
        df = self.indicators.calculate_cpgw_indicators(df)
        
        return df
        
    def run_strategy(self, strategy_name: str, df: pd.DataFrame) -> Dict:
        """运行特定策略"""
        strategy_map = {
            'momentum': MomentumRiskStrategy,
            'forecast': MarketForecastStrategy,
            'cpgw': CPGWStrategy
        }
        
        if strategy_name not in strategy_map:
            logger.error(f"未知策略: {strategy_name}")
            return {}
            
        strategy = strategy_map[strategy_name](df, self.config)
        strategy.run_backtest()
        return strategy.get_results()
        
    def run_parallel_backtest(self, stocks: List[str], strategy: str, start_date: dt.datetime) -> pd.DataFrame:
        """并行执行回测"""
        def process_stock(stock: str) -> Dict:
            try:
                df = self.prepare_data(stock, start_date, dt.datetime.now())
                if df.empty:
                    logger.warning(f"股票 {stock} 数据为空")
                    return None
                    
                results = self.run_strategy(strategy, df)
                results['Stock Code'] = stock
                return results
            except Exception as e:
                logger.error(f"处理股票 {stock} 时出错: {str(e)}")
                return None
                
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_stock, stocks))
            
        # 过滤掉None结果并转换为DataFrame
        valid_results = [r for r in results if r is not None]
        return pd.DataFrame(valid_results)

def save_results(df: pd.DataFrame, strategy: str):
    """保存回测结果"""
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    filename = output_dir / f"{strategy}_nasdaq100_backtest_results_{dt.datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"结果已保存到: {filename}")

def main():
    """主函数"""
    try:
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 初始化配置
        config = StrategyConfig()
        runner = BacktestRunner(config)
        
        # 获取股票列表 - 修改这里使用正确的表名和查询
        stock_list = pd.read_sql(
            "SELECT DISTINCT Code FROM stock_code_time ORDER BY Code",
            runner.db.engine
        )
        if stock_list.empty:
            logger.error("无法获取股票列表")
            return
            
        # 设置回测起始日期
        start_date = dt.datetime(2022, 1, 1)
        
        # 运行所有策略
        strategies = ['momentum', 'forecast', 'cpgw']
        for strategy in strategies:
            logger.info(f"开始运行 {strategy} 策略回测...")
            
            results_df = runner.run_parallel_backtest(
                stock_list['Code'].tolist(),
                strategy,
                start_date
            )
            
            if not results_df.empty:
                save_results(results_df, strategy)
                
                # 输出策略统计
                logger.info(f"\n{strategy} 策略统计:")
                logger.info(f"总交易数: {results_df['Total Trades'].sum()}")
                logger.info(f"平均胜率: {results_df['Win Rate'].mean():.2%}")
                logger.info(f"平均收益: {results_df['Total Return'].mean():.2f}%")
                
        logger.info("所有回测完成")
        
    except Exception as e:
        logger.error(f"运行回测时出错: {str(e)}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()
