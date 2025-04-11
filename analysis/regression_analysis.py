import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
from typing import Dict, List, Optional
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitor.data_manager import DataManager
from monitor.market_monitor import MarketMonitor
from strategy.uss_gold_triangle_strategy import GoldTriangleStrategy
from strategy.uss_momentum_strategy import MomentumStrategy
from strategy.uss_niuniu_strategy import NiuniuStrategy
from strategy.uss_tdi_strategy import TDIStrategy
from strategy.uss_market_forecast_strategy import MarketForecastStrategy
from strategy.uss_cpgw_strategy import CPGWStrategy
from strategy.uss_volume_strategy import VolumeStrategy
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StrategyAnalysis")

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'mose'
}

def get_stock_data_from_db(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """从数据库获取股票数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
    """
    try:
        # 创建数据库连接
        engine = DataManager(DB_CONFIG).engine
        
        # 构建SQL查询
        query = text("""
            SELECT Date, Open, High, Low, Close, Volume, AdjClose
            FROM stock_time_code
            WHERE Code = :symbol
            AND Date BETWEEN :start_date AND :end_date
            ORDER BY Date ASC
        """)
        
        # 执行查询
        data = pd.read_sql(query, engine, params={
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        if data.empty:
            logger.warning(f"没有找到股票 {symbol} 在 {start_date} 到 {end_date} 期间的数据")
            return None
            
        # 设置日期索引
        data.set_index('Date', inplace=True)
        
        # 重命名列名
        data.columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        
        return data
        
    except Exception as e:
        logger.error(f"从数据库获取股票 {symbol} 数据时出错: {e}")
        return None

class StrategyAnalysis:
    """策略分析类，使用多个策略分析股票"""
    
    def __init__(self, data_manager: DataManager = None, market_monitor: MarketMonitor = None):
        """初始化策略分析器"""
        self.data_manager = data_manager or DataManager(DB_CONFIG)
        self.market_monitor = market_monitor or MarketMonitor()
        
        # 初始化策略
        self.strategies = {
            'GoldTriangle': GoldTriangleStrategy(),
            'Momentum': MomentumStrategy(),
            'Niuniu': NiuniuStrategy(),
            'TDI': TDIStrategy(),
            'MarketForecast': MarketForecastStrategy(),
            'CPGW': CPGWStrategy(),
            'Volume': VolumeStrategy()
        }
        
        # 每个策略需要的最小数据量
        self.strategy_min_data = {
            'GoldTriangle': 100,
            'Momentum': 200,
            'Niuniu': 100,
            'TDI': 100,
            'MarketForecast': 100,
            'CPGW': 100,
            'Volume': 100
        }
        
    def analyze_stock(self, symbol: str, lookback_days: int = 300) -> Dict:
        """分析单只股票
        
        Args:
            symbol: 股票代码
            lookback_days: 回看天数，默认300天以确保有足够的数据进行分析
        """
        try:
            # 获取历史数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            data = get_stock_data_from_db(symbol, start_date, end_date)
            
            if data is None or data.empty:
                logger.warning(f"无法获取 {symbol} 的历史数据")
                return {}
                
            results = {
                'symbol': symbol,
                'current_price': data['close'].iloc[-1],
                'data_length': len(data),
                'signals': {},
                'indicators': {}
            }
            
            # 运行每个策略
            for strategy_name, strategy in self.strategies.items():
                try:
                    # 检查数据量是否足够
                    min_data_required = self.strategy_min_data.get(strategy_name, 100)
                    if len(data) < min_data_required:
                        logger.warning(f"策略 {strategy_name} 需要至少 {min_data_required} 条数据，当前只有 {len(data)} 条")
                        continue
                        
                    # 计算指标
                    indicators = strategy.calculate_indicators(data)
                    if indicators is not None and not indicators.empty:
                        # 生成信号
                        signals = strategy.generate_signals(indicators)
                        if signals is not None and not signals.empty:
                            # 获取最新信号
                            latest_signal = signals['signal'].iloc[-1]
                            
                            # 计算策略绩效
                            if hasattr(strategy, 'evaluate_performance'):
                                performance = strategy.evaluate_performance(data, signals)
                            else:
                                # 计算简单的绩效指标
                                returns = data['close'].pct_change()
                                signal_returns = signals['signal'].shift(1) * returns
                                performance = {
                                    'sharpe_ratio': signal_returns.mean() / signal_returns.std() if signal_returns.std() != 0 else 0,
                                    'annual_return': (1 + signal_returns.mean()) ** 252 - 1 if not signal_returns.empty else 0,
                                    'max_drawdown': (signal_returns.cumsum() - signal_returns.cumsum().expanding().max()).min()
                                }
                            
                            # 计算信号强度
                            signal_strength = latest_signal
                            if strategy_name == 'GoldTriangle':
                                # 黄金三角策略的信号强度基于价格趋势
                                price_trend = data['close'].iloc[-1] / data['close'].iloc[-20] - 1
                                signal_strength = latest_signal * (1 + abs(price_trend))
                            elif strategy_name == 'Momentum':
                                # 动量策略的信号强度基于RSI和MACD
                                rsi = indicators['RSI'].iloc[-1]
                                macd_hist = indicators['MACD_hist'].iloc[-1]
                                signal_strength = latest_signal * (1 + abs(rsi - 50) / 50) * (1 + abs(macd_hist))
                            elif strategy_name == 'CPGW':
                                # CPGW策略的信号强度基于趋势强度和波动率
                                if 'trend_strength' in indicators:
                                    trend_strength = indicators['trend_strength'].iloc[-1]
                                    volatility = data['close'].pct_change().std() * np.sqrt(252)  # 年化波动率
                                    signal_strength = latest_signal * (1 + abs(trend_strength)) * (1 + volatility)
                            
                            results['signals'][strategy_name] = {
                                'signal': signal_strength,
                                'raw_signal': latest_signal,
                                'performance': performance
                            }
                            
                            # 保存主要指标
                            results['indicators'][strategy_name] = {
                                col: indicators[col].iloc[-1]
                                for col in indicators.columns
                                if col not in data.columns
                            }
                            
                except Exception as e:
                    logger.error(f"运行策略 {strategy_name} 于股票 {symbol} 时出错: {e}")
                    
            # 计算综合信号
            if results['signals']:
                # 计算策略权重
                strategy_weights = {}
                total_weight = 0
                
                for strategy_name, signal_data in results['signals'].items():
                    # 使用夏普比率作为权重
                    weight = max(0, signal_data['performance'].get('sharpe_ratio', 0))
                    total_weight += weight
                    strategy_weights[strategy_name] = weight
                
                # 归一化权重
                if total_weight > 0:
                    for strategy_name in strategy_weights:
                        strategy_weights[strategy_name] /= total_weight
                
                # 保存策略权重
                results['strategy_weights'] = strategy_weights
                
                # 使用动态权重计算综合信号
                weighted_signals = []
                for strategy_name, signal_data in results['signals'].items():
                    signal = signal_data['signal']
                    weight = strategy_weights[strategy_name]
                    weighted_signals.append(signal * weight)
                
                results['composite_signal'] = sum(weighted_signals)
                    
            return results
            
        except Exception as e:
            logger.error(f"分析股票 {symbol} 时出错: {e}")
            return {}
            
    def analyze_all_stocks(self) -> List[Dict]:
        """分析所有监控的股票"""
        results = []
        try:
            # 获取监控的股票列表
            stocks = pd.DataFrame({
                'symbol': ['NVDA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AMZN', 'ASML', 'TSLA']
            })
            
            if stocks.empty:
                logger.warning("没有可分析的股票")
                return results
                
            logger.info(f"开始分析 {len(stocks)} 只股票")
            
            # 设置动态的时间范围
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # 分析每只股票
            for _, stock in stocks.iterrows():
                symbol = stock['symbol']
                data = get_stock_data_from_db(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    result = {
                        'symbol': symbol,
                        'current_price': data['close'].iloc[-1],
                        'data_length': len(data),
                        'signals': {},
                        'indicators': {}
                    }
                    
                    # 运行每个策略
                    for strategy_name, strategy in self.strategies.items():
                        try:
                            # 检查数据量是否足够
                            min_data_required = self.strategy_min_data.get(strategy_name, 100)
                            if len(data) < min_data_required:
                                logger.warning(f"策略 {strategy_name} 需要至少 {min_data_required} 条数据，当前只有 {len(data)} 条")
                                continue
                                
                            # 计算指标
                            indicators = strategy.calculate_indicators(data)
                            if indicators is not None and not indicators.empty:
                                # 生成信号
                                signals = strategy.generate_signals(indicators)
                                if signals is not None and not signals.empty:
                                    # 获取最新信号
                                    latest_signal = signals['signal'].iloc[-1]
                                    
                                    # 计算策略绩效
                                    if hasattr(strategy, 'evaluate_performance'):
                                        performance = strategy.evaluate_performance(data, signals)
                                    else:
                                        # 计算简单的绩效指标
                                        returns = data['close'].pct_change()
                                        signal_returns = signals['signal'].shift(1) * returns
                                        performance = {
                                            'sharpe_ratio': signal_returns.mean() / signal_returns.std() if signal_returns.std() != 0 else 0,
                                            'annual_return': (1 + signal_returns.mean()) ** 252 - 1 if not signal_returns.empty else 0,
                                            'max_drawdown': (signal_returns.cumsum() - signal_returns.cumsum().expanding().max()).min()
                                        }
                                    
                                    # 计算信号强度
                                    signal_strength = latest_signal
                                    if strategy_name == 'GoldTriangle':
                                        # 黄金三角策略的信号强度基于价格趋势
                                        price_trend = data['close'].iloc[-1] / data['close'].iloc[-20] - 1
                                        signal_strength = latest_signal * (1 + abs(price_trend))
                                    elif strategy_name == 'Momentum':
                                        # 动量策略的信号强度基于RSI和MACD
                                        rsi = indicators['RSI'].iloc[-1]
                                        macd_hist = indicators['MACD_hist'].iloc[-1]
                                        signal_strength = latest_signal * (1 + abs(rsi - 50) / 50) * (1 + abs(macd_hist))
                                    elif strategy_name == 'CPGW':
                                        # CPGW策略的信号强度基于趋势强度和波动率
                                        if 'trend_strength' in indicators:
                                            trend_strength = indicators['trend_strength'].iloc[-1]
                                            volatility = data['close'].pct_change().std() * np.sqrt(252)  # 年化波动率
                                            signal_strength = latest_signal * (1 + abs(trend_strength)) * (1 + volatility)
                                    
                                    result['signals'][strategy_name] = {
                                        'signal': signal_strength,
                                        'raw_signal': latest_signal,
                                        'performance': performance
                                    }
                                    
                                    # 保存主要指标
                                    result['indicators'][strategy_name] = {
                                        col: indicators[col].iloc[-1]
                                        for col in indicators.columns
                                        if col not in data.columns
                                    }
                                    
                        except Exception as e:
                            logger.error(f"运行策略 {strategy_name} 于股票 {symbol} 时出错: {e}")
                            
                    # 计算综合信号
                    if result['signals']:
                        # 计算策略权重
                        strategy_weights = {}
                        total_weight = 0
                        
                        for strategy_name, signal_data in result['signals'].items():
                            # 使用夏普比率作为权重
                            weight = max(0, signal_data['performance'].get('sharpe_ratio', 0))
                            total_weight += weight
                            strategy_weights[strategy_name] = weight
                        
                        # 归一化权重
                        if total_weight > 0:
                            for strategy_name in strategy_weights:
                                strategy_weights[strategy_name] /= total_weight
                        
                        # 保存策略权重
                        result['strategy_weights'] = strategy_weights
                        
                        # 使用动态权重计算综合信号
                        weighted_signals = []
                        for strategy_name, signal_data in result['signals'].items():
                            signal = signal_data['signal']
                            weight = strategy_weights[strategy_name]
                            weighted_signals.append(signal * weight)
                        
                        result['composite_signal'] = sum(weighted_signals)
                        
                    results.append(result)
                    
            # 按综合信号强度排序
            results.sort(key=lambda x: abs(x.get('composite_signal', 0)), reverse=True)
            
            # 分类结果
            strong_signals = []
            medium_signals = []
            weak_signals = []
            
            for result in results:
                signal = abs(result.get('composite_signal', 0))
                if signal >= 1.0:
                    strong_signals.append(result)
                elif signal >= 0.5:
                    medium_signals.append(result)
                else:
                    weak_signals.append(result)
                    
            logger.info("分析完成")
            logger.info(f"强信号股票: {[r['symbol'] for r in strong_signals]}")
            logger.info(f"中等信号股票: {[r['symbol'] for r in medium_signals]}")
            logger.info(f"弱信号股票: {[r['symbol'] for r in weak_signals]}")
            
            return results
            
        except Exception as e:
            logger.error(f"分析所有股票时出错: {e}")
            return results
            
    def calculate_strategy_weights(self, data: pd.DataFrame, signals: Dict) -> Dict[str, float]:
        """计算每个策略的权重
        
        Args:
            data: 股票数据
            signals: 各个策略的信号和绩效数据
        """
        try:
            # 计算波动率
            volatility = data['close'].pct_change().std() * np.sqrt(252)
            
            # 计算趋势强度
            sma_20 = data['close'].rolling(window=20).mean()
            sma_60 = data['close'].rolling(window=60).mean()
            trend_strength = (sma_20.iloc[-1] / sma_60.iloc[-1] - 1)
            
            # 计算成交量趋势
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_trend = (volume_ma.iloc[-1] / volume_ma.iloc[-20] - 1)
            
            # 初始化权重
            weights = {}
            
            for strategy_name, signal_data in signals.items():
                # 基础权重：使用夏普比率
                weight = max(0, signal_data['performance'].get('sharpe_ratio', 0))
                
                # 根据市场条件调整权重
                if strategy_name == 'GoldTriangle':
                    # 趋势明显时增加权重
                    weight *= (1 + abs(trend_strength))
                elif strategy_name == 'Momentum':
                    # 波动率高时增加权重
                    weight *= (1 + volatility)
                elif strategy_name == 'CPGW':
                    # 成交量趋势明显时增加权重
                    weight *= (1 + abs(volume_trend))
                    
                weights[strategy_name] = weight
                
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
                
            return weights
            
        except Exception as e:
            logger.error(f"计算策略权重时出错: {e}")
            return {strategy_name: 1.0 / len(signals) for strategy_name in signals.keys()}

def main():
    """主函数"""
    try:
        # 初始化策略分析器
        analyzer = StrategyAnalysis()
        
        # 分析所有股票
        results = analyzer.analyze_all_stocks()
        
        # 根据综合信号强度对股票进行分类
        strong_signals = []
        medium_signals = []
        weak_signals = []
        
        for result in results:
            if abs(result.get('composite_signal', 0)) > 0.7:
                strong_signals.append(result['symbol'])
            elif abs(result.get('composite_signal', 0)) > 0.3:
                medium_signals.append(result['symbol'])
            else:
                weak_signals.append(result['symbol'])
                
        logger.info(f"强信号股票: {strong_signals}")
        logger.info(f"中等信号股票: {medium_signals}")
        logger.info(f"弱信号股票: {weak_signals}")
        
        # 打印详细分析结果
        print("\n=== 策略分析结果 ===")
        print(f"分析时间: {datetime.now()}")
        print(f"分析股票数量: {len(results)}\n")
        
        # 打印强信号股票的详细信息
        print("强信号股票 (|signal| > 0.7):\n")
        for result in results:
            if abs(result.get('composite_signal', 0)) > 0.7:
                print(f"股票: {result['symbol']}")
                print(f"当前价格: {result['current_price']:.2f}")
                print(f"综合信号: {result['composite_signal']:.2f}")
                print("策略权重:")
                for strategy_name, weight in result['strategy_weights'].items():
                    print(f"  {strategy_name}: {weight:.2f}")
                print("各策略信号:")
                for strategy_name, signal_data in result['signals'].items():
                    print(f"  {strategy_name}: {signal_data['signal']:.2f}")
                print()
                
        # 打印中等信号股票的详细信息
        if medium_signals:
            print("\n中等信号股票 (0.3 < |signal| <= 0.7):\n")
            for result in results:
                if 0.3 < abs(result.get('composite_signal', 0)) <= 0.7:
                    print(f"股票: {result['symbol']}")
                    print(f"当前价格: {result['current_price']:.2f}")
                    print(f"综合信号: {result['composite_signal']:.2f}")
                    print("策略权重:")
                    for strategy_name, weight in result['strategy_weights'].items():
                        print(f"  {strategy_name}: {weight:.2f}")
                    print()
                    
        # 打印弱信号股票列表
        if weak_signals:
            print("\n弱信号股票 (|signal| <= 0.3):")
            print(", ".join(weak_signals))
            
    except Exception as e:
        logger.error(f"执行分析时出错: {e}")

if __name__ == "__main__":
    main() 