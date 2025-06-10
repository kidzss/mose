import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
import os

from openbb import obb
from ...strategy.strategy_base import Strategy, MarketRegime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenBBSectorRotationStrategy(Strategy):
    """
    利用OpenBB实现的行业轮动策略
    
    该策略基于经济周期理论，在不同的市场环境下对不同行业进行轮动投资:
    1. 牛市初期 - 加权金融、工业和材料行业
    2. 牛市中期 - 加权科技、可选消费和通讯服务
    3. 牛市后期 - 加权必需消费、医疗保健和公用事业
    4. 熊市 - 加权防御性行业（必需消费、公用事业）和黄金
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """初始化策略"""
        default_params = {
            'market_index': 'SPY',            # 用于分析市场环境的指数
            'lookback_days': 120,             # 分析历史数据的天数
            'rebalance_frequency': 30,        # 重新平衡的天数
            'stocks_per_sector': 3,           # 每个行业选择的股票数量
            'max_positions': 15,              # 最大持仓数量
            'position_sizing': 0.05,          # 单个头寸规模
            'stop_loss_pct': 0.07,            # 止损百分比
            'sector_allocations': {           # 行业配置权重
                'bull_early': {               # 牛市初期
                    'financial_services': 0.30,
                    'industrials': 0.25,
                    'basic_materials': 0.20,
                    'energy': 0.15,
                    'real_estate': 0.10,
                },
                'bull_mid': {                 # 牛市中期
                    'technology': 0.35,
                    'consumer_cyclical': 0.25,
                    'communication_services': 0.20,
                    'financial_services': 0.10,
                    'industrials': 0.10,
                },
                'bull_late': {                # 牛市后期
                    'consumer_defensive': 0.25,
                    'healthcare': 0.25,
                    'utilities': 0.20,
                    'technology': 0.15,
                    'communication_services': 0.15,
                },
                'bear': {                     # 熊市
                    'consumer_defensive': 0.35,
                    'utilities': 0.30,
                    'healthcare': 0.20,
                    'real_estate': 0.15,
                },
                'ranging': {                  # 震荡市
                    'technology': 0.20,
                    'healthcare': 0.20,
                    'financial_services': 0.15,
                    'consumer_defensive': 0.15,
                    'industrials': 0.10,
                    'consumer_cyclical': 0.10,
                    'utilities': 0.10,
                },
            },
            'technical_params': {            # 技术指标参数
                'ma_short': 20,
                'ma_long': 50,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            },
        }
        
        # 合并用户参数
        if params:
            for key, value in params.items():
                if key in default_params and isinstance(default_params[key], dict) and isinstance(value, dict):
                    default_params[key].update(value)
                else:
                    default_params[key] = value
        
        super().__init__("OpenBB Sector Rotation Strategy", default_params)
        self.market_phase = None  # 市场周期阶段
        self.market_regime = MarketRegime.UNKNOWN  # 当前市场环境
        self.last_analysis_date = None  # 上次分析日期
        self.sector_performance = {}  # 各行业表现
        self.sector_stocks = {}  # 各行业选择的股票
        
    def analyze_market_regime(self, data: pd.DataFrame = None) -> MarketRegime:
        """
        使用OpenBB分析当前市场环境
        
        Returns:
            MarketRegime: 当前市场环境枚举
        """
        try:
            # 获取市场指数数据
            market_index = self.parameters['market_index']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.parameters['lookback_days'])
            
            # 使用OpenBB获取历史数据
            market_data = obb.equity.price.historical(
                symbol=market_index,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            ).to_df()
            
            # 计算技术指标
            market_data['MA20'] = market_data['close'].rolling(window=20).mean()
            market_data['MA50'] = market_data['close'].rolling(window=50).mean()
            market_data['MA200'] = market_data['close'].rolling(window=200).mean()
            market_data['volatility'] = market_data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # 获取最新数据点
            latest = market_data.iloc[-1]
            
            # 判断市场趋势
            if latest['MA20'] > latest['MA50'] > latest['MA200']:
                # 牛市趋势
                if latest['volatility'] > 0.20:  # 高波动性
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.BULLISH
            elif latest['MA20'] < latest['MA50'] < latest['MA200']:
                # 熊市趋势
                return MarketRegime.BEARISH
            elif max(latest['MA20'], latest['MA50'], latest['MA200']) - min(latest['MA20'], latest['MA50'], latest['MA200']) < latest['close'] * 0.05:
                # 均线接近（震荡市）
                return MarketRegime.RANGING
            elif latest['volatility'] < 0.10:
                # 低波动性
                return MarketRegime.LOW_VOLATILITY
            else:
                # 无法明确判断
                return MarketRegime.UNKNOWN
                
        except Exception as e:
            logger.error(f"分析市场环境时出错: {str(e)}")
            return MarketRegime.UNKNOWN
    
    def determine_market_phase(self) -> str:
        """
        确定当前市场所处的经济周期阶段
        
        Returns:
            str: 市场周期阶段 ('bull_early', 'bull_mid', 'bull_late', 'bear', 'ranging')
        """
        market_regime = self.get_market_regime()
        
        # 获取经济指标
        try:
            # 获取失业率
            unemployment = obb.economy.unemployment().to_df()
            latest_unemployment = unemployment.iloc[-1]['value'] if not unemployment.empty else None
            unemployment_trend = unemployment.iloc[-3:]['value'].diff().mean() if len(unemployment) >= 3 else 0
            
            # 获取CPI（通胀率）
            cpi = obb.economy.cpi().to_df()
            latest_cpi = cpi.iloc[-1]['value'] if not cpi.empty else None
            cpi_trend = cpi.iloc[-3:]['value'].diff().mean() if len(cpi) >= 3 else 0
            
            # 获取利率
            interest_rate = obb.economy.fred_series(series_id="FEDFUNDS").to_df()
            latest_interest_rate = interest_rate.iloc[-1]['value'] if not interest_rate.empty else None
            interest_rate_trend = interest_rate.iloc[-3:]['value'].diff().mean() if len(interest_rate) >= 3 else 0
            
            # 根据市场环境和经济指标确定周期阶段
            if market_regime == MarketRegime.BULLISH:
                if interest_rate_trend > 0 and unemployment_trend < 0:
                    # 利率上升、失业率下降 -> 牛市中期
                    return 'bull_mid'
                elif interest_rate_trend < 0 and cpi_trend < 0:
                    # 利率下降、通胀下降 -> 牛市初期
                    return 'bull_early'
                else:
                    # 其他情况 -> 牛市后期
                    return 'bull_late'
            elif market_regime == MarketRegime.BEARISH:
                return 'bear'
            elif market_regime == MarketRegime.RANGING:
                return 'ranging'
            else:
                # 默认使用震荡市场配置
                return 'ranging'
                
        except Exception as e:
            logger.warning(f"确定市场阶段时出错: {str(e)}")
            
            # 根据市场环境直接确定阶段
            if market_regime == MarketRegime.BULLISH:
                return 'bull_mid'
            elif market_regime == MarketRegime.BEARISH:
                return 'bear'
            else:
                return 'ranging'
    
    def analyze_sector_performance(self) -> Dict[str, float]:
        """
        分析不同行业的表现
        
        Returns:
            Dict[str, float]: 行业及其表现分数
        """
        try:
            sectors = [
                'technology',
                'healthcare',
                'financial_services',
                'consumer_cyclical',
                'consumer_defensive',
                'industrials',
                'utilities',
                'energy',
                'basic_materials',
                'real_estate',
                'communication_services'
            ]
            
            performance = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 分析近30天表现
            
            for sector in sectors:
                try:
                    # 使用筛选器获取该行业的股票
                    stocks = obb.equity.screener(
                        sector=sector,
                        limit=20
                    ).to_df()
                    
                    if stocks.empty:
                        logger.warning(f"没有找到行业 {sector} 的股票")
                        performance[sector] = 0
                        continue
                    
                    # 获取行业ETF或前几只股票的表现
                    returns = []
                    for symbol in stocks['symbol'].head(5):
                        try:
                            stock_data = obb.equity.price.historical(
                                symbol=symbol,
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d')
                            ).to_df()
                            
                            if not stock_data.empty:
                                # 计算股票回报率
                                stock_return = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1
                                returns.append(stock_return)
                        except Exception as e:
                            logger.warning(f"获取股票 {symbol} 数据时出错: {str(e)}")
                    
                    # 计算行业平均回报率
                    if returns:
                        sector_return = sum(returns) / len(returns)
                    else:
                        sector_return = 0
                        
                    # 保存行业表现
                    performance[sector] = sector_return
                    
                except Exception as e:
                    logger.error(f"分析行业 {sector} 表现时出错: {str(e)}")
                    performance[sector] = 0
            
            # 归一化行业表现分数（转换为0-1之间）
            if performance:
                min_perf = min(performance.values())
                max_perf = max(performance.values())
                
                if max_perf > min_perf:
                    for sector in performance:
                        performance[sector] = (performance[sector] - min_perf) / (max_perf - min_perf)
                
            self.sector_performance = performance
            return performance
            
        except Exception as e:
            logger.error(f"分析行业表现时出错: {str(e)}")
            return {}
    
    def select_sector_stocks(self, sector: str, count: int = 3) -> List[str]:
        """
        选择特定行业内表现最好的股票
        
        Args:
            sector: 行业名称
            count: 选择的股票数量
            
        Returns:
            List[str]: 选中的股票代码列表
        """
        try:
            # 使用筛选器获取该行业的股票
            stocks = obb.equity.screener(
                sector=sector,
                mktcap_min=1000000000,  # 最小市值10亿
                price_min=10,           # 最低价格$10
                volume_min=500000,      # 最小成交量50万
                limit=50                # 最多获取50只股票
            ).to_df()
            
            if stocks.empty:
                logger.warning(f"没有找到行业 {sector} 的股票")
                return []
            
            # 获取这些股票的表现数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 分析近30天表现
            
            stock_metrics = []
            for symbol in stocks['symbol']:
                try:
                    stock_data = obb.equity.price.historical(
                        symbol=symbol,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    ).to_df()
                    
                    if stock_data.empty:
                        continue
                    
                    # 计算基本指标
                    returns = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1
                    volatility = stock_data['close'].pct_change().std() * np.sqrt(252)
                    volume_avg = stock_data['volume'].mean()
                    
                    # 计算动量和RSI
                    stock_data = self.calculate_indicators(stock_data)
                    latest = stock_data.iloc[-1]
                    rsi = latest.get('RSI', 50)
                    
                    # 计算综合得分 (更高的回报率、更低的波动性、更高的成交量得分更高)
                    score = (returns * 0.5) + (1/volatility * 0.3) + (volume_avg/1000000 * 0.2)
                    
                    # 根据RSI进行调整（避免超买超卖股票）
                    if rsi > 70 or rsi < 30:
                        score *= 0.7
                    
                    stock_metrics.append({
                        'symbol': symbol,
                        'returns': returns,
                        'volatility': volatility,
                        'volume_avg': volume_avg,
                        'rsi': rsi,
                        'score': score
                    })
                    
                except Exception as e:
                    logger.warning(f"分析股票 {symbol} 时出错: {str(e)}")
            
            # 按得分对股票排序并选择前几只
            stock_metrics.sort(key=lambda x: x['score'], reverse=True)
            selected_stocks = [item['symbol'] for item in stock_metrics[:count]]
            
            return selected_stocks
            
        except Exception as e:
            logger.error(f"选择行业 {sector} 的股票时出错: {str(e)}")
            return []
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标（实现抽象方法）
        
        Args:
            data: 价格数据DataFrame
            
        Returns:
            带有技术指标的DataFrame
        """
        if data.empty:
            return data
            
        df = data.copy()
        tech_params = self.parameters['technical_params']
        
        # 计算基本技术指标
        df['MA_short'] = df['close'].rolling(window=tech_params['ma_short']).mean()
        df['MA_long'] = df['close'].rolling(window=tech_params['ma_long']).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=tech_params['rsi_period']).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=tech_params['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算波动率
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号（实现抽象方法）
        
        Args:
            data: 包含技术指标的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        if data.empty:
            return data
            
        df = data.copy()
        tech_params = self.parameters['technical_params']
        
        # 确保必要的指标已计算
        if 'MA_short' not in df.columns or 'MA_long' not in df.columns or 'RSI' not in df.columns:
            df = self.calculate_indicators(df)
        
        # 初始化信号列
        df['signal'] = 0
        
        # 生成基本信号
        df.loc[(df['MA_short'] > df['MA_long']) & (df['RSI'] < 60), 'signal'] = 1  # 买入信号
        df.loc[(df['MA_short'] < df['MA_long']) | (df['RSI'] > 70), 'signal'] = -1  # 卖出信号
        
        return df
    
    def get_market_regime(self, data: pd.DataFrame = None) -> MarketRegime:
        """
        获取当前市场环境
        
        Args:
            data: 价格数据
            
        Returns:
            当前市场环境
        """
        # 如果没有保存的市场环境或者上次分析已经过期，重新分析
        current_date = datetime.now().date()
        if (self.market_regime == MarketRegime.UNKNOWN or 
            self.last_analysis_date is None or 
            (current_date - self.last_analysis_date).days >= 1):
            
            self.market_regime = self.analyze_market_regime(data)
            self.last_analysis_date = current_date
            
        return self.market_regime
    
    def run_strategy(self) -> Dict[str, Any]:
        """
        运行行业轮动策略
        
        Returns:
            策略结果字典
        """
        results = {
            'market_regime': None,
            'market_phase': None,
            'sector_performance': {},
            'selected_sectors': {},
            'portfolio': [],
            'timestamp': datetime.now()
        }
        
        try:
            # 1. 分析市场环境
            market_regime = self.get_market_regime()
            results['market_regime'] = market_regime.value
            
            # 2. 确定市场周期阶段
            market_phase = self.determine_market_phase()
            self.market_phase = market_phase
            results['market_phase'] = market_phase
            
            # 3. 分析各行业表现
            sector_performance = self.analyze_sector_performance()
            results['sector_performance'] = {k: float(v) for k, v in sector_performance.items()}
            
            # 4. 根据市场阶段选择行业配置
            sector_allocations = self.parameters['sector_allocations'][market_phase]
            results['selected_sectors'] = sector_allocations
            
            # 5. 为每个选定的行业选择股票
            portfolio = []
            for sector, allocation in sector_allocations.items():
                # 跳过表现太差的行业
                sector_score = sector_performance.get(sector, 0)
                if sector_score < 0.3 and market_phase != 'bear':  # 在熊市中仍保留防御性行业
                    continue
                
                # 为行业选择股票
                stocks_count = self.parameters['stocks_per_sector']
                sector_stocks = self.select_sector_stocks(sector, stocks_count)
                
                if not sector_stocks:
                    continue
                
                # 计算每只股票的权重
                stock_weight = allocation / len(sector_stocks)
                
                # 添加到投资组合
                for symbol in sector_stocks:
                    portfolio.append({
                        'symbol': symbol,
                        'sector': sector,
                        'weight': float(stock_weight),
                        'action': 'BUY',
                    })
            
            # 限制投资组合大小
            portfolio = portfolio[:self.parameters['max_positions']]
            
            # 重新计算权重以确保总和为1
            total_weight = sum(item['weight'] for item in portfolio)
            if total_weight > 0:
                for item in portfolio:
                    item['weight'] = float(item['weight'] / total_weight)
            
            # 获取每只股票的当前价格
            for item in portfolio:
                try:
                    quote = obb.equity.price.quote(symbol=item['symbol']).to_df()
                    current_price = quote['price'].iloc[0] if not quote.empty else None
                    
                    if current_price:
                        item['price'] = float(current_price)
                        item['stop_loss'] = float(current_price * (1 - self.parameters['stop_loss_pct']))
                except Exception as e:
                    logger.warning(f"获取股票 {item['symbol']} 价格时出错: {str(e)}")
            
            results['portfolio'] = portfolio
            
            return results
            
        except Exception as e:
            logger.error(f"运行策略时出错: {str(e)}")
            results['error'] = str(e)
            return results

def main():
    """运行行业轮动策略示例"""
    try:
        # 创建策略实例
        strategy = OpenBBSectorRotationStrategy()
        
        # 运行策略
        results = strategy.run_strategy()
        
        # 打印结果
        print(f"\n=== 行业轮动策略结果 ===")
        print(f"市场环境: {results['market_regime']}")
        print(f"市场阶段: {results['market_phase']}")
        
        print("\n--- 行业表现 ---")
        for sector, score in sorted(results['sector_performance'].items(), key=lambda x: x[1], reverse=True):
            print(f"{sector}: {score:.2f}")
        
        print("\n--- 选定行业配置 ---")
        for sector, allocation in results['selected_sectors'].items():
            print(f"{sector}: {allocation:.2%}")
        
        print("\n--- 推荐投资组合 ---")
        print(f"总股票数: {len(results['portfolio'])}")
        
        # 创建表格格式
        fmt = "{:<8} {:<25} {:<10} {:<8} {:<10}"
        print(fmt.format("股票", "行业", "权重", "价格", "止损价"))
        print("-" * 70)
        
        for stock in results['portfolio']:
            price_str = f"${stock['price']:.2f}" if 'price' in stock else "N/A"
            stop_loss_str = f"${stock['stop_loss']:.2f}" if 'stop_loss' in stock else "N/A"
            
            print(fmt.format(
                stock['symbol'],
                stock['sector'],
                f"{stock['weight']:.2%}",
                price_str,
                stop_loss_str
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"运行行业轮动策略示例时出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 