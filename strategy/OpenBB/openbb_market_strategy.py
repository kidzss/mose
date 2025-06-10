import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple

from openbb import obb
from ...strategy.strategy_base import Strategy, MarketRegime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenBBMarketStrategy(Strategy):
    """
    利用OpenBB的市场分析和股票筛选功能的策略
    
    该策略将OpenBB的数据分析能力与现有策略框架集成，用于:
    1. 市场环境分析
    2. 股票筛选
    3. 技术指标计算
    4. 基本面分析
    5. 宏观经济因素整合
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """初始化策略"""
        default_params = {
            'market_index': 'SPY',           # 用于分析市场环境的指数
            'lookback_days': 120,            # 分析历史数据的天数
            'screening_criteria': {          # 股票筛选标准
                'market_cap_min': 1000000000,  # 最小市值 (10亿)
                'price_min': 10,              # 最低价格
                'volume_min': 500000,         # 最小成交量
                'beta_min': 0.5,              # 最小Beta值
                'beta_max': 2.0,              # 最大Beta值
            },
            'sector_weights': {              # 各行业的权重
                'technology': 0.25,
                'healthcare': 0.15,
                'consumer_cyclical': 0.15,
                'financial_services': 0.15,
                'industrials': 0.10,
                'communication_services': 0.10,
                'consumer_defensive': 0.05,
                'energy': 0.05
            },
            'technical_params': {            # 技术指标参数
                'ma_short': 20,
                'ma_long': 50,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volatility_period': 20,
            },
            'rebalance_frequency': 30,       # 重新平衡投资组合的天数
            'max_positions': 20,             # 最大持仓数量
            'stop_loss_pct': 0.07,           # 止损百分比
            'take_profit_pct': 0.20,         # 止盈百分比
            'position_sizing': 0.05,         # 单个头寸规模（总资本百分比）
        }
        
        # 合并用户参数
        if params:
            for key, value in params.items():
                if key in default_params and isinstance(default_params[key], dict) and isinstance(value, dict):
                    default_params[key].update(value)
                else:
                    default_params[key] = value
        
        super().__init__("OpenBB Market Strategy", default_params)
        self.screened_stocks = []  # 筛选后的股票列表
        self.market_regime = MarketRegime.UNKNOWN  # 当前市场环境
        self.last_analysis_date = None  # 上次分析日期
        self.economic_indicators = {}  # 经济指标
        
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
    
    def screen_stocks(self) -> List[str]:
        """
        使用OpenBB筛选符合条件的股票
        
        Returns:
            List[str]: 筛选后的股票代码列表
        """
        try:
            criteria = self.parameters['screening_criteria']
            
            # 使用OpenBB的股票筛选功能
            screener_results = obb.equity.screener(
                mktcap_min=criteria.get('market_cap_min', 1000000000),
                price_min=criteria.get('price_min', 10),
                volume_min=criteria.get('volume_min', 500000),
                beta_min=criteria.get('beta_min', 0.5),
                beta_max=criteria.get('beta_max', 2.0),
                limit=100  # 限制结果数量
            ).to_df()
            
            # 检查是否有结果
            if screener_results.empty:
                logger.warning("股票筛选没有结果")
                return []
            
            # 返回筛选后的股票代码列表
            self.screened_stocks = screener_results['symbol'].tolist()
            return self.screened_stocks
            
        except Exception as e:
            logger.error(f"筛选股票时出错: {str(e)}")
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
        df['volatility'] = df['close'].pct_change().rolling(window=tech_params['volatility_period']).std() * np.sqrt(252)
        
        # 使用OpenBB计算MACD指标
        try:
            # 注意：这里假设数据的索引是日期格式
            symbol = df.get('symbol', ['UNKNOWN'])[0] if 'symbol' in df.columns else 'UNKNOWN'
            macd_data = obb.technical.macd(
                data=df,
                target='close',
                fast=12,
                slow=26,
                signal=9
            ).to_df()
            
            # 合并MACD结果到原始数据
            if not macd_data.empty:
                df['MACD'] = macd_data['MACD']
                df['MACD_signal'] = macd_data['MACD_signal']
                df['MACD_histogram'] = macd_data['MACD_histogram']
        except Exception as e:
            logger.warning(f"计算MACD指标时出错: {str(e)}")
        
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
        df['signal_strength'] = 0.0
        
        # 根据当前市场环境调整信号生成策略
        market_regime = self.get_market_regime(df)
        
        # 根据市场环境生成交易信号
        if market_regime == MarketRegime.BULLISH:
            # 牛市策略 - 关注突破和趋势跟踪
            df.loc[(df['MA_short'] > df['MA_long']) & 
                   (df['close'] > df['MA_short']) & 
                   (df['RSI'] < 60), 'signal'] = 1  # 买入信号
                   
            df.loc[(df['MA_short'] < df['MA_long']) | 
                   (df['RSI'] > 75), 'signal'] = -1  # 卖出信号
                   
        elif market_regime == MarketRegime.BEARISH:
            # 熊市策略 - 更保守，关注反转信号
            df.loc[(df['MA_short'] > df['MA_long']) & 
                   (df['RSI'] < 40) & 
                   (df['close'] > df['close'].shift(1)), 'signal'] = 1  # 买入信号
                   
            df.loc[(df['MA_short'] < df['MA_long']) | 
                   (df['close'] < df['MA_long']), 'signal'] = -1  # 卖出信号
                   
        elif market_regime == MarketRegime.RANGING:
            # 震荡市策略 - 关注超买超卖
            df.loc[(df['RSI'] < tech_params['rsi_oversold']) & 
                   (df['close'] > df['MA_short']), 'signal'] = 1  # 买入信号
                   
            df.loc[(df['RSI'] > tech_params['rsi_overbought']), 'signal'] = -1  # 卖出信号
            
        elif market_regime == MarketRegime.VOLATILE:
            # 高波动性策略 - 更短的时间周期，更快的进出
            df.loc[(df['close'] > df['MA_short']) & 
                   (df['RSI'] < 50) & 
                   (df['MACD'] > df['MACD_signal'] if 'MACD' in df.columns else True), 'signal'] = 1  # 买入信号
                   
            df.loc[(df['close'] < df['MA_short']) | 
                   (df['RSI'] > 65), 'signal'] = -1  # 卖出信号
                   
        else:  # 默认策略
            # 综合策略
            df.loc[(df['MA_short'] > df['MA_long']) & 
                   (df['RSI'] < 60) & 
                   (df['close'] > df['close'].shift(1)), 'signal'] = 1  # 买入信号
                   
            df.loc[(df['MA_short'] < df['MA_long']) | 
                   (df['RSI'] > 70), 'signal'] = -1  # 卖出信号
        
        # 计算信号强度
        # RSI信号强度（0-1之间）
        rsi_strength = (70 - df['RSI']) / 40  # RSI为30时强度为1，RSI为70时强度为0
        rsi_strength = rsi_strength.clip(0, 1)  # 限制在0-1之间
        
        # 均线信号强度
        ma_strength = (df['MA_short'] - df['MA_long']) / df['MA_long']
        ma_strength = (ma_strength + 0.05) / 0.1  # 归一化，当短期均线比长期均线高5%时，强度为0.5
        ma_strength = ma_strength.clip(0, 1)
        
        # 组合信号强度
        df['signal_strength'] = 0.6 * rsi_strength + 0.4 * ma_strength
        
        # 调整信号强度方向
        df.loc[df['signal'] == -1, 'signal_strength'] = 1 - df.loc[df['signal'] == -1, 'signal_strength']
        
        return df
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
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
    
    def get_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """
        确定头寸规模
        
        Args:
            data: 价格数据
            signal: 信号强度
            
        Returns:
            头寸规模（占总资本的比例）
        """
        base_size = self.parameters['position_sizing']
        
        # 获取当前市场环境
        market_regime = self.get_market_regime(data)
        
        # 根据市场环境调整头寸规模
        if market_regime == MarketRegime.BULLISH:
            # 牛市可以增加头寸规模
            adjusted_size = base_size * 1.2
        elif market_regime == MarketRegime.BEARISH:
            # 熊市减少头寸规模
            adjusted_size = base_size * 0.8
        elif market_regime == MarketRegime.VOLATILE:
            # 高波动性市场减少头寸规模
            adjusted_size = base_size * 0.7
        else:
            adjusted_size = base_size
        
        # 根据信号强度进一步调整
        if abs(signal) > 0:
            signal_adjustment = 0.5 + (abs(signal) * 0.5)  # 信号强度从0.5到1
        else:
            signal_adjustment = 0.5  # 默认值
            
        final_size = adjusted_size * signal_adjustment
        
        # 确保头寸规模不超过最大限制
        return min(final_size, base_size * 1.5)
    
    def get_economic_indicators(self) -> Dict[str, float]:
        """
        获取重要经济指标
        
        Returns:
            经济指标字典
        """
        try:
            # 获取当前日期
            current_date = datetime.now()
            
            # 使用OpenBB获取联邦基金利率
            fed_rate = obb.economy.fred_series(series_id="FEDFUNDS").to_df()
            latest_fed_rate = fed_rate.iloc[-1]['value'] if not fed_rate.empty else None
            
            # 获取CPI数据
            cpi = obb.economy.cpi().to_df()
            latest_cpi = cpi.iloc[-1]['value'] if not cpi.empty else None
            
            # 获取失业率
            unemployment = obb.economy.unemployment().to_df()
            latest_unemployment = unemployment.iloc[-1]['value'] if not unemployment.empty else None
            
            # 更新经济指标字典
            self.economic_indicators = {
                'fed_rate': latest_fed_rate,
                'cpi': latest_cpi,
                'unemployment': latest_unemployment,
                'last_update': current_date
            }
            
            return self.economic_indicators
            
        except Exception as e:
            logger.error(f"获取经济指标时出错: {str(e)}")
            return self.economic_indicators
    
    def run_strategy(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        运行完整策略流程
        
        Args:
            symbols: 要分析的股票列表，如果为None则使用筛选的股票
            
        Returns:
            策略结果字典
        """
        results = {
            'market_regime': None,
            'economic_indicators': {},
            'stock_signals': {},
            'portfolio_recommendation': [],
            'timestamp': datetime.now()
        }
        
        try:
            # 1. 分析市场环境
            market_regime = self.analyze_market_regime()
            results['market_regime'] = market_regime.value
            
            # 2. 获取经济指标
            economic_indicators = self.get_economic_indicators()
            results['economic_indicators'] = economic_indicators
            
            # 3. 筛选股票（如果没有提供）
            if symbols is None:
                symbols = self.screen_stocks()
                
            if not symbols:
                logger.warning("没有股票可供分析")
                return results
            
            # 4. 分析每只股票并生成信号
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # 获取60天的数据用于分析
            
            for symbol in symbols:
                try:
                    # 获取股票数据
                    stock_data = obb.equity.price.historical(
                        symbol=symbol,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    ).to_df()
                    
                    if stock_data.empty:
                        logger.warning(f"没有找到股票 {symbol} 的数据")
                        continue
                    
                    # 计算指标和信号
                    stock_data = self.calculate_indicators(stock_data)
                    stock_data = self.generate_signals(stock_data)
                    
                    # 获取最新信号
                    latest = stock_data.iloc[-1]
                    signal = latest.get('signal', 0)
                    signal_strength = latest.get('signal_strength', 0)
                    
                    # 获取公司基本信息
                    try:
                        company_info = obb.equity.profile(symbol=symbol).to_df()
                        sector = company_info.get('sector', ['Unknown'])[0] if not company_info.empty else 'Unknown'
                        industry = company_info.get('industry', ['Unknown'])[0] if not company_info.empty else 'Unknown'
                    except:
                        sector = "Unknown"
                        industry = "Unknown"
                    
                    # 保存结果
                    results['stock_signals'][symbol] = {
                        'signal': int(signal),
                        'signal_strength': float(signal_strength),
                        'last_price': float(latest['close']),
                        'sector': sector,
                        'industry': industry,
                        'rsi': float(latest.get('RSI', 0)),
                        'volatility': float(latest.get('volatility', 0)),
                    }
                    
                    # 如果有买入信号，添加到投资组合建议
                    if signal > 0 and signal_strength > 0.6:
                        position_size = self.get_position_size(stock_data, signal_strength)
                        
                        results['portfolio_recommendation'].append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'strength': float(signal_strength),
                            'price': float(latest['close']),
                            'position_size': float(position_size),
                            'stop_loss': float(latest['close'] * (1 - self.parameters['stop_loss_pct'])),
                            'take_profit': float(latest['close'] * (1 + self.parameters['take_profit_pct'])),
                            'sector': sector,
                            'industry': industry,
                        })
                    elif signal < 0 and signal_strength > 0.6:
                        results['portfolio_recommendation'].append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'strength': float(signal_strength),
                            'price': float(latest['close']),
                            'sector': sector,
                            'industry': industry,
                        })
                        
                except Exception as e:
                    logger.error(f"分析股票 {symbol} 时出错: {str(e)}")
                    continue
            
            # 5. 对投资组合建议进行排序（按信号强度）
            results['portfolio_recommendation'] = sorted(
                results['portfolio_recommendation'], 
                key=lambda x: x['strength'], 
                reverse=True
            )
            
            # 6. 限制投资组合大小
            results['portfolio_recommendation'] = results['portfolio_recommendation'][:self.parameters['max_positions']]
            
            return results
            
        except Exception as e:
            logger.error(f"运行策略时出错: {str(e)}")
            results['error'] = str(e)
            return results 