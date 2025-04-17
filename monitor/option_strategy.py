import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class OptionProtectionStrategy:
    def __init__(self, symbol='SPY', portfolio_value=100000):
        """
        初始化期权保护策略
        :param symbol: 标的股票代码
        :param portfolio_value: 投资组合价值
        """
        self.symbol = symbol
        self.portfolio_value = portfolio_value
        self.underlying = yf.Ticker(symbol)
        
    async def analyze_market_condition(self):
        """分析市场条件，判断是否需要保护"""
        try:
            # 获取最近10天的数据以获得更好的技术指标
            hist = self.underlying.history(period='10d')
            if hist.empty:
                return False, "无法获取历史数据"
                
            # 计算技术指标
            current_price = hist['Close'].iloc[-1]
            sma_5 = hist['Close'].rolling(window=5).mean().iloc[-1]
            sma_10 = hist['Close'].rolling(window=10).mean().iloc[-1]
            rsi = self._calculate_rsi(hist['Close'])
            
            # 计算波动率
            daily_returns = hist['Close'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252) * 100  # 年化波动率
            
            # 判断是否需要保护
            price_below_key_level = current_price < 4900
            price_below_ma = current_price < sma_5 and current_price < sma_10
            high_volatility = volatility > 20  # 波动率大于20%
            
            needs_protection = (
                price_below_key_level and  # 跌破关键点位
                (price_below_ma or high_volatility)  # 技术面恶化或波动率高
            )
            
            analysis_result = {
                'current_price': float(current_price),
                'sma_5': float(sma_5),
                'sma_10': float(sma_10),
                'rsi': float(rsi),
                'volatility': float(volatility),
                'price_below_key_level': price_below_key_level,
                'price_below_ma': price_below_ma,
                'high_volatility': high_volatility,
                'message': '市场条件满足保护需求' if needs_protection else '当前市场条件不需要保护'
            }
            
            return needs_protection, analysis_result
                
        except Exception as e:
            logger.error(f"分析市场条件时出错: {str(e)}")
            return False, f"分析出错: {str(e)}"
            
    def _calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        try:
            # 计算价格变化
            delta = prices.diff()
            
            # 分离上涨和下跌
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # 计算平均上涨和下跌
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # 处理除以零的情况
            avg_loss = avg_loss.replace(0, 0.00001)
            
            # 计算相对强度
            rs = avg_gain / avg_loss
            
            # 计算RSI
            rsi = 100 - (100 / (1 + rs))
            
            # 获取最后一个有效值
            last_valid = rsi.dropna().iloc[-1]
            return float(last_valid)
            
        except Exception as e:
            logger.error(f"计算RSI时出错: {str(e)}")
            return 50.0  # 返回中性值
        
    async def get_option_chain(self):
        """获取期权链数据"""
        try:
            # 获取下个月到期的期权
            expirations = self.underlying.options
            next_month = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            closest_expiry = min(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(next_month, '%Y-%m-%d')))
            
            # 获取期权链
            opt = self.underlying.option_chain(closest_expiry)
            puts = opt.puts
            
            # 选择执行价格为4900的put期权
            target_strike = 4900
            closest_strike = min(puts['strike'], key=lambda x: abs(x - target_strike))
            put_option = puts[puts['strike'] == closest_strike].iloc[0]
            
            return {
                'expiration': closest_expiry,
                'strike': closest_strike,
                'last_price': put_option['lastPrice'],
                'bid': put_option['bid'],
                'ask': put_option['ask'],
                'volume': put_option['volume'],
                'open_interest': put_option['openInterest']
            }
            
        except Exception as e:
            logger.error(f"获取期权链时出错: {str(e)}")
            return None
            
    async def calculate_position_size(self, option_price):
        """计算合适的仓位大小"""
        # 使用投资组合价值的10%进行保护
        protection_value = self.portfolio_value * 0.1
        # 计算需要购买的期权数量
        position_size = int(protection_value / (option_price * 100))  # 每个期权合约代表100股
        return max(1, position_size)  # 至少购买1个合约
        
    async def generate_trading_signal(self):
        """生成交易信号"""
        try:
            # 分析市场条件
            needs_protection, market_analysis = await self.analyze_market_condition()
            
            if not needs_protection:
                return {
                    'action': 'hold',
                    'reason': market_analysis['message'],
                    'details': market_analysis
                }
                
            # 获取期权链数据
            option_data = await self.get_option_chain()
            if not option_data:
                return {
                    'action': 'hold',
                    'reason': '无法获取期权数据',
                    'details': None
                }
                
            # 计算仓位大小
            position_size = await self.calculate_position_size(option_data['last_price'])
            
            return {
                'action': 'buy_put',
                'reason': '市场条件满足保护需求',
                'details': {
                    'market_analysis': market_analysis,
                    'option_data': option_data,
                    'position_size': position_size,
                    'total_cost': position_size * option_data['last_price'] * 100,
                    'risk_management': {
                        'max_loss': position_size * option_data['last_price'] * 100,
                        'exit_condition': 'SPY > 5000',
                        'time_stop': '到期前一周'
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"生成交易信号时出错: {str(e)}")
            return {
                'action': 'error',
                'reason': str(e),
                'details': None
            } 