import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from data.data_interface import YahooFinanceDataSource
from data.market_state_analyzer import MarketStateAnalyzer

logger = logging.getLogger(__name__)

class PortfolioAdvisor:
    def __init__(self):
        self.data_source = YahooFinanceDataSource()
        self.market_analyzer = MarketStateAnalyzer()
        
    async def analyze_market_trend(self, symbol: str, days: int = 30) -> Dict:
        """分析市场趋势"""
        try:
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 获取历史数据
            df = await self.data_source.get_historical_data(symbol, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return {'error': f"无法获取 {symbol} 的数据"}
                
            # 计算关键指标
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            df['RSI'] = self._calculate_rsi(df['close'])
            df['MACD'], df['Signal'], df['Hist'] = self._calculate_macd(df['close'])
            
            # 计算趋势强度
            trend_strength = self._calculate_trend_strength(df)
            
            # 获取市场状态
            market_states = await self.market_analyzer.analyze_market_states(df)
            
            # 综合分析
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            return {
                'symbol': symbol,
                'current_price': latest['close'],
                'price_change': (latest['close'] - prev['close']) / prev['close'],
                'volume_change': (latest['volume'] - prev['volume']) / prev['volume'],
                'trend_strength': trend_strength,
                'rsi': latest['RSI'],
                'macd_hist': latest['Hist'],
                'sma20_trend': 1 if latest['SMA20'] > prev['SMA20'] else -1,
                'sma50_trend': 1 if latest['SMA50'] > prev['SMA50'] else -1,
                'market_state': market_states[-1] if market_states else None
            }
        except Exception as e:
            logger.error(f"分析 {symbol} 时出错: {str(e)}")
            return {'error': str(e)}
            
    async def generate_portfolio_advice(self, positions: Dict[str, Dict], watchlist: List[str]) -> Dict:
        """生成投资组合建议"""
        advice = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'market_overview': "市场处于下跌趋势，建议谨慎操作",
            'position_advice': [],
            'watchlist_advice': [],
            'portfolio_actions': []
        }
        
        # 分析所有持仓
        position_analyses = {}
        for symbol in positions:
            analysis = await self.analyze_market_trend(symbol)
            position_analyses[symbol] = analysis
            
        # 分析观察列表
        watchlist_analyses = {}
        for symbol in watchlist:
            analysis = await self.analyze_market_trend(symbol)
            watchlist_analyses[symbol] = analysis
            
        # 评估每个持仓
        for symbol, analysis in position_analyses.items():
            if 'error' in analysis:
                continue
                
            position_info = positions[symbol]
            avg_price = position_info.get('avg_price', 0)
            current_price = analysis['current_price']
            pnl = (current_price - avg_price) / avg_price if avg_price > 0 else 0
            
            # 计算综合得分
            score = self._calculate_position_score(analysis)
            
            advice_item = {
                'symbol': symbol,
                'current_price': current_price,
                'pnl': pnl,
                'score': score,
                'action': self._determine_position_action(score, pnl, analysis)
            }
            
            advice['position_advice'].append(advice_item)
            
        # 评估观察列表中的股票
        for symbol, analysis in watchlist_analyses.items():
            if 'error' in analysis:
                continue
                
            # 计算综合得分
            score = self._calculate_position_score(analysis)
            
            advice_item = {
                'symbol': symbol,
                'current_price': analysis['current_price'],
                'score': score,
                'action': self._determine_watchlist_action(score, analysis)
            }
            
            advice['watchlist_advice'].append(advice_item)
            
        # 生成投资组合级别的建议
        advice['portfolio_actions'] = self._generate_portfolio_actions(
            advice['position_advice'],
            advice['watchlist_advice']
        )
        
        return advice
        
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
        
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度"""
        # 使用价格、成交量和技术指标计算趋势强度
        price_trend = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        volume_trend = (df['volume'] - df['volume'].shift(20)) / df['volume'].shift(20)
        rsi_trend = (df['RSI'] - 50) / 50
        macd_trend = df['Hist'] / df['close']
        
        # 综合各个指标
        strength = (
            price_trend * 0.4 +
            volume_trend * 0.2 +
            rsi_trend * 0.2 +
            macd_trend * 0.2
        )
        
        return float(strength.iloc[-1])
        
    def _calculate_position_score(self, analysis: Dict) -> float:
        """计算持仓得分"""
        score = 0
        
        # 趋势分数 (-2 到 2)
        score += analysis['trend_strength'] * 2
        
        # RSI分数 (-1 到 1)
        rsi = analysis['rsi']
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1
        
        # MACD分数 (-1 到 1)
        score += np.sign(analysis['macd_hist'])
        
        # 移动平均线分数 (-2 到 2)
        score += analysis['sma20_trend']
        score += analysis['sma50_trend']
        
        return score
        
    def _determine_position_action(self, score: float, pnl: float, analysis: Dict) -> Dict:
        """确定持仓操作建议"""
        action = {
            'type': 'hold',
            'reason': [],
            'urgency': 'low'
        }
        
        # 止损条件
        if pnl < -0.15:  # 亏损超过15%
            action['type'] = 'sell'
            action['reason'].append('触发止损线')
            action['urgency'] = 'high'
            return action
            
        # 止盈条件
        if pnl > 0.20 and score < 0:  # 盈利超过20%且评分转负
            action['type'] = 'sell'
            action['reason'].append('获利了结')
            action['urgency'] = 'medium'
            return action
            
        # 基于得分的建议
        if score <= -5:
            action['type'] = 'sell'
            action['reason'].append('技术指标严重恶化')
            action['urgency'] = 'high'
        elif score <= -3:
            action['type'] = 'reduce'
            action['reason'].append('技术指标转弱')
            action['urgency'] = 'medium'
        elif score >= 5:
            action['type'] = 'add'
            action['reason'].append('技术指标强势')
            action['urgency'] = 'medium'
        else:
            action['reason'].append('维持观望')
            
        return action
        
    def _determine_watchlist_action(self, score: float, analysis: Dict) -> Dict:
        """确定观察股票操作建议"""
        action = {
            'type': 'watch',
            'reason': [],
            'urgency': 'low'
        }
        
        if score >= 6:
            action['type'] = 'buy'
            action['reason'].append('多项指标显示强势买入信号')
            action['urgency'] = 'high'
        elif score >= 4:
            action['type'] = 'prepare_buy'
            action['reason'].append('技术指标转强，建议准备买入')
            action['urgency'] = 'medium'
        elif score <= -4:
            action['type'] = 'remove_watch'
            action['reason'].append('技术指标恶化，建议移出观察名单')
            action['urgency'] = 'low'
            
        return action
        
    def _generate_portfolio_actions(self, position_advice: List[Dict], watchlist_advice: List[Dict]) -> List[Dict]:
        """生成投资组合级别的操作建议"""
        actions = []
        
        # 统计需要卖出和减仓的持仓
        sell_positions = [p for p in position_advice if p['action']['type'] in ['sell', 'reduce']]
        
        # 统计推荐买入的观察股票
        buy_candidates = [w for w in watchlist_advice if w['action']['type'] in ['buy', 'prepare_buy']]
        
        # 如果有紧急卖出建议
        urgent_sells = [p for p in sell_positions if p['action']['urgency'] == 'high']
        if urgent_sells:
            actions.append({
                'type': 'portfolio_alert',
                'urgency': 'high',
                'message': '以下持仓建议立即卖出：',
                'symbols': [p['symbol'] for p in urgent_sells],
                'reasons': [p['action']['reason'] for p in urgent_sells]
            })
            
        # 投资组合再平衡建议
        if sell_positions and buy_candidates:
            actions.append({
                'type': 'portfolio_rebalance',
                'urgency': 'medium',
                'message': '建议进行以下调仓操作：',
                'sell_symbols': [p['symbol'] for p in sell_positions],
                'buy_symbols': [w['symbol'] for w in buy_candidates[:len(sell_positions)]]
            })
            
        # 整体市场建议
        avg_position_score = np.mean([p['score'] for p in position_advice])
        if avg_position_score < -2:
            actions.append({
                'type': 'portfolio_defense',
                'urgency': 'high',
                'message': '当前市场环境不佳，建议提高现金持仓比例，减少风险敞口'
            })
        elif avg_position_score > 2:
            actions.append({
                'type': 'portfolio_offense',
                'urgency': 'medium',
                'message': '市场出现机会，可以适当增加仓位'
            })
            
        return actions 