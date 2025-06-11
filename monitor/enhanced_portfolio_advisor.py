import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import asyncio

# 导入我们的核心策略
from strategy.strategy_factory import StrategyFactory
from strategy.combined_strategy import CombinedStrategy
from strategy.niuniu_strategy_v3 import NiuniuStrategyV3
from strategy.tdi_strategy import TDIStrategy
from strategy.cpgw_strategy import CPGWStrategy

logger = logging.getLogger(__name__)

class EnhancedPortfolioAdvisor:
    """
    增强的投资组合顾问
    集成了优化后的策略系统，提供更准确的持股分析
    """

    def __init__(self):
        """初始化增强投资组合顾问"""
        self.strategy_factory = StrategyFactory()
        
        # 初始化核心策略
        self.combined_strategy = self.strategy_factory.create_combined_strategy()
        self.niuniu_strategy = NiuniuStrategyV3()
        self.tdi_strategy = TDIStrategy()
        self.cpgw_strategy = CPGWStrategy()
        
        logger.info("✅ 增强投资组合顾问初始化完成")

    async def analyze_portfolio_with_strategies(self, positions: Dict[str, Dict], 
                                               watchlist: List[str] = None) -> Dict[str, Any]:
        """
        使用策略系统分析投资组合
        
        Args:
            positions: 持仓信息 {'symbol': {'shares': int, 'avg_price': float, 'weight': float}}
            watchlist: 观察列表
            
        Returns:
            完整的投资组合分析报告
        """
        logger.info("🔍 开始策略驱动的投资组合分析...")
        
        analysis_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'portfolio_overview': {},
            'position_analysis': {},
            'watchlist_analysis': {},
            'strategy_signals': {},
            'portfolio_recommendations': [],
            'risk_assessment': {}
        }
        
        # 1. 分析持仓股票
        logger.info("📊 分析持仓股票...")
        for symbol, position_info in positions.items():
            try:
                position_analysis = await self._analyze_position_with_strategy(symbol, position_info)
                analysis_results['position_analysis'][symbol] = position_analysis
            except Exception as e:
                logger.error(f"❌ 分析持仓 {symbol} 失败: {e}")
                
        # 2. 分析观察列表
        if watchlist:
            logger.info("👀 分析观察列表...")
            for symbol in watchlist:
                try:
                    watch_analysis = await self._analyze_watchlist_stock(symbol)
                    analysis_results['watchlist_analysis'][symbol] = watch_analysis
                except Exception as e:
                    logger.error(f"❌ 分析观察股票 {symbol} 失败: {e}")
        
        # 3. 生成投资组合级别的建议
        analysis_results['portfolio_recommendations'] = self._generate_portfolio_recommendations(
            analysis_results['position_analysis'],
            analysis_results['watchlist_analysis']
        )
        
        # 4. 风险评估
        analysis_results['risk_assessment'] = self._assess_portfolio_risk(
            analysis_results['position_analysis']
        )
        
        logger.info("✅ 投资组合分析完成")
        return analysis_results

    async def _analyze_position_with_strategy(self, symbol: str, position_info: Dict) -> Dict[str, Any]:
        """使用策略分析单个持仓"""
        try:
            # 获取股票数据
            data = await self._get_stock_data(symbol, days=60)
            if data is None or len(data) < 30:
                return {'error': f'{symbol} 数据不足'}
            
            # 使用组合策略分析
            strategy_signals = self.combined_strategy.generate_signals(data)
            current_signal = strategy_signals['signal'].iloc[-1] if not strategy_signals['signal'].empty else 0
            signal_strength = strategy_signals['signal_strength'].iloc[-1] if 'signal_strength' in strategy_signals.columns else 0
            
            # 获取各策略的单独信号
            niuniu_signals = self.niuniu_strategy.generate_signals(data)
            tdi_signals = self.tdi_strategy.generate_signals(data)
            cpgw_signals = self.cpgw_strategy.generate_signals(data)
            
            current_price = data['close'].iloc[-1]
            avg_price = position_info['avg_price']
            shares = position_info['shares']
            
            # 计算盈亏
            unrealized_pnl = (current_price - avg_price) / avg_price
            unrealized_amount = (current_price - avg_price) * shares
            
            # 生成持仓建议
            position_recommendation = self._generate_position_recommendation(
                current_signal, signal_strength, unrealized_pnl, 
                {'niuniu': niuniu_signals, 'tdi': tdi_signals, 'cpgw': cpgw_signals}
            )
            
            # 计算止损止盈建议
            stop_loss = self.combined_strategy.get_stop_loss(data, current_price, 1 if current_signal > 0 else -1)
            take_profit = self.combined_strategy.get_take_profit(data, current_price, 1 if current_signal > 0 else -1)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'position_info': position_info,
                'unrealized_pnl_percent': unrealized_pnl,
                'unrealized_pnl_amount': unrealized_amount,
                'strategy_analysis': {
                    'combined_signal': int(current_signal),
                    'signal_strength': float(signal_strength),
                    'individual_signals': {
                        'niuniu': int(niuniu_signals['signal'].iloc[-1]) if 'signal' in niuniu_signals.columns else 0,
                        'tdi': int(tdi_signals['signal'].iloc[-1]) if 'signal' in tdi_signals.columns else 0,
                        'cpgw': int(cpgw_signals['signal'].iloc[-1]) if 'signal' in cpgw_signals.columns else 0
                    }
                },
                'recommendation': position_recommendation,
                'risk_management': {
                    'stop_loss_price': stop_loss,
                    'take_profit_price': take_profit,
                    'stop_loss_percent': (stop_loss - current_price) / current_price,
                    'take_profit_percent': (take_profit - current_price) / current_price
                },
                'technical_indicators': {
                    'rsi': data['RSI'].iloc[-1] if 'RSI' in data.columns else None,
                    'macd': data['MACD'].iloc[-1] if 'MACD' in data.columns else None,
                    'volatility': data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
                }
            }
            
        except Exception as e:
            logger.error(f"分析持仓 {symbol} 出错: {e}")
            return {'error': str(e)}

    async def _analyze_watchlist_stock(self, symbol: str) -> Dict[str, Any]:
        """分析观察列表中的股票"""
        try:
            # 获取股票数据
            data = await self._get_stock_data(symbol, days=60)
            if data is None or len(data) < 30:
                return {'error': f'{symbol} 数据不足'}
            
            # 使用组合策略分析
            strategy_signals = self.combined_strategy.generate_signals(data)
            current_signal = strategy_signals['signal'].iloc[-1] if not strategy_signals['signal'].empty else 0
            signal_strength = strategy_signals['signal_strength'].iloc[-1] if 'signal_strength' in strategy_signals.columns else 0
            
            current_price = data['close'].iloc[-1]
            
            # 生成买入建议
            buy_recommendation = self._generate_buy_recommendation(current_signal, signal_strength, data)
            
            # 计算建议仓位大小
            position_size = self.combined_strategy.get_position_size(data, int(current_signal))
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'strategy_analysis': {
                    'combined_signal': int(current_signal),
                    'signal_strength': float(signal_strength)
                },
                'recommendation': buy_recommendation,
                'suggested_position_size': position_size,
                'entry_price': current_price,
                'technical_indicators': {
                    'rsi': data['RSI'].iloc[-1] if 'RSI' in data.columns else None,
                    'volatility': data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
                }
            }
            
        except Exception as e:
            logger.error(f"分析观察股票 {symbol} 出错: {e}")
            return {'error': str(e)}

    def _generate_position_recommendation(self, signal: float, strength: float, 
                                        pnl: float, individual_signals: Dict) -> Dict[str, Any]:
        """生成持仓操作建议"""
        recommendation = {
            'action': 'hold',
            'confidence': 'medium',
            'reasons': [],
            'urgency': 'low'
        }
        
        # 止损条件 - 最高优先级
        if pnl < -0.15:  # 亏损超过15%
            recommendation.update({
                'action': 'sell',
                'confidence': 'high',
                'urgency': 'high',
                'reasons': ['触发15%止损线']
            })
            return recommendation
        
        # 策略信号分析
        if signal > 0 and strength > 0.7:
            if pnl > 0.10:  # 已有盈利的情况下，可以继续持有
                recommendation.update({
                    'action': 'hold',
                    'confidence': 'high',
                    'reasons': ['策略信号强势', f'当前盈利{pnl:.1%}']
                })
            else:
                recommendation.update({
                    'action': 'add',
                    'confidence': 'medium',
                    'reasons': ['策略信号转强，建议加仓']
                })
        elif signal < 0 and strength > 0.7:
            if pnl > 0.20:  # 盈利超过20%，策略转弱，建议获利了结
                recommendation.update({
                    'action': 'sell',
                    'confidence': 'high',
                    'urgency': 'medium',
                    'reasons': ['获利了结', '策略信号转弱']
                })
            else:
                recommendation.update({
                    'action': 'reduce',
                    'confidence': 'medium',
                    'reasons': ['策略信号转弱，建议减仓']
                })
        else:
            # 信号不明确或强度不够
            if pnl < -0.10:  # 亏损超过10%且信号不明确
                recommendation.update({
                    'action': 'reduce',
                    'confidence': 'medium',
                    'reasons': ['亏损较大且信号不明确']
                })
        
        return recommendation

    def _generate_buy_recommendation(self, signal: float, strength: float, data: pd.DataFrame) -> Dict[str, Any]:
        """生成买入建议"""
        recommendation = {
            'action': 'watch',
            'confidence': 'low',
            'reasons': [],
            'urgency': 'low'
        }
        
        if signal > 0 and strength > 0.8:
            recommendation.update({
                'action': 'buy',
                'confidence': 'high',
                'urgency': 'high',
                'reasons': ['强势买入信号', f'信号强度{strength:.2f}']
            })
        elif signal > 0 and strength > 0.6:
            recommendation.update({
                'action': 'prepare_buy',
                'confidence': 'medium',
                'urgency': 'medium',
                'reasons': ['中等强度买入信号']
            })
        elif signal < 0 and strength > 0.7:
            recommendation.update({
                'action': 'avoid',
                'confidence': 'high',
                'reasons': ['策略显示卖出信号，建议避免买入']
            })
        
        return recommendation

    def _generate_portfolio_recommendations(self, position_analysis: Dict, 
                                          watchlist_analysis: Dict) -> List[Dict[str, Any]]:
        """生成投资组合级别的建议"""
        recommendations = []
        
        # 统计各种操作建议
        sell_positions = []
        reduce_positions = []
        add_positions = []
        buy_candidates = []
        
        for symbol, analysis in position_analysis.items():
            if 'recommendation' not in analysis:
                continue
                
            action = analysis['recommendation']['action']
            if action == 'sell':
                sell_positions.append(symbol)
            elif action == 'reduce':
                reduce_positions.append(symbol)
            elif action == 'add':
                add_positions.append(symbol)
        
        for symbol, analysis in watchlist_analysis.items():
            if 'recommendation' not in analysis:
                continue
                
            action = analysis['recommendation']['action']
            if action in ['buy', 'prepare_buy']:
                buy_candidates.append(symbol)
        
        # 生成建议
        if sell_positions:
            recommendations.append({
                'type': 'urgent_action',
                'title': '建议卖出',
                'symbols': sell_positions,
                'reason': '策略信号显示应该卖出这些持仓'
            })
        
        if reduce_positions:
            recommendations.append({
                'type': 'risk_management',
                'title': '建议减仓',
                'symbols': reduce_positions,
                'reason': '降低风险敞口'
            })
        
        if buy_candidates:
            recommendations.append({
                'type': 'opportunity',
                'title': '买入机会',
                'symbols': buy_candidates,
                'reason': '策略信号显示这些股票有买入机会'
            })
        
        if add_positions:
            recommendations.append({
                'type': 'position_optimization',
                'title': '建议加仓',
                'symbols': add_positions,
                'reason': '策略信号转强，可以增加仓位'
            })
        
        return recommendations

    def _assess_portfolio_risk(self, position_analysis: Dict) -> Dict[str, Any]:
        """评估投资组合风险"""
        risk_metrics = {
            'overall_risk': 'medium',
            'high_risk_positions': [],
            'risk_factors': [],
            'suggestions': []
        }
        
        high_volatility_positions = []
        large_loss_positions = []
        
        for symbol, analysis in position_analysis.items():
            if 'error' in analysis:
                continue
                
            # 检查高波动率
            if 'technical_indicators' in analysis and analysis['technical_indicators']['volatility'] > 0.4:
                high_volatility_positions.append(symbol)
            
            # 检查大幅亏损
            if analysis.get('unrealized_pnl_percent', 0) < -0.1:
                large_loss_positions.append(symbol)
        
        if high_volatility_positions:
            risk_metrics['high_risk_positions'].extend(high_volatility_positions)
            risk_metrics['risk_factors'].append('部分持仓波动率过高')
        
        if large_loss_positions:
            risk_metrics['risk_factors'].append('部分持仓存在较大亏损')
            risk_metrics['suggestions'].append('考虑止损或减仓高亏损持仓')
        
        # 评估整体风险等级
        risk_score = len(high_volatility_positions) + len(large_loss_positions) * 2
        if risk_score >= 4:
            risk_metrics['overall_risk'] = 'high'
        elif risk_score >= 2:
            risk_metrics['overall_risk'] = 'medium'
        else:
            risk_metrics['overall_risk'] = 'low'
        
        return risk_metrics

    async def _get_stock_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
            
            # 标准化列名
            data.columns = [col.lower() for col in data.columns]
            return data
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None

    def get_strategy_summary(self) -> Dict[str, Any]:
        """获取策略配置摘要"""
        return {
            'active_strategies': self.strategy_factory.get_core_strategies(),
            'combined_strategy_config': self.combined_strategy.get_strategy_summary(),
            'analysis_capabilities': [
                '持仓盈亏分析',
                '策略信号分析', 
                '风险管理建议',
                '止损止盈计算',
                '仓位建议'
            ]
        } 