import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 导入我们的核心策略
from strategy.strategy_factory import StrategyFactory
from strategy.combined_strategy import CombinedStrategy
from strategy.niuniu_strategy_v3 import NiuniuStrategyV3
from strategy.tdi_strategy import TDIStrategy
from strategy.cpgw_strategy import CPGWStrategy

logger = logging.getLogger(__name__)

@dataclass
class ScreeningResult:
    """筛选结果数据类"""
    symbol: str
    current_price: float
    strategy_score: float
    individual_scores: Dict[str, float]
    recommendation: str
    confidence: float
    position_size: float
    technical_indicators: Dict[str, float]
    risk_metrics: Dict[str, float]
    reasons: List[str]

class EnhancedStockScreener:
    """
    增强的股票筛选器
    集成了优化的策略系统，提供更准确的股票筛选
    """

    def __init__(self, stock_universe: List[str] = None):
        """
        初始化增强股票筛选器
        
        Args:
            stock_universe: 股票池，如果为None则使用默认股票池
        """
        self.strategy_factory = StrategyFactory()
        
        # 初始化核心策略
        self.combined_strategy = self.strategy_factory.create_combined_strategy()
        self.niuniu_strategy = NiuniuStrategyV3()
        self.tdi_strategy = TDIStrategy()
        self.cpgw_strategy = CPGWStrategy()
        
        # 设置股票池
        self.stock_universe = stock_universe or self._get_default_stock_universe()
        
        # 筛选标准
        self.screening_criteria = {
            'min_signal_strength': 0.6,
            'min_strategy_score': 0.7,
            'max_volatility': 0.5,
            'min_liquidity': 1000000,  # 最小日均交易量
            'min_price': 5.0,          # 最小价格
            'max_price': 1000.0,       # 最大价格
            'consensus_required': 2     # 至少需要2个策略同意
        }
        
        logger.info(f"✅ 增强股票筛选器初始化完成，股票池包含 {len(self.stock_universe)} 只股票")

    def _get_default_stock_universe(self) -> List[str]:
        """获取默认股票池"""
        # 包含一些流行的美股和中国概念股
        return [
            # 科技股
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CRM', 'NFLX',
            # 芯片股
            'AMD', 'INTC', 'QCOM', 'MU', 'AVGO', 'TXN', 'AMAT', 'LRCX',
            # 中国概念股
            'BABA', 'JD', 'PDD', 'BILI', 'NIO', 'XPEV', 'LI', 'BIDU',
            # 传统行业
            'JPM', 'BAC', 'KO', 'PEP', 'JNJ', 'PFE', 'DIS', 'WMT',
            # 能源和原材料
            'XOM', 'CVX', 'GLD', 'SLV'
        ]

    async def screen_stocks(self, max_results: int = 20, 
                           filter_criteria: Dict[str, Any] = None) -> List[ScreeningResult]:
        """
        筛选股票
        
        Args:
            max_results: 最大结果数量
            filter_criteria: 额外的筛选条件
            
        Returns:
            筛选结果列表，按策略评分排序
        """
        logger.info(f"🔍 开始筛选 {len(self.stock_universe)} 只股票...")
        
        # 合并筛选条件
        criteria = self.screening_criteria.copy()
        if filter_criteria:
            criteria.update(filter_criteria)
        
        # 并行分析股票
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for symbol in self.stock_universe:
                future = executor.submit(self._analyze_stock_for_screening, symbol, criteria)
                futures.append((symbol, future))
            
            # 收集结果
            for symbol, future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                        logger.debug(f"✅ 成功分析 {symbol}")
                except Exception as e:
                    logger.warning(f"⚠️ 分析 {symbol} 失败: {e}")
        
        # 按策略评分排序
        results.sort(key=lambda x: x.strategy_score, reverse=True)
        
        # 应用额外过滤
        filtered_results = self._apply_advanced_filters(results, criteria)
        
        logger.info(f"✅ 筛选完成，找到 {len(filtered_results)} 只符合条件的股票")
        return filtered_results[:max_results]

    def _analyze_stock_for_screening(self, symbol: str, criteria: Dict[str, Any]) -> Optional[ScreeningResult]:
        """分析单只股票用于筛选"""
        try:
            # 获取股票数据
            data = self._get_stock_data_sync(symbol, days=60)
            if data is None or len(data) < 30:
                return None
            
            current_price = data['close'].iloc[-1]
            
            # 基本筛选条件
            if not self._passes_basic_screening(data, current_price, criteria):
                return None
            
            # 使用策略分析
            strategy_signals = self.combined_strategy.generate_signals(data)
            current_signal = strategy_signals['signal'].iloc[-1] if not strategy_signals['signal'].empty else 0
            signal_strength = strategy_signals['signal_strength'].iloc[-1] if 'signal_strength' in strategy_signals.columns else 0
            
            # 获取各策略的单独信号
            niuniu_signals = self.niuniu_strategy.generate_signals(data)
            tdi_signals = self.tdi_strategy.generate_signals(data)
            cpgw_signals = self.cpgw_strategy.generate_signals(data)
            
            # 计算个别策略评分
            individual_scores = {
                'niuniu': float(niuniu_signals['signal'].iloc[-1]) if 'signal' in niuniu_signals.columns else 0,
                'tdi': float(tdi_signals['signal'].iloc[-1]) if 'signal' in tdi_signals.columns else 0,
                'cpgw': float(cpgw_signals['signal'].iloc[-1]) if 'signal' in cpgw_signals.columns else 0
            }
            
            # 计算综合策略评分
            positive_signals = sum(1 for score in individual_scores.values() if score > 0)
            strategy_score = signal_strength if positive_signals >= criteria['consensus_required'] else 0
            
            # 如果策略评分不够，跳过
            if strategy_score < criteria['min_strategy_score']:
                return None
            
            # 计算技术指标
            technical_indicators = self._calculate_technical_indicators(data)
            
            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(data)
            
            # 生成推荐和原因
            recommendation, confidence, reasons = self._generate_recommendation(
                current_signal, signal_strength, individual_scores, technical_indicators
            )
            
            # 计算建议仓位大小
            position_size = self.combined_strategy.get_position_size(data, int(current_signal))
            
            return ScreeningResult(
                symbol=symbol,
                current_price=current_price,
                strategy_score=strategy_score,
                individual_scores=individual_scores,
                recommendation=recommendation,
                confidence=confidence,
                position_size=position_size,
                technical_indicators=technical_indicators,
                risk_metrics=risk_metrics,
                reasons=reasons
            )
            
        except Exception as e:
            logger.debug(f"分析股票 {symbol} 时出错: {e}")
            return None

    def _passes_basic_screening(self, data: pd.DataFrame, current_price: float, 
                               criteria: Dict[str, Any]) -> bool:
        """基本筛选条件检查"""
        try:
            # 价格筛选
            if current_price < criteria['min_price'] or current_price > criteria['max_price']:
                return False
            
            # 流动性筛选
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            if avg_volume < criteria['min_liquidity']:
                return False
            
            # 波动率筛选
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            if volatility > criteria['max_volatility']:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"基本筛选检查出错: {e}")
            return False

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标"""
        try:
            indicators = {}
            
            # RSI
            if 'RSI' in data.columns:
                indicators['rsi'] = float(data['RSI'].iloc[-1])
            
            # MACD
            if 'MACD' in data.columns:
                indicators['macd'] = float(data['MACD'].iloc[-1])
            
            # 布林带位置
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                bb_position = (data['close'].iloc[-1] - data['BB_lower'].iloc[-1]) / (data['BB_upper'].iloc[-1] - data['BB_lower'].iloc[-1])
                indicators['bb_position'] = float(bb_position)
            
            # 移动平均线趋势
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            indicators['sma20_trend'] = float((sma_20.iloc[-1] - sma_20.iloc[-5]) / sma_20.iloc[-5])
            indicators['sma50_trend'] = float((sma_50.iloc[-1] - sma_50.iloc[-10]) / sma_50.iloc[-10])
            
            # 价格动量
            indicators['price_momentum_5d'] = float((data['close'].iloc[-1] - data['close'].iloc[-6]) / data['close'].iloc[-6])
            indicators['price_momentum_20d'] = float((data['close'].iloc[-1] - data['close'].iloc[-21]) / data['close'].iloc[-21])
            
            return indicators
            
        except Exception as e:
            logger.debug(f"计算技术指标出错: {e}")
            return {}

    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算风险指标"""
        try:
            metrics = {}
            
            # 波动率
            returns = data['close'].pct_change().dropna()
            metrics['volatility'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252))
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            metrics['max_drawdown'] = float(drawdown.min())
            
            # 夏普比率（简化版本）
            if len(returns) > 20:
                excess_returns = returns.mean() * 252  # 年化收益
                volatility_annual = returns.std() * np.sqrt(252)
                metrics['sharpe_ratio'] = float(excess_returns / volatility_annual) if volatility_annual > 0 else 0
            
            # VaR (5% 置信度)
            if len(returns) > 20:
                metrics['var_5pct'] = float(np.percentile(returns, 5))
            
            return metrics
            
        except Exception as e:
            logger.debug(f"计算风险指标出错: {e}")
            return {}

    def _generate_recommendation(self, signal: float, strength: float, 
                                individual_scores: Dict[str, float], 
                                technical_indicators: Dict[str, float]) -> Tuple[str, float, List[str]]:
        """生成推荐和原因"""
        reasons = []
        
        if signal > 0 and strength > 0.8:
            recommendation = "强烈买入"
            confidence = 0.9
            reasons.append(f"策略信号强烈买入 (强度: {strength:.2f})")
        elif signal > 0 and strength > 0.6:
            recommendation = "买入"
            confidence = 0.7
            reasons.append(f"策略信号买入 (强度: {strength:.2f})")
        elif signal > 0 and strength > 0.4:
            recommendation = "考虑买入"
            confidence = 0.5
            reasons.append(f"策略信号偏多 (强度: {strength:.2f})")
        else:
            recommendation = "观望"
            confidence = 0.3
            reasons.append("策略信号不明确")
        
        # 添加策略一致性分析
        positive_strategies = sum(1 for score in individual_scores.values() if score > 0)
        if positive_strategies >= 2:
            reasons.append(f"{positive_strategies}/3 个策略给出买入信号")
            confidence += 0.1
        
        # 添加技术指标支持
        if 'rsi' in technical_indicators:
            rsi = technical_indicators['rsi']
            if rsi < 30:
                reasons.append("RSI显示超卖")
                confidence += 0.05
            elif rsi > 70:
                reasons.append("RSI显示超买，需谨慎")
                confidence -= 0.1
        
        if 'price_momentum_5d' in technical_indicators:
            momentum = technical_indicators['price_momentum_5d']
            if momentum > 0.02:
                reasons.append("短期价格动量良好")
                confidence += 0.05
            elif momentum < -0.02:
                reasons.append("短期价格动量较弱")
                confidence -= 0.05
        
        confidence = min(max(confidence, 0.0), 1.0)  # 确保在0-1范围内
        
        return recommendation, confidence, reasons

    def _apply_advanced_filters(self, results: List[ScreeningResult], 
                               criteria: Dict[str, Any]) -> List[ScreeningResult]:
        """应用高级过滤条件"""
        filtered = []
        
        for result in results:
            # 只保留买入建议
            if result.recommendation not in ["强烈买入", "买入", "考虑买入"]:
                continue
            
            # 信号强度过滤
            if result.strategy_score < criteria['min_signal_strength']:
                continue
            
            # 风险过滤
            if result.risk_metrics.get('volatility', 0) > criteria['max_volatility']:
                continue
            
            filtered.append(result)
        
        return filtered

    def _get_stock_data_sync(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """同步获取股票数据"""
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
            logger.debug(f"获取 {symbol} 数据失败: {e}")
            return None

    def get_screening_report(self, results: List[ScreeningResult]) -> Dict[str, Any]:
        """生成筛选报告"""
        if not results:
            return {'message': '没有找到符合条件的股票'}
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_stocks_analyzed': len(self.stock_universe),
            'stocks_passed_screening': len(results),
            'top_recommendations': [],
            'statistics': {}
        }
        
        # 生成推荐列表
        for result in results[:10]:  # 取前10个
            report['top_recommendations'].append({
                'symbol': result.symbol,
                'price': result.current_price,
                'recommendation': result.recommendation,
                'confidence': result.confidence,
                'strategy_score': result.strategy_score,
                'position_size': result.position_size,
                'reasons': result.reasons
            })
        
        # 统计信息
        recommendations = [r.recommendation for r in results]
        report['statistics'] = {
            'avg_strategy_score': np.mean([r.strategy_score for r in results]),
            'avg_confidence': np.mean([r.confidence for r in results]),
            'recommendation_distribution': {
                '强烈买入': recommendations.count('强烈买入'),
                '买入': recommendations.count('买入'),
                '考虑买入': recommendations.count('考虑买入')
            }
        }
        
        return report

    def update_screening_criteria(self, new_criteria: Dict[str, Any]) -> None:
        """更新筛选标准"""
        self.screening_criteria.update(new_criteria)
        logger.info(f"筛选标准已更新: {new_criteria}")

    def get_strategy_summary(self) -> Dict[str, Any]:
        """获取策略配置摘要"""
        return {
            'active_strategies': self.strategy_factory.get_core_strategies(),
            'combined_strategy_config': self.combined_strategy.get_strategy_summary(),
            'stock_universe_size': len(self.stock_universe),
            'screening_criteria': self.screening_criteria
        } 