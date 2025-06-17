"""
风险优化器
实现仓位风险约束、行业分散化、相关性控制等风险管理功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskOptimizer:
    """风险优化器"""
    
    def __init__(self, risk_config: Dict = None):
        """
        初始化风险优化器
        
        Args:
            risk_config: 风险配置参数
        """
        self.risk_config = risk_config or {
            'max_single_position': 0.20,      # 单一股票最大权重
            'max_sector_exposure': 0.35,      # 单一行业最大权重
            'max_correlation_weight': 0.40,   # 高相关股票总权重限制
            'correlation_threshold': 0.7,     # 高相关判断阈值
            'max_volatility': 0.25,          # 组合最大波动率
            'min_diversification': 5,        # 最少持仓数量
            'lookback_period': 252,          # 风险计算回看期
            'rebalance_threshold': 0.05      # 再平衡阈值
        }
        
        # 行业映射
        self.sector_mapping = {
            'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'NVDA': 'Technology', 'AMD': 'Technology', 'ADBE': 'Technology',
            'TSLA': 'Automotive', 'F': 'Automotive', 'GM': 'Automotive',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'EOG': 'Energy',
            'KO': 'Consumer', 'PG': 'Consumer', 'WMT': 'Consumer',
            'HD': 'Retail', 'AMZN': 'Retail', 'TGT': 'Retail',
            'PHM': 'Real Estate', 'CF': 'Materials'
        }
        
        self.correlation_cache = {}
        self.volatility_cache = {}
        
    def calculate_portfolio_risk(self, weights: Dict[str, float], 
                               price_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        计算投资组合风险指标
        
        Args:
            weights: 股票权重字典
            price_data: 价格数据字典
            
        Returns:
            风险指标字典
        """
        try:
            # 获取价格数据
            if price_data is None:
                price_data = self._fetch_price_data(list(weights.keys()))
                
            # 计算收益率矩阵
            returns_data = {}
            for symbol, data in price_data.items():
                if symbol in weights and not data.empty:
                    returns_data[symbol] = data['Close'].pct_change().dropna()
                    
            # 创建收益率矩阵
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                logger.warning("没有有效的收益率数据")
                return {}
                
            # 计算协方差矩阵
            cov_estimator = LedoitWolf()
            cov_matrix = cov_estimator.fit(returns_df).covariance_
            cov_df = pd.DataFrame(cov_matrix, index=returns_df.columns, 
                                columns=returns_df.columns)
            
            # 对齐权重和协方差矩阵
            common_assets = list(set(weights.keys()) & set(cov_df.index))
            weight_array = np.array([weights[asset] for asset in common_assets])
            cov_aligned = cov_df.loc[common_assets, common_assets].values
            
            # 计算投资组合风险
            portfolio_variance = np.dot(weight_array.T, np.dot(cov_aligned, weight_array))
            portfolio_volatility = np.sqrt(portfolio_variance * 252)  # 年化波动率
            
            # 计算其他风险指标
            risk_metrics = {
                'portfolio_volatility': portfolio_volatility,
                'portfolio_variance': portfolio_variance,
                'max_drawdown': self._calculate_max_drawdown(returns_df, weights),
                'var_95': self._calculate_var(returns_df, weights, 0.05),
                'correlation_matrix': returns_df.corr().to_dict(),
                'diversification_ratio': self._calculate_diversification_ratio(
                    weight_array, cov_aligned, returns_df[common_assets].std().values
                )
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"计算投资组合风险失败: {e}")
            return {}
    
    def check_risk_constraints(self, weights: Dict[str, float]) -> Dict:
        """
        检查风险约束违规情况
        
        Args:
            weights: 股票权重字典
            
        Returns:
            违规检查结果
        """
        violations = []
        recommendations = []
        
        try:
            # 检查单一持仓限制
            for symbol, weight in weights.items():
                if weight > self.risk_config['max_single_position']:
                    violations.append(f"{symbol}权重{weight:.1%}超过限制{self.risk_config['max_single_position']:.1%}")
                    recommendations.append(f"建议将{symbol}权重降至{self.risk_config['max_single_position']:.1%}以下")
            
            # 检查行业集中度
            sector_weights = self._calculate_sector_weights(weights)
            for sector, weight in sector_weights.items():
                if weight > self.risk_config['max_sector_exposure']:
                    violations.append(f"{sector}行业权重{weight:.1%}超过限制{self.risk_config['max_sector_exposure']:.1%}")
                    recommendations.append(f"建议降低{sector}行业配置或增加其他行业股票")
            
            # 检查相关性风险
            high_corr_risk = self._check_correlation_risk(weights)
            if high_corr_risk['violation']:
                violations.append(high_corr_risk['message'])
                recommendations.append(high_corr_risk['recommendation'])
            
            # 检查分散化程度
            if len(weights) < self.risk_config['min_diversification']:
                violations.append(f"持仓数量{len(weights)}低于最低要求{self.risk_config['min_diversification']}")
                recommendations.append("建议增加持仓数量以提高分散化程度")
            
            return {
                'violations': violations,
                'recommendations': recommendations,
                'risk_score': self._calculate_risk_score(violations),
                'sector_weights': sector_weights
            }
            
        except Exception as e:
            logger.error(f"检查风险约束失败: {e}")
            return {'violations': [], 'recommendations': [], 'risk_score': 0.5}
    
    def optimize_portfolio_weights(self, current_weights: Dict[str, float],
                                 target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        在风险约束下优化投资组合权重
        
        Args:
            current_weights: 当前权重
            target_weights: 目标权重
            
        Returns:
            优化后的权重
        """
        try:
            # 合并所有股票
            all_symbols = list(set(current_weights.keys()) | set(target_weights.keys()))
            
            # 设置初始权重
            x0 = np.array([current_weights.get(symbol, 0) for symbol in all_symbols])
            
            # 设置目标权重
            target_array = np.array([target_weights.get(symbol, 0) for symbol in all_symbols])
            
            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 权重和为1
            ]
            
            # 边界条件
            bounds = []
            for i, symbol in enumerate(all_symbols):
                max_weight = min(self.risk_config['max_single_position'], 
                               target_array[i] + 0.05)  # 允许小幅超配
                bounds.append((0, max_weight))
            
            # 目标函数：最小化与目标权重的偏差，同时考虑风险
            def objective(x):
                # 权重偏差惩罚
                weight_penalty = np.sum((x - target_array) ** 2)
                
                # 行业集中度惩罚
                sector_penalty = self._calculate_sector_penalty(x, all_symbols)
                
                # 单一持仓惩罚
                position_penalty = np.sum(np.maximum(x - self.risk_config['max_single_position'], 0) ** 2)
                
                return weight_penalty + sector_penalty * 2 + position_penalty * 5
            
            # 优化
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = {symbol: weight for symbol, weight in 
                                   zip(all_symbols, result.x) if weight > 0.001}
                
                # 再次归一化
                total_weight = sum(optimized_weights.values())
                optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
                
                logger.info("权重优化成功")
                return optimized_weights
            else:
                logger.warning("权重优化失败，返回目标权重")
                return target_weights
                
        except Exception as e:
            logger.error(f"优化投资组合权重失败: {e}")
            return target_weights
    
    def suggest_rebalancing(self, current_weights: Dict[str, float],
                          target_weights: Dict[str, float]) -> Dict:
        """
        建议再平衡操作
        
        Args:
            current_weights: 当前权重
            target_weights: 目标权重
            
        Returns:
            再平衡建议
        """
        try:
            suggestions = []
            weight_changes = {}
            
            # 计算权重变化
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())
            for symbol in all_symbols:
                current = current_weights.get(symbol, 0)
                target = target_weights.get(symbol, 0)
                change = target - current
                
                if abs(change) > self.risk_config['rebalance_threshold']:
                    weight_changes[symbol] = {
                        'current': current,
                        'target': target,
                        'change': change,
                        'action': 'buy' if change > 0 else 'sell'
                    }
            
            # 生成建议
            for symbol, change_info in weight_changes.items():
                if change_info['action'] == 'buy':
                    suggestions.append(f"增持{symbol}: {change_info['current']:.1%} → {change_info['target']:.1%}")
                else:
                    suggestions.append(f"减持{symbol}: {change_info['current']:.1%} → {change_info['target']:.1%}")
            
            # 检查再平衡后的风险
            post_rebalance_risk = self.check_risk_constraints(target_weights)
            
            return {
                'suggestions': suggestions,
                'weight_changes': weight_changes,
                'requires_rebalancing': len(weight_changes) > 0,
                'post_rebalance_risk': post_rebalance_risk
            }
            
        except Exception as e:
            logger.error(f"生成再平衡建议失败: {e}")
            return {'suggestions': [], 'weight_changes': {}, 'requires_rebalancing': False}
    
    def _calculate_sector_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """计算行业权重"""
        sector_weights = {}
        
        for symbol, weight in weights.items():
            sector = self.sector_mapping.get(symbol, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
        return sector_weights
    
    def _calculate_sector_penalty(self, weights: np.ndarray, symbols: List[str]) -> float:
        """计算行业集中度惩罚"""
        sector_weights = {}
        
        for i, symbol in enumerate(symbols):
            sector = self.sector_mapping.get(symbol, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]
        
        penalty = 0
        for sector, weight in sector_weights.items():
            if weight > self.risk_config['max_sector_exposure']:
                penalty += (weight - self.risk_config['max_sector_exposure']) ** 2
                
        return penalty
    
    def _check_correlation_risk(self, weights: Dict[str, float]) -> Dict:
        """检查相关性风险"""
        try:
            # 获取相关性数据
            symbols = list(weights.keys())
            correlation_matrix = self._get_correlation_matrix(symbols)
            
            if correlation_matrix is None:
                return {'violation': False, 'message': '', 'recommendation': ''}
            
            # 检查高相关股票权重
            high_corr_total = 0
            high_corr_pairs = []
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    corr = correlation_matrix.loc[symbol1, symbol2]
                    if abs(corr) > self.risk_config['correlation_threshold']:
                        pair_weight = weights[symbol1] + weights[symbol2]
                        high_corr_total += pair_weight
                        high_corr_pairs.append((symbol1, symbol2, corr, pair_weight))
            
            if high_corr_total > self.risk_config['max_correlation_weight']:
                return {
                    'violation': True,
                    'message': f"高相关股票总权重{high_corr_total:.1%}超过限制{self.risk_config['max_correlation_weight']:.1%}",
                    'recommendation': f"建议降低相关性高的股票配置: {[pair[:2] for pair in high_corr_pairs]}",
                    'high_corr_pairs': high_corr_pairs
                }
            
            return {'violation': False, 'message': '', 'recommendation': ''}
            
        except Exception as e:
            logger.error(f"检查相关性风险失败: {e}")
            return {'violation': False, 'message': '', 'recommendation': ''}
    
    def _get_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """获取相关性矩阵"""
        try:
            # 检查缓存
            cache_key = tuple(sorted(symbols))
            if cache_key in self.correlation_cache:
                return self.correlation_cache[cache_key]
            
            # 获取价格数据
            price_data = self._fetch_price_data(symbols)
            
            # 计算相关性
            returns_data = {}
            for symbol, data in price_data.items():
                if not data.empty:
                    returns_data[symbol] = data['Close'].pct_change().dropna()
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                return None
                
            correlation_matrix = returns_df.corr()
            
            # 缓存结果
            self.correlation_cache[cache_key] = correlation_matrix
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"计算相关性矩阵失败: {e}")
            return None
    
    def _fetch_price_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """获取价格数据"""
        price_data = {}
        
        for symbol in symbols:
            try:
                data = yf.download(symbol, period=period, interval="1d")
                if not data.empty:
                    price_data[symbol] = data
            except Exception as e:
                logger.error(f"获取{symbol}价格数据失败: {e}")
                
        return price_data
    
    def _calculate_max_drawdown(self, returns_df: pd.DataFrame, weights: Dict[str, float]) -> float:
        """计算最大回撤"""
        try:
            # 计算组合收益率
            common_assets = list(set(weights.keys()) & set(returns_df.columns))
            portfolio_returns = sum(returns_df[asset] * weights[asset] for asset in common_assets)
            
            # 计算累计收益率
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # 计算回撤
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            
            return abs(drawdown.min())
            
        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            return 0.0
    
    def _calculate_var(self, returns_df: pd.DataFrame, weights: Dict[str, float], alpha: float) -> float:
        """计算风险价值VaR"""
        try:
            # 计算组合收益率
            common_assets = list(set(weights.keys()) & set(returns_df.columns))
            portfolio_returns = sum(returns_df[asset] * weights[asset] for asset in common_assets)
            
            # 计算VaR
            var = np.percentile(portfolio_returns, alpha * 100)
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return 0.0
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray, 
                                       individual_vols: np.ndarray) -> float:
        """计算分散化比率"""
        try:
            # 加权平均波动率
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            # 组合波动率
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 分散化比率
            if portfolio_vol > 0:
                return weighted_avg_vol / portfolio_vol
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"计算分散化比率失败: {e}")
            return 1.0
    
    def _calculate_risk_score(self, violations: List[str]) -> float:
        """计算风险评分"""
        if not violations:
            return 1.0  # 无违规，最高分
        elif len(violations) <= 2:
            return 0.7  # 轻微违规
        elif len(violations) <= 4:
            return 0.4  # 中等违规
        else:
            return 0.1  # 严重违规


if __name__ == "__main__":
    # 测试风险优化器
    optimizer = RiskOptimizer()
    
    # 示例权重
    test_weights = {
        'AAPL': 0.15, 'GOOGL': 0.20, 'NVDA': 0.18, 'AMD': 0.22,
        'TSLA': 0.10, 'PFE': 0.08, 'EOG': 0.07
    }
    
    print("检查风险约束...")
    risk_check = optimizer.check_risk_constraints(test_weights)
    
    print("风险检查结果:")
    print(f"风险评分: {risk_check['risk_score']:.2f}")
    print("违规项目:", risk_check['violations'])
    print("建议:", risk_check['recommendations'])
    print("行业权重:", risk_check['sector_weights']) 