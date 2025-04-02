import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
from scipy import stats

from strategy.strategy_factory import StrategyFactory
from strategy_optimizer.data_processors.feature_importance import FeatureImportanceAnalyzer

logger = logging.getLogger(__name__)

class SignalExtractor:
    """
    信号提取器
    
    从特征中提取交易信号，支持多种信号提取方式和优化方法。
    """
    
    def __init__(
        self,
        standardize: bool = True,
        smooth_window: Optional[int] = None,
        threshold_method: str = "percentile",
        threshold_params: Dict[str, Any] = None,
    ):
        """
        初始化信号提取器
        
        参数:
            standardize: 是否对信号进行标准化
            smooth_window: 平滑窗口大小，None表示不平滑
            threshold_method: 阈值方法，可选"percentile"、"std"、"fixed"
            threshold_params: 阈值参数，根据method不同而不同
                - "percentile": {"upper": 80, "lower": 20} 上下分位数
                - "std": {"n_std": 1.0} 标准差倍数
                - "fixed": {"upper": 0.5, "lower": -0.5} 固定阈值
        """
        self.standardize = standardize
        self.smooth_window = smooth_window
        self.threshold_method = threshold_method
        
        # 设置默认阈值参数
        default_params = {
            "percentile": {"upper": 80, "lower": 20},
            "std": {"n_std": 1.0},
            "fixed": {"upper": 0.5, "lower": -0.5}
        }
        
        if threshold_params is None:
            self.threshold_params = default_params.get(threshold_method, {})
        else:
            self.threshold_params = threshold_params
            
        self.scaler = StandardScaler() if standardize else None
        self.feature_importance_analyzer = None
        self.logger = logging.getLogger(__name__)
        
    def extract_from_feature(
        self, 
        feature_data: pd.DataFrame, 
        feature_name: str,
        return_processed: bool = False
    ) -> Tuple[pd.Series, Optional[pd.Series]]:
        """
        从单个特征中提取信号
        
        参数:
            feature_data: 特征数据DataFrame
            feature_name: 特征列名
            return_processed: 是否返回处理后的特征数据
            
        返回:
            信号序列，可选处理后的特征序列
        """
        # 获取特征数据
        feature = feature_data[feature_name].copy()
        
        # 处理缺失值 - 使用更新的方法
        feature = feature.ffill().bfill()
        if feature.isna().any():
            self.logger.warning(f"特征 {feature_name} 存在无法填充的缺失值，将使用0填充")
            feature = feature.fillna(0)
            
        # 标准化
        if self.standardize:
            feature_values = feature.values.reshape(-1, 1)
            feature_standardized = pd.Series(
                self.scaler.fit_transform(feature_values).flatten(),
                index=feature.index
            )
        else:
            feature_standardized = feature
            
        # 平滑处理
        if self.smooth_window is not None and self.smooth_window > 1:
            feature_processed = feature_standardized.rolling(
                window=self.smooth_window, center=False
            ).mean()
            # 填充开始的NaN值 - 使用更新的方法
            feature_processed = feature_processed.bfill()
        else:
            feature_processed = feature_standardized
            
        # 根据阈值生成信号
        signal = self._generate_signal_by_threshold(feature_processed)
        
        if return_processed:
            return signal, feature_processed
        else:
            return signal, None
        
    def extract_from_multiple_features(
        self, 
        feature_data: pd.DataFrame, 
        feature_names: List[str], 
        method: str = "average",
        weights: Optional[List[float]] = None
    ) -> pd.Series:
        """
        从多个特征中提取信号
        
        参数:
            feature_data: 特征数据DataFrame
            feature_names: 特征列名列表
            method: 合并方法，"average"、"weighted"、"vote"或"top_n"
            weights: 权重列表，仅当method为"weighted"时使用
            
        返回:
            合并后的信号序列
        """
        if not feature_names:
            raise ValueError("特征列表不能为空")
            
        # 提取每个特征的信号
        signals = {}
        processed_features = {}
        
        for feature in feature_names:
            signal, processed = self.extract_from_feature(
                feature_data, feature, return_processed=True
            )
            signals[feature] = signal
            processed_features[feature] = processed
            
        # 合并信号
        if method == "average":
            # 简单平均
            combined_signal = pd.DataFrame(signals).mean(axis=1)
            
        elif method == "weighted":
            # 加权平均
            if weights is None or len(weights) != len(feature_names):
                raise ValueError("weights必须与feature_names长度相同")
                
            # 归一化权重
            weights_normalized = np.array(weights) / np.sum(weights)
            
            # 加权平均
            combined_signal = pd.Series(0, index=feature_data.index)
            for i, feature in enumerate(feature_names):
                combined_signal += signals[feature] * weights_normalized[i]
                
        elif method == "vote":
            # 投票法
            combined_signal = pd.DataFrame(signals).apply(
                lambda row: 1 if (row > 0).sum() > (row < 0).sum() else 
                           (-1 if (row < 0).sum() > (row > 0).sum() else 0),
                axis=1
            )
            
        elif method == "top_n":
            # 使用前N个特征
            if weights is None or len(weights) != len(feature_names):
                raise ValueError("使用top_n方法时，weights参数必须提供特征重要性排名")
                
            # 获取重要性排序
            importance_order = np.argsort(weights)[::-1]
            top_feature_idx = importance_order[0]
            
            # 使用最重要的特征
            combined_signal = signals[feature_names[top_feature_idx]]
            
        else:
            raise ValueError(f"不支持的合并方法: {method}")
            
        return combined_signal
        
    def optimize_signal_weights(
        self, 
        feature_data: pd.DataFrame, 
        feature_names: List[str],
        target: pd.Series,
        method: str = "correlation",
        cv: int = 5
    ) -> List[float]:
        """
        优化信号权重
        
        参数:
            feature_data: 特征数据DataFrame
            feature_names: 特征列名列表
            target: 目标变量（如收益率）
            method: 权重优化方法，"correlation"或"importance"
            cv: 交叉验证折数
            
        返回:
            优化后的权重列表
        """
        if method == "correlation":
            # 使用特征与目标的相关性作为权重
            weights = []
            
            for feature_name in feature_names:
                # 提取处理后的特征
                _, processed_feature = self.extract_from_feature(
                    feature_data, feature_name, return_processed=True
                )
                
                # 计算与目标的相关性
                corr = processed_feature.corr(target)
                weights.append(abs(corr))  # 使用相关性绝对值
                
        elif method == "importance":
            # 使用特征重要性作为权重
            if self.feature_importance_analyzer is None:
                self.feature_importance_analyzer = FeatureImportanceAnalyzer()
                
            # 计算特征重要性
            self.feature_importance_analyzer.fit(
                feature_data[feature_names], target, cv=cv
            )
            
            # 获取特征重要性
            importance_df = self.feature_importance_analyzer.importances['summary']
            
            # 将特征名称与重要性匹配
            weights = []
            for feature_name in feature_names:
                importance = importance_df[
                    importance_df['feature'] == feature_name
                ]['importance_mean'].values
                
                if len(importance) > 0:
                    weights.append(importance[0])
                else:
                    self.logger.warning(f"特征 {feature_name} 未找到重要性信息，使用默认值1.0")
                    weights.append(1.0)
                    
        else:
            raise ValueError(f"不支持的权重优化方法: {method}")
            
        # 归一化权重
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
            
        return weights.tolist()
        
    def analyze_signal_performance(
        self, 
        signal: pd.Series, 
        price_data: pd.Series,
        position_mode: str = "discrete",
        transaction_cost: float = 0.0,
        plot: bool = False
    ) -> Dict[str, Any]:
        """
        分析信号性能
        
        参数:
            signal: 信号序列
            price_data: 价格序列
            position_mode: 持仓模式，"discrete"或"continuous"
            transaction_cost: 交易成本比例
            plot: 是否绘制性能图表
            
        返回:
            性能指标字典
        """
        # 检查数据对齐
        signal = signal.reindex(price_data.index)
        
        # 计算收益率
        returns = price_data.pct_change().fillna(0)
        
        # 根据持仓模式计算策略收益
        if position_mode == "discrete":
            # 离散信号(1,0,-1)
            positions = signal.shift(1).fillna(0)
            strategy_returns = positions * returns
            
            # 计算换手率
            position_changes = positions.diff().abs()
            turnover = position_changes.mean()
            
        elif position_mode == "continuous":
            # 连续信号(浮点数)
            positions = signal.shift(1).fillna(0)
            strategy_returns = positions * returns
            
            # 计算换手率
            position_changes = positions.diff().abs()
            turnover = position_changes.mean()
            
        else:
            raise ValueError(f"不支持的持仓模式: {position_mode}")
            
        # 计算交易成本
        if transaction_cost > 0:
            # 简化模型：每次仓位变化都收取成本
            transaction_costs = position_changes * transaction_cost
            strategy_returns = strategy_returns - transaction_costs
            
        # 计算性能指标
        cumulative_returns = (1 + strategy_returns).cumprod()
        benchmark_cum_returns = (1 + returns).cumprod()
        
        # 年化收益率
        n_trading_days = 252  # 假设一年有252个交易日
        n_days = len(signal)
        n_years = n_days / n_trading_days
        
        annual_return = (cumulative_returns.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
        
        # 计算最大回撤
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 计算夏普比率
        sharpe_ratio = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(n_trading_days)) if strategy_returns.std() > 0 else 0
        
        # 计算信息比率
        excess_returns = strategy_returns - returns
        information_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(n_trading_days)) if excess_returns.std() > 0 else 0
        
        # 计算胜率
        win_days = (strategy_returns > 0).sum()
        total_days = (strategy_returns != 0).sum()
        win_rate = win_days / total_days if total_days > 0 else 0
        
        # 绘制性能图表
        if plot:
            plt.figure(figsize=(12, 10))
            
            # 绘制累积收益
            plt.subplot(2, 2, 1)
            cumulative_returns.plot(label='策略')
            benchmark_cum_returns.plot(label='基准')
            plt.title('累积收益')
            plt.legend()
            plt.grid(True)
            
            # 绘制回撤
            plt.subplot(2, 2, 2)
            drawdown.plot()
            plt.title('回撤')
            plt.grid(True)
            
            # 绘制信号和价格
            plt.subplot(2, 2, 3)
            ax1 = plt.gca()
            ax1.plot(price_data, 'b-', label='价格')
            ax2 = ax1.twinx()
            ax2.plot(signal, 'r-', label='信号')
            plt.title('信号和价格')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)
            plt.grid(True)
            
            # 绘制每日收益分布
            plt.subplot(2, 2, 4)
            sns.histplot(strategy_returns, kde=True)
            plt.title('收益分布')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        # 返回性能指标
        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'turnover': turnover,
            'cumulative_return': cumulative_returns.iloc[-1] - 1,
            'volatility': strategy_returns.std() * np.sqrt(n_trading_days),
            'total_days': n_days,
            'benchmark_return': benchmark_cum_returns.iloc[-1] - 1,
            'daily_returns': strategy_returns,
            'cumulative_returns': cumulative_returns
        }
        
    def _generate_signal_by_threshold(self, feature: pd.Series) -> pd.Series:
        """
        根据阈值生成信号
        
        参数:
            feature: 特征序列
            
        返回:
            信号序列
        """
        method = self.threshold_method
        
        if method == "percentile":
            upper_pct = self.threshold_params.get("upper", 80)
            lower_pct = self.threshold_params.get("lower", 20)
            
            upper_threshold = np.percentile(feature, upper_pct)
            lower_threshold = np.percentile(feature, lower_pct)
            
        elif method == "std":
            n_std = self.threshold_params.get("n_std", 1.0)
            mean = feature.mean()
            std = feature.std()
            
            upper_threshold = mean + n_std * std
            lower_threshold = mean - n_std * std
            
        elif method == "fixed":
            upper_threshold = self.threshold_params.get("upper", 0.5)
            lower_threshold = self.threshold_params.get("lower", -0.5)
            
        else:
            raise ValueError(f"不支持的阈值方法: {method}")
            
        # 生成信号
        signal = pd.Series(0, index=feature.index)
        signal[feature > upper_threshold] = 1
        signal[feature < lower_threshold] = -1
        
        return signal
    
    def analyze_signals_by_market_regime(
        self, 
        signals: pd.DataFrame, 
        market_regimes: pd.Series,
        target_returns: pd.Series
    ) -> pd.DataFrame:
        """
        按市场环境分析各信号的表现
        
        参数:
            signals: 包含信号数据的DataFrame
            market_regimes: 市场环境Series，索引与signals一致
            target_returns: 目标收益率Series，索引与signals一致
            
        返回:
            按市场环境分组的信号表现DataFrame
        """
        results = []
        
        # 确保数据对齐
        aligned_data = pd.concat([signals, market_regimes, target_returns], axis=1)
        aligned_data.columns = list(signals.columns) + ['market_regime', 'returns']
        
        # 按市场环境分组
        for regime in aligned_data['market_regime'].unique():
            regime_data = aligned_data[aligned_data['market_regime'] == regime]
            
            if len(regime_data) < 10:  # 至少需要10个观测值
                logger.warning(f"市场环境 {regime} 的样本数量不足: {len(regime_data)}")
                continue
                
            regime_results = {'market_regime': regime, 'sample_count': len(regime_data)}
            
            # 计算每个信号与收益率的相关性
            for col in signals.columns:
                signal_values = regime_data[col].values
                return_values = regime_data['returns'].values
                
                # 去除NaN值
                mask = ~(np.isnan(signal_values) | np.isnan(return_values))
                signal_values = signal_values[mask]
                return_values = return_values[mask]
                
                if len(signal_values) < 10:
                    continue
                    
                # 计算相关性和p值
                correlation, p_value = pearsonr(signal_values, return_values)
                regime_results[f'{col}_corr'] = correlation
                regime_results[f'{col}_pvalue'] = p_value
                
            results.append(regime_results)
            
        return pd.DataFrame(results)
    
    def get_signal_performance_report(self) -> pd.DataFrame:
        """
        获取所有信号的性能报告
        
        返回:
            信号性能报告DataFrame
        """
        # 这个方法需要在具体的回测环境中实现
        # 作为占位符，我们返回一个空的DataFrame
        return pd.DataFrame(columns=[
            'signal_name', 'strategy', 'type', 'time_scale',
            'avg_return', 'sharpe', 'win_rate', 'correlation'
        ])
        
    def plot_signal_components(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        strategy_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        绘制策略的信号组件和价格图表
        
        参数:
            data: 价格数据DataFrame
            signals: 信号数据DataFrame
            strategy_name: 策略名称
            start_date: 开始日期，可选
            end_date: 结束日期，可选
            figsize: 图表大小
            
        返回:
            图表对象
        """
        # 过滤信号
        strategy_signals = {
            col: signals[col] for col in signals.columns 
            if col.startswith(f"{strategy_name}_")
        }
        
        if not strategy_signals:
            logger.warning(f"没有找到策略 {strategy_name} 的信号")
            return None
            
        # 准备数据
        plot_data = data.copy()
        for name, signal in strategy_signals.items():
            plot_data[name] = signal
            
        # 应用日期过滤
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
            
        # 绘图
        fig, axs = plt.subplots(len(strategy_signals) + 1, 1, figsize=figsize, sharex=True)
        
        # 绘制价格图
        ax0 = axs[0]
        ax0.plot(plot_data.index, plot_data['close'], label='收盘价')
        ax0.set_title(f"{strategy_name} 策略 - 价格和信号组件")
        ax0.legend()
        ax0.grid(True)
        
        # 绘制信号组件
        for i, (name, signal) in enumerate(strategy_signals.items(), 1):
            short_name = name.replace(f"{strategy_name}_", "")
            axs[i].plot(plot_data.index, plot_data[name], label=short_name, color='blue')
            axs[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axs[i].axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
            axs[i].axhline(y=-0.5, color='red', linestyle='--', alpha=0.3)
            axs[i].set_ylim(-1.1, 1.1)
            axs[i].set_ylabel(short_name)
            axs[i].grid(True)
            
        plt.tight_layout()
        return fig 