import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import logging
import talib
from typing import Dict, Any

class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, strategy, data: pd.DataFrame):
        self.strategy = strategy
        self.data = data
        self.logger = logging.getLogger(__name__)
        
        # 性能指标权重
        self.metric_weights = {
            'sharpe_ratio': 0.4,
            'sortino_ratio': 0.3,
            'total_return': 0.3
        }
        
        # 无风险利率（年化）
        self.risk_free_rate = 0.02
        
        # 参数搜索空间
        self.param_space = {
            'fast_period': (5, 30),  # 快速周期
            'slow_period': (20, 90),  # 慢速周期
            'rsi_period': (3, 14),  # RSI周期
            'macd_fast': (3, 12),  # MACD快线
            'macd_slow': (12, 26),  # MACD慢线
            'macd_signal': (3, 9),  # MACD信号线
            'adx_period': (3, 14),  # ADX周期
            'adx_threshold': (10, 25),  # ADX阈值
            'atr_period': (3, 14),  # ATR周期
            'atr_multiplier': (0.5, 2.0),  # ATR乘数
            'volume_threshold': (0.5, 2.0),  # 成交量阈值
            'signal_threshold': (0.5, 0.8),  # 信号阈值
            'profit_target': (0.15, 0.25),  # 止盈目标
            'stop_loss': (-0.15, -0.05),  # 止损目标
            'trailing_stop': (0.03, 0.1)  # 追踪止损
        }
        
        # 参数类型映射
        self.param_types = {
            'fast_period': int,
            'slow_period': int,
            'rsi_period': int,
            'macd_fast': int,
            'macd_slow': int,
            'macd_signal': int,
            'adx_period': int,
            'adx_threshold': int,
            'atr_period': int,
            'atr_multiplier': float,
            'volume_threshold': float,
            'signal_threshold': float,
            'profit_target': float,
            'stop_loss': float,
            'trailing_stop': float
        }
        
    def calculate_indicators(self, data):
        """计算技术指标"""
        df = data.copy()
        
        # 计算RSI
        df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # 计算MACD
        macd, signal, hist = talib.MACD(df['close'].values, 
                                      fastperiod=12, 
                                      slowperiod=26, 
                                      signalperiod=9)
        df['MACD'] = macd
        df['Signal'] = signal
        
        # 计算ADX
        df['ADX'] = talib.ADX(df['high'].values, 
                             df['low'].values, 
                             df['close'].values, 
                             timeperiod=14)
        
        # 计算ATR
        df['ATR'] = talib.ATR(df['high'].values,
                             df['low'].values,
                             df['close'].values,
                             timeperiod=14)
        
        # 计算成交量比率
        df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # 计算趋势强度
        df['Trend_Strength'] = abs(df['close'] - df['close'].rolling(window=20).mean()) / df['ATR']
        
        # 计算动量强度
        df['Momentum_Strength'] = df['close'].pct_change(periods=10)
        
        # 计算成交量强度
        df['Volume_Strength'] = df['volume'].pct_change(periods=10)
        
        # 计算波动率
        df['Volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        return df
        
    def prepare_features(self):
        """准备训练数据的特征和标签"""
        try:
            # 计算技术指标
            df = self.calculate_indicators(self.data)
            
            # 删除包含NaN的行
            df = df.dropna()
            
            if len(df) < 2:
                raise ValueError("Insufficient data after calculating indicators")
            
            # 准备特征
            features = df[['RSI', 'ADX', 'MACD', 'Signal', 'Volume_Ratio',
                         'Trend_Strength', 'Momentum_Strength', 'Volume_Strength',
                         'Volatility', 'ATR']].values[:-1]  # 排除最后一行用于计算未来收益
            
            # 计算未来收益作为标签
            future_returns = df['close'].pct_change().shift(-1).values[:-1]
            labels = (future_returns > 0).astype(int)
            
            if len(features) == 0:
                raise ValueError("No valid features were generated")
                
            return features, labels
            
        except Exception as e:
            self.logger.error(f"Error in prepare_features: {str(e)}")
            raise
        
    def train_xgboost(self):
        """训练XGBoost模型"""
        X, y = self.prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # 训练XGBoost模型，增加模型复杂度
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,  # 增加树的数量
            learning_rate=0.05,  # 降低学习率
            max_depth=8,  # 增加树的深度
            min_child_weight=3,  # 增加最小子节点权重
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,  # 添加gamma参数
            reg_alpha=0.1,  # 添加L1正则化
            reg_lambda=1.0,  # 添加L2正则化
            random_state=42
        )
        
        # 使用eval_set进行验证
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        return model
        
    def create_trading_env(self):
        """创建交易环境"""
        class TradingEnv(gym.Env):
            def __init__(self, data, strategy):
                super().__init__()
                self.data = data.copy()
                self.strategy = strategy
                self.current_step = 0
                self.position = 0
                self.balance = 100000
                
                # 定义动作空间
                self.action_space = gym.spaces.Discrete(3)  # 买入、卖出、持有
                
                # 定义观察空间
                self.observation_space = gym.spaces.Box(
                    low=-10, high=10, shape=(10,), dtype=np.float32
                )
                
                # 预处理数据
                self._preprocess_data()
                
            def _preprocess_data(self):
                """预处理数据，处理缺失值和异常值"""
                try:
                    # 计算技术指标
                    df = self.strategy.calculate_indicators(self.data)
                    
                    # 处理缺失值
                    df = df.ffill().bfill()
                    
                    # 处理无穷值
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.ffill().bfill()
                    
                    # 归一化数据
                    for col in df.columns:
                        if col not in ['open', 'high', 'low', 'close', 'volume']:
                            mean = df[col].mean()
                            std = df[col].std()
                            if std != 0:
                                df[col] = (df[col] - mean) / std
                    
                    self.processed_data = df
                except Exception as e:
                    print(f"数据预处理错误: {str(e)}")
                    raise
                
            def reset(self):
                self.current_step = 0
                self.position = 0
                self.balance = 100000
                return self._get_observation()
                
            def step(self, action):
                try:
                    # 执行交易
                    reward = self._execute_action(action)
                    
                    # 更新状态
                    self.current_step += 1
                    done = self.current_step >= len(self.processed_data) - 1
                    next_obs = self._get_observation()
                    
                    return next_obs, reward, done, {}
                except Exception as e:
                    print(f"执行步骤错误: {str(e)}")
                    raise
                
            def _get_observation(self):
                """获取当前市场状态"""
                try:
                    obs = np.array([
                        self.processed_data['RSI'].iloc[self.current_step],
                        self.processed_data['ADX'].iloc[self.current_step],
                        self.processed_data['MACD'].iloc[self.current_step],
                        self.processed_data['Signal'].iloc[self.current_step],
                        self.processed_data['Volume_Ratio'].iloc[self.current_step],
                        self.processed_data['Trend_Strength'].iloc[self.current_step],
                        self.processed_data['Momentum_Strength'].iloc[self.current_step],
                        self.processed_data['Volume_Strength'].iloc[self.current_step],
                        self.processed_data['Volatility'].iloc[self.current_step],
                        self.processed_data['ATR'].iloc[self.current_step]
                    ], dtype=np.float32)
                    
                    # 确保没有NaN值，并将值限制在[-10, 10]范围内
                    obs = np.nan_to_num(obs, nan=0.0)
                    obs = np.clip(obs, -10, 10)
                    return obs
                except Exception as e:
                    print(f"获取观察值错误: {str(e)}")
                    raise
                
            def _execute_action(self, action):
                try:
                    current_price = self.processed_data['close'].iloc[self.current_step]
                    
                    if action == 0:  # 买入
                        if self.position <= 0:
                            self.position = 1
                            return 0
                    elif action == 1:  # 卖出
                        if self.position >= 0:
                            self.position = -1
                            return 0
                    
                    # 计算收益
                    next_price = self.processed_data['close'].iloc[self.current_step + 1]
                    return self.position * (next_price / current_price - 1)
                except Exception as e:
                    print(f"执行动作错误: {str(e)}")
                    raise
        
        return TradingEnv(self.data, self.strategy)
        
    def train_rl_agent(self):
        """训练强化学习代理"""
        env = self.create_trading_env()
        env = DummyVecEnv([lambda: env])
        
        # 使用更简单的PPO配置
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
            gamma=0.99,
            verbose=1
        )
        
        # 减少训练步数到5000步
        model.learn(total_timesteps=5000, progress_bar=True)
        return model
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算夏普比率"""
        if len(returns) == 0:
            return -1.0
        
        # 处理无效值
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算超额收益
        excess_returns = returns - self.risk_free_rate / 252
        
        # 计算年化收益率和波动率
        annual_return = np.mean(excess_returns) * 252
        annual_volatility = np.std(excess_returns) * np.sqrt(252)
        
        # 如果波动率接近0，返回一个合理的负值
        if annual_volatility < 1e-6:
            return -1.0
        
        # 计算夏普比率并限制范围
        sharpe = annual_return / annual_volatility
        return np.clip(sharpe, -5.0, 5.0)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """计算索提诺比率"""
        if len(returns) == 0:
            return -1.0
        
        # 处理无效值
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算超额收益
        excess_returns = returns - self.risk_free_rate / 252
        
        # 计算年化收益率
        annual_return = np.mean(excess_returns) * 252
        
        # 计算下行波动率
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 5.0  # 如果没有下行风险，返回一个合理的正值
        
        downside_std = np.std(downside_returns) * np.sqrt(252)
        
        # 如果下行波动率接近0，返回一个合理的正值
        if downside_std < 1e-6:
            return 5.0
        
        # 计算索提诺比率并限制范围
        sortino = annual_return / downside_std
        return np.clip(sortino, -5.0, 5.0)

    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """计算总收益率"""
        if len(returns) == 0:
            return -1.0
        
        # 处理无效值
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算累积收益率
        cumulative_return = np.prod(1 + returns) - 1
        
        # 限制总收益率的范围
        return np.clip(cumulative_return, -0.5, 2.0)

    def _evaluate_parameters(self, params: Dict[str, Any]) -> float:
        """评估单组参数"""
        try:
            # 设置策略参数
            for param, value in params.items():
                setattr(self.strategy, param, value)
            
            # 执行回测
            returns = self.strategy.backtest(self.data)
            
            if len(returns) == 0:
                return -1.0
            
            # 处理无效值
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 计算性能指标
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            total_return = self._calculate_total_return(returns)
            
            # 确保所有指标都是有效的数值
            if np.isnan(sharpe_ratio) or np.isnan(sortino_ratio) or np.isnan(total_return):
                return -1.0
            
            # 计算综合得分
            score = (
                self.metric_weights['sharpe_ratio'] * sharpe_ratio +
                self.metric_weights['sortino_ratio'] * sortino_ratio +
                self.metric_weights['total_return'] * total_return
            )
            
            # 限制最终得分的范围
            return np.clip(score, -5.0, 5.0)
            
        except Exception as e:
            self.logger.error(f"参数评估出错: {str(e)}")
            return -1.0

    def optimize_parameters(self) -> Dict[str, Any]:
        """优化策略参数"""
        try:
            best_score = float('-inf')
            best_params = {}
            
            # 将参数分为三组：趋势参数、动量参数和风险管理参数
            trend_params = ['fast_period', 'slow_period', 'adx_period', 'adx_threshold']
            momentum_params = ['rsi_period', 'macd_fast', 'macd_slow', 'macd_signal']
            risk_params = ['atr_period', 'atr_multiplier', 'volume_threshold', 'signal_threshold',
                         'profit_target', 'stop_loss', 'trailing_stop']
            
            # 优化趋势参数组合
            self.logger.info("优化趋势参数组合...")
            trend_combinations = []
            for fast in np.linspace(self.param_space['fast_period'][0], self.param_space['fast_period'][1], 4):
                for slow in np.linspace(self.param_space['slow_period'][0], self.param_space['slow_period'][1], 4):
                    for adx_p in np.linspace(self.param_space['adx_period'][0], self.param_space['adx_period'][1], 3):
                        for adx_t in np.linspace(self.param_space['adx_threshold'][0], self.param_space['adx_threshold'][1], 3):
                            if fast < slow:  # 确保快速周期小于慢速周期
                                trend_combinations.append({
                                    'fast_period': int(fast),
                                    'slow_period': int(slow),
                                    'adx_period': int(adx_p),
                                    'adx_threshold': int(adx_t)
                                })
            
            # 测试趋势参数组合
            for params in trend_combinations:
                score = self._evaluate_parameters(params)
                self.logger.info(f"趋势参数: {params}, 性能: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_params.update(params)
            
            # 优化动量参数组合
            self.logger.info("优化动量参数组合...")
            momentum_combinations = []
            for rsi in np.linspace(self.param_space['rsi_period'][0], self.param_space['rsi_period'][1], 4):
                for macd_f in np.linspace(self.param_space['macd_fast'][0], self.param_space['macd_fast'][1], 3):
                    for macd_s in np.linspace(self.param_space['macd_slow'][0], self.param_space['macd_slow'][1], 3):
                        for macd_sig in np.linspace(self.param_space['macd_signal'][0], self.param_space['macd_signal'][1], 3):
                            if macd_f < macd_s:  # 确保MACD快线周期小于慢线周期
                                momentum_combinations.append({
                                    'rsi_period': int(rsi),
                                    'macd_fast': int(macd_f),
                                    'macd_slow': int(macd_s),
                                    'macd_signal': int(macd_sig)
                                })
            
            # 使用最佳趋势参数测试动量参数组合
            for params in momentum_combinations:
                params.update(best_params)  # 合并之前找到的最佳趋势参数
                score = self._evaluate_parameters(params)
                self.logger.info(f"动量参数: {params}, 性能: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_params.update(params)
            
            # 优化风险管理参数
            self.logger.info("优化风险管理参数...")
            risk_combinations = []
            for atr_p in np.linspace(self.param_space['atr_period'][0], self.param_space['atr_period'][1], 3):
                for atr_m in np.linspace(self.param_space['atr_multiplier'][0], self.param_space['atr_multiplier'][1], 3):
                    for vol_t in np.linspace(self.param_space['volume_threshold'][0], self.param_space['volume_threshold'][1], 3):
                        for sig_t in np.linspace(self.param_space['signal_threshold'][0], self.param_space['signal_threshold'][1], 3):
                            for profit in np.linspace(self.param_space['profit_target'][0], self.param_space['profit_target'][1], 3):
                                for stop in np.linspace(self.param_space['stop_loss'][0], self.param_space['stop_loss'][1], 3):
                                    for trail in np.linspace(self.param_space['trailing_stop'][0], self.param_space['trailing_stop'][1], 3):
                                        risk_combinations.append({
                                            'atr_period': int(atr_p),
                                            'atr_multiplier': float(atr_m),
                                            'volume_threshold': float(vol_t),
                                            'signal_threshold': float(sig_t),
                                            'profit_target': float(profit),
                                            'stop_loss': float(stop),
                                            'trailing_stop': float(trail)
                                        })
            
            # 使用最佳趋势和动量参数测试风险管理参数组合
            for params in risk_combinations:
                params.update(best_params)  # 合并之前找到的最佳参数
                score = self._evaluate_parameters(params)
                self.logger.info(f"风险参数: {params}, 性能: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_params.update(params)
            
            # 最后评估所有参数组合
            self.logger.info("评估最佳参数组合...")
            final_score = self._evaluate_parameters(best_params)
            self.logger.info(f"最终性能指标: {final_score:.4f}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"参数优化过程中出错: {str(e)}")
            raise 