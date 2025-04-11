import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sqlalchemy import create_engine
import yfinance as yf
from strategy_optimizer.market_state import MarketStateAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取市场数据"""
    try:
        # 使用 yfinance 获取数据
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # 打印基本信息
        print(f"\n获取到的数据基本信息:")
        print(f"日期范围: {df.index.min()} 到 {df.index.max()}")
        print(f"总记录数: {len(df)}")
        print(f"列名: {df.columns.tolist()}")
        
        # 检查数据连续性
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        missing_dates = date_range.difference(df.index)
        if len(missing_dates) > 0:
            print(f"\n警告: 发现 {len(missing_dates)} 个缺失交易日")
            print("缺失日期:", missing_dates.tolist())
        
        # 数据清洗
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 计算技术指标
        # 1. 移动平均线
        for window in [5, 10, 20, 60]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA{window}_slope'] = df[f'MA{window}'].pct_change()
        
        # 2. 布林带
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # 3. 波动率
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=60).mean()
        
        # 4. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['Signal']
        
        # 6. KDJ
        low_9 = df['Low'].rolling(window=9).min()
        high_9 = df['High'].rolling(window=9).max()
        rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
        df['K'] = rsv.rolling(window=3).mean()
        df['D'] = df['K'].rolling(window=3).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        # 7. 收益率特征
        df['returns'] = df['Close'].pct_change()
        df['returns_skew'] = df['returns'].rolling(window=20).skew()
        df['returns_kurt'] = df['returns'].rolling(window=20).kurt()
        
        # 8. 价格特征
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        
        # 9. 成交量特征
        df['volume_ma5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma20']
        
        # 删除包含 NaN 的行
        df = df.dropna()
        
        # 打印数据统计信息
        print("\n数据统计信息:")
        print(df.describe())
        
        return df
        
    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        raise

class TestNiuniuStrategyV2(unittest.TestCase):
    """测试牛牛策略V2版本"""
    
    @classmethod
    def setUpClass(cls):
        """测试前的准备工作"""
        # 使用当前日期作为结束日期
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = '2020-01-01'
        
        # 获取市场数据
        cls.market_data = get_market_data('BABA', start_date, end_date)
        
        # 创建市场状态分析器
        cls.analyzer = MarketStateAnalyzer()
        
        # 分析市场状态
        market_states = cls.analyzer.analyze(cls.market_data)
        
        # 将市场状态列表转换为DataFrame
        states_data = []
        for state in market_states:
            data = {
                'timestamp': state.timestamp,
                'state_type': state.state_type.value,
                'confidence': state.confidence,
                **state.features
            }
            states_data.append(data)
        
        cls.market_states = pd.DataFrame(states_data)
        cls.market_states.set_index('timestamp', inplace=True)
        
        # 添加收益率
        cls.market_states['returns'] = cls.market_data['Close'].pct_change()
        
        # 准备特征和标签
        cls.features = cls.market_states.drop(['returns', 'state_type'], axis=1)
        cls.labels = cls.market_states['returns']
        
        # 分离数值型和分类型特征
        numeric_features = cls.features.select_dtypes(include=['float64', 'int64'])
        categorical_features = cls.features.select_dtypes(include=['object'])
        
        # 对分类型特征进行编码
        for col in categorical_features.columns:
            le = LabelEncoder()
            cls.features[col] = le.fit_transform(cls.features[col].astype(str))
        
        # 标准化数值型特征
        scaler = StandardScaler()
        cls.features_scaled = pd.DataFrame(
            scaler.fit_transform(numeric_features),
            columns=numeric_features.columns,
            index=numeric_features.index
        )
        
        # 合并数值型和分类型特征
        cls.features_scaled = pd.concat([cls.features_scaled, categorical_features], axis=1)
    
    def test_strategy_performance(self):
        """测试策略性能"""
        try:
            logger.info("\n=== 测试牛牛策略V2性能 ===")
            
            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            cv_results = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(self.features_scaled)):
                logger.info(f"\n正在评估第 {fold + 1} 折...")
                
                # 将训练集分为训练集和验证集
                train_size = int(len(train_idx) * 0.8)
                train_data = self.features_scaled.iloc[train_idx[:train_size]]
                train_labels = self.labels.iloc[train_idx[:train_size]]
                valid_data = self.features_scaled.iloc[train_idx[train_size:]]
                valid_labels = self.labels.iloc[train_idx[train_size:]]
                test_data = self.features_scaled.iloc[test_idx]
                test_labels = self.labels.iloc[test_idx]
                
                # 训练前的数据清洗
                # 1. 删除包含 NaN 的样本
                mask = ~(train_data.isna().any(axis=1) | train_labels.isna())
                train_data = train_data[mask]
                train_labels = train_labels[mask]
                
                mask = ~(valid_data.isna().any(axis=1) | valid_labels.isna())
                valid_data = valid_data[mask]
                valid_labels = valid_labels[mask]
                
                mask = ~(test_data.isna().any(axis=1) | test_labels.isna())
                test_data = test_data[mask]
                test_labels = test_labels[mask]
                
                # 2. 删除包含无穷大值的样本
                numeric_train = train_data.select_dtypes(include=[np.number])
                mask = ~(np.isinf(numeric_train).any(axis=1) | np.isinf(train_labels))
                train_data = train_data[mask]
                train_labels = train_labels[mask]
                
                numeric_valid = valid_data.select_dtypes(include=[np.number])
                mask = ~(np.isinf(numeric_valid).any(axis=1) | np.isinf(valid_labels))
                valid_data = valid_data[mask]
                valid_labels = valid_labels[mask]
                
                numeric_test = test_data.select_dtypes(include=[np.number])
                mask = ~(np.isinf(numeric_test).any(axis=1) | np.isinf(test_labels))
                test_data = test_data[mask]
                test_labels = test_labels[mask]
                
                # 3. 限制标签值的范围
                train_labels = train_labels.clip(lower=-0.1, upper=0.1)
                valid_labels = valid_labels.clip(lower=-0.1, upper=0.1)
                test_labels = test_labels.clip(lower=-0.1, upper=0.1)
                
                # 创建XGBoost模型（调整参数）
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    learning_rate=0.005,  # 降低学习率
                    n_estimators=500,     # 增加树的数量
                    max_depth=5,          # 增加树的深度
                    min_child_weight=1,   # 降低最小子节点权重
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    random_state=42,
                    early_stopping_rounds=50,  # 添加早停
                    eval_metric=['rmse', 'mae']  # 添加评估指标
                )
                
                # 训练模型
                model.fit(
                    train_data, train_labels,
                    eval_set=[(valid_data, valid_labels)],
                    verbose=False
                )
                
                # 评估XGBoost模型
                predictions = model.predict(test_data)
                
                # 计算性能指标
                fold_results = {
                    'mse': float(mean_squared_error(test_labels, predictions)),
                    'mae': float(mean_absolute_error(test_labels, predictions)),
                    'r2': float(r2_score(test_labels, predictions)),
                    'direction_accuracy': float((np.sign(test_labels) == np.sign(predictions)).mean()),
                    'test_size': len(test_idx),
                    'train_size': len(train_idx)
                }
                
                # 计算收益率预测的相关系数
                correlation = float(np.corrcoef(test_labels, predictions)[0, 1])
                fold_results['correlation'] = correlation
                
                # 计算预测值的统计特征
                fold_results['pred_mean'] = float(predictions.mean())
                fold_results['pred_std'] = float(predictions.std())
                fold_results['true_mean'] = float(test_labels.mean())
                fold_results['true_std'] = float(test_labels.std())
                
                # 计算信息系数（IC）
                ic = float(pd.Series(predictions).corr(pd.Series(test_labels)))
                fold_results['ic'] = ic
                
                # 计算累积收益
                strategy_returns = pd.Series(predictions).shift(1).fillna(0) * test_labels
                cumulative_returns = (1 + strategy_returns).cumprod()
                fold_results['final_return'] = float(cumulative_returns.iloc[-1] - 1)
                fold_results['sharpe_ratio'] = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
                
                cv_results.append(fold_results)
                
                logger.info(f"\n第 {fold + 1} 折性能:")
                for metric, value in fold_results.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
                # 输出特征重要性
                if fold == 0:  # 只在第一折输出
                    importance = pd.DataFrame({
                        'feature': self.features_scaled.columns,
                        'importance': model.feature_importances_
                    })
                    importance = importance.sort_values('importance', ascending=False)
                    logger.info("\n特征重要性（前10个）:")
                    for _, row in importance.head(10).iterrows():
                        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # 计算平均性能
            avg_performance = {}
            for metric in cv_results[0].keys():
                avg_performance[metric] = float(np.mean([r[metric] for r in cv_results]))
            
            # 计算标准差
            std_performance = {}
            for metric in cv_results[0].keys():
                std_performance[metric] = float(np.std([r[metric] for r in cv_results]))
            
            # 保存完整的测试结果
            performance_report = {
                'cv_results': cv_results,
                'avg_performance': avg_performance,
                'std_performance': std_performance
            }
            
            logger.info("\n平均性能:")
            for metric, value in avg_performance.items():
                logger.info(f"  {metric}: {value:.4f} ± {std_performance[metric]:.4f}")
            
            # 保存测试结果
            self.save_test_results(performance_report)
                
        except Exception as e:
            logger.error(f"测试过程中出错: {str(e)}")
            raise
        
    def save_test_results(self, performance_report):
        """保存测试结果"""
        import json
        import os
        
        try:
            # 创建输出目录
            output_dir = 'test_outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存结果
            output_file = os.path.join(output_dir, 'niuniu_v2_test_results.json')
            with open(output_file, 'w') as f:
                json.dump(performance_report, f, indent=2)
            
            logger.info(f"\n测试结果已保存至: {output_file}")
            
        except Exception as e:
            logger.error(f"保存测试结果时出错: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main() 