#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信号组合示例

展示如何使用SignalExtractor提取信号，并使用XGBoostSignalCombiner组合信号预测未来收益
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# 导入自定义模块
from strategy_optimizer.utils.signal_extractor import SignalExtractor
from strategy_optimizer.models.xgboost_model import XGBoostSignalCombiner
from strategy_optimizer.utils import DataGenerator

def load_sample_data():
    """
    加载或生成样本数据
    
    此函数尝试加载样本数据，如果不存在则生成合成数据
    
    返回:
        价格数据 DataFrame 和收益率 Series
    """
    # 尝试从文件加载样本数据，如果文件不存在则生成合成数据
    sample_data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    
    if os.path.exists(sample_data_path):
        print(f"从 {sample_data_path} 加载样本数据...")
        # 从CSV文件加载数据
        price_data = pd.read_csv(sample_data_path, index_col=0, parse_dates=True)
        price_data.sort_index(inplace=True)
    else:
        print("生成合成样本数据...")
        # 使用数据生成器创建合成数据
        generator = DataGenerator(seed=42)
        signals, returns = generator.generate_synthetic_data(
            n_samples=750,  # 3年的交易日
            n_signals=5,
            signal_strength={0: 0.7, 1: 0.5, 2: 0.3},
            noise_level=0.2,
            start_date="2018-01-01"
        )
        
        # 创建价格数据 (基于收益率反向计算)
        initial_price = 100.0
        prices = (1 + returns).cumprod() * initial_price
        
        # 创建OHLCV数据
        price_data = pd.DataFrame({
            'open': prices * (1 - np.random.randn(len(prices)) * 0.005),
            'high': 0.0,
            'low': 0.0,
            'close': prices,
            'volume': np.abs(np.random.randn(len(prices)) * 1000000 + 500000)
        }, index=returns.index)
        
        # 填充high和low
        for i in range(len(price_data)):
            idx = price_data.index[i]
            price_data.loc[idx, 'high'] = max(price_data.loc[idx, 'open'], price_data.loc[idx, 'close']) * (1 + abs(np.random.randn()) * 0.008)
            price_data.loc[idx, 'low'] = min(price_data.loc[idx, 'open'], price_data.loc[idx, 'close']) * (1 - abs(np.random.randn()) * 0.008)
        
        # 保存样本数据供以后使用
        price_data.to_csv(sample_data_path)
    
    # 计算目标变量：未来的1天收益率
    returns = price_data['close'].pct_change().shift(-1)
    
    return price_data, returns

def visualize_data(price_data, returns):
    """
    可视化数据
    
    参数:
        price_data: 价格数据 DataFrame
        returns: 收益率 Series
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制收盘价
    plt.subplot(2, 1, 1)
    plt.plot(price_data.index, price_data['close'], label='收盘价')
    plt.title('价格数据')
    plt.ylabel('价格')
    plt.grid(True)
    plt.legend()
    
    # 绘制收益率
    plt.subplot(2, 1, 2)
    plt.plot(returns.index, returns, label='1日收益率', color='green')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    plt.title('收益率')
    plt.ylabel('收益率')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('price_and_returns.png')
    print("数据可视化已保存为 'price_and_returns.png'")

def extract_signals(price_data):
    """
    从价格数据中提取交易信号
    
    参数:
        price_data: 价格数据 DataFrame
        
    返回:
        包含所有信号的 DataFrame
    """
    print("\n提取交易信号...")
    
    # 初始化信号提取器
    extractor = SignalExtractor(price_data)
    
    # 提取各类信号
    extractor.extract_trend_signals()
    extractor.extract_momentum_signals()
    extractor.extract_volatility_signals()
    extractor.extract_volume_signals()
    extractor.extract_support_resistance_signals()
    
    # 获取所有信号
    signals_df = extractor.get_signals()
    
    # 打印信号信息
    print(f"- 提取的信号总数: {len(signals_df.columns)}")
    
    # 按类别统计信号数量
    category_counts = {}
    for signal, metadata in extractor.metadata.items():
        category = metadata["category"]
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    
    for category, count in category_counts.items():
        print(f"- {category}: {count}个信号")
    
    return signals_df, extractor

def engineer_features(signals_df, extractor, returns):
    """
    特征工程：对信号进行处理和选择
    
    参数:
        signals_df: 信号 DataFrame
        extractor: 信号提取器实例
        returns: 收益率 Series
        
    返回:
        处理后的特征 DataFrame
    """
    print("\n特征工程...")
    
    # 1. 标准化信号
    normalized_signals = extractor.normalize_signals(method="zscore")
    
    # 2. 去除包含过多NaN值的特征
    threshold = 0.95  # 要求至少95%的数据是非NaN
    valid_columns = normalized_signals.columns[normalized_signals.notna().mean() >= threshold]
    cleaned_signals = normalized_signals[valid_columns].copy()
    
    # 3. 填充剩余的NaN值
    cleaned_signals.fillna(0, inplace=True)
    
    # 4. 计算信号与目标的相关性
    correlation_df = extractor.rank_signals_by_correlation(returns)
    
    # 打印信息
    print(f"- 标准化和清洗后的信号数量: {len(cleaned_signals.columns)}")
    print(f"- 与收益率相关性最高的前5个信号:")
    for i, row in correlation_df.head(5).iterrows():
        print(f"  {row['signal']}: 相关性 = {row['correlation']:.4f}")
    
    return cleaned_signals

def train_and_evaluate_model(features_df, returns, test_size=0.3):
    """
    训练和评估信号组合模型
    
    参数:
        features_df: 特征 DataFrame
        returns: 收益率 Series
        test_size: 测试集比例
        
    返回:
        训练好的模型和性能指标
    """
    print("\n训练信号组合模型...")
    
    # 确保特征和目标变量的索引一致
    common_index = features_df.index.intersection(returns.index)
    X = features_df.loc[common_index]
    y = returns.loc[common_index]
    
    # 划分训练集和测试集 (按时间顺序)
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"- 训练集大小: {len(X_train)}个样本 ({train_size / len(X) * 100:.1f}%)")
    print(f"- 测试集大小: {len(X_test)}个样本 ({test_size * 100:.1f}%)")
    
    # 初始化XGBoost模型
    model = XGBoostSignalCombiner(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=30,
        verbose=False
    )
    
    # 评估模型
    train_perf = model.train_performance
    test_perf = model.test_performance
    
    print("\n模型评估结果:")
    print("训练集性能:")
    for metric in ['sign_accuracy', 'sharpe_ratio', 'annual_return', 'max_drawdown']:
        print(f"- {metric}: {train_perf[metric]:.4f}")
    
    print("\n测试集性能:")
    for metric in ['sign_accuracy', 'sharpe_ratio', 'annual_return', 'max_drawdown']:
        print(f"- {metric}: {test_perf[metric]:.4f}")
    
    # 特征重要性分析
    importance = model.get_feature_importance(plot=False, top_n=10)
    
    print("\n最重要的10个特征:")
    for i, row in importance.iterrows():
        print(f"- {row['feature']}: {row['importance']:.4f}")
    
    # 生成特征重要性图
    plt.figure(figsize=(10, 8))
    plt.barh(importance['feature'].head(15), importance['importance'].head(15))
    plt.xlabel('重要性')
    plt.title('顶部15个特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("特征重要性图已保存为 'feature_importance.png'")
    
    # 绘制预测性能
    model.plot_performance(X_test, y_test, plot_type='cumulative_returns')
    
    # 绘制性能图表
    plt.figure(figsize=(12, 10))
    
    # 绘制预测vs实际值
    plt.subplot(2, 1, 1)
    y_pred = model.predict(X_test)
    plt.plot(y_test.index, y_test.values, label='实际收益率', color='blue', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='预测收益率', color='red', alpha=0.7)
    plt.title('预测vs实际收益率')
    plt.ylabel('收益率')
    plt.legend()
    plt.grid(True)
    
    # 绘制累积收益对比
    plt.subplot(2, 1, 2)
    cum_actual = (1 + y_test).cumprod() - 1
    
    # 计算模型信号产生的累积收益
    position = np.sign(y_pred)
    strategy_returns = position * y_test
    cum_strategy = (1 + strategy_returns).cumprod() - 1
    
    # 买入持有基准
    plt.plot(y_test.index, cum_actual, label='买入持有', color='blue')
    plt.plot(y_test.index, cum_strategy, label='模型策略', color='green')
    plt.title('累积收益对比')
    plt.xlabel('日期')
    plt.ylabel('累积收益率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    print("模型性能图已保存为 'model_performance.png'")
    
    return model, train_perf, test_perf, importance

def generate_report(model, price_data, signals_df, train_perf, test_perf, importance, start_date=None, end_date=None):
    """
    生成模型报告
    
    参数:
        model: 训练好的XGBoost模型
        price_data: 价格数据
        signals_df: 信号数据
        train_perf: 训练集性能
        test_perf: 测试集性能
        importance: 特征重要性
        start_date: 开始日期
        end_date: 结束日期
    """
    from datetime import datetime
    
    print("\n生成报告...")
    
    # 创建报告目录
    report_dir = "signal_model_report"
    os.makedirs(report_dir, exist_ok=True)
    
    # 创建一个HTML报告文件
    report_file = os.path.join(report_dir, "report.html")
    
    # 获取数据范围
    if start_date is None:
        start_date = price_data.index[0]
    if end_date is None:
        end_date = price_data.index[-1]
    
    # 构建HTML报告内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>信号组合模型报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .performance {{ display: flex; justify-content: space-between; }}
            .performance-table {{ width: 48%; }}
            .images {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>信号组合模型报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>数据摘要</h2>
        <p>日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}</p>
        <p>样本数量: {len(price_data)}</p>
        <p>信号数量: {len(signals_df.columns)}</p>
        
        <h2>模型性能</h2>
        <div class="performance">
            <div class="performance-table">
                <h3>训练集性能</h3>
                <table>
                    <tr><th>指标</th><th>值</th></tr>
                    <tr><td>方向性准确率</td><td>{train_perf['sign_accuracy']:.4f}</td></tr>
                    <tr><td>夏普比率</td><td>{train_perf['sharpe_ratio']:.4f}</td></tr>
                    <tr><td>年化收益率</td><td>{train_perf['annual_return']:.4f}</td></tr>
                    <tr><td>最大回撤</td><td>{train_perf['max_drawdown']:.4f}</td></tr>
                    <tr><td>均方误差</td><td>{train_perf['mse']:.6f}</td></tr>
                    <tr><td>R²</td><td>{train_perf['r2']:.4f}</td></tr>
                </table>
            </div>
            
            <div class="performance-table">
                <h3>测试集性能</h3>
                <table>
                    <tr><th>指标</th><th>值</th></tr>
                    <tr><td>方向性准确率</td><td>{test_perf['sign_accuracy']:.4f}</td></tr>
                    <tr><td>夏普比率</td><td>{test_perf['sharpe_ratio']:.4f}</td></tr>
                    <tr><td>年化收益率</td><td>{test_perf['annual_return']:.4f}</td></tr>
                    <tr><td>最大回撤</td><td>{test_perf['max_drawdown']:.4f}</td></tr>
                    <tr><td>均方误差</td><td>{test_perf['mse']:.6f}</td></tr>
                    <tr><td>R²</td><td>{test_perf['r2']:.4f}</td></tr>
                </table>
            </div>
        </div>
        
        <h2>顶部特征重要性</h2>
        <table>
            <tr><th>特征</th><th>重要性</th></tr>
    """
    
    # 添加特征重要性表格
    for i, row in importance.head(15).iterrows():
        html_content += f"""
            <tr><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>
        """
    
    html_content += """
        </table>
        
        <h2>模型可视化</h2>
        <div class="images">
            <h3>价格数据和收益率</h3>
            <img src="../price_and_returns.png" alt="价格数据和收益率">
            
            <h3>特征重要性</h3>
            <img src="../feature_importance.png" alt="特征重要性">
            
            <h3>模型性能</h3>
            <img src="../model_performance.png" alt="模型性能">
        </div>
        
        <h2>结论和建议</h2>
        <p>这个模型通过结合多个技术指标信号，尝试预测市场的未来走势。以下是根据模型评估得出的主要结论：</p>
        <ul>
    """
    
    # 添加结论
    if test_perf['sign_accuracy'] > 0.55:
        html_content += f"<li>模型在预测市场方向上表现较好，方向性预测准确率达到 {test_perf['sign_accuracy']:.1%}。</li>"
    else:
        html_content += f"<li>模型在预测市场方向上的准确率为 {test_perf['sign_accuracy']:.1%}，仍有改进空间。</li>"
    
    if test_perf['sharpe_ratio'] > 1.0:
        html_content += f"<li>模型策略的风险调整收益表现良好，夏普比率为 {test_perf['sharpe_ratio']:.2f}。</li>"
    else:
        html_content += f"<li>模型策略的风险调整收益一般，夏普比率为 {test_perf['sharpe_ratio']:.2f}。</li>"
    
    # 添加重要特征分析
    top_features = importance['feature'].head(3).tolist()
    html_content += f"<li>最重要的三个信号特征是：{', '.join(top_features)}。</li>"
    
    # 添加建议
    html_content += """
        </ul>
        
        <p>改进建议：</p>
        <ul>
            <li>考虑添加市场状态特征，在不同市场环境下调整模型行为。</li>
            <li>探索更多的信号组合方式，可能会提高预测准确性。</li>
            <li>尝试特征选择方法，减少噪声特征的影响。</li>
            <li>考虑使用集成方法，如Stacking，结合多个模型的优势。</li>
        </ul>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"报告已生成，保存在 {report_file}")

def main():
    """主函数"""
    # 设置随机种子保持结果一致
    np.random.seed(42)
    
    # 加载数据
    price_data, returns = load_sample_data()
    
    # 可视化数据
    visualize_data(price_data, returns)
    
    # 提取交易信号
    signals_df, extractor = extract_signals(price_data)
    
    # 特征工程
    features_df = engineer_features(signals_df, extractor, returns)
    
    # 训练和评估模型
    model, train_perf, test_perf, importance = train_and_evaluate_model(features_df, returns)
    
    # 生成报告
    generate_report(model, price_data, signals_df, train_perf, test_perf, importance)
    
    print("\n示例完成！")


if __name__ == "__main__":
    main() 