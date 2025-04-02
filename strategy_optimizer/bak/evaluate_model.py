import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import json
import logging
import talib
import matplotlib
import seaborn as sns
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_optimizer.models.transformer import StrategyTransformer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path, config):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StrategyTransformer(config['model_config']).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"成功加载模型 {model_path}, 训练到epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"训练损失: {checkpoint.get('train_loss', 'unknown')}, 验证损失: {checkpoint.get('val_loss', 'unknown')}")
    
    return model, device

def calculate_indicators(df):
    """直接计算需要的技术指标"""
    df_copy = df.copy()
    
    # 确保列名小写
    df_copy.columns = [col.lower() for col in df_copy.columns]
    
    # 计算SMA指标
    df_copy['sma_20'] = talib.SMA(df_copy['close'], timeperiod=20)
    df_copy['sma_50'] = talib.SMA(df_copy['close'], timeperiod=50)
    df_copy['sma_200'] = talib.SMA(df_copy['close'], timeperiod=200)
    
    # 计算RSI
    df_copy['rsi'] = talib.RSI(df_copy['close'], timeperiod=14)
    
    # 计算MACD
    df_copy['macd'], df_copy['macd_signal'], df_copy['macd_hist'] = talib.MACD(
        df_copy['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # 计算布林带
    df_copy['bb_upper'], df_copy['bb_middle'], df_copy['bb_lower'] = talib.BBANDS(
        df_copy['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    
    # 填充缺失值
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df_copy

def prepare_test_data(symbols, config):
    """准备测试数据 - 使用自定义方法而非DataProcessor"""
    try:
        # 计算日期范围 - 使用最近60天的数据作为测试
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 获取更多数据以确保有足够的历史
        
        all_features = []
        valid_symbols = []
        sequence_length = config['model_config']['sequence_length']
        
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'sma_200',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower'
        ]
        
        for symbol in symbols:
            try:
                logger.info(f"处理股票 {symbol} 的测试数据")
                
                # 使用yfinance获取股票数据
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                    end=end_date.strftime('%Y-%m-%d'))
                
                if df.empty:
                    logger.warning(f"股票 {symbol} 数据为空，跳过")
                    continue
                
                # 计算需要的技术指标
                df = calculate_indicators(df)
                
                # 检查是否有足够的数据
                if len(df) < sequence_length:
                    logger.warning(f"股票 {symbol} 数据不足 {sequence_length} 条，跳过")
                    continue
                    
                # 检查是否所有所需的特征都存在
                missing_cols = [col for col in feature_columns if col not in df.columns]
                if missing_cols:
                    logger.warning(f"股票 {symbol} 缺少特征: {missing_cols}")
                    # 为缺失的列填充0值
                    for col in missing_cols:
                        df[col] = 0
                
                # 选择特征列
                df_features = df[feature_columns]
                
                # 数据标准化
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(df_features)
                
                # 创建序列数据
                sequences = []
                for i in range(len(normalized_data) - sequence_length + 1):
                    seq = normalized_data[i:(i + sequence_length)]
                    sequences.append(seq)
                
                if not sequences:
                    logger.warning(f"股票 {symbol} 没有生成有效的序列数据")
                    continue
                
                # 只保留最后一个序列用于预测
                X_last = sequences[-1:]
                X_last_np = np.array(X_last)
                
                all_features.append(X_last_np)
                valid_symbols.append(symbol)
                logger.info(f"股票 {symbol} 测试数据准备完成，特征形状: {X_last_np.shape}")
                
            except Exception as e:
                logger.error(f"处理股票 {symbol} 数据时出错: {e}")
                continue
        
        if not all_features:
            raise ValueError("没有有效的测试数据")
        
        # 合并数据
        X_test = np.vstack(all_features)
        logger.info(f"测试数据准备完成，形状: {X_test.shape}")
        
        return X_test, valid_symbols
        
    except Exception as e:
        logger.error(f"准备测试数据时出错: {e}")
        raise

def adjust_diversity(raw_predictions, diversity_factor=0.5):
    """
    调整预测结果的多样性
    
    参数:
        raw_predictions: 模型原始预测输出，形状为(batch_size, num_strategies)
        diversity_factor: 多样性因子，0表示保持原始预测，1表示最大多样性
        
    返回:
        调整后的预测
    """
    # 深拷贝原始预测
    adjusted = raw_predictions.copy()
    
    # 基于排名的调整
    for i in range(len(adjusted)):
        # 获取策略排名
        ranking = np.argsort(adjusted[i])[::-1]  # 从大到小排序
        
        # 创建基于排名的值
        rank_values = np.linspace(1.0, 0.1, len(ranking))
        
        # 混合原始预测和排名值
        blend = adjusted[i] * (1 - diversity_factor) + rank_values * diversity_factor
        
        # 进行归一化
        blend = blend / blend.sum()
        
        # 为确保主要策略仍然突出，对最高的两个策略值进行轻微调整
        top_idx = ranking[0]
        second_idx = ranking[1]
        blend[top_idx] *= 1.2
        blend[second_idx] *= 1.1
        
        # 再次归一化
        adjusted[i] = blend / blend.sum()
    
    return adjusted

def evaluate_model(model, X_test, symbols, device, diversity_factor=0.5):
    """评估模型"""
    model.eval()
    predictions = []
    
    # 转换数据为PyTorch张量
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(X_tensor)
        raw_predictions = outputs.cpu().numpy()
    
    logger.info(f"原始预测完成，形状: {raw_predictions.shape}")
    
    # 调整预测的多样性
    adjusted_predictions = adjust_diversity(raw_predictions, diversity_factor)
    logger.info(f"调整后的预测完成，形状: {adjusted_predictions.shape}")
    
    # 获取策略名称
    strategy_names = [
        "GoldTriangle", "Momentum", "TDI", 
        "MarketForecast", "CPGW", "Niuniu"
    ]
    
    # 创建结果DataFrame
    results = []
    for i, symbol in enumerate(symbols):
        # 提取原始预测值
        raw_weights = {
            strategy: float(raw_predictions[i, j]) 
            for j, strategy in enumerate(strategy_names) if j < raw_predictions.shape[1]
        }
        
        # 提取调整后的预测值
        adjusted_weights = {
            strategy: float(adjusted_predictions[i, j]) 
            for j, strategy in enumerate(strategy_names) if j < adjusted_predictions.shape[1]
        }
        
        # 添加到结果中
        results.append({
            'Symbol': symbol,
            **{f"{k}_raw": v for k, v in raw_weights.items()},
            **{f"{k}": v for k, v in adjusted_weights.items()}
        })
    
    return pd.DataFrame(results)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估策略优化模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output', type=str, default='strategy_weights.csv', help='输出文件路径')
    parser.add_argument('--diversity', type=float, default=0.5, help='策略多样性因子(0-1)')
    args = parser.parse_args()
    
    try:
        # 加载配置
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # 获取股票列表
        symbols = config['data_config']['symbols'][:5]  # 简单起见只取前5个进行测试
        logger.info(f"将评估以下股票的策略权重: {', '.join(symbols)}")
        
        # 加载模型
        model, device = load_model(args.model_path, config)
        
        # 准备测试数据
        X_test, test_symbols = prepare_test_data(symbols, config)
        
        # 评估模型
        results_df = evaluate_model(model, X_test, test_symbols, device, args.diversity)
        
        # 保存结果
        results_df.to_csv(args.output, index=False)
        logger.info(f"结果已保存到 {args.output}")
        
        # 打印结果
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print("\n== 策略优化结果 ==")
        print(results_df[['Symbol'] + [col for col in results_df.columns if not col.endswith('_raw') and col != 'Symbol']])
        
        # 调用可视化模块展示权重分布
        try:
            import matplotlib.pyplot as plt
            # 创建可视化图表
            visualization_output = args.output.replace('.csv', '_viz.png')
            summary_output = args.output.replace('.csv', '_summary.png')
            
            # 使用简单的可视化直接在这里展示
            strategy_cols = [col for col in results_df.columns if not col.endswith('_raw') and col != 'Symbol']
            plt.figure(figsize=(12, 10))
            
            # 热图展示
            plt.subplot(2, 1, 1)
            data_pivot = results_df.pivot(index='Symbol', columns=strategy_cols, values=strategy_cols)
            
            # 清理数据透视表
            data_heatmap = results_df.set_index('Symbol')[strategy_cols]
            sns.heatmap(data_heatmap, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('股票策略权重分布热图', fontsize=16)
            
            # 每只股票策略权重占比
            plt.subplot(2, 1, 2)
            data_melted = pd.melt(results_df, id_vars=['Symbol'], value_vars=strategy_cols, 
                                  var_name='策略', value_name='权重')
            sns.barplot(x='Symbol', y='权重', hue='策略', data=data_melted)
            plt.title('各股票策略权重占比', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(visualization_output, dpi=300)
            logger.info(f"权重可视化已保存至 {visualization_output}")
            
            # 策略总体权重分布
            plt.figure(figsize=(10, 8))
            avg_weights = results_df[strategy_cols].mean()
            
            plt.subplot(1, 2, 1)
            plt.pie(avg_weights, labels=avg_weights.index, autopct='%1.1f%%')
            plt.title('策略平均权重分布', fontsize=16)
            
            plt.subplot(1, 2, 2)
            bars = plt.barh(avg_weights.index, avg_weights.values)
            plt.title('策略平均权重排名', fontsize=16)
            for bar in bars:
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{bar.get_width():.3f}', va='center')
                        
            plt.tight_layout()
            plt.savefig(summary_output, dpi=300)
            logger.info(f"策略权重总结已保存至 {summary_output}")
            
            # 输出一些关键分析
            best_strategy = avg_weights.idxmax()
            logger.info(f"整体最佳策略: {best_strategy} (平均权重: {avg_weights.max():.3f})")
            
            # 找出每只股票的最佳策略
            best_per_stock = {}
            for i, row in results_df.iterrows():
                symbol = row['Symbol']
                strategy_weights = {col: row[col] for col in strategy_cols}
                best = max(strategy_weights.items(), key=lambda x: x[1])
                best_per_stock[symbol] = best
            
            logger.info("各股票的最佳策略:")
            for symbol, (strategy, weight) in best_per_stock.items():
                logger.info(f"{symbol}: {strategy} (权重: {weight:.3f})")
                
        except Exception as e:
            logger.warning(f"创建可视化时出错: {e}")
        
    except Exception as e:
        logger.error(f"评估模型时出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 