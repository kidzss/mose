import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='可视化策略权重')
    parser.add_argument('--input', type=str, default='strategy_weights.csv', help='输入权重CSV文件路径')
    parser.add_argument('--output', type=str, default='strategy_weights_viz.png', help='输出图表文件路径')
    args = parser.parse_args()
    
    # 读取数据
    df = pd.read_csv(args.input)
    
    # 提取权重列（不包括_raw后缀和Symbol列）
    weight_cols = [col for col in df.columns if not col.endswith('_raw') and col != 'Symbol']
    
    # 检查是否有权重列
    if not weight_cols:
        print("错误：找不到权重列，请检查输入文件格式")
        return
    
    # 创建可视化数据框
    plot_data = df[['Symbol'] + weight_cols].copy()
    
    # 设置风格和画布
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 10))
    
    # 1. 柱状图：每个股票的策略权重分布
    plt.subplot(2, 1, 1)
    plot_data_melted = pd.melt(plot_data, id_vars=['Symbol'], var_name='Strategy', value_name='Weight')
    ax = sns.barplot(data=plot_data_melted, x='Symbol', y='Weight', hue='Strategy')
    plt.title('每个股票的策略权重分布', fontsize=16)
    plt.ylabel('权重', fontsize=12)
    plt.xlabel('股票代码', fontsize=12)
    plt.legend(title='策略')
    
    # 添加数值标签
    for i, p in enumerate(ax.patches):
        if p.get_height() > 0.05:  # 只显示大于0.05的权重
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    # 2. 热力图：不同股票和策略之间的权重关系
    plt.subplot(2, 1, 2)
    pivot_data = plot_data.set_index('Symbol')
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('策略权重热力图', fontsize=16)
    plt.xlabel('策略', fontsize=12)
    plt.ylabel('股票代码', fontsize=12)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f'可视化结果已保存至 {args.output}')
    
    # 3. 创建表格展示总体策略权重占比
    strategy_avg = pivot_data.mean()
    plt.figure(figsize=(10, 6))
    
    # 饼图：策略权重平均分布
    plt.subplot(1, 2, 1)
    plt.pie(strategy_avg, labels=strategy_avg.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('策略权重平均分布', fontsize=16)
    
    # 条形图：策略权重平均值
    plt.subplot(1, 2, 2)
    bars = plt.barh(strategy_avg.index, strategy_avg.values)
    plt.title('策略权重平均值', fontsize=16)
    plt.xlabel('平均权重', fontsize=12)
    
    # 添加数值标签
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.3f}', va='center')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(args.output.replace('.png', '_summary.png'), dpi=300)
    print(f'策略权重总结已保存至 {args.output.replace(".png", "_summary.png")}')
    
    # 分析结果
    print("\n策略权重分析:")
    print(f"平均策略权重:\n{strategy_avg}")
    
    # 找出每个股票的最佳策略
    best_strategies = pivot_data.idxmax(axis=1)
    print(f"\n每个股票的最佳策略:\n{best_strategies}")
    
    # 找出整体最重要的策略
    most_important = strategy_avg.idxmax()
    print(f"\n整体最重要的策略: {most_important} (平均权重: {strategy_avg.max():.3f})")
    
    # 找出整体最不重要的策略
    least_important = strategy_avg.idxmin()
    print(f"整体最不重要的策略: {least_important} (平均权重: {strategy_avg.min():.3f})")

if __name__ == "__main__":
    main() 