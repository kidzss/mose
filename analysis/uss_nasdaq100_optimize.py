import pandas as pd
from scipy.optimize import minimize


def calculate_weighted_return(weights, returns):
    """计算加权总收益率"""
    return -sum(weights[i] * returns[i] for i in range(len(weights)))  # 负值用于最小化


def constraint(weights):
    """约束条件：权重和为1"""
    return sum(weights) - 1


def optimize_weights(df):
    """优化权重以最大化总收益率"""
    # 提取收益率
    returns = df['Total Return(%)'].values

    # 初始权重
    initial_weights = [1 / 3, 1 / 3, 1 / 3]

    # 定义约束条件
    cons = {'type': 'eq', 'fun': constraint}

    # 设置权重的边界
    bounds = [(0, 1) for _ in range(len(initial_weights))]

    # 优化
    result = minimize(calculate_weighted_return, initial_weights, args=(returns,),
                      constraints=cons, bounds=bounds)

    return result.x, -result.fun  # 返回最优权重和最大化后的总收益率


def main():
    # 读取预测结果
    df = pd.read_csv("top_stocks_by_total_return.csv")  # 替换为你的文件路径

    # 优化权重
    optimal_weights, max_return = optimize_weights(df)

    print("Optimal Weights:", optimal_weights)
    print("Max Total Return(%):", max_return)


if __name__ == "__main__":
    main()
