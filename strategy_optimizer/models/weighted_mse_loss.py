import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    带权重的均方误差损失函数
    允许对不同样本或预测赋予不同的重要性权重
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        
    def forward(self, predictions, targets, weights=None):
        """
        计算加权MSE损失
        
        Args:
            predictions: 模型预测值
            targets: 真实目标值
            weights: 每个样本/时间步的权重。如果为None，则所有样本权重相等
            
        Returns:
            加权平均的MSE损失
        """
        if weights is None:
            # 如果没有提供权重，则使用普通MSE
            return torch.mean((predictions - targets) ** 2)
        
        # 确保权重形状与输入匹配
        if weights.shape != predictions.shape:
            weights = weights.expand_as(predictions)
            
        # 计算加权MSE
        squared_error = (predictions - targets) ** 2
        weighted_squared_error = squared_error * weights
        
        # 返回加权平均值
        return torch.sum(weighted_squared_error) / torch.sum(weights) 