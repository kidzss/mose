import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer

logger = logging.getLogger(__name__)

class StrategyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 基本参数
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.sequence_length = config['sequence_length']
        self.hidden_size = 32  # 减小隐藏层大小
        
        # 检查并修复input_size
        actual_input_size = 15  # 根据错误信息修正为实际输入特征维度
        self.logger = logging.getLogger(__name__) if 'logger' not in vars(self) else self.logger
        self.logger.info(f"Model initialized with config input_size={self.input_size}, "
                        f"but using actual_input_size={actual_input_size}")
        
        # 简单的网络结构
        self.encoder = nn.Sequential(
            nn.Linear(actual_input_size, self.hidden_size),  # 使用实际输入大小
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        batch_size = x.size(0)
        
        # 编码每个时间步
        encoded = []
        for t in range(x.size(1)):
            encoded.append(self.encoder(x[:, t]))
        encoded = torch.stack(encoded, dim=1)
        
        # LSTM处理序列
        lstm_out, _ = self.lstm(encoded)
        
        # 使用最后一个时间步的输出
        final_hidden = lstm_out[:, -1]
        
        # 解码得到输出
        output = self.decoder(final_hidden)
        
        return output

class SimpleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        return F.mse_loss(pred, target)

def train_epoch(model, optimizer, train_loader, criterion, device):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.info(f'Train Batch [{batch_idx}/{num_batches}] Loss: {loss.item():.6f}')
            
    return total_loss / num_batches

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
    return total_loss / len(val_loader) 