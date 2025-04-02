import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, config: dict):
        """初始化Transformer模型
        
        Args:
            config: 模型配置字典，包含以下键：
                - input_size: 输入特征维度
                - hidden_size: 隐藏层维度
                - num_heads: 注意力头数
                - num_layers: Transformer层数
                - dropout: Dropout比率
                - output_size: 输出维度
        """
        super().__init__()
        
        self.input_size = config.get('input_size', 12)  # 修改默认特征维度为12
        self.hidden_size = config.get('hidden_size', 128)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.35)
        self.output_size = config.get('output_size', 6)  # 默认6个策略
        
        # 特征投影层
        self.feature_projection = nn.Linear(self.input_size, self.hidden_size)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.hidden_size, self.dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Softmax(dim=-1)  # 确保输出是概率分布
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
            
        Returns:
            输出张量，形状为 (batch_size, output_size)
        """
        # 特征投影
        x = self.feature_projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 取序列的最后一个时间步
        x = x[:, -1, :]
        
        # 输出层
        x = self.output_layer(x)
        
        return x

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, embedding_dim)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 