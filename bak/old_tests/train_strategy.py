import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy_optimizer.data_processors.data_processor import DataProcessor
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(sequence_length: int, n_features: int) -> tf.keras.Model:
    """创建LSTM模型"""
    model = Sequential([
        LSTM(128, input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # 输出层使用sigmoid激活函数
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_strategy():
    # 初始化数据处理器
    dp = DataProcessor()
    
    # 设置训练参数
    sequence_length = 20  # 序列长度
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # 训练用的股票
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # 使用2年的数据
    
    logger.info("开始准备训练数据...")
    logger.info(f"时间范围: {start_date} 到 {end_date}")
    logger.info(f"股票列表: {symbols}")
    
    # 准备训练数据
    X, y = dp.prepare_training_data(symbols, start_date, end_date, sequence_length)
    
    if len(X) == 0 or len(y) == 0:
        logger.error("未能获取到训练数据")
        return
    
    # 将y转换为二分类标签（1表示买入信号，0表示其他）
    y_binary = (y > 0).astype(int)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, shuffle=False
    )
    
    logger.info(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
    
    # 创建和训练模型
    model = create_model(sequence_length, X.shape[2])
    
    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # 训练模型
    logger.info("开始训练模型...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估模型
    logger.info("评估模型性能...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    logger.info(f"测试集损失: {test_loss:.4f}")
    logger.info(f"测试集准确率: {test_accuracy:.4f}")
    
    # 保存模型
    model.save('momentum_strategy_model.h5')
    logger.info("模型已保存为 'momentum_strategy_model.h5'")

if __name__ == "__main__":
    train_strategy() 