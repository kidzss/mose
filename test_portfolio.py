import pandas as pd
import numpy as np
from backtest.backtest_engine import BacktestEngine
from strategy.combined_strategy import CombinedStrategy
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_from_mysql(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从MySQL加载数据"""
    try:
        # 连接MySQL数据库
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='mose'
        )
        
        # 查询数据
        query = f"""
        SELECT * FROM stock_code_time 
        WHERE Code = '{symbol}' 
        AND Date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY Date
        """
        
        # 读取数据到DataFrame
        df = pd.read_sql(query, conn)
        
        # 关闭连接
        conn.close()
        
        # 转换列名为小写
        df.columns = [col.lower() for col in df.columns]
        
        # 设置日期为索引
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"从MySQL加载数据失败: {str(e)}")
        return pd.DataFrame()

def main():
    """主函数"""
    try:
        # 设置时间范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # 从MySQL加载数据
        df = load_data_from_mysql('GOOG', start_date, end_date)
        if df.empty:
            logger.error("获取数据失败")
            return
            
        # 创建组合策略实例
        strategy = CombinedStrategy()
        
        # 创建回测引擎
        engine = BacktestEngine(df)
        engine.strategy = strategy
        
        # 运行回测
        engine.run()
        
        # 分析结果
        logger.info("\n=== 组合策略分析结果 ===")
        logger.info(f"\n1. 性能指标:")
        logger.info(f"总收益率: {engine.metrics['total_return']:.2%}")
        logger.info(f"年化收益率: {engine.metrics['annual_return']:.2%}")
        logger.info(f"夏普比率: {engine.metrics['sharpe_ratio']:.2f}")
        logger.info(f"最大回撤: {engine.metrics['max_drawdown']:.2%}")
        logger.info(f"总交易次数: {engine.metrics['total_trades']}")
        
        # 打印策略权重
        metadata = strategy.get_signal_metadata()
        logger.info("\n2. 策略权重:")
        for name, weight in metadata['weights'].items():
            logger.info(f"{name}: {weight:.2%}")
        
    except Exception as e:
        logger.error(f"运行组合策略时出错: {str(e)}")

if __name__ == "__main__":
    main() 