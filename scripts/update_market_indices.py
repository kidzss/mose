import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 市场指标配置
MARKET_INDICATORS = {
    'volatility': {
        'primary': '^VIX',  # VIX指数
        'alternatives': ['VXX', 'UVXY']  # 备选数据源
    },
    'broad_market': {
        'primary': 'SPY',  # S&P 500 ETF
        'alternatives': ['^GSPC', 'VOO']  # S&P 500指数和另一个ETF
    },
    'tech_market': {
        'primary': 'QQQ',  # 纳斯达克100 ETF
        'alternatives': ['^NDX', 'TQQQ']  # 纳斯达克100指数和杠杆ETF
    },
    'treasury': {
        'primary': 'TLT',  # 20年期国债ETF
        'alternatives': ['IEF', '^TNX']  # 10年期国债ETF和收益率
    },
    'sector_etfs': {
        'technology': 'XLK',
        'financials': 'XLF',
        'energy': 'XLE',
        'healthcare': 'XLV',
        'industrials': 'XLI',
        'consumer_discretionary': 'XLY',
        'consumer_staples': 'XLP',
        'materials': 'XLB',
        'utilities': 'XLU',
        'real_estate': 'XLRE'
    },
    'market_breadth': {
        'advance_decline': '^ADVN',  # 上涨/下跌股票数量
        'new_highs': '^NHNL',  # 新高/新低股票数量
        'arms_index': '^TRIN'  # ARMS指数
    },
    'sentiment': {
        'put_call_ratio': '^PCR',  # 看跌/看涨期权比率
        'fear_greed': '^VIX'  # 使用VIX作为恐慌指标
    }
}

def get_db_connection():
    """获取数据库连接"""
    return create_engine('mysql+pymysql://root@localhost/mose')

def validate_market_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """验证市场数据的质量
    
    Args:
        df: 待验证的数据框
        
    Returns:
        (is_valid, message): 数据是否有效及相关信息
    """
    try:
        # 检查是否有空值
        if df.isnull().sum().sum() > 0:
            return False, "数据包含空值"
            
        # 检查价格是否为负
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (df[col] < 0).any():
                return False, f"{col}列包含负值"
                
        # 检查成交量是否为负
        if (df['Volume'] < 0).any():
            return False, "成交量包含负值"
            
        # 检查价格逻辑
        invalid_price = (df['High'] < df['Low']) | (df['Open'] > df['High']) | (df['Open'] < df['Low']) | \
                       (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        if invalid_price.any():
            return False, "价格数据逻辑错误"
            
        # 检查日期连续性
        dates = pd.to_datetime(df['Date'])
        date_diff = dates.diff().dropna()
        if date_diff.max().days > 5:  # 允许最多5天的gap（考虑节假日）
            return False, "日期数据不连续"
            
        return True, "数据验证通过"
        
    except Exception as e:
        return False, f"数据验证过程出错: {str(e)}"

def get_latest_date(engine, symbol):
    """获取数据库中最新的数据日期"""
    try:
        query = """
        SELECT MAX(Date) as latest_date
        FROM stock_time_code
        WHERE Code = %s
        """
        result = pd.read_sql_query(query, engine, params=(symbol,))
        return result['latest_date'].iloc[0]
    except Exception as e:
        logger.error(f"获取{symbol}最新日期时出错: {e}")
        return None

def get_market_indicator_data(symbol: str, alternatives: List[str], start_date: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """获取市场指标数据，如果主要数据源失败则尝试备选数据源
    
    Args:
        symbol: 主要数据源代码
        alternatives: 备选数据源列表
        start_date: 起始日期
        
    Returns:
        (dataframe, actual_symbol): 数据框和实际使用的代码
    """
    try:
        # 首先尝试主要数据源
        logger.info(f"尝试获取{symbol}数据...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date)
        if not df.empty:
            logger.info(f"成功获取{symbol}数据")
            return df, symbol
            
        # 如果主要数据源失败，尝试备选数据源
        for alt_symbol in alternatives:
            logger.info(f"{symbol}数据获取失败，尝试获取{alt_symbol}数据...")
            ticker = yf.Ticker(alt_symbol)
            df = ticker.history(start=start_date)
            if not df.empty:
                logger.info(f"成功获取{alt_symbol}数据")
                return df, alt_symbol
                
        logger.error(f"{symbol}及其所有备选数据源都获取失败")
        return None, None
        
    except Exception as e:
        logger.error(f"获取{symbol}数据时出错: {e}")
        return None, None

def calculate_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算衍生指标
    
    Args:
        df: 原始数据框
        
    Returns:
        添加了衍生指标的数据框
    """
    try:
        # 计算日收益率
        df['Daily_Return'] = df['Close'].pct_change()
        
        # 计算波动率（20日）
        annualization_factor = np.full(len(df), np.sqrt(252))
        df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std().mul(annualization_factor)
        
        # 计算移动平均线
        for window in [5, 10, 20, 50, 200]:
            df[f'MA_{window}D'] = df['Close'].rolling(window=window).mean()
            
        # 计算相对强弱指标（RSI）
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14D'] = 100 - (100 / (1 + rs))
        
        # 计算成交量变化
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # 计算布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        
        return df
        
    except Exception as e:
        logger.error(f"计算衍生指标时出错: {e}")
        return df

def update_market_data(symbol: str, engine, alternatives: List[str] = None):
    """更新单个市场指数数据"""
    try:
        # 获取最新数据日期
        latest_date = get_latest_date(engine, symbol)
        
        # 设置开始日期
        if latest_date is None:
            start_date = '2015-01-01'  # 如果没有数据，从2015年开始
        else:
            start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
        # 如果已经是最新的，跳过
        if latest_date and latest_date >= datetime.now().date() - timedelta(days=1):
            logger.info(f"{symbol} 数据已经是最新的")
            return True
            
        # 获取数据
        df, actual_symbol = get_market_indicator_data(symbol, alternatives or [], start_date)
        if df is None:
            return False
            
        # 准备数据
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Dividends': 'Dividends',
            'Stock Splits': 'StockSplits',
            'Capital Gains': 'Capital_Gains'
        })
        
        # 添加代码列
        df['Code'] = actual_symbol
        
        # 转换日期格式
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # 验证数据
        is_valid, message = validate_market_data(df)
        if not is_valid:
            logger.error(f"{actual_symbol}数据验证失败: {message}")
            return False
            
        # 计算衍生指标
        df = calculate_derived_indicators(df)
        
        # 确保所有必需的列都存在
        required_columns = ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 
                          'Dividends', 'StockSplits', 'Capital_Gains']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0 if col not in ['Date', 'Code'] else None
        
        # 只选择需要的列
        df = df[required_columns]
        
        # 写入数据库
        df.to_sql('stock_time_code', engine, if_exists='append', index=False)
        
        logger.info(f"成功更新 {actual_symbol} 的数据，新增 {len(df)} 条记录")
        return True
        
    except Exception as e:
        logger.error(f"更新 {symbol} 数据时出错: {e}")
        return False

def main():
    """主函数"""
    # 获取数据库连接
    engine = get_db_connection()
    
    # 更新每个市场指标
    for category, indicators in MARKET_INDICATORS.items():
        logger.info(f"开始更新 {category} 类别的数据")
        
        if isinstance(indicators, dict) and 'primary' in indicators:
            # 处理带有备选数据源的指标
            symbol = indicators['primary']
            alternatives = indicators['alternatives']
            logger.info(f"正在更新 {symbol}")
            success = update_market_data(symbol, engine, alternatives)
            if success:
                logger.info(f"{symbol} 更新成功")
            else:
                logger.warning(f"{symbol} 更新失败")
        elif isinstance(indicators, dict):
            # 处理分类指标（如行业ETF）
            for name, symbol in indicators.items():
                logger.info(f"正在更新 {name}: {symbol}")
                success = update_market_data(symbol, engine)
                if success:
                    logger.info(f"{symbol} 更新成功")
                else:
                    logger.warning(f"{symbol} 更新失败")
                time.sleep(1)  # 避免请求过于频繁
                
        time.sleep(1)  # 在不同类别之间添加延迟
            
    logger.info("所有市场指标数据更新完成")

if __name__ == "__main__":
    main() 