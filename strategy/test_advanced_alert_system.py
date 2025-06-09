import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
import logging
from pprint import pprint

# 添加项目根目录到sys.path以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alert_system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlertSystemTest")

# 确保可以找到相关模块
logger.info(f"当前Python路径: {sys.path}")
logger.info(f"当前工作目录: {os.getcwd()}")
logger.info(f"脚本位置: {os.path.abspath(__file__)}")

# 导入系统相关模块
try:
    from strategy.market_environment_classifier import MarketEnvironment, MarketEnvironmentClassifier
    from strategy.dynamic_strategy_selector import DynamicStrategySelector
    from strategy.signal_quality_evaluator import SignalQualityEvaluator, SignalStrength
    from strategy.advanced_alert_system import AdvancedAlertSystem, AlertLevel, AlertCategory
    
    logger.info("成功导入所有必要模块")
except ImportError as e:
    logger.error(f"模块导入失败: {e}", exc_info=True)
    sys.exit(1)

def load_test_data(symbol='AAPL', lookback_days=200):
    """加载测试数据"""
    try:
        # 由于无法连接到数据库，创建模拟测试数据
        logger.info(f"创建 {symbol} 的模拟测试数据")
        
        # 创建日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 使用工作日
        
        # 随机生成价格序列
        np.random.seed(42)  # 固定随机种子以获得可重复的结果
        
        # 起始价格
        start_price = 150.0  # 模拟AAPL的价格范围
        
        # 创建一个随机游走的价格序列
        price_changes = np.random.normal(0.001, 0.02, len(date_range))  # 均值为0.1%，标准差为2%
        prices = start_price * np.cumprod(1 + price_changes)
        
        # 生成OHLCV数据
        close_prices = prices
        high_prices = close_prices * np.random.uniform(1.0, 1.05, len(date_range))
        low_prices = close_prices * np.random.uniform(0.95, 1.0, len(date_range))
        open_prices = low_prices + np.random.uniform(0, 1, len(date_range)) * (high_prices - low_prices)
        volumes = np.random.normal(1000000, 300000, len(date_range))
        volumes = np.abs(volumes).astype(int)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=date_range)
        
        logger.info(f"成功生成 {symbol} 的模拟数据，共 {len(df)} 条记录")
        return df
        
    except Exception as e:
        logger.error(f"创建模拟数据时出错: {str(e)}", exc_info=True)
        return None

def calculate_technical_indicators(data):
    """计算技术指标"""
    logger.info("开始计算技术指标")
    
    if data is None or len(data) == 0:
        logger.error("无法计算技术指标：数据为空")
        return None
        
    try:
        df = data.copy()
        
        # 移动平均线
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 波动率 (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(window=14).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = high_low
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
        minus_di = abs(100 * (minus_dm.rolling(14).sum() / atr))
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        logger.info("技术指标计算完成")
        
        # 检查是否有NaN值
        nan_check = df.isna().sum().sum()
        if nan_check > 0:
            logger.warning(f"计算出的技术指标中存在 {nan_check} 个NaN值")
            # 删除NaN值，但仅保留最后200行数据
            df = df.iloc[-200:].dropna()
            logger.info(f"删除NaN值后，剩余 {len(df)} 条记录")
        
        # 确保我们至少有60行数据用于测试
        if len(df) < 60:
            logger.error("数据量不足，至少需要60行有效数据")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"计算技术指标时出错: {str(e)}", exc_info=True)
        return None

def test_market_environment_classifier(data):
    """测试市场环境分类器"""
    logger.info("=== 测试市场环境分类器 ===")
    
    try:
        classifier = MarketEnvironmentClassifier()
        logger.info("成功创建市场环境分类器实例")
        
        result = classifier.classify_environment(data)
        logger.info(f"市场环境分类结果: {result['environment'].value}")
        logger.info(f"分类置信度: {result['confidence']:.2f}")
        
        if 'details' in result and 'reasons' in result['details']:
            logger.info("分类原因:")
            for i, reason in enumerate(result['details']['reasons'][:3]):  # 显示前3个原因
                logger.info(f"  {i+1}. {reason}")
                
        return result
        
    except Exception as e:
        logger.error(f"市场环境分类时出错: {str(e)}", exc_info=True)
        return None

def test_dynamic_strategy_selector(data):
    """测试动态策略选择器"""
    logger.info("=== 测试动态策略选择器 ===")
    
    try:
        selector = DynamicStrategySelector()
        logger.info("成功创建动态策略选择器实例")
        
        result = selector.get_best_strategy(data)
        logger.info(f"选择的主策略: {result['primary_strategy']}")
        logger.info(f"市场环境: {result['environment'].value}")
        logger.info(f"环境置信度: {result['confidence']:.2f}")
        
        if 'strategy_weights' in result:
            logger.info("策略权重分配:")
            for strategy, weight in result['strategy_weights'].items():
                logger.info(f"  - {strategy}: {weight:.2f}")
                
        return result
        
    except Exception as e:
        logger.error(f"策略选择时出错: {str(e)}", exc_info=True)
        return None

def test_signal_quality_evaluator(data, market_environment):
    """测试信号质量评估器"""
    logger.info("=== 测试信号质量评估器 ===")
    
    try:
        # 创建模拟信号
        current_price = data['close'].iloc[-1]
        logger.info(f"当前价格: {current_price:.2f}")
        
        signal_data = {
            'direction': 1,  # 1=买入, -1=卖出
            'entry_price': current_price,
            'stop_loss': current_price * 0.95,  # 5%止损
            'target_price': current_price * 1.15,  # 15%目标
            'indicator_signals': {
                'macd': 1,
                'rsi': 1,
                'sma_crossover': 1,
                'adx': 1,
                'bollinger_bands': 0
            }
        }
        logger.info(f"创建模拟买入信号: 入场价={current_price:.2f}, 止损={signal_data['stop_loss']:.2f}, 目标={signal_data['target_price']:.2f}")
        
        evaluator = SignalQualityEvaluator()
        logger.info("成功创建信号质量评估器实例")
        
        # 如果市场环境是UNKNOWN，使用RANGE_BOUND作为默认值进行测试
        if market_environment == MarketEnvironment.UNKNOWN:
            market_environment = MarketEnvironment.RANGE_BOUND
            logger.info(f"市场环境未知，使用 {market_environment.value} 进行测试")
        
        result = evaluator.evaluate_signal(signal_data, data, market_environment)
        logger.info(f"信号质量分数: {result['quality_score']:.2f}")
        logger.info(f"信号强度: {result['strength'].value}")
        logger.info(f"是否通过质量阈值: {result['passed_threshold']}")
        
        if 'dimension_scores' in result:
            logger.info("各维度评分:")
            for dimension, score in result['dimension_scores'].items():
                logger.info(f"  - {dimension}: {score:.2f}")
        
        if 'recommendations' in result:
            logger.info("评估建议:")
            for rec in result['recommendations']:
                logger.info(f"  - {rec}")
                
        return result
        
    except Exception as e:
        logger.error(f"信号质量评估时出错: {str(e)}", exc_info=True)
        return None

def test_advanced_alert_system(data, symbol='AAPL'):
    """测试高级提醒系统"""
    logger.info("=== 测试高级提醒系统 ===")
    
    try:
        # 创建模拟信号
        current_price = data['close'].iloc[-1]
        signal_data = {
            'direction': 1,  # 1=买入, -1=卖出
            'entry_price': current_price,
            'stop_loss': current_price * 0.95,  # 5%止损
            'target_price': current_price * 1.15,  # 15%目标
            'indicator_signals': {
                'macd': 1,
                'rsi': 1,
                'sma_crossover': 1,
                'adx': 1,
                'bollinger_bands': 0
            }
        }
        
        # 初始化提醒系统
        alert_system = AdvancedAlertSystem({
            'enable_notification_manager': False  # 测试时关闭通知，避免发送实际提醒
        })
        logger.info("成功创建高级提醒系统实例")
        
        # 1. 测试处理市场数据
        logger.info("处理市场数据...")
        market_alerts = alert_system.process_market_data(symbol, data)
        logger.info(f"生成了 {len(market_alerts)} 个市场提醒")
        
        if market_alerts:
            logger.info("市场提醒示例:")
            alert = market_alerts[0]
            logger.info(f"  标题: {alert['title']}")
            logger.info(f"  级别: {alert['level'].value}")
            logger.info(f"  类别: {alert['category'].value}")
            
        # 2. 测试处理交易信号
        logger.info("处理交易信号...")
        signal_result = alert_system.process_trading_signal(symbol, signal_data, data)
        
        logger.info(f"信号通过评估: {signal_result['passed']}")
        logger.info(f"信号质量分数: {signal_result['quality_score']:.2f}")
        logger.info(f"信号强度: {signal_result['strength'].value}")
        logger.info(f"生成提醒: {signal_result['alert_generated']}")
        
        if signal_result['alert_generated'] and 'alert' in signal_result:
            alert = signal_result['alert']
            logger.info(f"  标题: {alert['title']}")
            logger.info(f"  级别: {alert['level'].value}")
            logger.info("  消息示例:")
            message_lines = alert['message'].split('\n')[:5]
            for line in message_lines:
                logger.info(f"    {line}")
        
        # 3. 获取市场摘要
        logger.info("获取市场摘要...")
        summary = alert_system.get_market_summary(symbol, data)
        
        logger.info(f"当前价格: {summary['current_price']:.2f}")
        logger.info(f"价格变化: {summary['price_change']:.2f}%")
        logger.info(f"市场环境: {summary['environment'].value}")
        logger.info(f"环境置信度: {summary['environment_confidence']:.2f}")
        logger.info(f"主策略: {summary['primary_strategy']}")
        
        return {
            'market_alerts': market_alerts,
            'signal_result': signal_result,
            'market_summary': summary
        }
        
    except Exception as e:
        logger.error(f"测试高级提醒系统时出错: {str(e)}", exc_info=True)
        return None

def visualize_classification_result(data, environment_result):
    """可视化市场环境分类结果"""
    try:
        logger.info("开始生成市场环境分类可视化图表")
        
        # 设置中文字体支持
        import matplotlib
        import platform
        
        # 根据操作系统设置合适的中文字体
        system = platform.system()
        if system == "Windows":
            # Windows系统常用中文字体
            font_list = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi']
        elif system == "Darwin":  # macOS
            font_list = ['PingFang SC', 'Heiti SC', 'STHeiti']
        else:  # Linux
            font_list = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        
        # 尝试设置字体
        font_set = False
        for font_name in font_list:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                font_set = True
                logger.info(f"成功设置字体: {font_name}")
                break
            except:
                continue
        
        if not font_set:
            logger.warning("无法设置中文字体，使用英文显示")
            # 使用英文标签
            labels = {
                'close_price': 'Close Price',
                'ma_20': '20-day MA',
                'ma_50': '50-day MA', 
                'ma_200': '200-day MA',
                'market_env': 'Market Environment',
                'confidence': 'Confidence',
                'rsi_indicator': 'RSI Indicator'
            }
        else:
            # 使用中文标签
            labels = {
                'close_price': '收盘价',
                'ma_20': '20日均线',
                'ma_50': '50日均线',
                'ma_200': '200日均线',
                'market_env': '市场环境',
                'confidence': '置信度',
                'rsi_indicator': 'RSI指标'
            }
        
        plt.figure(figsize=(12, 8))
        
        # 绘制价格图
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['close'], label=labels['close_price'], linewidth=2)
        
        if 'sma_20' in data.columns:
            plt.plot(data.index, data['sma_20'], label=labels['ma_20'], alpha=0.7)
        if 'sma_50' in data.columns:
            plt.plot(data.index, data['sma_50'], label=labels['ma_50'], alpha=0.7)
        if 'sma_200' in data.columns:
            plt.plot(data.index, data['sma_200'], label=labels['ma_200'], alpha=0.7)
        
        # 添加环境标签
        env_name = environment_result['environment'].value if font_set else environment_result['environment'].name
        confidence = environment_result['confidence']
        
        if not np.isnan(confidence):
            title = f"{labels['market_env']}: {env_name} ({labels['confidence']}: {confidence:.2f})"
        else:
            title = f"{labels['market_env']}: {env_name}"
            
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制技术指标
        plt.subplot(2, 1, 2)
        if 'rsi' in data.columns:
            plt.plot(data.index, data['rsi'], label=labels['rsi_indicator'], color='purple')
            plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买线')
            plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='超卖线')
            plt.ylim(0, 100)
        
        plt.title(labels['rsi_indicator'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('market_classification_result.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        logger.info("已保存分类结果图表到 market_classification_result.png")
        
    except Exception as e:
        logger.error(f"生成可视化图表时出错: {str(e)}", exc_info=True)
    
def main():
    logger.info("=== 股票预测与提醒系统测试开始 ===")
    
    # 1. 加载测试数据
    symbol = 'AAPL'  # 可以修改为其他股票代码
    logger.info(f"测试股票代码: {symbol}")
    raw_data = load_test_data(symbol=symbol, lookback_days=400)  # 增加历史数据量
    
    if raw_data is None:
        logger.error("测试数据加载失败，测试终止")
        return
    
    # 2. 计算技术指标
    data = calculate_technical_indicators(raw_data)
    
    if data is None or len(data) < 60:
        logger.error("技术指标计算失败或数据量不足，测试终止")
        return
        
    logger.info(f"最终可用数据: {len(data)} 行")
    
    # 3. 测试市场环境分类
    environment_result = test_market_environment_classifier(data)
    if environment_result is None:
        logger.error("市场环境分类失败，测试终止")
        return
        
    # 4. 测试策略选择
    strategy_result = test_dynamic_strategy_selector(data)
    if strategy_result is None:
        logger.error("策略选择测试失败，测试终止")
        return
    
    # 5. 测试信号质量评估
    evaluation_result = test_signal_quality_evaluator(data, environment_result['environment'])
    if evaluation_result is None:
        logger.error("信号质量评估测试失败，测试终止")
        return
    
    # 6. 测试高级提醒系统
    alert_results = test_advanced_alert_system(data, symbol)
    if alert_results is None:
        logger.error("高级提醒系统测试失败，测试终止")
        return
    
    # 7. 可视化市场环境分类结果
    visualize_classification_result(data, environment_result)
    
    logger.info("=== 测试完成! ===")

if __name__ == "__main__":
    main() 