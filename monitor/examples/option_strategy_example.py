import asyncio
import logging
from monitor.option_strategy import OptionProtectionStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_option_strategy():
    """测试期权保护策略"""
    try:
        # 初始化策略
        strategy = OptionProtectionStrategy(symbol='SPY', portfolio_value=100000)
        
        # 生成交易信号
        signal = await strategy.generate_trading_signal()
        
        # 打印结果
        logger.info("\n市场分析:")
        if signal['details'] and isinstance(signal['details'], dict):
            market = signal['details'].get('market_analysis', {})
            logger.info(f"当前价格: ${market.get('current_price', 0):.2f}")
            logger.info(f"5日均线: ${market.get('sma_5', 0):.2f}")
            logger.info(f"10日均线: ${market.get('sma_10', 0):.2f}")
            logger.info(f"RSI指标: {market.get('rsi', 0):.1f}")
            logger.info(f"年化波动率: {market.get('volatility', 0):.1f}%")
            logger.info(f"价格是否低于关键点位(4900): {'是' if market.get('price_below_key_level') else '否'}")
            logger.info(f"价格是否低于均线: {'是' if market.get('price_below_ma') else '否'}")
            logger.info(f"波动率是否过高: {'是' if market.get('high_volatility') else '否'}")
            
        logger.info("\n交易信号:")
        logger.info(f"操作: {signal['action']}")
        logger.info(f"原因: {signal['reason']}")
        
        if signal['action'] == 'buy_put' and signal['details'].get('option_data'):
            opt = signal['details']['option_data']
            logger.info("\n期权详情:")
            logger.info(f"到期日: {opt['expiration']}")
            logger.info(f"执行价格: ${opt['strike']:.2f}")
            logger.info(f"期权价格: ${opt['last_price']:.2f}")
            logger.info(f"买入数量: {signal['details']['position_size']} 张合约")
            logger.info(f"总成本: ${signal['details']['total_cost']:.2f}")
            
            logger.info("\n风险管理:")
            risk = signal['details']['risk_management']
            logger.info(f"最大损失: ${risk['max_loss']:.2f}")
            logger.info(f"退出条件: {risk['exit_condition']}")
            logger.info(f"时间止损: {risk['time_stop']}")
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_option_strategy()) 