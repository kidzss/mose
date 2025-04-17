import asyncio
import logging
from datetime import datetime, timedelta
from monitor.portfolio_monitor import PortfolioMonitor
from data.data_interface import YahooFinanceRealTimeSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_stop_loss_levels():
    # 初始化数据接口
    data_source = YahooFinanceRealTimeSource()
    
    # 获取当前持仓
    positions = {
        "GOOG": {"cost_basis": 170.478, "shares": 59, "stop_loss": 0.12},
        "TSLA": {"cost_basis": 289.434, "shares": 28, "stop_loss": 0.15},
        "AMD": {"cost_basis": 123.737, "shares": 58, "stop_loss": 0.08},
        "NVDA": {"cost_basis": 138.843, "shares": 40, "stop_loss": 0.08},
        "PFE": {"cost_basis": 25.899, "shares": 80, "stop_loss": 0.10},
        "MSFT": {"cost_basis": 370.95, "shares": 3, "stop_loss": 0.12},
        "TMDX": {"cost_basis": 101.75, "shares": 13, "stop_loss": 0.15}
    }
    
    # 获取实时价格
    symbols = list(positions.keys())
    current_prices = await data_source.get_realtime_data(symbols)
    
    # 计算每个股票的当前状态
    for symbol, data in positions.items():
        if symbol not in current_prices or current_prices[symbol].empty:
            logger.warning(f"无法获取 {symbol} 的实时数据")
            continue
            
        current_price = current_prices[symbol]['close'].iloc[-1]
        cost_basis = data['cost_basis']
        stop_loss = data['stop_loss']
        stop_loss_price = cost_basis * (1 - stop_loss)
        current_loss = (current_price - cost_basis) / cost_basis
        
        logger.info(f"\n{symbol} 状态:")
        logger.info(f"成本价: ${cost_basis:.2f}")
        logger.info(f"当前价: ${current_price:.2f}")
        logger.info(f"止损价: ${stop_loss_price:.2f}")
        logger.info(f"当前盈亏: {current_loss:.2%}")
        logger.info(f"距离止损点: {(current_price - stop_loss_price) / cost_basis:.2%}")
        
        if current_price <= stop_loss_price:
            logger.warning(f"⚠️ {symbol} 已达到止损点！")
        elif current_loss < -0.05:  # 亏损超过5%
            logger.warning(f"⚠️ {symbol} 接近止损点，当前亏损 {current_loss:.2%}")

if __name__ == "__main__":
    asyncio.run(check_stop_loss_levels()) 