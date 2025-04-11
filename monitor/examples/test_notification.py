import sys
import os
import datetime as dt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from monitor.notification_manager import NotificationManager
from data.data_interface import DataInterface
from data.data_validator import DataValidator

def test_notifications():
    """测试通知系统的各种功能"""
    print("开始测试通知系统...")
    
    # 初始化组件
    notification_manager = NotificationManager()
    data_interface = DataInterface()
    data_validator = DataValidator()
    
    # 获取实时数据
    symbols = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN"]
    print("获取实时市场数据...")
    
    # 获取最新的市场数据
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=60)  # 获取最近60天的数据以确保有足够的数据点
    market_data = {}
    
    for symbol in symbols:
        try:
            # 获取历史数据
            df = data_interface.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                # 验证数据
                validated_df, report = data_validator.validate_data(df)
                if report['validation_passed']:
                    print(f"获取到 {symbol} 的有效数据")
                    market_data[symbol] = validated_df.iloc[-1].to_dict()  # 获取最新一条数据
                    # 添加前一天的数据用于计算变化
                    if len(validated_df) > 1:
                        market_data[symbol]['prev_close'] = validated_df.iloc[-2]['close']
                        market_data[symbol]['prev_volume'] = validated_df.iloc[-2]['volume']
                        # 添加技术指标数据
                        for col in validated_df.columns:
                            if col not in ['open', 'high', 'low', 'close', 'volume', 'prev_close', 'prev_volume']:
                                market_data[symbol][col] = validated_df.iloc[-1][col]
                else:
                    print(f"警告: {symbol} 的数据验证未通过")
            else:
                print(f"警告: 未能获取到 {symbol} 的数据")
        except Exception as e:
            print(f"获取 {symbol} 数据时出错: {e}")
    
    if not market_data:
        print("未能获取到任何市场数据，测试终止")
        return
    
    # 1. 测试交易信号提醒
    if "AAPL" in market_data:
        print("测试交易信号提醒...")
        data = market_data["AAPL"]
        notification_manager.send_trade_signal(
            stock="AAPL",
            action="买入",
            price=data["close"],
            reason=f"技术指标显示买入信号，RSI: {data.get('RSI', 'N/A')}",
            confidence=0.85
        )
    
    # 2. 测试市场状况提醒
    print("测试市场状况提醒...")
    try:
        market_status = {
            "market_condition": "震荡上行",
            "risk_level": "中等",
            "opportunity_sectors": ["科技", "新能源", "医疗健康"],
            "trend": "上升趋势",
            "recommendation": "逢低买入"
        }
        notification_manager.send_market_alert(market_status)
    except Exception as e:
        print(f"获取市场状况时出错: {e}")
    
    # 3. 测试波动性提醒
    if "TSLA" in market_data:
        print("测试波动性提醒...")
        data = market_data["TSLA"]
        try:
            # 计算波动率（使用当日高低价差）
            volatility = (data["high"] - data["low"]) / data["close"]
            # 使用过去的平均波动率
            avg_volatility = volatility * 0.8  # 假设当前波动率高于平均水平
            
            notification_manager.send_volatility_alert(
                stock="TSLA",
                volatility=volatility,
                avg_volatility=avg_volatility,
                additional_info={
                    "成交量": data["volume"],
                    "当前价格": data["close"],
                    "RSI": data.get("RSI", "N/A"),
                    "建议": "注意风险，考虑减仓" if volatility > avg_volatility * 1.5 else "保持观察"
                }
            )
        except Exception as e:
            print(f"处理TSLA波动性数据时出错: {e}")
    
    # 4. 测试价格变动提醒
    if "NVDA" in market_data:
        print("测试价格变动提醒...")
        data = market_data["NVDA"]
        try:
            if "prev_close" in data and "prev_volume" in data:
                price_change = (data["close"] - data["prev_close"]) / data["prev_close"]
                volume_change = (data["volume"] - data["prev_volume"]) / data["prev_volume"]
                notification_manager.send_price_alert(
                    stock="NVDA",
                    current_price=data["close"],
                    prev_price=data["prev_close"],
                    volume_change=volume_change
                )
        except Exception as e:
            print(f"处理NVDA价格数据时出错: {e}")
    
    # 5. 测试风险提醒
    if "AMD" in market_data:
        print("测试风险提醒...")
        data = market_data["AMD"]
        try:
            # 风险分析
            price_change = (data["close"] - data.get("prev_close", data["close"])) / data.get("prev_close", data["close"])
            volume_change = (data["volume"] - data.get("prev_volume", data["volume"])) / data.get("prev_volume", data["volume"])
            rsi = data.get("RSI", 50)
            
            risk_level = "低"
            risk_factors = []
            
            if abs(price_change) > 0.05:
                risk_level = "高"
                risk_factors.append(f"价格变动幅度较大: {price_change:.1%}")
            if volume_change > 2:
                risk_level = "高"
                risk_factors.append(f"成交量异常放大: {volume_change:.1%}")
            if rsi > 70:
                risk_level = "高"
                risk_factors.append(f"RSI超买: {rsi:.1f}")
            elif rsi < 30:
                risk_level = "高"
                risk_factors.append(f"RSI超卖: {rsi:.1f}")
                
            risk_details = {
                "description": "发现异常波动",
                "recommendation": "建议关注风险" if risk_level == "高" else "保持观察",
                "precautions": risk_factors
            }
            
            notification_manager.send_risk_alert(
                stock="AMD",
                risk_type="市场风险",
                risk_level=risk_level,
                details=risk_details
            )
        except Exception as e:
            print(f"处理AMD风险数据时出错: {e}")
    
    # 6. 测试批量提醒
    print("测试批量提醒...")
    batch_alerts = []
    for symbol, data in market_data.items():
        try:
            if "prev_close" in data:
                price_change = (data["close"] - data["prev_close"]) / data["prev_close"]
                if abs(price_change) > 0.03:  # 价格变动超过3%
                    batch_alerts.append({
                        "type": "price_alert",
                        "message": f"{symbol} 价格{('上涨' if price_change > 0 else '下跌')}{abs(price_change):.1%}"
                    })
            if "prev_volume" in data:
                volume_change = (data["volume"] - data["prev_volume"]) / data["prev_volume"]
                if volume_change > 1.0:  # 成交量翻倍
                    batch_alerts.append({
                        "type": "volume_alert",
                        "message": f"{symbol} 成交量放大{volume_change:.1%}"
                    })
        except Exception as e:
            print(f"处理{symbol}批量提醒数据时出错: {e}")
    
    if batch_alerts:
        notification_manager.send_batch_alerts(batch_alerts)
    
    print("通知系统测试完成！")

if __name__ == "__main__":
    test_notifications() 