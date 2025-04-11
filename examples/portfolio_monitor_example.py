from monitor.portfolio_monitor import PortfolioMonitor
import pandas as pd
from datetime import datetime, timedelta

def main():
    # 创建示例投资组合数据
    portfolio_data = {
        'AAPL': {
            'cost_basis': 170.0,  # 购买成本
            'weight': 0.4        # 投资组合权重
        },
        'MSFT': {
            'cost_basis': 310.0,
            'weight': 0.35
        },
        'GOOGL': {
            'cost_basis': 125.0,
            'weight': 0.25
        }
    }
    
    # 监控配置
    monitor_config = {
        "price_alert_threshold": 0.02,    # 价格变动提醒阈值 (2%)
        "loss_alert_threshold": 0.05,     # 亏损提醒阈值 (5%)
        "profit_target": 0.10,            # 止盈目标 (10%)
        "stop_loss": 0.15,                # 止损线 (15%)
        "check_interval": 300,            # 检查间隔（秒）
        "email_notifications": False      # 暂时关闭邮件通知
    }
    
    # 初始化 PortfolioMonitor
    monitor = PortfolioMonitor(portfolio_data, monitor_config)
    
    print("=== 投资组合监控演示 ===")
    
    # 计算并显示投资组合价值
    portfolio_value = monitor.calculate_portfolio_value()
    print(f"\n1. 当前投资组合总价值: ${portfolio_value:,.2f}")
    
    # 计算并显示收益率
    returns = monitor.calculate_returns()
    print(f"\n2. 当前收益率:")
    for symbol, ret in returns.items():
        print(f"   {symbol}: {ret:.2%}")
    
    # 获取风险分析
    risk_metrics = monitor.analyze_portfolio_risk()
    print("\n3. 风险分析:")
    print(f"   VaR (95%): {risk_metrics['var_95']:.2%}")
    print("\n   波动率:")
    for symbol, vol in risk_metrics['volatility'].items():
        print(f"   {symbol}: {vol:.2%}")
    print("\n   集中度风险:")
    for symbol, conc in risk_metrics['concentration_risk'].items():
        print(f"   {symbol}: {conc:.2%}")
    
    # 获取警报
    alerts = monitor.check_alerts()
    print("\n4. 当前警报:")
    if alerts:
        for alert in alerts:
            print(f"   - {alert}")
    else:
        print("   没有需要注意的警报")
    
    # 生成投资组合报告
    print("\n5. 详细报告:")
    print(monitor.generate_portfolio_report())

if __name__ == "__main__":
    main() 