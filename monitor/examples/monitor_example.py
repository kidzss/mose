import json
import time
from pathlib import Path
from monitor.portfolio_monitor import PortfolioMonitor

def main():
    # 加载配置文件
    config_path = Path(__file__).parent.parent / "configs" / "portfolio_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # 初始化投资组合监控器
    monitor = PortfolioMonitor(
        positions=config["positions"],
        monitor_config=config["monitor_config"]
    )

    try:
        while True:
            # 更新持仓信息
            monitor.update_positions()

            # 检查警报
            alerts = monitor.check_alerts()
            if alerts:
                print("\nNew Alerts:")
                for alert in alerts:
                    print(f"- {alert}")

            # 生成投资组合报告
            report = monitor.generate_portfolio_report()
            print("\nPortfolio Report:")
            print(report)

            # 分析风险
            risk_analysis = monitor.analyze_portfolio_risk()
            print("\nRisk Analysis:")
            print(risk_analysis)

            # 获取建议
            recommendations = monitor.get_portfolio_recommendations()
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"- {rec}")

            # 等待下一次检查
            time.sleep(config["monitor_config"]["check_interval"])

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

if __name__ == "__main__":
    main() 