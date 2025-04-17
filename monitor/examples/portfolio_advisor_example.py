import asyncio
import json
import logging
from monitor.portfolio_advisor import PortfolioAdvisor
from monitor.notification_manager import NotificationManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    # 创建顾问实例
    advisor = PortfolioAdvisor()
    notification_manager = NotificationManager()
    
    try:
        # 加载持仓和观察列表
        with open('monitor/configs/watchlist.json', 'r') as f:
            config = json.load(f)
            
        # 加载持仓信息
        with open('monitor/configs/positions.json', 'r') as f:
            positions = json.load(f)
            
        # 生成建议
        advice = await advisor.generate_portfolio_advice(
            positions=positions,
            watchlist=config['watchlist']
        )
        
        # 生成HTML报告
        html_report = generate_html_report(advice)
        
        # 发送通知
        notification_manager.send_batch_alerts([{
            'type': 'portfolio_advice',
            'message': html_report
        }])
        
        # 打印建议摘要
        print("\n=== 投资组合建议摘要 ===")
        print(f"生成时间: {advice['timestamp']}")
        print(f"市场概况: {advice['market_overview']}")
        
        print("\n持仓建议:")
        for item in advice['position_advice']:
            print(f"- {item['symbol']}: {item['action']['type']} ({item['action']['urgency']}) - {', '.join(item['action']['reason'])}")
            
        print("\n观察列表建议:")
        for item in advice['watchlist_advice']:
            print(f"- {item['symbol']}: {item['action']['type']} ({item['action']['urgency']}) - {', '.join(item['action']['reason'])}")
            
        print("\n投资组合级别建议:")
        for action in advice['portfolio_actions']:
            print(f"- [{action['urgency']}] {action['message']}")
            if 'symbols' in action:
                print(f"  涉及股票: {', '.join(action['symbols'])}")
            if 'sell_symbols' in action:
                print(f"  建议卖出: {', '.join(action['sell_symbols'])}")
                print(f"  建议买入: {', '.join(action['buy_symbols'])}")
                
    except Exception as e:
        logger.error(f"生成投资建议时出错: {str(e)}")
        
def generate_html_report(advice):
    """生成HTML格式的报告"""
    html = """
    <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; }
                .card {
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }
                .high { border-left: 4px solid #f44336; }
                .medium { border-left: 4px solid #ff9800; }
                .low { border-left: 4px solid #4CAF50; }
                .header {
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .metric { margin: 5px 0; }
                .action {
                    margin-top: 10px;
                    padding: 5px;
                    background-color: #f5f5f5;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <h1>投资组合分析报告</h1>
            <p>生成时间: {}</p>
            
            <div class="section">
                <h2>市场概况</h2>
                <p>{}</p>
            </div>
    """.format(advice['timestamp'], advice['market_overview'])
    
    # 添加投资组合级别建议
    html += """
        <div class="section">
            <h2>投资组合建议</h2>
    """
    
    for action in advice['portfolio_actions']:
        html += f"""
            <div class="card {action['urgency']}">
                <div class="header">{action['message']}</div>
        """
        if 'symbols' in action:
            html += f"<div class='metric'>涉及股票: {', '.join(action['symbols'])}</div>"
        if 'sell_symbols' in action:
            html += f"""
                <div class='metric'>建议卖出: {', '.join(action['sell_symbols'])}</div>
                <div class='metric'>建议买入: {', '.join(action['buy_symbols'])}</div>
            """
        html += "</div>"
        
    # 添加持仓建议
    html += """
        <div class="section">
            <h2>持仓建议</h2>
    """
    
    for item in advice['position_advice']:
        html += f"""
            <div class="card {item['action']['urgency']}">
                <div class="header">{item['symbol']}</div>
                <div class="metric">当前价格: ${item['current_price']:.2f}</div>
                <div class="metric">盈亏: {item['pnl']:.2%}</div>
                <div class="metric">综合评分: {item['score']:.2f}</div>
                <div class="action">
                    <strong>建议操作: {item['action']['type']}</strong><br>
                    原因: {', '.join(item['action']['reason'])}
                </div>
            </div>
        """
        
    # 添加观察列表建议
    html += """
        <div class="section">
            <h2>观察列表建议</h2>
    """
    
    for item in advice['watchlist_advice']:
        html += f"""
            <div class="card {item['action']['urgency']}">
                <div class="header">{item['symbol']}</div>
                <div class="metric">当前价格: ${item['current_price']:.2f}</div>
                <div class="metric">综合评分: {item['score']:.2f}</div>
                <div class="action">
                    <strong>建议操作: {item['action']['type']}</strong><br>
                    原因: {', '.join(item['action']['reason'])}
                </div>
            </div>
        """
        
    html += """
        </body>
    </html>
    """
    
    return html
    
if __name__ == "__main__":
    asyncio.run(main()) 