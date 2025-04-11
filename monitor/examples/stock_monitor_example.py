import time
import logging
import asyncio
from datetime import datetime
from monitor.stock_monitor import StockMonitor
from monitor.semiconductor_trading_strategy import SemiconductorTradingStrategy
from monitor.notification_system import NotificationSystem
from monitor.examples.market_bottom_analysis import MarketBottomAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def monitor_stocks():
    """监控股票"""
    stock_monitor = StockMonitor()
    semiconductor_strategy = SemiconductorTradingStrategy()
    market_analyzer = MarketBottomAnalyzer()
    notification_system = NotificationSystem()
    
    while True:
        try:
            # 更新持仓数据
            await stock_monitor.update_positions()
            
            # 检查警报
            alerts = await stock_monitor.check_alerts()
            
            # 分析半导体板块
            semiconductor_analysis = await semiconductor_strategy.analyze_semiconductor_sector()
            
            # 分析市场底部
            market_analysis = await market_analyzer.analyze_market_bottom()
            
            # 生成综合报告
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'alerts': alerts,
                'semiconductor_analysis': semiconductor_analysis,
                'market_analysis': market_analysis
            }
            
            # 发送邮件通知
            await notification_system.send_email(
                subject=f"股票监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                body=format_report(report),
                is_html=True
            )
            
            # 等待下一次检查
            await asyncio.sleep(300)  # 每5分钟检查一次
            
        except Exception as e:
            logger.error(f"监控过程中出错: {e}")
            await asyncio.sleep(60)  # 出错后等待1分钟再重试

def format_report(report):
    """格式化报告为HTML"""
    html = """<html>
<body style="font-family: Arial; margin: 20px;">
<div style="max-width: 1200px; margin: 0 auto;">
<h1>股票监控报告</h1>
<p>报告时间: {timestamp}</p>

<div style="margin-bottom: 30px;">
<h2>市场底部分析</h2>
<div style="padding: 10px; margin: 5px 0; border-radius: 4px; {market_style}">
<h3>市场状态: {market_state}</h3>
<p>风险等级: {risk_level}</p>
<p>信号: {signals}</p>
<p>RSI: {rsi}</p>
<p>MACD: {macd}</p>
<p>成交量变化: {volume_change}</p>
</div>
</div>

<div style="margin-bottom: 30px;">
<h2>半导体板块分析</h2>
<div style="padding: 10px; margin: 5px 0; border-radius: 4px; {sector_style}">
<h3>板块趋势: {sector_trend}</h3>
<p>健康度: {sector_health}</p>
<p>建议: {sector_recommendations}</p>
</div>

<h3>个股分析</h3>
<table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
<tr>
<th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd; background-color: #f2f2f2;">股票</th>
<th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd; background-color: #f2f2f2;">当前价格</th>
<th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd; background-color: #f2f2f2;">价格变化</th>
<th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd; background-color: #f2f2f2;">RSI</th>
<th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd; background-color: #f2f2f2;">MACD</th>
<th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd; background-color: #f2f2f2;">趋势</th>
<th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd; background-color: #f2f2f2;">建议</th>
</tr>
{semiconductor_stocks}
</table>
</div>

<div style="margin-bottom: 30px;">
<h2>波段分析</h2>
{wave_band_analysis}
</div>

<div style="margin-bottom: 30px;">
<h2>警报</h2>
{alerts}
</div>
</div>
</body>
</html>"""
    
    # 格式化市场分析部分
    market_analysis = report.get('market_analysis', {})
    market_state = "底部" if market_analysis.get('is_bottom', False) else "非底部"
    risk_level = market_analysis.get('risk_level', 'N/A')
    signals = ', '.join(market_analysis.get('signals', ['N/A']))
    rsi = market_analysis.get('rsi', 'N/A')
    macd = market_analysis.get('macd', 'N/A')
    volume_change = market_analysis.get('volume_change', 'N/A')
    
    # 格式化半导体分析部分
    semiconductor_analysis = report.get('semiconductor_analysis', {})
    sector_trend = semiconductor_analysis.get('trend', 'N/A')
    sector_health = semiconductor_analysis.get('sector_health', 0)
    sector_recommendations = ', '.join(semiconductor_analysis.get('recommendations', ['N/A']))
    
    # 格式化半导体股票表格
    semiconductor_stocks_html = ""
    for symbol, data in semiconductor_analysis.get('stocks', {}).items():
        # 安全地获取所有值，处理None值
        current_price = data.get('current_price')
        price_change = data.get('price_change', 'N/A')
        rsi_value = data.get('rsi')
        macd_status = data.get('macd_status', 'N/A')
        trend = data.get('trend', 'N/A')
        recommendations = data.get('recommendations', ['N/A'])
        
        # 格式化价格和RSI，处理None值
        current_price_str = f"${float(current_price):.2f}" if current_price is not None else 'N/A'
        rsi_str = f"{float(rsi_value):.2f}" if rsi_value is not None else 'N/A'
        
        semiconductor_stocks_html += f"<tr><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{symbol}</td><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{current_price_str}</td><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{price_change}</td><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{rsi_str}</td><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{macd_status}</td><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{trend}</td><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{', '.join(str(r) for r in recommendations)}</td></tr>"
    
    # 格式化波段分析部分
    wave_band_analysis_html = ""
    for symbol, data in semiconductor_analysis.get('stocks', {}).items():
        wave_band_alerts = data.get('wave_band_alerts', [])
        if wave_band_alerts:
            alerts_html = "".join([f'<div style="padding: 10px; margin: 5px 0; border-radius: 4px; {get_alert_style(alert.get("level", "info"))}">{alert.get("message", "N/A")}</div>' for alert in wave_band_alerts])
            wave_band_analysis_html += f'<div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 4px;"><h3>{symbol} 波段分析</h3>{alerts_html}</div>'
    
    # 格式化警报部分
    alerts_html = "".join([f'<div style="padding: 10px; margin: 5px 0; border-radius: 4px; {get_alert_style(alert.get("level", "info"))}">{alert.get("message", "N/A")}</div>' for alert in report.get('alerts', [])])
    
    # 获取样式
    market_style = get_alert_style({
        'high': 'danger',
        'medium': 'warning',
        'low': 'info'
    }.get(str(risk_level).lower(), 'info'))
    
    sector_style = get_alert_style({
        'bearish': 'danger',
        'neutral': 'warning',
        'bullish': 'success'
    }.get(str(sector_trend).lower(), 'info'))
    
    return html.format(
        timestamp=report.get('timestamp', 'N/A'),
        market_state=market_state,
        risk_level=risk_level,
        signals=signals,
        rsi=rsi,
        macd=macd,
        volume_change=volume_change,
        market_style=market_style,
        sector_style=sector_style,
        sector_trend=sector_trend,
        sector_health=sector_health,
        sector_recommendations=sector_recommendations,
        semiconductor_stocks=semiconductor_stocks_html,
        wave_band_analysis=wave_band_analysis_html,
        alerts=alerts_html
    )

def get_alert_style(level):
    """获取警报级别的样式"""
    styles = {
        'info': 'background-color: #e7f3fe; border-left: 6px solid #2196F3',
        'warning': 'background-color: #fff3cd; border-left: 6px solid #ffc107',
        'danger': 'background-color: #f8d7da; border-left: 6px solid #dc3545',
        'success': 'background-color: #d4edda; border-left: 6px solid #28a745'
    }
    return styles.get(level, styles['info'])

def format_alerts(alerts):
    """格式化警报信息"""
    if not alerts:
        return "<p>没有触发警报</p>"
        
    html = ""
    for alert in alerts:
        # 安全地获取警报级别，默认为 'info'
        alert_level = alert.get('level', 'info')
        message = alert.get('message', 'N/A')
        
        html += f"""
        <div class="alert {alert_level}">
            <p>{message}</p>
        </div>
        """
    return html

def format_semiconductor_stocks(stocks):
    """格式化半导体股票数据"""
    if not stocks:
        return "<tr><td colspan='7'>没有可用的半导体股票数据</td></tr>"
        
    html = ""
    for symbol, data in stocks.items():
        # 处理可能为None的值
        name = data.get('name', 'N/A')
        current_price = data.get('current_price', 0)
        price_change = data.get('price_change', '0%')
        rsi = data.get('rsi', 0)
        macd_status = data.get('macd_status', 'N/A')
        trend = data.get('trend', 'N/A')
        
        # 安全地格式化数值
        try:
            current_price_str = f"${float(current_price):.2f}"
        except (TypeError, ValueError):
            current_price_str = 'N/A'
            
        try:
            rsi_str = f"{float(rsi):.2f}"
        except (TypeError, ValueError):
            rsi_str = 'N/A'
        
        html += f"""
        <tr>
            <td>{symbol}</td>
            <td>{name}</td>
            <td>{current_price_str}</td>
            <td>{price_change}</td>
            <td>{rsi_str}</td>
            <td>{macd_status}</td>
            <td>{trend}</td>
        </tr>
        """
    return html

async def main():
    """主函数"""
    try:
        await monitor_stocks()
    except KeyboardInterrupt:
        logger.info("监控程序已停止")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 