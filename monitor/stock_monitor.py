import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional
import logging
import asyncio
from data.data_interface import DataInterface, RealTimeDataSource
from data.data_validator import DataValidator
from monitor.notification_manager import NotificationManager
import json
import pytz
import time as time_module

class StockMonitor:
    def __init__(self, refresh_interval: int = 60):
        """
        初始化股票监控器
        :param refresh_interval: 刷新间隔（秒）
        """
        self.refresh_interval = refresh_interval
        self.monitored_stocks = set()
        self.watchlist_stocks = set()
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据接口
        try:
            from data.data_interface import YahooFinanceDataSource
            self.data_interface = YahooFinanceDataSource()
        except ImportError:
            self.logger.error("无法初始化Yahoo Finance数据源")
            raise
        
        # 初始化数据验证器
        self.data_validator = DataValidator()
        
        # 初始化通知管理器
        self.notification_manager = NotificationManager()
        
        # 上次检查时间
        self.last_check_time: Dict[str, datetime] = {}
        # 上次检查的数据
        self.last_check_data: Dict[str, pd.DataFrame] = {}
        
        # 加载持仓数据
        self.positions = self._load_positions()
        
        self.load_watchlist()
        
    def _load_positions(self) -> Dict[str, Dict]:
        """加载持仓数据"""
        try:
            # 从data模块获取持仓数据
            from data.data_loader import DataLoader
            data_loader = DataLoader()
            positions = data_loader.load_positions()
            return positions
        except Exception as e:
            self.logger.error(f"加载持仓数据失败: {e}")
            return {}
        
    def load_watchlist(self):
        """加载监控列表和观察列表"""
        try:
            with open('monitor/configs/watchlist.json', 'r') as f:
                config = json.load(f)
                self.monitored_stocks = set(config['current_positions'])
                self.watchlist_stocks = set(config['watchlist'])
                self.logger.info(f"已加载持仓股票: {self.monitored_stocks}")
                self.logger.info(f"已加载观察列表: {self.watchlist_stocks}")
        except Exception as e:
            self.logger.error(f"加载监控列表失败: {str(e)}")
            
    def add_stock(self, symbol: str):
        """添加要监控的股票"""
        if symbol not in self.monitored_stocks:
            self.monitored_stocks.add(symbol)
            self.logger.info(f"添加监控股票: {symbol}")
            
    def remove_stock(self, symbol: str):
        """移除监控的股票"""
        if symbol in self.monitored_stocks:
            self.monitored_stocks.remove(symbol)
            self.logger.info(f"移除监控股票: {symbol}")
            
    async def get_stock_analysis(self, symbol: str) -> Dict:
        """
        获取单个股票的综合分析
        :param symbol: 股票代码
        :return: 包含各种分析指标的字典
        """
        try:
            # 获取实时数据
            if isinstance(self.data_interface, RealTimeDataSource):
                # 使用1分钟数据
                realtime_data = await self.data_interface.get_realtime_data([symbol], timeframe='1m')
                df = realtime_data.get(symbol)
            else:
                # 如果数据源不支持实时数据，获取最近的数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)  # 获取最近1天的数据
                df = self.data_interface.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1m'  # 使用1分钟数据
                )
            
            if df is None or df.empty:
                return {
                    "symbol": symbol,
                    "error": "无法获取数据",
                    "current_price": 0.0,
                    "price_change": "0.00%",
                    "volume_change": "0.00%",
                    "trend": "无数据",
                    "volume_status": "无数据",
                    "rsi": None,
                    "rsi_status": "无数据",
                    "macd_status": "无数据",
                    "position_size": 0,
                    "avg_price": 0.0,
                    "position_pnl": "0.00%",
                    "position_pnl_amount": "$0.00",
                    "score": 0,
                    "recommendations": ["无法获取数据，请检查数据源"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            # 验证数据
            validated_df, report = self.data_validator.validate_data(df)
            
            # 忽略adj_close的缺失值警告
            if 'missing_values' in report and 'adj_close' in report['missing_values']:
                del report['missing_values']['adj_close']
                if not report['missing_values']:
                    del report['missing_values']
                    
            if not report['validation_passed']:
                return {
                    "symbol": symbol,
                    "error": "数据验证失败",
                    "current_price": 0.0,
                    "price_change": "0.00%",
                    "volume_change": "0.00%",
                    "trend": "数据无效",
                    "volume_status": "数据无效",
                    "rsi": None,
                    "rsi_status": "数据无效",
                    "macd_status": "数据无效",
                    "position_size": 0,
                    "avg_price": 0.0,
                    "position_pnl": "0.00%",
                    "position_pnl_amount": "$0.00",
                    "score": 0,
                    "recommendations": ["数据验证失败，请检查数据质量"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            # 获取最新数据
            latest_data = validated_df.iloc[-1]
            prev_data = validated_df.iloc[-2] if len(validated_df) > 1 else latest_data
            
            # 计算各种指标
            price_change = (latest_data['close'] - prev_data['close']) / prev_data['close']
            volume_change = (latest_data['volume'] - prev_data['volume']) / prev_data['volume']
            
            # 计算技术指标
            rsi = latest_data.get('RSI', None)
            macd = latest_data.get('MACD', None)
            macd_signal = latest_data.get('MACD_Signal', None)
            
            # 判断趋势
            trend = "上升" if price_change > 0 else "下降"
            trend_strength = "强势" if abs(price_change) > 0.02 else "弱势"
            
            # 判断成交量
            volume_status = "放量" if volume_change > 0.5 else "缩量"
            
            # 判断RSI状态
            rsi_status = None
            if rsi is not None:
                if rsi > 70:
                    rsi_status = "超买"
                elif rsi < 30:
                    rsi_status = "超卖"
                    
            # 判断MACD状态
            macd_status = None
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    macd_status = "金叉"
                elif macd < macd_signal:
                    macd_status = "死叉"
                    
            # 获取持仓信息
            position = self.positions.get(symbol, {})
            position_size = position.get('size', 0)
            avg_price = position.get('avg_price', 0)
            
            # 计算持仓盈亏
            if position_size > 0 and avg_price > 0:
                position_pnl = (latest_data['close'] - avg_price) / avg_price
                position_pnl_amount = (latest_data['close'] - avg_price) * position_size
            else:
                position_pnl = 0
                position_pnl_amount = 0
                
            # 综合评分
            score = 0
            if price_change > 0:
                score += 1
            if volume_change > 0:
                score += 1
            if rsi_status == "超买":
                score -= 1
            elif rsi_status == "超卖":
                score += 1
            if macd_status == "金叉":
                score += 1
            elif macd_status == "死叉":
                score -= 1
                
            # 生成建议
            recommendations = []
            if score >= 3:
                recommendations.append("买入信号：技术指标显示强势，可以考虑买入")
            elif score <= -3:
                recommendations.append("卖出信号：技术指标显示弱势，建议谨慎操作或考虑卖出")
            else:
                recommendations.append("观望：等待更明确的信号")
                
            if rsi_status == "超买":
                recommendations.append("风险提示：RSI超买，注意回调风险")
            elif rsi_status == "超卖":
                recommendations.append("机会提示：RSI超卖，可能有反弹机会")
                
            if macd_status == "金叉":
                recommendations.append("技术提示：MACD金叉，短期趋势向好")
            elif macd_status == "死叉":
                recommendations.append("技术提示：MACD死叉，短期趋势向弱")
                
            # 根据持仓情况添加建议
            if position_size > 0:
                if position_pnl > 0.1:  # 盈利超过10%
                    recommendations.append("持仓建议：盈利可观，可以考虑部分止盈")
                elif position_pnl < -0.1:  # 亏损超过10%
                    recommendations.append("持仓建议：亏损较大，注意止损")
                elif position_pnl > 0:
                    recommendations.append("持仓建议：当前盈利，可以继续持有")
                else:
                    recommendations.append("持仓建议：当前亏损，注意风险")
                    
            return {
                "symbol": symbol,
                "current_price": latest_data['close'],
                "price_change": f"{price_change:.2%}",
                "volume_change": f"{volume_change:.2%}",
                "trend": f"{trend}{trend_strength}",
                "volume_status": volume_status,
                "rsi": rsi,
                "rsi_status": rsi_status,
                "macd_status": macd_status,
                "position_size": position_size,
                "avg_price": avg_price,
                "position_pnl": f"{position_pnl:.2%}",
                "position_pnl_amount": f"${position_pnl_amount:.2f}",
                "score": score,
                "recommendations": recommendations,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            self.logger.error(f"分析股票 {symbol} 时出错: {e}")
            return {"error": str(e)}
            
    async def monitor_stocks(self):
        """监控所有股票"""
        while True:
            try:
                if not self.is_market_open():
                    self.logger.info("市场已收盘，生成每日总结报告...")
                    analysis_results = []
                    
                    # 分析持仓股票
                    for symbol in self.monitored_stocks:
                        try:
                            result = await self.get_stock_analysis(symbol)
                            if result:
                                analysis_results.append(result)
                        except Exception as e:
                            self.logger.error(f"分析股票 {symbol} 失败: {str(e)}")
                            
                    # 分析观察列表股票
                    for symbol in self.watchlist_stocks:
                        try:
                            result = await self.get_stock_analysis(symbol)
                            if result:
                                analysis_results.append(result)
                        except Exception as e:
                            self.logger.error(f"分析股票 {symbol} 失败: {str(e)}")
                            
                    # 生成并发送每日总结
                    summary = self.generate_daily_summary(analysis_results)
                    html_report = self._generate_html_report(analysis_results, summary=True)
                    self.notification_manager.send_batch_alerts([{
                        "type": "daily_summary",
                        "message": html_report
                    }])
                    
                    # 计算到明天开盘的等待时间
                    now = datetime.now(pytz.timezone('America/New_York'))
                    tomorrow = now.date() + timedelta(days=1)
                    next_open = datetime.combine(tomorrow, time(9, 30))
                    next_open = pytz.timezone('America/New_York').localize(next_open)
                    sleep_time = (next_open - now).total_seconds()
                    
                    self.logger.info(f"市场已收盘，等待 {sleep_time/3600:.2f} 小时后重新开始监控...")
                    time_module.sleep(sleep_time)
                    continue
                    
                analysis_results = []
                
                # 分析持仓股票
                for symbol in self.monitored_stocks:
                    try:
                        result = await self.get_stock_analysis(symbol)
                        if result:
                            analysis_results.append(result)
                    except Exception as e:
                        self.logger.error(f"分析股票 {symbol} 失败: {str(e)}")
                        
                # 分析观察列表股票
                for symbol in self.watchlist_stocks:
                    try:
                        result = await self.get_stock_analysis(symbol)
                        if result:
                            analysis_results.append(result)
                    except Exception as e:
                        self.logger.error(f"分析股票 {symbol} 失败: {str(e)}")
                        
                if analysis_results:
                    html_report = self._generate_html_report(analysis_results)
                    self.notification_manager.send_batch_alerts([{
                        "type": "comprehensive_report",
                        "message": html_report
                    }])
                    
                time_module.sleep(self.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"监控过程出错: {str(e)}")
                time_module.sleep(self.refresh_interval)
                
    def is_market_open(self):
        """检查市场是否开盘"""
        now = datetime.now(pytz.timezone('America/New_York'))
        market_open = time(9, 30)  # 9:30 AM ET
        market_close = time(16, 0)  # 4:00 PM ET
        
        # 检查是否是工作日
        if now.weekday() >= 5:  # 5是周六，6是周日
            return False
            
        # 检查是否在交易时间内
        current_time = now.time()
        return market_open <= current_time <= market_close
        
    def generate_daily_summary(self, analysis_results):
        """生成每日总结报告"""
        if not analysis_results:
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "positions": [],
                "watchlist": [],
                "market_summary": {
                    "total_positions": 0,
                    "profitable_positions": 0,
                    "losing_positions": 0,
                    "total_pnl": 0.0
                }
            }
            
        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "positions": [],
            "watchlist": [],
            "market_summary": {
                "total_positions": len([r for r in analysis_results if r["symbol"] in self.monitored_stocks]),
                "profitable_positions": 0,
                "losing_positions": 0,
                "total_pnl": 0.0
            }
        }
        
        for result in analysis_results:
            if "error" in result:
                continue
                
            try:
                position_data = {
                    "symbol": result["symbol"],
                    "current_price": result.get("current_price", 0.0),
                    "price_change": result.get("price_change", "0.00%"),
                    "position_pnl": result.get("position_pnl", "0.00%"),
                    "position_pnl_amount": result.get("position_pnl_amount", "$0.00"),
                    "recommendations": result.get("recommendations", ["无数据"])
                }
                
                if result["symbol"] in self.monitored_stocks:
                    summary["positions"].append(position_data)
                    pnl = float(position_data["position_pnl"].strip('%')) if position_data["position_pnl"] else 0
                    if pnl > 0:
                        summary["market_summary"]["profitable_positions"] += 1
                    else:
                        summary["market_summary"]["losing_positions"] += 1
                    pnl_amount = float(position_data["position_pnl_amount"].strip('$')) if position_data["position_pnl_amount"] else 0
                    summary["market_summary"]["total_pnl"] += pnl_amount
                else:
                    summary["watchlist"].append(position_data)
            except Exception as e:
                self.logger.error(f"处理股票 {result.get('symbol', 'unknown')} 的总结数据时出错: {str(e)}")
                continue
                
        return summary
        
    def _generate_html_report(self, analysis_results, summary=False):
        """生成HTML格式的报告"""
        if not analysis_results:
            return """
            <html>
                <body>
                    <h1>无有效数据</h1>
                    <p>未能获取到任何股票数据，请检查数据源或网络连接。</p>
                </body>
            </html>
            """
            
        html = """
        <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .stock-card { 
                        border: 1px solid #ddd; 
                        padding: 15px; 
                        margin: 10px 0; 
                        border-radius: 5px;
                        background-color: #f9f9f9;
                    }
                    .position { background-color: #e6f7ff; }
                    .watchlist { background-color: #f0fff0; }
                    .header { 
                        font-size: 1.2em; 
                        font-weight: bold; 
                        margin-bottom: 10px;
                        color: #333;
                    }
                    .metric { margin: 5px 0; }
                    .recommendations { 
                        margin-top: 10px; 
                        padding: 10px;
                        background-color: #fff;
                        border-radius: 3px;
                    }
                    .recommendation { 
                        margin: 5px 0;
                        padding: 5px;
                        border-left: 3px solid #4CAF50;
                    }
                    .warning { border-left-color: #ff9800; }
                    .danger { border-left-color: #f44336; }
                    .summary {
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #f5f5f5;
                        border-radius: 5px;
                    }
                    .error-card {
                        background-color: #ffebee;
                        border-left: 3px solid #f44336;
                    }
                </style>
            </head>
            <body>
        """
        
        if summary:
            positions_count = len([r for r in analysis_results if r["symbol"] in self.monitored_stocks and "error" not in r])
            profitable_positions = len([r for r in analysis_results if r["symbol"] in self.monitored_stocks and "error" not in r and float(r.get("position_pnl", "0%").strip('%')) > 0])
            losing_positions = positions_count - profitable_positions
            
            html += f"""
                <div class="summary">
                    <h2>每日市场总结 - {datetime.now().strftime("%Y-%m-%d")}</h2>
                    <p>总持仓数: {positions_count}</p>
                    <p>盈利持仓: {profitable_positions}</p>
                    <p>亏损持仓: {losing_positions}</p>
                </div>
            """
            
        for result in analysis_results:
            if "error" in result:
                html += f"""
                    <div class="stock-card error-card">
                        <div class="header">{result["symbol"]}</div>
                        <div class="metric">状态: {result["error"]}</div>
                    </div>
                """
                continue
                
            try:
                stock_type = "position" if result["symbol"] in self.monitored_stocks else "watchlist"
                
                # 安全地获取值，处理None的情况
                rsi_value = result.get('rsi')
                rsi_display = f"{rsi_value:.2f}" if rsi_value is not None else "N/A"
                rsi_status = result.get('rsi_status', '')
                macd_status = result.get('macd_status', '无信号')
                
                html += f"""
                    <div class="stock-card {stock_type}">
                        <div class="header">{result["symbol"]}</div>
                        <div class="metric">当前价格: ${result.get("current_price", 0.0):.2f}</div>
                        <div class="metric">价格变动: {result.get("price_change", "0.00%")}</div>
                        <div class="metric">成交量变动: {result.get("volume_change", "0.00%")}</div>
                        <div class="metric">趋势: {result.get("trend", "无数据")}</div>
                        <div class="metric">RSI: {rsi_display} ({rsi_status})</div>
                        <div class="metric">MACD: {macd_status}</div>
                """
                
                if result["symbol"] in self.monitored_stocks:
                    html += f"""
                        <div class="metric">持仓数量: {result.get("position_size", 0)}</div>
                        <div class="metric">平均成本: ${result.get("avg_price", 0.0):.2f}</div>
                        <div class="metric">持仓盈亏: {result.get("position_pnl", "0.00%")} ({result.get("position_pnl_amount", "$0.00")})</div>
                    """
                    
                html += """
                        <div class="recommendations">
                """
                
                recommendations = result.get("recommendations", ["无建议"])
                for rec in recommendations:
                    rec_class = "recommendation"
                    if "风险" in rec or "卖出" in rec:
                        rec_class += " danger"
                    elif "机会" in rec or "买入" in rec:
                        rec_class += " warning"
                    html += f'<div class="{rec_class}">{rec}</div>'
                    
                html += """
                        </div>
                        <div class="metric">更新时间: {}</div>
                    </div>
                """.format(result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            except Exception as e:
                self.logger.error(f"生成股票 {result.get('symbol', 'unknown')} 的HTML报告时出错: {str(e)}")
                html += f"""
                    <div class="stock-card error-card">
                        <div class="header">{result.get("symbol", "未知股票")}</div>
                        <div class="metric">状态: 生成报告时出错 - {str(e)}</div>
                    </div>
                """
                
        html += """
            </body>
        </html>
        """
        
        return html 