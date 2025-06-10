#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import schedule
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
import platform

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.market_environment_classifier import MarketEnvironmentClassifier
from strategy.dynamic_strategy_selector import DynamicStrategySelector
from strategy.signal_quality_evaluator import SignalQualityEvaluator
from strategy.advanced_alert_system import AdvancedAlertSystem
from monitor.notification_manager import NotificationManager
from monitor.data_fetcher import DataFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedDailyReport")

class AdvancedDailyReportGenerator:
    """高级每日股票分析报告生成器"""
    
    def __init__(self, config=None):
        """初始化报告生成器"""
        self.config = config or {}
        self.watchlist = self.config.get('watchlist', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        
        # 初始化核心组件
        self.market_classifier = MarketEnvironmentClassifier()
        self.strategy_selector = DynamicStrategySelector()
        self.signal_evaluator = SignalQualityEvaluator()
        self.alert_system = AdvancedAlertSystem()
        self.notification_manager = NotificationManager()
        self.data_fetcher = DataFetcher()
        
        # 设置中文字体支持
        self._setup_chinese_font()
        
        logger.info("高级日报生成器初始化完成")
    
    def _setup_chinese_font(self):
        """设置中文字体支持"""
        try:
            system = platform.system()
            if system == "Windows":
                font_list = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi']
            elif system == "Darwin":  # macOS
                font_list = ['PingFang SC', 'Heiti SC', 'STHeiti']
            else:  # Linux
                font_list = ['WenQuanYi Micro Hei', 'DejaVu Sans']
            
            for font_name in font_list:
                try:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    logger.info(f"成功设置字体: {font_name}")
                    break
                except:
                    continue
        except Exception as e:
            logger.warning(f"设置中文字体失败: {e}")
    
    def _get_market_data(self, symbol: str, days: int = 400) -> Optional[pd.DataFrame]:
        """获取股票市场数据"""
        try:
            # 这里应该调用实际的数据获取接口
            # 现在使用模拟数据作为示例
            dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
            np.random.seed(42)  # 确保可重复性
            
            # 模拟股价数据
            initial_price = 150 + np.random.random() * 50
            price_changes = np.random.normal(0, 0.02, days)
            prices = [initial_price]
            
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1))  # 确保价格不为负
            
            data = pd.DataFrame({
                'date': dates,
                'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, days)
            })
            
            data.set_index('date', inplace=True)
            return self._calculate_technical_indicators(data)
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # 移动平均线
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            data['sma_200'] = data['close'].rolling(window=200).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # 布林带
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            
            return data.dropna()
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return data
    
    def _analyze_single_stock(self, symbol: str) -> Dict:
        """分析单只股票"""
        logger.info(f"开始分析 {symbol}")
        
        # 获取数据
        data = self._get_market_data(symbol)
        if data is None or len(data) < 60:
            logger.warning(f"{symbol} 数据不足，跳过分析")
            return None
        
        result = {
            'symbol': symbol,
            'current_price': data['close'].iloc[-1],
            'price_change': ((data['close'].iloc[-1] / data['close'].iloc[-2]) - 1) * 100,
            'volume': data['volume'].iloc[-1],
            'volume_change': ((data['volume'].iloc[-1] / data['volume'].iloc[-20:].mean()) - 1) * 100
        }
        
        try:
            # 市场环境分析
            env_result = self.market_classifier.classify_environment(data)
            result['market_environment'] = {
                'classification': env_result['environment'].value,
                'confidence': env_result.get('confidence', 0),
                'reasons': env_result.get('reasons', [])
            }
            
            # 策略建议
            strategy_result = self.strategy_selector.get_best_strategy(data)
            result['strategy_recommendation'] = {
                'primary_strategy': strategy_result['primary_strategy'],
                'environment': strategy_result['market_environment'].value,
                'strategy_weights': strategy_result['strategy_weights']
            }
            
            # 生成交易信号进行评估
            signal_data = self._generate_test_signal(data)
            if signal_data:
                signal_evaluation = self.signal_evaluator.evaluate_signal(
                    signal_data, data, env_result['environment']
                )
                result['signal_analysis'] = {
                    'quality_score': signal_evaluation['quality_score'],
                    'strength': signal_evaluation['strength'].value,
                    'passed_threshold': signal_evaluation['passed_threshold'],
                    'dimension_scores': signal_evaluation.get('dimension_scores', {}),
                    'recommendations': signal_evaluation.get('recommendations', [])
                }
            
            # 生成可视化图表
            chart_path = self._generate_stock_chart(symbol, data, env_result)
            result['chart_path'] = chart_path
            
            logger.info(f"{symbol} 分析完成")
            return result
            
        except Exception as e:
            logger.error(f"分析 {symbol} 时出错: {e}")
            return result
    
    def _generate_test_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """生成测试用的交易信号"""
        try:
            current_price = data['close'].iloc[-1]
            return {
                'direction': 1,  # 买入信号
                'entry_price': current_price,
                'stop_loss': current_price * 0.95,
                'target_price': current_price * 1.15,
                'indicator_signals': {
                    'macd': 1 if data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] else -1,
                    'rsi': 1 if data['rsi'].iloc[-1] < 70 else -1,
                    'sma_crossover': 1 if data['close'].iloc[-1] > data['sma_20'].iloc[-1] else -1,
                    'bollinger_bands': 0  # 中性
                }
            }
        except:
            return None
    
    def _generate_stock_chart(self, symbol: str, data: pd.DataFrame, env_result: Dict) -> str:
        """生成股票分析图表"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # 价格图
            axes[0].plot(data.index[-60:], data['close'].iloc[-60:], label='收盘价', linewidth=2)
            axes[0].plot(data.index[-60:], data['sma_20'].iloc[-60:], label='20日均线', alpha=0.7)
            axes[0].plot(data.index[-60:], data['sma_50'].iloc[-60:], label='50日均线', alpha=0.7)
            
            env_name = env_result['environment'].value
            confidence = env_result.get('confidence', 0)
            if not np.isnan(confidence):
                title = f"{symbol} - 市场环境: {env_name} (置信度: {confidence:.2f})"
            else:
                title = f"{symbol} - 市场环境: {env_name}"
            
            axes[0].set_title(title)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # RSI图
            axes[1].plot(data.index[-60:], data['rsi'].iloc[-60:], label='RSI', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买线')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='超卖线')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # MACD图
            axes[2].plot(data.index[-60:], data['macd'].iloc[-60:], label='MACD', color='blue')
            axes[2].plot(data.index[-60:], data['macd_signal'].iloc[-60:], label='Signal', color='red')
            axes[2].bar(data.index[-60:], (data['macd'] - data['macd_signal']).iloc[-60:], 
                       label='Histogram', alpha=0.3)
            axes[2].set_ylabel('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"已生成 {symbol} 分析图表: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"生成 {symbol} 图表失败: {e}")
            return ""
    
    def _generate_html_report(self, analysis_results: List[Dict]) -> str:
        """生成HTML格式的报告"""
        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .summary {{ margin: 20px 0; padding: 15px; background-color: #e7f3ff; border-radius: 5px; }}
                .stock-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .positive {{ color: #28a745; font-weight: bold; }}
                .negative {{ color: #dc3545; font-weight: bold; }}
                .neutral {{ color: #6c757d; }}
                .chart-container {{ text-align: center; margin: 15px 0; }}
                .recommendations {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .metrics-table th, .metrics-table td {{ 
                    border: 1px solid #ddd; padding: 8px; text-align: left; 
                }}
                .metrics-table th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 每日股票分析报告</h1>
                <p><strong>生成时间:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                <p><strong>分析股票数量:</strong> {len([r for r in analysis_results if r])} 只</p>
            </div>
        """
        
        # 市场概览
        html += """
            <div class="summary">
                <h2>📊 市场概览</h2>
                <p>基于高级市场环境分类器和动态策略选择器的分析结果</p>
            </div>
        """
        
        # 个股分析
        for result in analysis_results:
            if not result:
                continue
                
            symbol = result['symbol']
            price_change = result['price_change']
            price_class = 'positive' if price_change >= 0 else 'negative'
            
            html += f"""
            <div class="stock-section">
                <h3>📈 {symbol} 分析报告</h3>
                
                <table class="metrics-table">
                    <tr>
                        <th>指标</th>
                        <th>数值</th>
                        <th>状态</th>
                    </tr>
                    <tr>
                        <td>当前价格</td>
                        <td>${result['current_price']:.2f}</td>
                        <td class="{price_class}">{price_change:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>成交量变化</td>
                        <td>{result['volume']:,}</td>
                        <td>{result['volume_change']:+.1f}%</td>
                    </tr>
                    <tr>
                        <td>市场环境</td>
                        <td>{result['market_environment']['classification']}</td>
                        <td>置信度: {result['market_environment']['confidence']:.2f}</td>
                    </tr>
                    <tr>
                        <td>推荐策略</td>
                        <td>{result['strategy_recommendation']['primary_strategy']}</td>
                        <td>适合当前环境</td>
                    </tr>
            """
            
            if 'signal_analysis' in result:
                signal = result['signal_analysis']
                signal_class = 'positive' if signal['passed_threshold'] else 'neutral'
                html += f"""
                    <tr>
                        <td>信号质量</td>
                        <td class="{signal_class}">{signal['quality_score']:.2f}</td>
                        <td>{signal['strength']}</td>
                    </tr>
                """
            
            html += "</table>"
            
            # 分析原因
            if result['market_environment']['reasons']:
                html += "<div class='recommendations'><h4>📋 分析依据:</h4><ul>"
                for reason in result['market_environment']['reasons'][:5]:  # 只显示前5个原因
                    html += f"<li>{reason}</li>"
                html += "</ul></div>"
            
            # 投资建议
            if 'signal_analysis' in result and result['signal_analysis']['recommendations']:
                html += "<div class='recommendations'><h4>💡 投资建议:</h4><ul>"
                for rec in result['signal_analysis']['recommendations']:
                    html += f"<li>{rec}</li>"
                html += "</ul></div>"
            
            html += "</div>"
        
        # 免责声明
        html += """
            <div style="margin-top: 30px; padding: 15px; background-color: #fff3cd; border-radius: 5px;">
                <h4>⚠️ 免责声明</h4>
                <p>本报告基于技术分析和历史数据生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def generate_daily_report(self) -> bool:
        """生成每日报告"""
        try:
            logger.info("开始生成每日高级股票分析报告")
            
            # 分析所有关注的股票
            analysis_results = []
            for symbol in self.watchlist:
                result = self._analyze_single_stock(symbol)
                analysis_results.append(result)
                time.sleep(1)  # 避免请求过于频繁
            
            # 生成HTML报告
            html_report = self._generate_html_report(analysis_results)
            
            # 发送邮件
            subject = f"📊 每日股票分析报告 - {datetime.now().strftime('%Y年%m月%d日')}"
            
            # 这里应该调用邮件发送功能
            # self.notification_manager.alert_system.send_email(subject, html_report)
            
            # 暂时保存到文件用于调试
            report_file = f"daily_report_{datetime.now().strftime('%Y%m%d')}.html"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            logger.info(f"每日报告生成完成，已保存到: {report_file}")
            return True
            
        except Exception as e:
            logger.error(f"生成每日报告失败: {e}")
            return False

def setup_daily_report_schedule(config=None):
    """设置每日报告定时任务"""
    report_generator = AdvancedDailyReportGenerator(config)
    
    # 设置为交易日收盘后30分钟发送 (美股时间下午4:30)
    # 对应北京时间凌晨5:30 (夏令时) 或 6:30 (冬令时)
    schedule.every().monday.at("17:30").do(report_generator.generate_daily_report)
    schedule.every().tuesday.at("17:30").do(report_generator.generate_daily_report)
    schedule.every().wednesday.at("17:30").do(report_generator.generate_daily_report)
    schedule.every().thursday.at("17:30").do(report_generator.generate_daily_report)
    schedule.every().friday.at("17:30").do(report_generator.generate_daily_report)
    
    logger.info("每日报告定时任务已设置 - 工作日17:30发送")
    
    return report_generator

def main():
    """主函数"""
    try:
        # 配置
        config = {
            'watchlist': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        }
        
        # 设置定时任务
        report_generator = setup_daily_report_schedule(config)
        
        # 立即生成一份报告用于测试
        logger.info("生成测试报告...")
        report_generator.generate_daily_report()
        
        # 开始定时任务循环
        logger.info("启动定时任务，等待调度...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
            
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序运行错误: {e}")

if __name__ == "__main__":
    main() 