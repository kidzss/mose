import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import yfinance as yf
from jinja2 import Template
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import talib

class ReportGenerator:
    """投资组合报告生成器"""
    
    def __init__(self):
        self.report_template = Template("""
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
                .summary { margin: 20px 0; }
                .position-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .position-table th, .position-table td { 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }
                .position-table th { background-color: #f8f9fa; }
                .profit { color: #28a745; }
                .loss { color: #dc3545; }
                .risk-metrics { margin: 20px 0; }
                .recommendations { 
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                .market-summary { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>每日投资组合报告</h2>
                <p>生成时间: {{ timestamp }}</p>
            </div>

            <div class="summary">
                <h3>投资组合概览</h3>
                <p>总市值: ${{ "{:,.2f}".format(total_value) }}</p>
                <p>今日收益: <span class="{{ 'profit' if daily_return >= 0 else 'loss' }}">
                    {{ '{:+.2f}%'.format(daily_return * 100) }}</span></p>
                <p>累计收益: <span class="{{ 'profit' if total_return >= 0 else 'loss' }}">
                    {{ '{:+.2f}%'.format(total_return * 100) }}</span></p>
            </div>

            <div class="market-summary">
                <h3>市场概览</h3>
                <p>标普500今日表现: {{ '{:+.2f}%'.format(sp500_return * 100) }}</p>
                <p>纳斯达克今日表现: {{ '{:+.2f}%'.format(nasdaq_return * 100) }}</p>
                <p>相对大盘超额收益: <span class="{{ 'profit' if relative_return >= 0 else 'loss' }}">
                    {{ '{:+.2f}%'.format(relative_return * 100) }}</span></p>
            </div>

            <h3>持仓明细</h3>
            <table class="position-table">
                <tr>
                    <th>股票代码</th>
                    <th>现价</th>
                    <th>成本价</th>
                    <th>持仓数量</th>
                    <th>市值</th>
                    <th>仓位占比</th>
                    <th>今日收益</th>
                    <th>累计收益</th>
                </tr>
                {% for position in positions %}
                <tr>
                    <td>{{ position.symbol }}</td>
                    <td>${{ "{:.2f}".format(position.current_price) }}</td>
                    <td>${{ "{:.2f}".format(position.cost_basis) }}</td>
                    <td>{{ "{:,.0f}".format(position.shares) }}</td>
                    <td>${{ "{:,.2f}".format(position.market_value) }}</td>
                    <td>{{ "{:.2f}%".format(position.weight * 100) }}</td>
                    <td class="{{ 'profit' if position.daily_return >= 0 else 'loss' }}">
                        {{ '{:+.2f}%'.format(position.daily_return * 100) }}</td>
                    <td class="{{ 'profit' if position.total_return >= 0 else 'loss' }}">
                        {{ '{:+.2f}%'.format(position.total_return * 100) }}</td>
                </tr>
                {% endfor %}
            </table>

            <div class="risk-metrics">
                <h3>风险指标</h3>
                <p>投资组合贝塔系数: {{ "{:.2f}".format(portfolio_beta) }}</p>
                <p>夏普比率: {{ "{:.2f}".format(sharpe_ratio) }}</p>
                <p>最大回撤: {{ "{:.2f}%".format(max_drawdown * 100) }}</p>
                <p>年化波动率: {{ "{:.2f}%".format(volatility * 100) }}</p>
                <p>在险价值(95%): {{ "{:.2f}%".format(var_95 * 100) }}</p>
            </div>

            {% if alerts %}
            <div class="alerts">
                <h3>今日预警信息</h3>
                <ul>
                {% for alert in alerts %}
                    <li>{{ alert }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}

            {% if recommendations %}
            <div class="recommendations">
                <h3>投资建议</h3>
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
        </body>
        </html>
        """)
        self.logger = logging.getLogger(__name__)

    def _calculate_market_returns(self) -> tuple:
        """计算大盘指数收益率"""
        try:
            # 使用新的时间频率标识符
            sp500 = yf.download('^GSPC', period='2d', interval='1d')
            nasdaq = yf.download('^IXIC', period='2d', interval='1d')
            
            # 使用新的 pct_change 参数
            sp500_return = sp500['Close'].pct_change(fill_method=None).iloc[-1]
            nasdaq_return = nasdaq['Close'].pct_change(fill_method=None).iloc[-1]
            
            return sp500_return, nasdaq_return
        except Exception as e:
            print(f"获取市场数据时出错: {e}")
            return 0.0, 0.0

    def _calculate_risk_metrics(self, portfolio_monitor) -> Dict:
        """计算风险指标"""
        try:
            # 获取SP500收益率作为市场基准
            sp500 = yf.download('^GSPC', period='1y')['Close'].pct_change().dropna()
            
            # 计算投资组合的每日收益率
            portfolio_returns = []
            for symbol, position in portfolio_monitor.positions.items():
                if symbol in portfolio_monitor.historical_data:
                    returns = portfolio_monitor.historical_data[symbol]['Close'].pct_change().dropna()
                    weight = position['weight']
                    portfolio_returns.append(returns * weight)
            
            if portfolio_returns:
                portfolio_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
                
                # 计算Beta
                covariance = np.cov(portfolio_returns, sp500)[0][1]
                market_variance = np.var(sp500)
                beta = covariance / market_variance if market_variance != 0 else 1.0
                
                # 计算夏普比率 (假设无风险利率为2%)
                risk_free_rate = 0.02
                excess_returns = portfolio_returns - risk_free_rate/252
                sharpe = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std() if portfolio_returns.std() != 0 else 0
                
                # 计算最大回撤
                cumulative_returns = (1 + portfolio_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = cumulative_returns/rolling_max - 1
                max_drawdown = drawdowns.min()
                
                # 计算波动率
                volatility = portfolio_returns.std() * np.sqrt(252)
                
                # 计算VaR
                var_95 = np.percentile(portfolio_returns, 5)
                
                return {
                    'beta': beta,
                    'sharpe': sharpe,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'var_95': var_95
                }
            
            return {
                'beta': 1.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'var_95': 0.0
            }
        except Exception as e:
            print(f"计算风险指标时出错: {e}")
            return {
                'beta': 1.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'var_95': 0.0
            }

    def _generate_market_analysis(self) -> Dict:
        """生成市场分析"""
        try:
            # 获取主要指数数据
            indices = {
                'SPY': yf.download('^GSPC', period='5d'),
                'QQQ': yf.download('^IXIC', period='5d'),
                'VIX': yf.download('^VIX', period='5d')
            }
            
            # 分析市场情绪
            vix_change = (indices['VIX']['Close'].iloc[-1] / indices['VIX']['Close'].iloc[-2] - 1) * 100
            market_sentiment = "谨慎" if vix_change > 5 else "中性" if vix_change > -5 else "乐观"
            
            return {
                'market_sentiment': market_sentiment,
                'vix_change': vix_change,
                'sp500_5d_return': (indices['SPY']['Close'].iloc[-1] / indices['SPY']['Close'].iloc[-5] - 1) * 100,
                'nasdaq_5d_return': (indices['QQQ']['Close'].iloc[-1] / indices['QQQ']['Close'].iloc[-5] - 1) * 100
            }
        except Exception as e:
            print(f"生成市场分析时出错: {e}")
            return {
                'market_sentiment': "未知",
                'vix_change': 0,
                'sp500_5d_return': 0,
                'nasdaq_5d_return': 0
            }

    def _generate_position_analysis(self, positions: List[Dict]) -> Dict:
        """生成持仓分析"""
        try:
            # 分类持仓
            sectors = {
                'GOOG': '科技巨头',
                'MSFT': '科技巨头',
                'TSLA': '新能源车',
                'AMD': '半导体',
                'NVDA': '半导体',
                'PFE': '医疗保健',
                'TMDX': '医疗保健'
            }
            
            # 按板块统计
            sector_stats = {}
            for pos in positions:
                sector = sectors.get(pos['symbol'], '其他')
                if sector not in sector_stats:
                    sector_stats[sector] = {
                        'market_value': 0,
                        'total_return': 0,
                        'daily_return': 0,
                        'positions': []
                    }
                sector_stats[sector]['market_value'] += pos['market_value']
                sector_stats[sector]['positions'].append(pos)
                
            # 生成建议
            recommendations = []
            high_risk_positions = []
            
            for pos in positions:
                # 检查波动率
                if abs(pos['daily_return']) > 0.02:
                    high_risk_positions.append(pos['symbol'])
                
                # 检查大幅亏损
                if pos['total_return'] < -0.2:
                    recommendations.append({
                        'symbol': pos['symbol'],
                        'type': 'warning',
                        'message': f"{pos['symbol']}已累计亏损{pos['total_return']*100:.2f}%，建议关注"
                    })
                
                # 检查集中度
                if pos['weight'] > 0.3:
                    recommendations.append({
                        'symbol': pos['symbol'],
                        'type': 'risk',
                        'message': f"{pos['symbol']}占比{pos['weight']*100:.2f}%，建议适度分散"
                    })
                    
            return {
                'sector_stats': sector_stats,
                'high_risk_positions': high_risk_positions,
                'recommendations': recommendations
            }
        except Exception as e:
            print(f"生成持仓分析时出错: {e}")
            return {
                'sector_stats': {},
                'high_risk_positions': [],
                'recommendations': []
            }

    def _generate_strategy_recommendations(self, position_analysis: Dict, market_analysis: Dict) -> str:
        """生成策略建议"""
        try:
            recommendations = []
            
            # 根据市场情绪生成整体建议
            if market_analysis['market_sentiment'] == "谨慎":
                recommendations.append("当前市场波动加大，建议提高现金仓位，对高波动品种采取防御策略")
            elif market_analysis['market_sentiment'] == "乐观":
                recommendations.append("市场情绪向好，可以适度加仓优质标的，但注意风险控制")
            
            # 针对高风险持仓的建议
            for symbol in position_analysis['high_risk_positions']:
                if symbol in ['AMD', 'NVDA']:
                    recommendations.append(f"{symbol}波动较大，建议设置阶梯式止损位，可以考虑在大跌时分批加仓")
                elif symbol in ['TSLA', 'TMDX']:
                    recommendations.append(f"{symbol}风险较高，建议在反弹时适度减仓，降低仓位")
            
            # 针对板块的建议
            for sector, stats in position_analysis['sector_stats'].items():
                if sector == '半导体':
                    recommendations.append("半导体板块短期波动较大，建议关注行业基本面变化和需求情况")
                elif sector == '医疗保健':
                    recommendations.append("医疗保健板块可以作为防御性配置，建议保持适度仓位")
            
            return recommendations
        except Exception as e:
            print(f"生成策略建议时出错: {e}")
            return []

    def generate_daily_report(self, portfolio_monitor) -> str:
        """生成每日投资组合报告"""
        try:
            # 获取市场数据
            sp500_return, nasdaq_return = self._calculate_market_returns()
            
            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(portfolio_monitor)
            
            # 准备持仓数据
            positions = []
            total_value = 0
            daily_returns = []
            
            for symbol, position in portfolio_monitor.positions.items():
                if symbol in portfolio_monitor.current_prices:
                    current_price = portfolio_monitor.current_prices[symbol]
                    cost_basis = position['cost_basis']
                    shares = position['shares']
                    market_value = shares * current_price
                    total_value += market_value
                    
                    # 计算日收益率
                    hist_data = portfolio_monitor.historical_data.get(symbol)
                    daily_return = 0
                    if hist_data is not None and len(hist_data) >= 2:
                        daily_return = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-2]) - 1
                        daily_returns.append(daily_return * position['weight'])
                    
                    positions.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'cost_basis': cost_basis,
                        'shares': shares,
                        'market_value': market_value,
                        'weight': position['weight'],
                        'daily_return': daily_return,
                        'total_return': (current_price - cost_basis) / cost_basis
                    })
            
            # 生成分析
            market_analysis = self._generate_market_analysis()
            position_analysis = self._generate_position_analysis(positions)
            strategy_recommendations = self._generate_strategy_recommendations(position_analysis, market_analysis)
            
            # 计算投资组合整体收益
            portfolio_daily_return = sum(daily_returns) if daily_returns else 0
            portfolio_total_return = sum((p['market_value'] - p['shares'] * p['cost_basis']) for p in positions) / sum(p['shares'] * p['cost_basis'] for p in positions)
            
            # 更新HTML模板
            self.report_template = Template("""
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                    .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
                    .summary { margin: 20px 0; }
                    .position-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    .position-table th, .position-table td { 
                        border: 1px solid #ddd; 
                        padding: 12px; 
                        text-align: left; 
                    }
                    .position-table th { background-color: #f8f9fa; }
                    .profit { color: #28a745; }
                    .loss { color: #dc3545; }
                    .risk-metrics { margin: 20px 0; }
                    .analysis-section {
                        background-color: #f8f9fa;
                        padding: 20px;
                        border-radius: 5px;
                        margin: 20px 0;
                    }
                    .recommendation {
                        background-color: #e9ecef;
                        padding: 15px;
                        margin: 10px 0;
                        border-left: 4px solid #007bff;
                    }
                    .warning {
                        border-left-color: #ffc107;
                    }
                    .risk {
                        border-left-color: #dc3545;
                    }
                    .market-analysis {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                    }
                    .market-card {
                        background-color: white;
                        padding: 15px;
                        border-radius: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>每日投资组合诊断报告</h2>
                    <p>生成时间: {{ timestamp }}</p>
                </div>

                <div class="analysis-section">
                    <h3>市场环境分析</h3>
                    <div class="market-analysis">
                        <div class="market-card">
                            <h4>市场情绪</h4>
                            <p>当前市场情绪: {{ market_analysis.market_sentiment }}</p>
                            <p>VIX变动: {{ '{:+.2f}%'.format(market_analysis.vix_change) }}</p>
                        </div>
                        <div class="market-card">
                            <h4>主要指数表现</h4>
                            <p>标普500五日: {{ '{:+.2f}%'.format(market_analysis.sp500_5d_return) }}</p>
                            <p>纳斯达克五日: {{ '{:+.2f}%'.format(market_analysis.nasdaq_5d_return) }}</p>
                        </div>
                    </div>
                </div>

                <div class="summary">
                    <h3>投资组合概览</h3>
                    <p>总市值: ${{ "{:,.2f}".format(total_value) }}</p>
                    <p>今日收益: <span class="{{ 'profit' if daily_return >= 0 else 'loss' }}">
                        {{ '{:+.2f}%'.format(daily_return * 100) }}</span></p>
                    <p>累计收益: <span class="{{ 'profit' if total_return >= 0 else 'loss' }}">
                        {{ '{:+.2f}%'.format(total_return * 100) }}</span></p>
                </div>

                <div class="analysis-section">
                    <h3>板块分析</h3>
                    {% for sector, stats in position_analysis.sector_stats.items() %}
                    <div class="market-card">
                        <h4>{{ sector }}</h4>
                        <p>总市值: ${{ "{:,.2f}".format(stats.market_value) }}</p>
                        <p>持仓: {{ ", ".join(pos.symbol for pos in stats.positions) }}</p>
                    </div>
                    {% endfor %}
                </div>

                <h3>个股持仓明细</h3>
                <table class="position-table">
                    <tr>
                        <th>股票代码</th>
                        <th>现价</th>
                        <th>成本价</th>
                        <th>持仓数量</th>
                        <th>市值</th>
                        <th>仓位占比</th>
                        <th>今日收益</th>
                        <th>累计收益</th>
                    </tr>
                    {% for position in positions %}
                    <tr>
                        <td>{{ position.symbol }}</td>
                        <td>${{ "{:.2f}".format(position.current_price) }}</td>
                        <td>${{ "{:.2f}".format(position.cost_basis) }}</td>
                        <td>{{ "{:,.0f}".format(position.shares) }}</td>
                        <td>${{ "{:,.2f}".format(position.market_value) }}</td>
                        <td>{{ "{:.2f}%".format(position.weight * 100) }}</td>
                        <td class="{{ 'profit' if position.daily_return >= 0 else 'loss' }}">
                            {{ '{:+.2f}%'.format(position.daily_return * 100) }}</td>
                        <td class="{{ 'profit' if position.total_return >= 0 else 'loss' }}">
                            {{ '{:+.2f}%'.format(position.total_return * 100) }}</td>
                    </tr>
                    {% endfor %}
                </table>

                <div class="analysis-section">
                    <h3>风险指标分析</h3>
                    <p>投资组合贝塔系数: {{ "{:.2f}".format(portfolio_beta) }}</p>
                    <p>夏普比率: {{ "{:.2f}".format(sharpe_ratio) }}</p>
                    <p>最大回撤: {{ "{:.2f}%".format(max_drawdown * 100) }}</p>
                    <p>年化波动率: {{ "{:.2f}%".format(volatility * 100) }}</p>
                    <p>在险价值(95%): {{ "{:.2f}%".format(var_95 * 100) }}</p>
                </div>

                <div class="analysis-section">
                    <h3>投资建议</h3>
                    {% for rec in strategy_recommendations %}
                    <div class="recommendation">
                        {{ rec }}
                    </div>
                    {% endfor %}
                    
                    {% if position_analysis.recommendations %}
                    <h4>个股风险提示</h4>
                    {% for rec in position_analysis.recommendations %}
                    <div class="recommendation {{ rec.type }}">
                        {{ rec.message }}
                    </div>
                    {% endfor %}
                    {% endif %}
                </div>

                {% if alerts %}
                <div class="analysis-section">
                    <h3>今日预警信息</h3>
                    <ul>
                    {% for alert in alerts %}
                        <li>{{ alert }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </body>
            </html>
            """)
            
            # 生成报告
            report = self.report_template.render(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                total_value=total_value,
                daily_return=portfolio_daily_return,
                total_return=portfolio_total_return,
                positions=positions,
                portfolio_beta=risk_metrics['beta'],
                sharpe_ratio=risk_metrics['sharpe'],
                max_drawdown=risk_metrics['max_drawdown'],
                volatility=risk_metrics['volatility'],
                var_95=risk_metrics['var_95'],
                alerts=portfolio_monitor.check_alerts(),
                market_analysis=market_analysis,
                position_analysis=position_analysis,
                strategy_recommendations=strategy_recommendations
            )
            
            return report
        except Exception as e:
            print(f"生成报告时出错: {e}")
            return f"生成报告时出错: {str(e)}"

    def generate_report(self, analysis_result: Dict, df: pd.DataFrame, symbol: str) -> Dict:
        """
        生成分析报告
        :param analysis_result: 分析结果
        :param df: 股票数据
        :param symbol: 股票代码
        :return: 报告数据
        """
        try:
            # 生成图表
            charts = self._generate_charts(df, analysis_result)
            
            # 生成多时间周期分析
            timeframe_analysis = self._analyze_timeframes(df)
            
            # 生成移动端优化报告
            mobile_report = self._generate_mobile_report(analysis_result)
            
            return {
                'charts': charts,
                'timeframe_analysis': timeframe_analysis,
                'mobile_report': mobile_report,
                'analysis_result': analysis_result
            }
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            return {}
            
    def _generate_charts(self, df: pd.DataFrame, analysis_result: Dict) -> Dict:
        """
        生成图表
        :param df: 股票数据
        :param analysis_result: 分析结果
        :return: 图表数据
        """
        try:
            charts = {}
            
            # 生成K线图
            charts['candlestick'] = self._generate_candlestick_chart(df, analysis_result)
            
            # 生成技术指标图
            charts['indicators'] = self._generate_indicators_chart(df, analysis_result)
            
            # 生成成交量图
            charts['volume'] = self._generate_volume_chart(df)
            
            # 生成斐波那契回调图
            charts['fibonacci'] = self._generate_fibonacci_chart(df, analysis_result)
            
            return charts
            
        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")
            return {}
            
    def _generate_candlestick_chart(self, df: pd.DataFrame, analysis_result: Dict) -> Dict:
        """
        生成K线图
        """
        try:
            fig = go.Figure()
            
            # 添加K线
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线'
            ))
            
            # 添加布林带
            bb = analysis_result.get('technical_indicators', {}).get('bollinger_bands', {})
            if bb:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[bb['upper']] * len(df),
                    name='布林带上轨',
                    line=dict(color='rgba(255, 0, 0, 0.5)')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[bb['middle']] * len(df),
                    name='布林带中轨',
                    line=dict(color='rgba(0, 0, 255, 0.5)')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[bb['lower']] * len(df),
                    name='布林带下轨',
                    line=dict(color='rgba(0, 255, 0, 0.5)')
                ))
                
            # 添加止损止盈线
            if analysis_result.get('stop_loss_price'):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[analysis_result['stop_loss_price']] * len(df),
                    name='止损价',
                    line=dict(color='rgba(255, 0, 0, 0.8)', dash='dash')
                ))
            if analysis_result.get('take_profit_price'):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[analysis_result['take_profit_price']] * len(df),
                    name='止盈价',
                    line=dict(color='rgba(0, 255, 0, 0.8)', dash='dash')
                ))
                
            # 更新布局
            fig.update_layout(
                title='K线图',
                xaxis_title='日期',
                yaxis_title='价格',
                template='plotly_white',
                height=600
            )
            
            return fig.to_dict()
            
        except Exception as e:
            self.logger.error(f"生成K线图失败: {e}")
            return {}
            
    def _generate_indicators_chart(self, df: pd.DataFrame, analysis_result: Dict) -> Dict:
        """
        生成技术指标图
        """
        try:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            
            # 添加KDJ
            kdj = analysis_result.get('technical_indicators', {}).get('kdj', {})
            if kdj:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[kdj['k']] * len(df),
                    name='K值',
                    line=dict(color='blue')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[kdj['d']] * len(df),
                    name='D值',
                    line=dict(color='red')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[kdj['j']] * len(df),
                    name='J值',
                    line=dict(color='green')
                ), row=1, col=1)
                
            # 添加RSI
            rsi = talib.RSI(df['close'].values)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=rsi,
                name='RSI',
                line=dict(color='purple')
            ), row=2, col=1)
            
            # 更新布局
            fig.update_layout(
                title='技术指标',
                height=600,
                template='plotly_white'
            )
            
            return fig.to_dict()
            
        except Exception as e:
            self.logger.error(f"生成技术指标图失败: {e}")
            return {}
            
    def _generate_volume_chart(self, df: pd.DataFrame) -> Dict:
        """生成成交量图"""
        try:
            # 确保列名小写
            df = df.copy()
            df.columns = df.columns.str.lower()
            
            # 确保成交量数据是 double 类型
            if 'volume' not in df.columns:
                self.logger.warning("数据中缺少成交量列")
                return {}
                
            # 转换成交量为数值类型
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # 处理可能的 NaN 值
            df['volume'] = df['volume'].fillna(0)
            
            # 创建成交量图
            volume_fig = go.Figure()
            volume_fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'].astype(float),  # 确保是 float 类型
                    name='Volume',
                    marker_color='rgba(0, 128, 255, 0.7)'
                )
            )
            
            # 设置布局
            volume_fig.update_layout(
                title='Volume',
                xaxis_title='Date',
                yaxis_title='Volume',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return volume_fig.to_dict()
        except Exception as e:
            self.logger.error(f"生成成交量图失败: {str(e)}")
            return {}
            
    def _generate_fibonacci_chart(self, df: pd.DataFrame, analysis_result: Dict) -> Dict:
        """
        生成斐波那契回调图
        """
        try:
            fig = go.Figure()
            
            # 添加K线
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线'
            ))
            
            # 添加斐波那契回调位
            fib = analysis_result.get('technical_indicators', {}).get('fibonacci_levels', {})
            if fib and fib.get('levels'):
                levels = fib['levels']
                for level, price in levels.items():
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=[price] * len(df),
                        name=f'斐波那契{level}',
                        line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash')
                    ))
                    
            # 更新布局
            fig.update_layout(
                title='斐波那契回调分析',
                xaxis_title='日期',
                yaxis_title='价格',
                template='plotly_white',
                height=600
            )
            
            return fig.to_dict()
            
        except Exception as e:
            self.logger.error(f"生成斐波那契回调图失败: {e}")
            return {}
            
    def _analyze_timeframes(self, df: pd.DataFrame) -> Dict:
        """
        分析多时间周期
        :param df: 股票数据
        :return: 多时间周期分析结果
        """
        try:
            # 转换为不同时间周期
            daily_df = df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            weekly_df = df.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            monthly_df = df.resample('ME').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return {
                'daily': self._analyze_single_timeframe(daily_df, '日线'),
                'weekly': self._analyze_single_timeframe(weekly_df, '周线'),
                'monthly': self._analyze_single_timeframe(monthly_df, '月线')
            }
            
        except Exception as e:
            self.logger.error(f"分析多时间周期失败: {e}")
            return {}
            
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        分析单个时间周期
        :param df: 股票数据
        :param timeframe: 时间周期
        :return: 分析结果
        """
        try:
            # 计算收益率
            returns = df['close'].pct_change(fill_method=None)
            
            # 计算波动率
            volatility = returns.std() * np.sqrt(252)
            
            # 计算趋势
            trend = self._calculate_trend(df['close'])
            
            return {
                'timeframe': timeframe,
                'returns': returns.mean(),
                'volatility': volatility,
                'trend': trend
            }
            
        except Exception as e:
            self.logger.error(f"分析{timeframe}失败: {e}")
            return {}
            
    def _generate_mobile_report(self, analysis_result: Dict) -> Dict:
        """
        生成移动端优化报告
        :param analysis_result: 分析结果
        :return: 移动端报告数据
        """
        try:
            # 提取关键信息
            score = analysis_result.get('score', 0)
            risk_level = analysis_result.get('risk_level', 'normal')
            recommendations = analysis_result.get('recommendations', [])
            
            # 生成简洁的建议
            short_recommendations = []
            for rec in recommendations[:3]:  # 只取前3条建议
                short_recommendations.append(rec)
                
            return {
                'score': score,
                'risk_level': risk_level,
                'recommendations': short_recommendations,
                'technical_indicators': self._simplify_technical_indicators(
                    analysis_result.get('technical_indicators', {})
                )
            }
            
        except Exception as e:
            self.logger.error(f"生成移动端报告失败: {e}")
            return {}
            
    def _simplify_technical_indicators(self, indicators: Dict) -> Dict:
        """
        简化技术指标数据
        :param indicators: 技术指标数据
        :return: 简化后的技术指标数据
        """
        try:
            simplified = {}
            
            # 简化布林带数据
            bb = indicators.get('bollinger_bands', {})
            if bb:
                simplified['bollinger_bands'] = {
                    'position': bb.get('position', 0),
                    'signal': 'upper' if bb.get('position', 0) > 0.8 else 
                             'lower' if bb.get('position', 0) < 0.2 else 'middle'
                }
                
            # 简化KDJ数据
            kdj = indicators.get('kdj', {})
            if kdj:
                simplified['kdj'] = {
                    'signal': kdj.get('signal', 'neutral')
                }
                
            # 简化成交量数据
            volume = indicators.get('volume_indicators', {})
            if volume:
                simplified['volume'] = {
                    'trend': 'increasing' if volume.get('volume_ratio', 1) > 1 else 'decreasing'
                }
                
            return simplified
            
        except Exception as e:
            self.logger.error(f"简化技术指标数据失败: {e}")
            return {}

    def _calculate_trend(self, data: pd.Series) -> str:
        """
        计算趋势方向
        :param data: 价格数据序列
        :return: 趋势方向 ('上升', '下降', '震荡')
        """
        try:
            if len(data) < 2:
                return '震荡'
            
            # 计算短期和长期移动平均线
            short_ma = data.rolling(window=5).mean()
            long_ma = data.rolling(window=20).mean()
            
            # 计算趋势强度
            current_short = short_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            prev_short = short_ma.iloc[-2]
            prev_long = long_ma.iloc[-2]
            
            # 判断趋势
            if current_short > current_long and current_short > prev_short:
                return '上升'
            elif current_short < current_long and current_short < prev_short:
                return '下降'
            else:
                return '震荡'
            
        except Exception as e:
            self.logger.error(f"计算趋势时出错: {e}")
            return '震荡'

    def generate_email_report(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        生成邮件报告
        
        参数:
            data: 包含分析结果的字典
            
        返回:
            包含HTML和纯文本内容的字典
        """
        try:
            # 生成HTML内容
            html_content = self._generate_html_content(data)
            
            # 生成纯文本内容
            text_content = self._generate_text_content(data)
            
            return {
                'html_content': html_content,
                'text_content': text_content
            }
            
        except Exception as e:
            self.logger.error(f"生成邮件报告时出错: {str(e)}")
            return {
                'html_content': f"生成报告时出错: {str(e)}",
                'text_content': f"生成报告时出错: {str(e)}"
            }

    def _generate_html_content(self, data: Dict[str, Any]) -> str:
        """
        生成HTML内容
        :param data: 分析结果
        :return: HTML内容
        """
        try:
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; margin-top: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .alert {{ color: #d9534f; }}
                    .warning {{ color: #f0ad4e; }}
                    .info {{ color: #5bc0de; }}
                    .success {{ color: #5cb85c; }}
                </style>
            </head>
            <body>
                <h1>市场监控报告</h1>
                <p>报告生成时间: {data.get('timestamp', '')}</p>
                
                <h2>市场概览</h2>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>值</th>
                        <th>状态</th>
                    </tr>
                    <tr>
                        <td>市场趋势</td>
                        <td>{data.get('market_trend', {}).get('trend', '')}</td>
                        <td class="{data.get('market_trend', {}).get('trend_class', '')}">
                            {data.get('market_trend', {}).get('trend', '')}
                        </td>
                    </tr>
                    <tr>
                        <td>波动性</td>
                        <td>{data.get('volatility', {}).get('level', '')}</td>
                        <td class="{data.get('volatility', {}).get('level_class', '')}">
                            {data.get('volatility', {}).get('level', '')}
                        </td>
                    </tr>
                    <tr>
                        <td>风险水平</td>
                        <td>{data.get('risk_level', {}).get('level', '')}</td>
                        <td class="{data.get('risk_level', {}).get('level_class', '')}">
                            {data.get('risk_level', {}).get('level', '')}
                        </td>
                    </tr>
                </table>
                
                <h2>股票分析</h2>
                <table>
                    <tr>
                        <th>股票代码</th>
                        <th>价格</th>
                        <th>变化</th>
                        <th>波动率</th>
                        <th>仓位建议</th>
                        <th>推荐操作</th>
                    </tr>
            """
            
            # 添加股票分析数据
            for symbol, stock_data in data.get('stock_analysis', {}).items():
                html_content += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{stock_data.get('price', '')}</td>
                        <td class="{stock_data.get('change_class', '')}">
                            {stock_data.get('change', '')}
                        </td>
                        <td>{stock_data.get('volatility', '')}</td>
                        <td>{stock_data.get('position_size', '')}</td>
                        <td>
                            <ul>
                """
                
                # 添加推荐信息
                for rec in stock_data.get('recommendations', []):
                    html_content += f"<li>{rec}</li>"
                    
                html_content += """
                            </ul>
                        </td>
                    </tr>
                """
                
            html_content += """
                </table>
                
                <h2>警报信息</h2>
                <table>
                    <tr>
                        <th>时间</th>
                        <th>股票</th>
                        <th>类型</th>
                        <th>消息</th>
                        <th>状态</th>
                    </tr>
            """
            
            # 添加警报信息
            for alert in data.get('alerts', []):
                html_content += f"""
                    <tr>
                        <td>{alert.get('timestamp', '')}</td>
                        <td>{alert.get('symbol', '')}</td>
                        <td>{alert.get('type', '')}</td>
                        <td>{alert.get('message', '')}</td>
                        <td class="{alert.get('status_class', '')}">
                            {alert.get('status', '')}
                        </td>
                    </tr>
                """
                
            html_content += """
                </table>
                
                <h2>市场情绪</h2>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>值</th>
                        <th>状态</th>
                    </tr>
            """
            
            # 添加市场情绪数据
            sentiment = data.get('market_sentiment', {})
            html_content += f"""
                <tr>
                    <td>整体情绪</td>
                    <td>{sentiment.get('overall', '')}</td>
                    <td class="{sentiment.get('overall_class', '')}">
                        {sentiment.get('overall', '')}
                    </td>
                </tr>
                <tr>
                    <td>新闻情绪</td>
                    <td>{sentiment.get('news', '')}</td>
                    <td class="{sentiment.get('news_class', '')}">
                        {sentiment.get('news', '')}
                    </td>
                </tr>
                <tr>
                    <td>社交媒体情绪</td>
                    <td>{sentiment.get('social', '')}</td>
                    <td class="{sentiment.get('social_class', '')}">
                        {sentiment.get('social', '')}
                    </td>
                </tr>
            """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"生成HTML内容时出错: {str(e)}")
            return f"生成HTML内容时出错: {str(e)}"

    def _generate_text_content(self, data: Dict[str, Any]) -> str:
        """
        生成纯文本内容
        :param data: 分析结果
        :return: 纯文本内容
        """
        try:
            text_content = f"""
市场监控报告
============
报告生成时间: {data.get('timestamp', '')}

市场概览
--------
市场趋势: {data.get('market_trend', {}).get('trend', '')}
波动性: {data.get('volatility', {}).get('level', '')}
风险水平: {data.get('risk_level', {}).get('level', '')}

股票分析
--------
"""
            
            # 添加股票分析数据
            for symbol, stock_data in data.get('stock_analysis', {}).items():
                text_content += f"""
{symbol}:
- 价格: {stock_data.get('price', '')}
- 变化: {stock_data.get('change', '')}
- 波动率: {stock_data.get('volatility', '')}
- 仓位建议: {stock_data.get('position_size', '')}
- 推荐操作:
"""
                
                # 添加推荐信息
                for rec in stock_data.get('recommendations', []):
                    text_content += f"  * {rec}\n"
                    
            text_content += """
警报信息
--------
"""
            
            # 添加警报信息
            for alert in data.get('alerts', []):
                text_content += f"""
时间: {alert.get('timestamp', '')}
股票: {alert.get('symbol', '')}
类型: {alert.get('type', '')}
消息: {alert.get('message', '')}
状态: {alert.get('status', '')}
"""
                
            text_content += """
市场情绪
--------
"""
            
            # 添加市场情绪数据
            sentiment = data.get('market_sentiment', {})
            text_content += f"""
整体情绪: {sentiment.get('overall', '')}
新闻情绪: {sentiment.get('news', '')}
社交媒体情绪: {sentiment.get('social', '')}
"""
            
            return text_content
            
        except Exception as e:
            self.logger.error(f"生成纯文本内容时出错: {str(e)}")
            return f"生成纯文本内容时出错: {str(e)}" 