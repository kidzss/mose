import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, List
import yfinance as yf
from jinja2 import Template
import logging

# 添加路径以导入宏观分析模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))
try:
    from analysis.portfolio_macro_integration import PortfolioMacroIntegration
    MACRO_ANALYSIS_AVAILABLE = True
except ImportError:
    MACRO_ANALYSIS_AVAILABLE = False
    logging.warning("宏观分析模块不可用，将跳过宏观分析部分")

class ReportGenerator:
    """投资组合报告生成器"""
    
    def __init__(self, config=None):
        """初始化报告生成器"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化宏观分析器
        self.macro_integration = None
        if MACRO_ANALYSIS_AVAILABLE:
            try:
                self.macro_integration = PortfolioMacroIntegration()
                self.logger.info("宏观分析模块初始化成功")
            except Exception as e:
                self.logger.error(f"宏观分析模块初始化失败: {e}")
                self.macro_integration = None
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
                .macro-analysis { 
                    background-color: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 20px 0; 
                    border-left: 4px solid #007bff;
                }
                .macro-score { margin-bottom: 15px; }
                .macro-components { margin: 15px 0; }
                .sector-impact { margin: 15px 0; }
                .sector-impact p { margin: 5px 0; }
                .recommendations div { 
                    font-size: 0.9em; 
                    margin: 2px 0; 
                    padding: 2px 5px;
                    background-color: #fff3cd;
                    border-radius: 3px;
                }
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

            {% if macro_analysis %}
            <div class="macro-analysis">
                <h3>🌍 宏观环境分析</h3>
                <div class="macro-score">
                    <p><strong>宏观得分:</strong> 
                        <span class="{{ 'profit' if macro_analysis.macro_score >= 0.6 else 'loss' if macro_analysis.macro_score < 0.4 else '' }}">
                            {{ "{:.2f}".format(macro_analysis.macro_score) }}/1.00 ({{ "{:.0f}".format(macro_analysis.macro_score * 100) }}分)
                        </span>
                    </p>
                    <p><strong>环境建议:</strong> {{ macro_analysis.recommendation }}</p>
                </div>
                
                {% if macro_analysis.components %}
                <div class="macro-components">
                    <h4>📊 关键宏观指标</h4>
                    {% if macro_analysis.components.interest_rate %}
                    <p><strong>💹 利率环境:</strong> {{ macro_analysis.components.interest_rate.description }} 
                       (得分: {{ "{:.1f}".format(macro_analysis.components.interest_rate.curve_score) }})</p>
                    {% endif %}
                    
                    {% if macro_analysis.components.market_sentiment %}
                    <p><strong>📈 市场情绪:</strong> VIX {{ "{:.1f}".format(macro_analysis.components.market_sentiment.vix_level) }}, 
                       投资者情绪{{ macro_analysis.components.market_sentiment.sentiment_desc }}</p>
                    {% endif %}
                    
                    {% if macro_analysis.components.dollar_strength %}
                    <p><strong>💵 美元强度:</strong> DXY {{ "{:.1f}".format(macro_analysis.components.dollar_strength.dxy_level) }}, 
                       {{ macro_analysis.components.dollar_strength.trend_desc }}</p>
                    {% endif %}
                </div>
                {% endif %}
                
                {% if macro_analysis.sector_impact %}
                <div class="sector-impact">
                    <h4>🏭 行业影响评估</h4>
                    {% for sector, score in macro_analysis.sector_impact.items() %}
                    <p><strong>{{ sector }}:</strong> 
                        <span class="{{ 'profit' if score >= 0.6 else 'loss' if score < 0.4 else '' }}">
                            {{ "{:.2f}".format(score) }}分
                        </span>
                        {% if score >= 0.6 %}
                        - 宏观环境有利，可考虑增加配置
                        {% elif score < 0.4 %}
                        - 宏观环境不利，建议减少配置
                        {% else %}
                        - 宏观环境中性，维持现有配置
                        {% endif %}
                    </p>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endif %}

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
                    {% if macro_analysis %}
                    <th>宏观影响</th>
                    <th>操作建议</th>
                    {% endif %}
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
                    {% if macro_analysis and position.macro_impact %}
                    <td class="{{ 'profit' if position.macro_impact.impact_score >= 0.6 else 'loss' if position.macro_impact.impact_score < 0.4 else '' }}">
                        {{ "{:.2f}".format(position.macro_impact.impact_score) }}
                        {% if position.macro_impact.impact_level == 'positive' %}🟢
                        {% elif position.macro_impact.impact_level == 'negative' %}🔴
                        {% else %}🟡{% endif %}
                    </td>
                    <td class="recommendations">
                        {% if position.macro_impact.recommendations %}
                            {% for rec in position.macro_impact.recommendations[:2] %}
                                <div>{{ rec }}</div>
                            {% endfor %}
                        {% endif %}
                    </td>
                    {% elif macro_analysis %}
                    <td>-</td>
                    <td>-</td>
                    {% endif %}
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

    def _calculate_market_returns(self) -> tuple:
        """计算大盘指数收益率"""
        try:
            sp500 = yf.download('^GSPC', period='2d', auto_adjust=True)
            nasdaq = yf.download('^IXIC', period='2d', auto_adjust=True)
            
            sp500_return = (sp500['Close'].iloc[-1] / sp500['Close'].iloc[-2]) - 1
            nasdaq_return = (nasdaq['Close'].iloc[-1] / nasdaq['Close'].iloc[-2]) - 1
            
            return sp500_return, nasdaq_return
        except Exception as e:
            print(f"获取市场数据时出错: {e}")
            return 0.0, 0.0

    def _calculate_risk_metrics(self, portfolio_monitor) -> Dict:
        """计算风险指标"""
        try:
            # 获取SP500收益率作为市场基准
            sp500 = yf.download('^GSPC', period='1y', auto_adjust=True)['Close'].pct_change().dropna()
            
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
    
    def _get_macro_analysis(self) -> Dict:
        """获取宏观分析数据"""
        if not self.macro_integration:
            return None
        
        try:
            self.logger.info("开始获取宏观分析数据...")
            macro_report = self.macro_integration.generate_macro_report()
            
            if 'error' in macro_report:
                self.logger.error(f"宏观分析失败: {macro_report['error']}")
                return None
            
            # 提取需要的数据
            detailed_analysis = macro_report.get('detailed_analysis', {})
            macro_analysis_data = detailed_analysis.get('macro_analysis', {})
            sector_impact = detailed_analysis.get('sector_impact', {})
            portfolio_impact = detailed_analysis.get('portfolio_impact', {})
            
            # 构建简化的宏观分析数据结构
            macro_data = {
                'macro_score': macro_analysis_data.get('macro_score', 0),
                'recommendation': macro_analysis_data.get('recommendation', ''),
                'components': {},
                'sector_impact': sector_impact,
                'individual_impacts': portfolio_impact.get('individual_impacts', {})
            }
            
            # 处理宏观组件数据
            components = macro_analysis_data.get('components', {})
            
            # 利率环境
            if 'interest_rate' in components:
                rate_data = components['interest_rate']
                curve_shape = rate_data.get('curve_shape', 'unknown')
                
                # 生成描述
                if curve_shape == 'normal':
                    description = "收益率曲线正常，利率环境稳定"
                elif curve_shape == 'inverted':
                    description = "收益率曲线倒挂，经济可能面临衰退风险"
                else:
                    description = "收益率曲线平坦，经济增长动力不足"
                
                macro_data['components']['interest_rate'] = {
                    'description': description,
                    'curve_score': rate_data.get('curve_score', 0),
                    'curve_shape': curve_shape
                }
            
            # 市场情绪
            if 'market_sentiment' in components:
                sentiment_data = components['market_sentiment']
                vix_level = sentiment_data.get('vix_level', 0)
                vix_sentiment = sentiment_data.get('vix_sentiment', 'unknown')
                
                if vix_sentiment == 'low':
                    sentiment_desc = "乐观，投资者风险偏好较高"
                elif vix_sentiment == 'normal':
                    sentiment_desc = "正常，投资者相对理性"
                else:
                    sentiment_desc = "恐慌，投资者风险规避"
                
                macro_data['components']['market_sentiment'] = {
                    'vix_level': vix_level,
                    'sentiment_desc': sentiment_desc,
                    'vix_sentiment': vix_sentiment
                }
            
            # 美元强度
            if 'dollar_strength' in components:
                dollar_data = components['dollar_strength']
                dollar_trend = dollar_data.get('dollar_trend', 'unknown')
                dxy_level = dollar_data.get('dxy_level', 0)
                
                if dollar_trend == 'strong':
                    trend_desc = "美元走强，对新兴市场和大宗商品不利"
                elif dollar_trend == 'weak':
                    trend_desc = "美元走弱，有利于风险资产和新兴市场"
                else:
                    trend_desc = "美元震荡，对市场影响中性"
                
                macro_data['components']['dollar_strength'] = {
                    'dxy_level': dxy_level,
                    'trend_desc': trend_desc,
                    'dollar_trend': dollar_trend
                }
            
            self.logger.info("宏观分析数据获取成功")
            return macro_data
            
        except Exception as e:
            self.logger.error(f"获取宏观分析数据失败: {e}")
            return None

    def _generate_market_analysis(self) -> Dict:
        """生成市场分析"""
        try:
            # 获取主要指数数据
            indices = {
                'SPY': yf.download('^GSPC', period='5d', auto_adjust=True),
                'QQQ': yf.download('^IXIC', period='5d', auto_adjust=True),
                'VIX': yf.download('^VIX', period='5d', auto_adjust=True)
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
            
            # 获取宏观分析数据
            macro_analysis = self._get_macro_analysis()
            self.logger.info(f"宏观分析数据获取: {'成功' if macro_analysis else '失败或跳过'}")
            
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
                    
                    # 获取宏观影响数据
                    macro_impact = None
                    if macro_analysis and symbol in macro_analysis.get('individual_impacts', {}):
                        individual_impact = macro_analysis['individual_impacts'][symbol]
                        
                        # 生成简化的操作建议
                        recommendations = []
                        impact_score = individual_impact.get('impact_score', 0.5)
                        impact_level = individual_impact.get('impact_level', 'neutral')
                        
                        # 基于宏观影响生成建议
                        if impact_level == 'positive':
                            recommendations.append("宏观环境有利，可考虑逢低加仓")
                        elif impact_level == 'negative':
                            recommendations.append("宏观环境不利，建议降低仓位")
                        elif impact_level == 'very_negative':
                            recommendations.append("宏观环境恶劣，强烈建议减仓")
                        
                        # 基于收益率生成建议
                        total_return = (current_price - cost_basis) / cost_basis
                        if total_return > 0.15:
                            recommendations.append("盈利丰厚，建议部分止盈")
                        elif total_return < -0.08:
                            recommendations.append("亏损较大，需重点关注")
                        
                        # 基于仓位权重生成建议
                        if position['weight'] > 0.25:
                            recommendations.append("仓位过重，建议分散投资")
                        
                        macro_impact = {
                            'impact_score': impact_score,
                            'impact_level': impact_level,
                            'recommendations': recommendations
                        }
                    
                    positions.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'cost_basis': cost_basis,
                        'shares': shares,
                        'market_value': market_value,
                        'weight': position['weight'],
                        'daily_return': daily_return,
                        'total_return': (current_price - cost_basis) / cost_basis,
                        'macro_impact': macro_impact
                    })
            
            # 生成分析
            market_analysis = self._generate_market_analysis()
            position_analysis = self._generate_position_analysis(positions)
            strategy_recommendations = self._generate_strategy_recommendations(position_analysis, market_analysis)
            
            # 计算投资组合整体收益
            portfolio_daily_return = sum(daily_returns) if daily_returns else 0
            portfolio_total_return = sum((p['market_value'] - p['shares'] * p['cost_basis']) for p in positions) / sum(p['shares'] * p['cost_basis'] for p in positions)
            
            # 生成报告
            report = self.report_template.render(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                total_value=total_value,
                daily_return=portfolio_daily_return,
                total_return=portfolio_total_return,
                sp500_return=sp500_return,
                nasdaq_return=nasdaq_return,
                relative_return=portfolio_daily_return - sp500_return,
                positions=positions,
                portfolio_beta=risk_metrics['beta'],
                sharpe_ratio=risk_metrics['sharpe'],
                max_drawdown=risk_metrics['max_drawdown'],
                volatility=risk_metrics['volatility'],
                var_95=risk_metrics['var_95'],
                alerts=portfolio_monitor.check_alerts(),
                recommendations=strategy_recommendations,
                market_analysis=market_analysis,
                position_analysis=position_analysis,
                strategy_recommendations=strategy_recommendations,
                macro_analysis=macro_analysis
            )
            
            return report
        except Exception as e:
            print(f"生成报告时出错: {e}")
            return f"生成报告时出错: {str(e)}"

    def generate_report(self, analysis_result: Dict, data: pd.DataFrame, symbol: str) -> Dict:
        """生成分析报告"""
        try:
            report = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': analysis_result,
                'data_summary': {
                    'last_price': data['close'].iloc[-1] if not data.empty else None,
                    'price_change': (data['close'].iloc[-1] / data['close'].iloc[-2] - 1) if len(data) > 1 else None,
                    'volume': data['volume'].iloc[-1] if not data.empty else None,
                    'volume_change': (data['volume'].iloc[-1] / data['volume'].iloc[-2] - 1) if len(data) > 1 else None
                }
            }
            return report
        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}")
            return {} 