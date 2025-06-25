#!/usr/bin/env python3
"""
股票类型分析器 - 根据股票特征调整分析权重
"""

class StockTypeAnalyzer:
    """股票类型分析器，为不同类型股票分配合适的分析权重"""
    
    def __init__(self):
        # 定义股票类型及其特征
        self.stock_types = {
            'wave_trading': {  # 波段交易股票（如TSLA）
                'name': '波段交易股',
                'weights': {
                    'technical': 0.50,    # 技术面50%
                    'fundamental': 0.40,  # 基本面40%
                    'sentiment': 0.10     # 市场情绪10%
                },
                'characteristics': [
                    '高波动性（日波动>3%）',
                    '情绪驱动明显',
                    '技术走势清晰',
                    '基本面滞后性强'
                ],
                'key_indicators': ['RSI', 'MACD', '成交量', '突破信号'],
                'risk_level': 'high'
            },
            'value_investing': {  # 价值投资股票（如BRK.B）
                'name': '价值投资股',
                'weights': {
                    'technical': 0.20,    # 技术面20%
                    'fundamental': 0.70,  # 基本面70%
                    'sentiment': 0.10     # 市场情绪10%
                },
                'characteristics': [
                    '低波动性',
                    '基本面驱动',
                    '长期投资价值',
                    '分红稳定'
                ],
                'key_indicators': ['PE', 'PB', 'ROE', '分红率'],
                'risk_level': 'low'
            },
            'growth_stock': {  # 成长股（如NVDA）
                'name': '成长股',
                'weights': {
                    'technical': 0.35,    # 技术面35%
                    'fundamental': 0.55,  # 基本面55%
                    'sentiment': 0.10     # 市场情绪10%
                },
                'characteristics': [
                    '中等波动性',
                    '业绩增长驱动',
                    '估值相对合理',
                    '行业前景好'
                ],
                'key_indicators': ['EPS增长', 'PEG', '营收增长', '行业地位'],
                'risk_level': 'medium'
            },
            'defensive_stock': {  # 防御性股票（如JNJ, PG）
                'name': '防御性股票',
                'weights': {
                    'technical': 0.15,    # 技术面15%
                    'fundamental': 0.75,  # 基本面75%
                    'sentiment': 0.10     # 市场情绪10%
                },
                'characteristics': [
                    '极低波动性',
                    '稳定现金流',
                    '抗周期性强',
                    '分红持续'
                ],
                'key_indicators': ['分红率', '现金流', '债务率', '市场份额'],
                'risk_level': 'very_low'
            }
        }
        
        # 股票分类规则
        self.classification_rules = {
            'TSLA': 'wave_trading',
            'GME': 'wave_trading', 
            'AMC': 'wave_trading',
            'NVDA': 'growth_stock',
            'AAPL': 'growth_stock',
            'MSFT': 'growth_stock',
            'GOOGL': 'growth_stock',
            'AMD': 'growth_stock',
            'BRK-B': 'value_investing',
            'JPM': 'value_investing',
            'JNJ': 'defensive_stock',
            'PG': 'defensive_stock',
            'KO': 'defensive_stock',
            'MRK': 'defensive_stock'
        }
    
    def classify_stock(self, symbol: str, price_data=None, financial_data=None) -> str:
        """
        分类股票类型
        
        Args:
            symbol: 股票代码
            price_data: 价格数据（用于计算波动率）
            financial_data: 财务数据
            
        Returns:
            股票类型
        """
        # 首先检查预定义分类
        if symbol in self.classification_rules:
            return self.classification_rules[symbol]
        
        # 基于数据特征自动分类
        if price_data is not None:
            return self._classify_by_features(symbol, price_data, financial_data)
        
        # 默认分类
        return 'growth_stock'
    
    def _classify_by_features(self, symbol: str, price_data, financial_data) -> str:
        """基于特征自动分类股票"""
        try:
            # 计算波动率（20日标准差）
            # 兼容不同数据源的列名（Close/close）
            close_col = 'Close' if 'Close' in price_data.columns else 'close'
            returns = price_data[close_col].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * 100
            
            # 波动率分类
            if volatility > 4.0:  # 日波动率>4%
                return 'wave_trading'
            elif volatility < 1.5:  # 日波动率<1.5%
                if financial_data and financial_data.get('dividend_yield', 0) > 3:
                    return 'defensive_stock'
                else:
                    return 'value_investing'
            else:  # 中等波动率
                return 'growth_stock'
                
        except Exception as e:
            print(f"自动分类失败: {e}")
            return 'growth_stock'
    
    def get_analysis_weights(self, symbol: str, price_data=None, financial_data=None) -> dict:
        """
        获取股票的分析权重
        
        Returns:
            包含技术面、基本面、市场情绪权重的字典
        """
        stock_type = self.classify_stock(symbol, price_data, financial_data)
        return self.stock_types[stock_type]['weights']
    
    def get_stock_info(self, symbol: str, price_data=None, financial_data=None) -> dict:
        """获取股票完整信息"""
        stock_type = self.classify_stock(symbol, price_data, financial_data)
        stock_info = self.stock_types[stock_type].copy()
        stock_info['type'] = stock_type
        stock_info['symbol'] = symbol
        return stock_info
    
    def calculate_comprehensive_score(self, symbol: str, technical_score: float, 
                                    fundamental_score: float, sentiment_score: float,
                                    price_data=None, financial_data=None) -> dict:
        """
        计算综合评分
        
        Args:
            symbol: 股票代码
            technical_score: 技术面评分 (0-10)
            fundamental_score: 基本面评分 (0-10)
            sentiment_score: 市场情绪评分 (0-10)
            
        Returns:
            综合分析结果
        """
        weights = self.get_analysis_weights(symbol, price_data, financial_data)
        stock_info = self.get_stock_info(symbol, price_data, financial_data)
        
        # 计算加权综合评分
        comprehensive_score = (
            technical_score * weights['technical'] +
            fundamental_score * weights['fundamental'] +
            sentiment_score * weights['sentiment']
        )
        
        # 生成评级
        if comprehensive_score >= 8.0:
            rating = '强烈买入'
            rating_color = 'excellent'
        elif comprehensive_score >= 6.5:
            rating = '买入'
            rating_color = 'good'
        elif comprehensive_score >= 5.0:
            rating = '持有'
            rating_color = 'neutral'
        elif comprehensive_score >= 3.5:
            rating = '减持'
            rating_color = 'poor'
        else:
            rating = '卖出'
            rating_color = 'critical'
        
        return {
            'symbol': symbol,
            'stock_type': stock_info['type'],
            'stock_name': stock_info['name'],
            'weights': weights,
            'scores': {
                'technical': technical_score,
                'fundamental': fundamental_score,
                'sentiment': sentiment_score,
                'comprehensive': comprehensive_score
            },
            'rating': rating,
            'rating_color': rating_color,
            'risk_level': stock_info['risk_level'],
            'key_indicators': stock_info['key_indicators'],
            'characteristics': stock_info['characteristics'],
            'analysis_focus': self._get_analysis_focus(stock_info['type']),
            'trading_strategy': self._get_trading_strategy(stock_info['type'])
        }
    
    def _get_analysis_focus(self, stock_type: str) -> list:
        """获取分析重点"""
        focus_map = {
            'wave_trading': [
                '关注技术突破信号',
                '监控成交量变化',
                '跟踪市场情绪指标',
                '设置严格止损位'
            ],
            'value_investing': [
                '深度财务分析',
                '估值水平评估',
                '行业竞争地位',
                '长期投资价值'
            ],
            'growth_stock': [
                '业绩增长持续性',
                '行业发展前景',
                '技术走势确认',
                '估值合理性'
            ],
            'defensive_stock': [
                '分红稳定性',
                '现金流质量',
                '抗风险能力',
                '长期持有价值'
            ]
        }
        return focus_map.get(stock_type, [])
    
    def _get_trading_strategy(self, stock_type: str) -> dict:
        """获取交易策略建议"""
        strategy_map = {
            'wave_trading': {
                'position_size': '中等仓位（5-15%）',
                'holding_period': '短期-中期（1-6个月）',
                'stop_loss': '严格止损（-8%）',
                'take_profit': '分批获利（+15%, +25%）',
                'rebalance_frequency': '每周检查'
            },
            'value_investing': {
                'position_size': '重仓持有（10-25%）',
                'holding_period': '长期（1-5年）',
                'stop_loss': '宽松止损（-20%）',
                'take_profit': '长期持有',
                'rebalance_frequency': '每季度检查'
            },
            'growth_stock': {
                'position_size': '标准仓位（8-20%）',
                'holding_period': '中长期（6个月-3年）',
                'stop_loss': '适中止损（-12%）',
                'take_profit': '趋势跟踪',
                'rebalance_frequency': '每月检查'
            },
            'defensive_stock': {
                'position_size': '稳定仓位（5-15%）',
                'holding_period': '长期（2-10年）',
                'stop_loss': '很宽松（-25%）',
                'take_profit': '分红再投资',
                'rebalance_frequency': '每半年检查'
            }
        }
        return strategy_map.get(stock_type, {}) 