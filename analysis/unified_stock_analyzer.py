"""
统一股票分析服务
Unified Stock Analysis Service

整合所有分析功能，提供统一的分析接口
供不同的UI系统调用，避免代码重复
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 导入现有的分析模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.financial_analyzer import FinancialAnalyzer
from monitor.stock_type_analyzer import StockTypeAnalyzer
from analysis.liquidity_analyzer import LiquidityAnalyzer


class UnifiedStockAnalyzer:
    """统一股票分析服务"""
    
    def __init__(self):
        """初始化分析服务"""
        self.financial_analyzer = FinancialAnalyzer()
        self.stock_type_analyzer = StockTypeAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        
        # 缓存配置
        self.cache_duration = 300  # 5分钟缓存
        self.cache = {}
        
        # 技术指标配置
        self.tech_config = {
            'rsi_period': 14,
            'ma_periods': [5, 20, 50, 200],
            'bollinger_period': 20,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
    
    def get_comprehensive_analysis(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        获取股票的综合分析
        
        Args:
            symbol: 股票代码
            force_refresh: 强制刷新缓存
            
        Returns:
            包含所有分析结果的字典
        """
        # 检查缓存
        cache_key = f"{symbol}_comprehensive"
        if not force_refresh and self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            print(f"🔍 正在分析 {symbol}...")
            
            # 1. 获取基础数据
            basic_data = self._get_basic_data(symbol)
            if not basic_data:
                return self._create_error_response(symbol, "无法获取基础数据")
            
            # 2. 技术面分析
            technical_analysis = self._analyze_technical(symbol, basic_data)
            
            # 3. 基本面分析
            fundamental_analysis = self._analyze_fundamental(symbol)
            
            # 4. 流动性分析
            liquidity_analysis = self._analyze_liquidity(symbol, basic_data)
            
            # 5. 智能增强分析
            enhanced_analysis = self._analyze_enhanced(symbol, basic_data, fundamental_analysis)
            
            # 6. 股票类型分析
            stock_type_analysis = self._analyze_stock_type(symbol, basic_data, fundamental_analysis)
            
            # 7. 右侧交易分析
            right_side_analysis = self._analyze_right_side_trading(symbol, technical_analysis, basic_data)
            
            # 8. 市场环境分析
            market_environment = self._analyze_market_environment(symbol, basic_data)
            
            # 9. 综合评分和建议
            comprehensive_rating = self._calculate_comprehensive_rating(
                technical_analysis, fundamental_analysis, enhanced_analysis, stock_type_analysis
            )
            
            # 整合所有分析结果
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'basic_info': basic_data['info'],
                'current_price': basic_data['current_price'],
                'prev_close': basic_data['prev_close'],
                'change': basic_data['change'],
                'change_pct': basic_data['change_pct'],
                'market_environment': market_environment,
                'technical_analysis': technical_analysis,
                'fundamental_analysis': fundamental_analysis,
                'liquidity_analysis': liquidity_analysis,
                'enhanced_analysis': enhanced_analysis,
                'stock_type_analysis': stock_type_analysis,
                'right_side_analysis': right_side_analysis,
                'comprehensive_rating': comprehensive_rating,
                'trading_suggestions': self._generate_trading_suggestions(
                    symbol, technical_analysis, fundamental_analysis, comprehensive_rating
                )
            }
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            print(f"✅ {symbol} 分析完成")
            return result
            
        except Exception as e:
            print(f"❌ {symbol} 分析失败: {e}")
            return self._create_error_response(symbol, str(e))
    
    def _get_basic_data(self, symbol: str) -> Optional[Dict]:
        """获取基础市场数据"""
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取历史数据（6个月用于技术分析）
            hist = ticker.history(period='6mo', interval='1d')
            if hist.empty or len(hist) < 30:
                return None
            
            # 获取股票信息
            info = ticker.info
            
            # 获取实时价格和前一交易日收盘价
            current_price = hist['Close'].iloc[-1]
            
            # 尝试从info获取更准确的前一日收盘价
            prev_close = info.get('regularMarketPreviousClose', info.get('previousClose'))
            if prev_close is None or prev_close == 0:
                # 如果info中没有，则使用历史数据的前一天
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # 计算价格变化
            change = current_price - prev_close
            change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
            
            # 调试信息
            print(f"{symbol} 价格信息: 当前={current_price:.2f}, 前收={prev_close:.2f}, 涨跌={change:+.2f} ({change_pct:+.2f}%)")
            
            return {
                'hist': hist,
                'info': info,
                'current_price': current_price,
                'prev_close': prev_close,
                'change': change,
                'change_pct': change_pct
            }
            
        except Exception as e:
            print(f"获取 {symbol} 基础数据失败: {e}")
            return None
    
    def _analyze_technical(self, symbol: str, basic_data: Dict) -> Dict:
        """技术面分析"""
        hist = basic_data['hist']
        current_price = basic_data['current_price']
        
        try:
            # 计算技术指标
            indicators = {}
            
            # 当前价格
            indicators['current_price'] = current_price
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(hist['Close'])
            
            # 移动平均线
            for period in self.tech_config['ma_periods']:
                if len(hist) >= period:
                    indicators[f'ma_{period}'] = hist['Close'].rolling(period).mean().iloc[-1]
            
            # MACD
            macd_data = self._calculate_macd(hist['Close'])
            indicators.update(macd_data)
            
            # 布林带
            bollinger_data = self._calculate_bollinger_bands(hist['Close'])
            indicators.update(bollinger_data)
            
            # 成交量分析
            volume_data = self._analyze_volume(hist)
            indicators.update(volume_data)
            
            # 52周数据
            indicators['high_52w'] = hist['High'].max()
            indicators['low_52w'] = hist['Low'].min()
            indicators['position_52w'] = ((current_price - indicators['low_52w']) / 
                                        (indicators['high_52w'] - indicators['low_52w']) * 100) if indicators['high_52w'] != indicators['low_52w'] else 50
            
            # 技术评分
            tech_score = self._calculate_technical_score(indicators, current_price)
            
            # 推荐策略
            strategy = self._determine_strategy(indicators, current_price)
            
            return {
                'indicators': indicators,
                'score': tech_score,
                'rating': self._get_technical_rating(tech_score),
                'strategy': strategy,
                'signal_strength': self._calculate_signal_strength(indicators),
                'trend_analysis': self._analyze_trend(indicators, current_price)
            }
            
        except Exception as e:
            print(f"技术分析失败 {symbol}: {e}")
            return {'error': str(e)}
    
    def _analyze_fundamental(self, symbol: str) -> Dict:
        """基本面分析"""
        try:
            # 使用现有的财务分析器
            result = self.financial_analyzer.analyze_stock(symbol)
            if not result:
                return {'error': '无法获取基本面数据'}
            
            # 调试输出
            print(f"财务分析结果: {result.get('total_score', 'N/A')}")
            
            # 确保数值转换正确
            total_score = result.get('total_score', 50)
            if isinstance(total_score, (int, float)) and total_score > 1:
                # 如果是百分制，转换为1分制
                overall_score = total_score / 100
            else:
                overall_score = total_score
            
            # 重新格式化结果以匹配界面需求
            dimensions = result.get('dimensions', {})
            formatted_result = {
                'valuation': dimensions.get('valuation', {'score': 0.5}),
                'profitability': dimensions.get('profitability', {'score': 0.5}),
                'growth': dimensions.get('growth', {'score': 0.5}),
                'financial_health': dimensions.get('financial_health', {'score': 0.5}),
                'analyst_sentiment': dimensions.get('analyst_sentiment', {'score': 0.5}),
                'overall_score': overall_score,
                'rating': result.get('overall_rating', '中性'),
                'investment_advice': result.get('investment_advice', {}),
                'company_info': self._format_company_info(result.get('basic_info', {}))
            }
            
            return formatted_result
            
        except Exception as e:
            print(f"基本面分析失败 {symbol}: {e}")
            return {'error': str(e)}
    
    def _analyze_liquidity(self, symbol: str, basic_data: Dict) -> Dict:
        """流动性分析"""
        try:
            # 使用现有的流动性分析器
            hist = basic_data['hist']
            current_price = basic_data['current_price']
            
            # 计算流动性指标
            avg_volume = hist['Volume'].tail(20).mean()
            current_volume = hist['Volume'].iloc[-1]
            
            # 价差分析（模拟）
            bid_ask_spread = 0.001  # 0.1% 模拟价差
            
            # 市值等级
            market_cap = basic_data['info'].get('marketCap', 0)
            market_cap_level = self._get_market_cap_level(market_cap)
            
            # 流动性评分
            liquidity_score = self._calculate_liquidity_score(
                avg_volume, current_volume, bid_ask_spread, market_cap
            )
            
            return {
                'score': liquidity_score,
                'risk_level': self._get_liquidity_risk_level(liquidity_score),
                'bid_ask_spread': bid_ask_spread,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'market_cap_level': market_cap_level,
                'analysis_points': self._generate_liquidity_points(liquidity_score, market_cap_level)
            }
            
        except Exception as e:
            print(f"流动性分析失败 {symbol}: {e}")
            return {'error': str(e)}
    
    def _analyze_enhanced(self, symbol: str, basic_data: Dict, fundamental_data: Dict) -> Dict:
        """智能增强分析"""
        try:
            # 结合基本面和技术面的增强分析
            overall_score = fundamental_data.get('overall_score', 0.5)
            
            # 成长性分析
            growth_score = fundamental_data.get('growth', {}).get('score', 0.5)
            
            # 行业比较（简化版）
            industry_performance = self._analyze_industry_performance(symbol, basic_data)
            
            # 风险分析
            risk_analysis = self._analyze_enhanced_risks(symbol, basic_data, fundamental_data)
            
            # 智能退出策略
            exit_strategy = self._generate_exit_strategy(symbol, basic_data)
            
            return {
                'overall_score': overall_score,
                'growth_analysis': {
                    'score': growth_score,
                    'rating': '优秀' if growth_score > 0.8 else '良好' if growth_score > 0.6 else '一般'
                },
                'industry_comparison': industry_performance,
                'risk_analysis': risk_analysis,
                'exit_strategy': exit_strategy,
                'smart_suggestions': self._generate_smart_suggestions(overall_score, growth_score)
            }
            
        except Exception as e:
            print(f"增强分析失败 {symbol}: {e}")
            return {'error': str(e)}
    
    def _analyze_stock_type(self, symbol: str, basic_data: Dict, fundamental_data: Dict) -> Dict:
        """智能股票类型分析"""
        try:
            # 使用现有的股票类型分析器
            price_data = basic_data['hist']
            financial_data = fundamental_data
            
            stock_info = self.stock_type_analyzer.get_stock_info(symbol, price_data, financial_data)
            
            # 计算综合评分
            technical_score = 8.0  # 模拟技术面评分
            fundamental_score = fundamental_data.get('overall_score', 0.5) * 10  # 转换为10分制
            sentiment_score = 4.0  # 模拟市场情绪评分
            
            comprehensive_score = self.stock_type_analyzer.calculate_comprehensive_score(
                symbol, technical_score, fundamental_score, sentiment_score, price_data, financial_data
            )
            
            return {
                'stock_type': stock_info.get('type', '成长股'),
                'risk_level': stock_info.get('risk_level', 'MEDIUM'),
                'comprehensive_score': comprehensive_score.get('score', 7.9),
                'technical_score': technical_score,
                'fundamental_score': fundamental_score,
                'sentiment_score': sentiment_score,
                'weights': comprehensive_score.get('weights', {}),
                'trading_strategy': stock_info.get('trading_strategy', {}),
                'stock_characteristics': stock_info.get('characteristics', [])
            }
            
        except Exception as e:
            print(f"股票类型分析失败 {symbol}: {e}")
            return {'error': str(e)}
    
    def _analyze_right_side_trading(self, symbol: str, technical_data: Dict, basic_data: Dict) -> Dict:
        """右侧交易分析"""
        try:
            indicators = technical_data.get('indicators', {})
            current_price = basic_data.get('current_price', 0)
            
            # 趋势确认
            trend_confirmed = self._is_trend_confirmed(indicators)
            
            # 突破确认
            breakout_confirmed = self._is_breakout_confirmed(indicators)
            
            # 成交量配合
            volume_ratio = indicators.get('volume_ratio', 1.0)
            volume_support = volume_ratio > 1.2
            
            # 调试信息
            print(f"右侧交易分析 - {symbol}:")
            print(f"  当前价格: {current_price}")
            print(f"  MA20: {indicators.get('ma_20', 'N/A')}")
            print(f"  MA50: {indicators.get('ma_50', 'N/A')}")
            print(f"  布林带上轨: {indicators.get('bb_upper', 'N/A')}")
            print(f"  成交量比率: {volume_ratio:.2f}")
            print(f"  趋势确认: {trend_confirmed}")
            print(f"  突破确认: {breakout_confirmed}")
            print(f"  成交量配合: {volume_support}")
            
            # 右侧交易评分 - 改进版本，即使不满足条件也给基础分
            right_side_score = 10  # 基础分
            
            if trend_confirmed:
                right_side_score += 35
            else:
                # 部分趋势分析
                ma_20 = indicators.get('ma_20', 0)
                if current_price > ma_20 and ma_20 > 0:
                    right_side_score += 15  # 价格在MA20上方
            
            if breakout_confirmed:
                right_side_score += 30
            else:
                # 接近突破给部分分数
                bb_upper = indicators.get('bb_upper', current_price)
                if current_price > bb_upper * 0.98:  # 接近上轨
                    right_side_score += 10
            
            if volume_support:
                right_side_score += 25
            elif volume_ratio > 1.0:
                right_side_score += int(volume_ratio * 10)  # 按比例给分
            
            # 决策建议
            if right_side_score >= 70:
                decision = "积极买入"
                decision_color = "🟢"
                decision_reason = "多项正面因素确认，符合右侧交易条件"
            elif right_side_score >= 40:
                decision = "观察等待"
                decision_color = "🟡"
                decision_reason = "趋势不明确，建议等待更清晰的信号"
            else:
                decision = "右侧等待"
                decision_color = "🔴"
                decision_reason = "抄底风险：股价仍在长期均线下方，可能继续下跌"
            
            return {
                'decision': decision,
                'decision_color': decision_color,
                'decision_reason': decision_reason,
                'score': right_side_score,
                'trend_confirmed': trend_confirmed,
                'breakout_confirmed': breakout_confirmed,
                'volume_support': volume_support,
                'core_principles': [
                    "趋势确认后再进入，不抄底不摸顶",
                    "等待突破确认，避免假突破陷阱",
                    "成交量必须配合，无量上涨不追",
                    "设置止损位，严格执行纪律"
                ]
            }
            
        except Exception as e:
            print(f"右侧交易分析失败 {symbol}: {e}")
            return {'error': str(e)}
    
    def _analyze_market_environment(self, symbol: str, basic_data: Dict) -> Dict:
        """市场环境分析"""
        try:
            # 分析市场大环境
            market_indices = ['^GSPC', '^IXIC', '^VIX']
            market_data = {}
            
            for index in market_indices:
                try:
                    index_ticker = yf.Ticker(index)
                    index_hist = index_ticker.history(period='5d')
                    if not index_hist.empty:
                        current_val = index_hist['Close'].iloc[-1]
                        prev_val = index_hist['Close'].iloc[-2] if len(index_hist) > 1 else current_val
                        change_pct = ((current_val - prev_val) / prev_val) * 100 if prev_val != 0 else 0
                        market_data[index] = {
                            'value': current_val,
                            'change_pct': change_pct
                        }
                except:
                    continue
            
            # 判断市场环境
            if '^GSPC' in market_data and '^IXIC' in market_data:
                sp500_change = market_data['^GSPC']['change_pct']
                nasdaq_change = market_data['^IXIC']['change_pct']
                
                if sp500_change > 1 and nasdaq_change > 1:
                    environment = "强势上升趋势"
                elif sp500_change > 0 and nasdaq_change > 0:
                    environment = "温和上升趋势"
                elif sp500_change < -1 and nasdaq_change < -1:
                    environment = "下降趋势"
                else:
                    environment = "震荡整理"
            else:
                environment = "数据不足"
            
            return {
                'environment': environment,
                'market_indices': market_data,
                'vix_level': market_data.get('^VIX', {}).get('value', 20),
                'market_sentiment': self._determine_market_sentiment(market_data)
            }
            
        except Exception as e:
            print(f"市场环境分析失败 {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_comprehensive_rating(self, technical: Dict, fundamental: Dict, 
                                      enhanced: Dict, stock_type: Dict) -> Dict:
        """计算综合评级"""
        try:
            # 获取各维度评分
            tech_score = technical.get('score', 50) / 100  # 转换为0-1
            fund_score = fundamental.get('overall_score', 0.5)
            enhanced_score = enhanced.get('overall_score', 0.5)
            type_score = stock_type.get('comprehensive_score', 7.9) / 10  # 转换为0-1
            
            # 动态权重（根据股票类型调整）
            weights = stock_type.get('weights', {
                'technical': 0.35,
                'fundamental': 0.55,
                'sentiment': 0.10
            })
            
            # 计算综合评分
            comprehensive_score = (
                tech_score * weights.get('technical', 0.35) +
                fund_score * weights.get('fundamental', 0.55) +
                enhanced_score * weights.get('sentiment', 0.10)
            )
            
            # 评级等级
            if comprehensive_score >= 0.8:
                rating = "强烈买入"
                confidence = 90
            elif comprehensive_score >= 0.65:
                rating = "买入"
                confidence = 80
            elif comprehensive_score >= 0.5:
                rating = "持有"
                confidence = 70
            elif comprehensive_score >= 0.35:
                rating = "减持"
                confidence = 75
            else:
                rating = "卖出"
                confidence = 80
            
            return {
                'overall_score': comprehensive_score,
                'rating': rating,
                'confidence': confidence,
                'component_scores': {
                    'technical': tech_score,
                    'fundamental': fund_score,
                    'enhanced': enhanced_score,
                    'stock_type': type_score
                },
                'weights': weights
            }
            
        except Exception as e:
            print(f"综合评级计算失败: {e}")
            return {'error': str(e)}
    
    def _generate_trading_suggestions(self, symbol: str, technical: Dict, 
                                    fundamental: Dict, rating: Dict) -> Dict:
        """生成交易建议"""
        try:
            overall_score = rating.get('overall_score', 0.5)
            rating_text = rating.get('rating', '持有')
            indicators = technical.get('indicators', {})
            current_price = indicators.get('current_price', 0)
            
            # 设置当前股票代码供策略生成函数使用
            self._current_symbol = symbol
            
            # 基础建议
            suggestions = {
                'primary_action': rating_text,
                'confidence': rating.get('confidence', 70),
                'position_size': self._recommend_position_size(overall_score),
                'holding_period': self._recommend_holding_period(overall_score),
                'stop_loss': self._calculate_stop_loss(technical),
                'take_profit': self._calculate_take_profit(technical, overall_score),
                'entry_timing': self._analyze_entry_timing(technical),
                'risk_warnings': self._generate_risk_warnings(technical, fundamental)
            }
            
            # 详细分析要点
            suggestions['analysis_points'] = self._generate_analysis_points(
                symbol, technical, fundamental, rating
            )
            
            # 新增：波段交易策略
            suggestions['swing_trading_strategy'] = self._generate_swing_trading_strategy(
                current_price, technical, overall_score
            )
            
            # 新增：买入价格指导
            suggestions['buy_price_guidance'] = self._generate_buy_price_guidance(
                current_price, technical, rating_text
            )
            
            return suggestions
            
        except Exception as e:
            print(f"交易建议生成失败: {e}")
            return {'error': str(e)}
    
    def _generate_swing_trading_strategy(self, current_price: float, technical: Dict, score: float) -> Dict:
        """生成波段交易策略"""
        try:
            indicators = technical.get('indicators', {})
            ma_20 = indicators.get('ma_20', current_price)
            ma_50 = indicators.get('ma_50', current_price)
            symbol = getattr(self, '_current_symbol', '')
            
            # 核心仓位和波段仓位比例
            core_position = 60  # 60%核心仓位长期持有
            swing_position = 40  # 40%波段仓位
            
            # TSLA特殊补仓策略
            if symbol == 'TSLA':
                add_positions = [
                    {
                        'price': 296,  # 第一批：试探性建仓
                        'percentage': '30%仓位 ($825)',
                        'reason': '试探性建仓验证支撑，测试市场反应'
                    },
                    {
                        'price': 290,  # 第二批：重仓加码 
                        'percentage': '40%仓位 ($1,100)',
                        'reason': '确认趋势后重仓买入，获取主要收益'
                    },
                    {
                        'price': 280,  # 第三批：极值收割
                        'percentage': '30%仓位 ($825)',
                        'reason': '极值区域收割，风险最低时加码'
                    }
                ]
                
                # TSLA波段卖出策略
                sell_positions = [
                    {
                        'price': 380,  # 短期目标
                        'percentage': '减持30%波段仓位',
                        'reason': '短期反弹目标，部分获利了结'
                    },
                    {
                        'price': 420,  # 中期目标
                        'percentage': '减持50%波段仓位',
                        'reason': '中期上涨目标，逐步减仓'
                    },
                    {
                        'price': 480,  # 长期高位
                        'percentage': '减持100%波段仓位',
                        'reason': '长期高位目标，清仓波段仓位'
                    }
                ]
                
                target_price = 450  # TSLA长期目标价
                upside_potential = ((target_price - current_price) / current_price) * 100
                
                return {
                    'core_position_pct': core_position,
                    'swing_position_pct': swing_position,
                    'add_positions': add_positions,
                    'sell_positions': sell_positions,
                    'target_price': target_price,
                    'upside_potential': upside_potential,
                    'time_horizon': '6-12个月',
                    'risk_control': {
                        'stop_loss_pct': 20.0,  # TSLA波动较大，止损设置宽松
                        'position_limit_pct': 10.0,
                        'total_investment': '$2,750 (约占总资产10%)',
                        'key_support': '$336 - 关键支撑不破看涨'
                    },
                    'strategy_note': '🎯 倒金字塔补仓策略：由小到大到小，试探-重仓-收割三步走'
                }
            
            # 通用策略（非TSLA股票）
            add_positions = [
                {
                    'price': current_price * 0.95,  # 当前价-5%
                    'percentage': '2-3股',
                    'reason': '健康回调，趋势完好'
                },
                {
                    'price': current_price * 0.90,  # 当前价-10%
                    'percentage': '3-4股', 
                    'reason': '深度回调，价值显现'
                },
                {
                    'price': ma_20 * 0.97 if ma_20 > 0 else current_price * 0.92,  # MA20附近
                    'percentage': '4-5股',
                    'reason': '技术支撑位，安全边际'
                }
            ]
            
            # 波段卖出策略（分3档）
            sell_positions = [
                {
                    'price': current_price * 1.08,  # 当前价+8%
                    'percentage': '减持30%波段仓位',
                    'reason': '短期获利，部分兑现'
                },
                {
                    'price': current_price * 1.15,  # 当前价+15%
                    'percentage': '减持50%波段仓位',
                    'reason': '中期目标，减仓操作'
                },
                {
                    'price': current_price * 1.25,  # 当前价+25%
                    'percentage': '减持100%波段仓位',
                    'reason': '高位获利，清仓波段'
                }
            ]
            
            # 长期目标
            target_price = current_price * (1.2 + score * 0.8)  # 根据评分调整目标价
            upside_potential = ((target_price - current_price) / current_price) * 100
            
            return {
                'core_position_pct': core_position,
                'swing_position_pct': swing_position,
                'add_positions': add_positions,
                'sell_positions': sell_positions,
                'target_price': target_price,
                'upside_potential': upside_potential,
                'time_horizon': '1-3年',
                'risk_control': {
                    'stop_loss_pct': 15.0,
                    'position_limit_pct': 10.0
                }
            }
            
        except Exception as e:
            print(f"波段交易策略生成失败: {e}")
            return {}
    
    def _generate_buy_price_guidance(self, current_price: float, technical: Dict, rating: str) -> Dict:
        """生成买入价格指导"""
        try:
            indicators = technical.get('indicators', {})
            ma_20 = indicators.get('ma_20', current_price)
            ma_50 = indicators.get('ma_50', current_price)
            bb_lower = indicators.get('bb_lower', current_price * 0.95)
            symbol = getattr(self, '_current_symbol', '')
            
            guidance = {
                'optimal_buy_zones': [],
                'current_assessment': '',
                'entry_strategy': ''
            }
            
            # TSLA特殊买入指导
            if symbol == 'TSLA':
                guidance['optimal_buy_zones'] = [
                    {
                        'price_range': '$296-300',
                        'allocation': '30%仓位 ($825)',
                        'reason': '第一批试探性建仓，验证支撑有效性'
                    },
                    {
                        'price_range': '$285-290', 
                        'allocation': '40%仓位 ($1,100)',
                        'reason': '第二批重仓买入，确认趋势后获取主要收益'
                    },
                    {
                        'price_range': '$273-280',
                        'allocation': '30%仓位 ($825)',
                        'reason': '第三批极值收割，风险最低时加码'
                    }
                ]
                
                if current_price > 336:
                    guidance['current_assessment'] = f'当前价位${current_price:.2f}高于关键支撑$336，等待回调到补仓区间'
                elif current_price <= 296:
                    guidance['current_assessment'] = f'当前价位${current_price:.2f}已进入补仓区间，可开始执行倒金字塔策略'
                else:
                    guidance['current_assessment'] = f'当前价位${current_price:.2f}接近补仓区间，密切关注'
                    
                guidance['entry_strategy'] = '倒金字塔补仓：30%-40%-30%，由小到大到小，分3批执行'
                
                return guidance
            
            # 通用买入指导（非TSLA股票）
            if rating in ['强烈买入', '买入']:
                # 积极买入区间
                guidance['optimal_buy_zones'] = [
                    {
                        'price_range': f"${current_price * 0.98:.2f} - ${current_price:.2f}",
                        'allocation': '30%',
                        'reason': '当前价位，分批建仓'
                    },
                    {
                        'price_range': f"${max(ma_20 * 0.98, bb_lower):.2f} - ${current_price * 0.95:.2f}",
                        'allocation': '40%',
                        'reason': '技术支撑位，重点加仓'
                    },
                    {
                        'price_range': f"${current_price * 0.90:.2f} - ${current_price * 0.93:.2f}",
                        'allocation': '30%',
                        'reason': '深度回调，价值凸显'
                    }
                ]
                
                if current_price > ma_20:
                    guidance['current_assessment'] = '当前价位偏高，建议等待回调'
                else:
                    guidance['current_assessment'] = '当前价位合理，可以开始建仓'
                    
                guidance['entry_strategy'] = '分批买入，逢低加仓'
                
            elif rating == '持有':
                guidance['optimal_buy_zones'] = [
                    {
                        'price_range': f"${current_price * 0.92:.2f} - ${current_price * 0.95:.2f}",
                        'allocation': '50%',
                        'reason': '等待更好的买入时机'
                    },
                    {
                        'price_range': f"${current_price * 0.85:.2f} - ${current_price * 0.90:.2f}",
                        'allocation': '50%',
                        'reason': '深度回调时考虑买入'
                    }
                ]
                
                guidance['current_assessment'] = '观望为主，等待更好机会'
                guidance['entry_strategy'] = '谨慎入场，严控仓位'
                
            else:  # 减持/卖出
                guidance['optimal_buy_zones'] = []
                guidance['current_assessment'] = '不建议在当前价位买入'
                guidance['entry_strategy'] = '暂时观望，等待明确信号'
            
            return guidance
            
        except Exception as e:
            print(f"买入价格指导生成失败: {e}")
            return {}
    
    # === 技术指标计算方法 ===
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        try:
            if len(prices) < period + 1:
                return 50
            
            delta = prices.diff().dropna()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gain = gains.rolling(window=period, min_periods=period).mean()
            avg_loss = losses.rolling(window=period, min_periods=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50
    
    def _calculate_macd(self, prices):
        """计算MACD"""
        try:
            exp1 = prices.ewm(span=self.tech_config['macd_fast']).mean()
            exp2 = prices.ewm(span=self.tech_config['macd_slow']).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=self.tech_config['macd_signal']).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd_line': float(macd_line.iloc[-1]),
                'signal_line': float(signal_line.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        except:
            return {'macd_line': 0, 'signal_line': 0, 'histogram': 0}
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """计算布林带"""
        try:
            if len(prices) < period:
                current_price = prices.iloc[-1]
                return {
                    'bb_upper': current_price * 1.02,
                    'bb_middle': current_price,
                    'bb_lower': current_price * 0.98
                }
            
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return {
                'bb_upper': float(upper.iloc[-1]),
                'bb_middle': float(middle.iloc[-1]),
                'bb_lower': float(lower.iloc[-1])
            }
        except:
            current_price = prices.iloc[-1]
            return {
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98
            }
    
    def _analyze_volume(self, hist):
        """成交量分析"""
        try:
            current_volume = hist['Volume'].iloc[-1]
            avg_volume_20 = hist['Volume'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_volume
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            return {
                'current_volume': int(current_volume),
                'avg_volume_20': int(avg_volume_20),
                'volume_ratio': float(volume_ratio)
            }
        except:
            return {'current_volume': 0, 'avg_volume_20': 0, 'volume_ratio': 1}
    
    # === 辅助方法 ===
    
    def _calculate_technical_score(self, indicators, current_price):
        """计算技术面评分"""
        score = 50  # 基础分
        
        # RSI评分
        rsi = indicators.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 20
        elif rsi < 30:
            score += 10  # 超卖有反弹机会
        elif rsi > 70:
            score -= 10  # 超买有回调风险
        
        # 移动平均线评分
        ma_20 = indicators.get('ma_20')
        ma_50 = indicators.get('ma_50')
        if ma_20 and ma_50:
            if current_price > ma_20 > ma_50:
                score += 20  # 多头排列
            elif current_price > ma_20:
                score += 10  # 短期强势
            elif current_price < ma_20 < ma_50:
                score -= 20  # 空头排列
        
        # 成交量评分
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 10  # 放量
        elif volume_ratio < 0.5:
            score -= 5   # 缩量
        
        return max(0, min(100, score))
    
    def _get_technical_rating(self, score):
        """根据技术评分获取评级"""
        if score >= 80:
            return "强势"
        elif score >= 60:
            return "偏强"
        elif score >= 40:
            return "中性"
        elif score >= 20:
            return "偏弱"
        else:
            return "弱势"
    
    def _determine_strategy(self, indicators, current_price):
        """确定推荐策略"""
        rsi = indicators.get('rsi', 50)
        ma_20 = indicators.get('ma_20')
        
        if rsi < 30:
            return "value_buying"  # 价值买入
        elif rsi > 70:
            return "profit_taking"  # 获利了结
        elif ma_20 and current_price > ma_20:
            return "trend_following"  # 趋势跟踪
        else:
            return "wait_and_see"  # 观望等待
    
    def _calculate_signal_strength(self, indicators):
        """计算信号强度"""
        strength = 0
        
        # RSI信号
        rsi = indicators.get('rsi', 50)
        if rsi < 25 or rsi > 75:
            strength += 2
        elif rsi < 35 or rsi > 65:
            strength += 1
        
        # MACD信号
        macd_line = indicators.get('macd_line', 0)
        signal_line = indicators.get('signal_line', 0)
        if abs(macd_line - signal_line) > 0.5:
            strength += 1
        
        # 成交量信号
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            strength += 2
        elif volume_ratio > 1.5:
            strength += 1
        
        return min(7, strength)  # 最高7分
    
    def _analyze_trend(self, indicators, current_price):
        """趋势分析"""
        ma_20 = indicators.get('ma_20')
        ma_50 = indicators.get('ma_50')
        
        if ma_20 and ma_50:
            if current_price > ma_20 > ma_50:
                return "上升趋势"
            elif current_price < ma_20 < ma_50:
                return "下降趋势"
            else:
                return "震荡趋势"
        else:
            return "趋势不明"
    
    def _format_company_info(self, company_info):
        """格式化公司信息"""
        return {
            'company_name': company_info.get('longName', 'N/A'),
            'sector': company_info.get('sector', 'N/A'),
            'industry': company_info.get('industry', 'N/A'),
            'market_cap': company_info.get('marketCap', 0),
            'employee_count': company_info.get('fullTimeEmployees', 0)
        }
    
    # === 缓存管理 ===
    
    def _is_cache_valid(self, cache_key):
        """检查缓存是否有效"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    def _cache_result(self, cache_key, data):
        """缓存结果"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def _create_error_response(self, symbol, error_msg):
        """创建错误响应"""
        return {
            'symbol': symbol,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
    
    # === 更多辅助方法（简化实现） ===
    
    def _get_market_cap_level(self, market_cap):
        """获取市值等级"""
        if market_cap > 200e9:
            return "MEGA"
        elif market_cap > 50e9:
            return "LARGE"
        elif market_cap > 10e9:
            return "MID"
        else:
            return "SMALL"
    
    def _calculate_liquidity_score(self, avg_volume, current_volume, spread, market_cap):
        """计算流动性评分"""
        base_score = 50
        
        # 成交量评分
        if avg_volume > 1e6:
            base_score += 20
        elif avg_volume > 500e3:
            base_score += 10
        
        # 价差评分
        if spread < 0.005:
            base_score += 20
        elif spread < 0.01:
            base_score += 10
        
        # 市值评分
        if market_cap > 50e9:
            base_score += 10
        
        return min(100, base_score)
    
    def _get_liquidity_risk_level(self, score):
        """获取流动性风险等级"""
        if score >= 80:
            return "LOW"
        elif score >= 60:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_liquidity_points(self, score, market_cap_level):
        """生成流动性分析要点"""
        points = []
        
        if score >= 80:
            points.append("流动性良好")
        elif score >= 60:
            points.append("流动性一般")
        else:
            points.append("流动性偏低，注意冲击成本")
        
        if market_cap_level in ["LARGE", "MEGA"]:
            points.append("大市值股票，机构参与度高")
        
        return points
    
    def _analyze_industry_performance(self, symbol, basic_data):
        """行业表现分析（简化版）"""
        return {
            'relative_performance': "行业内表现优秀",
            'industry_trend': "行业趋势向好",
            'comparison': "同行业中表现突出"
        }
    
    def _analyze_enhanced_risks(self, symbol, basic_data, fundamental_data):
        """增强风险分析"""
        risks = []
        risk_level = "MEDIUM"
        
        # 基于基本面评分判断风险
        overall_score = fundamental_data.get('overall_score', 0.5)
        if overall_score < 0.3:
            risks.append("基本面恶化风险")
            risk_level = "HIGH"
        elif overall_score > 0.8:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'identified_risks': risks,
            'overall_assessment': "无重大风险" if risk_level == "LOW" else "注意潜在风险"
        }
    
    def _generate_exit_strategy(self, symbol, basic_data):
        """生成退出策略"""
        return {
            'strategy_type': "动态止损止盈分析",
            'signal': "持有信号",
            'stop_loss_level': basic_data['current_price'] * 0.9,
            'take_profit_level': basic_data['current_price'] * 1.15
        }
    
    def _generate_smart_suggestions(self, overall_score, growth_score):
        """生成智能建议"""
        suggestions = []
        
        if growth_score > 0.8:
            suggestions.append("成长性优秀，具有较大上涨潜力")
        
        if overall_score > 0.7:
            suggestions.append("综合表现良好，建议持有")
        
        return suggestions
    
    def _is_trend_confirmed(self, indicators):
        """趋势确认"""
        ma_20 = indicators.get('ma_20', 0)
        ma_50 = indicators.get('ma_50', 0)
        current_price = indicators.get('current_price', 0)
        
        return current_price > ma_20 > ma_50
    
    def _is_breakout_confirmed(self, indicators):
        """突破确认"""
        current_price = indicators.get('current_price', 0)
        bb_upper = indicators.get('bb_upper', current_price)
        
        return current_price > bb_upper
    
    def _determine_market_sentiment(self, market_data):
        """判断市场情绪"""
        if '^VIX' in market_data:
            vix = market_data['^VIX']['value']
            if vix < 15:
                return "极度乐观"
            elif vix < 25:
                return "乐观"
            elif vix < 35:
                return "谨慎"
            else:
                return "恐慌"
        return "中性"
    
    def _recommend_position_size(self, score):
        """推荐仓位大小"""
        if score >= 0.8:
            return "标准仓位（15-25%）"
        elif score >= 0.6:
            return "标准仓位（8-20%）"
        elif score >= 0.4:
            return "小仓位（3-8%）"
        else:
            return "观望（0%）"
    
    def _recommend_holding_period(self, score):
        """推荐持有周期"""
        if score >= 0.7:
            return "中长期（6个月-3年）"
        elif score >= 0.5:
            return "中期（3-12个月）"
        else:
            return "短期（1-6个月）"
    
    def _calculate_stop_loss(self, technical):
        """计算止损位"""
        indicators = technical.get('indicators', {})
        ma_20 = indicators.get('ma_20', 0)
        current_price = indicators.get('current_price', 0)
        
        if ma_20 > 0:
            return ma_20 * 0.95  # MA20下方5%
        else:
            return current_price * 0.9  # 当前价格下方10%
    
    def _calculate_take_profit(self, technical, score):
        """计算止盈位"""
        indicators = technical.get('indicators', {})
        current_price = indicators.get('current_price', 0)
        
        if score >= 0.8:
            return current_price * 1.25  # 25%涨幅
        elif score >= 0.6:
            return current_price * 1.15  # 15%涨幅
        else:
            return current_price * 1.08  # 8%涨幅
    
    def _analyze_entry_timing(self, technical):
        """分析入场时机"""
        indicators = technical.get('indicators', {})
        rsi = indicators.get('rsi', 50)
        
        if rsi < 35:
            return "良好时机"
        elif rsi < 50:
            return "谨慎观察"
        else:
            return "等待回调"
    
    def _generate_risk_warnings(self, technical, fundamental):
        """生成风险警告"""
        warnings = []
        
        rsi = technical.get('indicators', {}).get('rsi', 50)
        if rsi > 75:
            warnings.append("技术面超买风险")
        
        overall_score = fundamental.get('overall_score', 0.5)
        if overall_score < 0.4:
            warnings.append("基本面偏弱风险")
        
        return warnings
    
    def _generate_analysis_points(self, symbol, technical, fundamental, rating):
        """生成分析要点"""
        points = []
        
        # 技术面要点
        rsi = technical.get('indicators', {}).get('rsi', 50)
        if rsi < 30:
            points.append(f"RSI({rsi:.1f})显示超卖状态")
        elif rsi > 70:
            points.append(f"RSI({rsi:.1f})显示超买状态")
        
        # 基本面要点
        fund_score = fundamental.get('overall_score', 0.5)
        if fund_score > 0.7:
            points.append("基本面表现优秀")
        elif fund_score < 0.4:
            points.append("基本面表现偏弱")
        
        # 综合建议
        overall_rating = rating.get('rating', '持有')
        points.append(f"综合建议：{overall_rating}")
        
        return points


# 为了兼容性，创建一个简化的分析函数
def analyze_stock_comprehensive(symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    股票综合分析的便捷函数
    
    Args:
        symbol: 股票代码
        force_refresh: 强制刷新缓存
        
    Returns:
        综合分析结果
    """
    analyzer = UnifiedStockAnalyzer()
    return analyzer.get_comprehensive_analysis(symbol, force_refresh)


if __name__ == "__main__":
    # 测试代码
    analyzer = UnifiedStockAnalyzer()
    result = analyzer.get_comprehensive_analysis("ADBE")
    
    if 'error' not in result:
        print(f"✅ {result['symbol']} 分析完成")
        print(f"📊 综合评级: {result['comprehensive_rating']['rating']}")
        print(f"🎯 技术评分: {result['technical_analysis']['score']}")
        print(f"💼 基本面评分: {result['fundamental_analysis']['overall_score']:.2f}")
    else:
        print(f"❌ 分析失败: {result['error']}")