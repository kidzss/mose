#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个人投资者自动化股票推荐系统

功能：
1. 每周自动筛选优质股票
2. 自动更新股票数据和财务数据
3. 发送个性化投资建议邮件
4. 适合个人投资者的风险控制

推荐频率：
- 每周筛选：每周日20:00
- 每月深度分析：每月第一个周日
- 季度策略调整：每季度第一个周日
"""

import os
import sys
import schedule
import time
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitor.phase2_professional_screener import Phase2ProfessionalScreener
from data.data_interface import DataInterface
from utils.unified_email_api import send_html

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('personal_investor_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PersonalInvestorAutomation")

class PersonalInvestorAutomation:
    """个人投资者自动化系统"""
    
    def __init__(self):
        self.screener = Phase2ProfessionalScreener()
        self.data_interface = DataInterface()
        
        # 个人投资者配置
        self.config = {
            'email': 'kidzss@gmail.com',
            'risk_tolerance': 'moderate',  # conservative, moderate, aggressive
            'max_position_size': 0.20,     # 单只股票最大仓位20%
            'min_quality_factor': 0.7,     # 最低质量因子
            'max_results': 15,             # 推荐股票数量
            'min_score': 60,               # 最低评分
            # 策略信号权重配置
            'strategy_weights': {
                'TDI': 0.3,        # TDI策略权重
                'NiuniuV3': 0.4,   # 牛牛策略权重（主要策略）
                'CPGW': 0.3        # CPGW策略权重
            },
            'strategy_score_threshold': 0.6,  # 策略信号最低分数阈值
            'enhanced_analysis_weight': 0.7,  # AI增强分析权重
            'strategy_signal_weight': 0.3     # 策略信号权重
        }
        
        # 初始化增强分析器（Phase 2新功能）
        self.enhanced_analyzer = None
        try:
            from monitor.enhanced_stock_analyzer import EnhancedStockAnalyzer
            from monitor.enhanced_stock_screener import EnhancedStockScreener
            self.enhanced_analyzer = EnhancedStockAnalyzer()
            self.enhanced_screener = EnhancedStockScreener()
            logger.info("✅ 增强分析器和筛选器集成成功")
        except Exception as e:
            logger.warning(f"增强分析器初始化失败: {e}")
            self.enhanced_screener = None
        
        # 初始化策略工厂（新增）
        self.strategy_factory = None
        try:
            from strategy.strategy_factory import StrategyFactory
            self.strategy_factory = StrategyFactory()
            logger.info("✅ 策略工厂初始化成功")
        except Exception as e:
            logger.warning(f"策略工厂初始化失败: {e}")
        
        # 初始化动态权重系统（新增）
        self.dynamic_weight_system = None
        try:
            from utils.dynamic_weight_system import DynamicWeightSystem
            self.dynamic_weight_system = DynamicWeightSystem()
            logger.info("✅ 动态权重系统初始化成功")
        except Exception as e:
            logger.warning(f"动态权重系统初始化失败: {e}")
        
        # 初始化AI策略优化集成系统（新增）
        self.ai_optimization_integration = None
        try:
            from ai_strategy_optimization_integration import AIStrategyOptimizationIntegration
            self.ai_optimization_integration = AIStrategyOptimizationIntegration()
            logger.info("✅ AI策略优化集成系统初始化成功")
        except Exception as e:
            logger.warning(f"AI策略优化集成系统初始化失败: {e}")
        
        # 加载个人配置
        self._load_personal_config()
        
        logger.info("�� 个人投资者自动化系统初始化完成")
    
    def _load_personal_config(self):
        """加载个人配置文件"""
        config_file = 'personal_investor_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    personal_config = json.load(f)
                    self.config.update(personal_config)
                logger.info("✅ 已加载个人配置文件")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")
        else:
            # 创建默认配置文件
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            'email': 'kidzss@gmail.com',
            'risk_tolerance': 'moderate',
            'max_position_size': 0.20,
            'min_quality_factor': 0.7,
            'max_results': 15,
            'min_score': 60,
            'investment_goals': {
                'time_horizon': '3-5年',
                'risk_preference': '中等风险',
                'investment_amount': '可承受20%亏损'
            },
            'preferred_sectors': ['科技', '消费', '医疗', '金融'],
            'excluded_sectors': ['能源', '原材料']  # 可选排除
        }
        
        try:
            with open('personal_investor_config.json', 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info("✅ 已创建默认配置文件: personal_investor_config.json")
        except Exception as e:
            logger.error(f"创建配置文件失败: {e}")
    
    def update_market_data(self):
        """更新市场数据"""
        try:
            logger.info("📊 开始更新市场数据...")
            
                        # 获取需要更新的股票列表 - 从数据库获取全量股票
            try:
                from data.data_interface import DataInterface
                db_interface = DataInterface()
                watchlist = db_interface.get_available_symbols()
                logger.info(f"📊 准备更新 {len(watchlist)} 只股票的数据")
                # 为了避免更新时间过长，只更新前100只最活跃的股票
                if len(watchlist) > 100:
                    watchlist = watchlist[:100]
                    logger.info(f"📊 为提高效率，仅更新前 {len(watchlist)} 只股票")
            except Exception as e:
                logger.warning(f"无法获取数据库股票池，使用默认股票池: {e}")
                watchlist = ['AMD', 'GOOGL', 'PFE', 'NVDA', 'TSLA', 'ADBE', 'MSFT', 'EOG', 'PHM', 'CF']
            
            # 更新股票数据
            updated_count = 0
            for symbol in watchlist:
                try:
                    # 获取最新数据
                    data = self.data_interface.get_data_for_strategy(symbol, lookback_days=60)
                    if data is not None and len(data) > 0:
                        updated_count += 1
                        logger.info(f"✅ {symbol} 数据已更新")
                except Exception as e:
                    logger.warning(f"更新 {symbol} 数据失败: {e}")
            
            logger.info(f"📊 数据更新完成，成功更新 {updated_count}/{len(watchlist)} 只股票")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"数据更新失败: {e}")
            return False
    
    def run_weekly_screening(self):
        """运行每周股票筛选"""
        try:
            logger.info("🎯 开始每周股票筛选...")
            
            # 更新数据
            self.update_market_data()
            
            # 根据风险偏好调整筛选参数
            if self.config['risk_tolerance'] == 'conservative':
                min_score = 65
                min_quality = 0.8
            elif self.config['risk_tolerance'] == 'aggressive':
                min_score = 55
                min_quality = 0.6
            else:  # moderate
                min_score = 60
                min_quality = 0.7
            
            # Phase 2: 使用增强筛选器（集成专家建议）
            if self.enhanced_screener:
                logger.info("🚀 使用增强筛选器进行智能选股...")
                
                # 获取候选股票池 - 从MySQL数据库获取全量股票
                try:
                    from data.data_interface import DataInterface
                    db_interface = DataInterface()
                    watchlist = db_interface.get_available_symbols()
                    logger.info(f"📊 从数据库获取到 {len(watchlist)} 只股票进行筛选")
                except Exception as e:
                    logger.warning(f"无法获取数据库股票池，使用默认股票池: {e}")
                    watchlist = ['AMD', 'GOOGL', 'PFE', 'NVDA', 'TSLA', 'ADBE', 'MSFT', 'EOG', 'PHM', 'CF', 
                               'AAPL', 'META', 'NFLX', 'CRM', 'ORCL']
                
                # 转换评分标准：0-100 -> 0-1
                enhanced_min_score = min_score / 100.0
                
                # 使用增强筛选器
                enhanced_results = self.enhanced_screener.screen_stocks(
                    watchlist, 
                    min_score=enhanced_min_score
                )
                
                # 转换为兼容格式并集成策略信号分析
                high_quality_results = []
                for result in enhanced_results[:self.config['max_results']]:
                    # 计算简化的夏普比率 (基于增强评分)
                    sharpe_ratio = result['enhanced_score'] * 2.0  # 简化计算，范围0-2
                    
                    # 获取策略信号分析
                    strategy_analysis = self._analyze_strategy_signals(
                        result['symbol'], 
                        current_price=result['current_price']
                    )
                    
                    # 融合AI增强分析分数和策略信号分数
                    enhanced_score = result['enhanced_score']
                    strategy_score = strategy_analysis['strategy_score']
                    fused_score = self._fuse_enhanced_and_strategy_scores(enhanced_score, strategy_score)
                    
                    stock_data = {
                        'symbol': result['symbol'],
                        'current_price': result['current_price'],
                        'enhanced_score': result['enhanced_score'],
                        'traditional_score': result['traditional_score'],
                        'growth_score': result['growth_score'], 
                        'industry_score': result['industry_score'],
                        'price_targets': result['price_targets'],
                        'enhanced_analysis': {
                            'recommendations': result['recommendations'],
                            'warnings': result['warnings']
                        },
                        'quality_factor': fused_score,  # 使用融合后的分数
                        'sharpe_ratio': sharpe_ratio,  # 添加夏普比率
                        # 新增策略信号信息
                        'strategy_analysis': strategy_analysis,
                        'fused_score': fused_score,
                        'strategy_score': strategy_score
                    }
                    high_quality_results.append(stock_data)
                
                logger.info(f"🎯 增强筛选完成，推荐 {len(high_quality_results)} 只股票")
                
            else:
                # 回退到传统筛选
                logger.info("📊 使用传统筛选器...")
                results = self.screener.screen_stocks_professional(
                    min_score=min_score,
                    max_results=self.config['max_results']
                )
                
                # 过滤高质量股票
                high_quality_results = [
                    stock for stock in results 
                    if stock['quality_factor'] >= min_quality
                ]
            
            # 应用额外的增强分析（如果可用）
            if self.enhanced_analyzer and high_quality_results and not self.enhanced_screener:
                logger.info("🔍 开始应用增强分析过滤...")
                enhanced_results = []
                
                for stock in high_quality_results:
                    try:
                        # 获取增强分析
                        enhanced_analysis = self.enhanced_analyzer.analyze_stock_enhanced(
                            stock['symbol'], 
                            current_price=stock.get('current_price')
                        )
                        
                        if enhanced_analysis:
                            # 添加增强分析结果到股票数据
                            stock['enhanced_analysis'] = enhanced_analysis
                            
                            # 应用风险过滤：排除高风险警告的股票
                            warnings = enhanced_analysis.get('warnings', [])
                            high_risk_warnings = [w for w in warnings if '高风险' in w or 'High Risk' in w]
                            
                            if not high_risk_warnings:
                                enhanced_results.append(stock)
                                logger.info(f"✅ {stock['symbol']} 通过增强分析过滤")
                            else:
                                logger.warning(f"⚠️ {stock['symbol']} 被过滤：{high_risk_warnings[0]}")
                        else:
                            # 如果无法获取增强分析，保留原始推荐
                            enhanced_results.append(stock)
                            
                    except Exception as e:
                        logger.warning(f"{stock['symbol']} 增强分析失败: {e}")
                        enhanced_results.append(stock)  # 失败时保留
                
                high_quality_results = enhanced_results
                logger.info(f"📊 增强分析完成，最终推荐 {len(high_quality_results)} 只股票")
            
            if high_quality_results:
                # 生成个性化投资建议
                self._generate_personalized_report(high_quality_results, 'weekly')
                logger.info(f"✅ 每周筛选完成，推荐 {len(high_quality_results)} 只高质量股票")
            else:
                logger.warning("⚠️ 未找到符合条件的高质量股票")
            
            return high_quality_results
            
        except Exception as e:
            logger.error(f"每周筛选失败: {e}")
            return []
    
    def run_monthly_analysis(self):
        """运行每月深度分析"""
        try:
            logger.info("📈 开始每月深度分析...")
            
            # 更新所有数据
            self.update_market_data()
            
            # 更严格的筛选标准
            results = self.screener.screen_stocks_professional(
                min_score=70,  # 更高标准
                max_results=10  # 更少但更精
            )
            
            # 生成深度分析报告
            self._generate_personalized_report(results, 'monthly')
            
            logger.info("✅ 每月深度分析完成")
            return results
            
        except Exception as e:
            logger.error(f"每月分析失败: {e}")
            return []
    
    def run_quarterly_strategy(self):
        """运行季度策略调整"""
        try:
            logger.info("🔄 开始季度策略调整...")
            
            # 全面数据更新
            self.update_market_data()
            
            # 市场环境分析
            market_analysis = self._analyze_market_environment()
            
            # 策略调整建议
            strategy_recommendations = self._generate_strategy_recommendations(market_analysis)
            
            # 生成季度报告
            self._generate_personalized_report([], 'quarterly', 
                                             market_analysis=market_analysis,
                                             strategy_recommendations=strategy_recommendations)
            
            logger.info("✅ 季度策略调整完成")
            
        except Exception as e:
            logger.error(f"季度策略调整失败: {e}")
    
    def _check_monthly_analysis(self):
        """检查是否应该执行月度分析"""
        try:
            today = datetime.now()
            # 检查是否是当月第一个周日
            if today.day <= 7:  # 前7天内
                logger.info("📅 检测到月度分析时间")
                self.run_monthly_analysis()
            else:
                logger.info("📅 本周不是月度分析时间")
        except Exception as e:
            logger.error(f"月度分析检查失败: {e}")
    
    def _check_quarterly_strategy(self):
        """检查是否应该执行季度策略调整"""
        try:
            today = datetime.now()
            # 检查是否是季度第一个月的第一个周日
            if today.month in [1, 4, 7, 10] and today.day <= 7:
                logger.info("📅 检测到季度策略调整时间")
                self.run_quarterly_strategy()
            else:
                logger.info("📅 本周不是季度策略调整时间")
        except Exception as e:
            logger.error(f"季度策略调整检查失败: {e}")
    
    def _analyze_market_environment(self):
        """分析市场环境"""
        try:
            # 获取VIX数据
            vix_data = self._get_vix_data()
            
            # 市场情绪分析
            sentiment = self._analyze_market_sentiment()
            
            # 行业轮动分析
            sector_rotation = self._analyze_sector_rotation()
            
            return {
                'vix_level': vix_data.get('current', 20),
                'market_sentiment': sentiment,
                'sector_rotation': sector_rotation,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"市场环境分析失败: {e}")
            return {}
    
    def _get_vix_data(self):
        """获取VIX数据"""
        try:
            # 简化版VIX数据获取
            return {'current': 20, 'change': 0.5, 'level': 'normal'}
        except:
            return {'current': 20, 'change': 0, 'level': 'unknown'}
    
    def _analyze_market_sentiment(self):
        """分析市场情绪"""
        try:
            # 简化版情绪分析
            return {
                'fear_greed_index': 50,
                'market_breadth': 0.6,
                'sentiment': 'neutral'
            }
        except:
            return {'sentiment': 'unknown'}
    
    def _analyze_sector_rotation(self):
        """分析行业轮动"""
        try:
            # 简化版行业轮动分析
            return {
                'leading_sectors': ['科技', '消费'],
                'lagging_sectors': ['能源', '原材料'],
                'rotation_phase': 'growth_to_value'
            }
        except:
            return {'rotation_phase': 'unknown'}
    
    def _generate_strategy_recommendations(self, market_analysis):
        """生成策略调整建议"""
        recommendations = []
        
        vix_level = market_analysis.get('vix_level', 20)
        sentiment = market_analysis.get('market_sentiment', {}).get('sentiment', 'neutral')
        
        if vix_level > 30:
            recommendations.append({
                'type': 'risk_reduction',
                'action': '增加防御性股票配置',
                'reason': f'VIX指数较高({vix_level})，市场波动性增加'
            })
        elif vix_level < 15:
            recommendations.append({
                'type': 'opportunity_seeking',
                'action': '可以适当增加成长股配置',
                'reason': f'VIX指数较低({vix_level})，市场相对稳定'
            })
        
        if sentiment == 'fear':
            recommendations.append({
                'type': 'contrarian',
                'action': '考虑逆向投资机会',
                'reason': '市场恐慌情绪，可能存在超跌机会'
            })
        
        return recommendations
    
    def _analyze_strategy_signals(self, symbol: str, current_price: float = None) -> Dict[str, Any]:
        """
        分析股票的策略信号
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            
        Returns:
            策略信号分析结果
        """
        if not self.strategy_factory:
            logger.warning(f"策略工厂未初始化，跳过策略信号分析: {symbol}")
            return {'strategy_score': 0.0, 'strategy_signals': {}, 'error': 'Strategy factory not available'}
        
        try:
            # 获取股票历史数据
            data = self.data_interface.get_data_for_strategy(symbol, lookback_days=60)
            if data is None or len(data) == 0:
                logger.warning(f"无法获取 {symbol} 的历史数据")
                return {'strategy_score': 0.0, 'strategy_signals': {}, 'error': 'No historical data'}
            
            # 转换为DataFrame格式
            import pandas as pd
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 初始化策略信号结果
            strategy_signals = {}
            strategy_scores = {}
            
            # 分析各个策略的信号
            strategy_weights = self.config['strategy_weights']
            
            for strategy_name, weight in strategy_weights.items():
                try:
                    # 创建策略实例
                    strategy = self.strategy_factory.create_strategy(strategy_name)
                    if strategy:
                        # 生成策略信号
                        signals_df = strategy.generate_signals(df.copy())
                        
                        # 获取最新信号
                        latest_signal = signals_df.iloc[-1] if not signals_df.empty else None
                        
                        if latest_signal is not None:
                            # 将信号转换为分数
                            signal_score = self._convert_signal_to_score(latest_signal, strategy_name)
                            strategy_scores[strategy_name] = signal_score
                            strategy_signals[strategy_name] = {
                                'signal': latest_signal.get('signal', 'unknown'),
                                'score': signal_score,
                                'strength': latest_signal.get('strength', 0.0),
                                'confidence': latest_signal.get('confidence', 0.0)
                            }
                            logger.debug(f"{symbol} {strategy_name} 信号: {signal_score:.3f}")
                        else:
                            strategy_scores[strategy_name] = 0.0
                            strategy_signals[strategy_name] = {'signal': 'unknown', 'score': 0.0}
                            
                except Exception as e:
                    logger.warning(f"{symbol} {strategy_name} 策略分析失败: {e}")
                    strategy_scores[strategy_name] = 0.0
                    strategy_signals[strategy_name] = {'signal': 'error', 'score': 0.0}
            
            # 使用AI策略优化集成系统的优化权重（如果可用）
            if self.ai_optimization_integration:
                try:
                    optimized_weights = self.ai_optimization_integration.get_optimized_weights(symbol)
                    if optimized_weights:
                        logger.info(f"🎯 使用优化权重: {symbol} - {optimized_weights}")
                        strategy_weights = optimized_weights
                    else:
                        logger.info(f"⚠️ 未找到 {symbol} 的优化权重，使用默认权重")
                except Exception as e:
                    logger.warning(f"获取优化权重失败: {e}")
            
            # 计算加权策略分数
            total_weight = sum(strategy_weights.values())
            weighted_strategy_score = 0.0
            
            if total_weight > 0:
                for strategy_name, weight in strategy_weights.items():
                    score = strategy_scores.get(strategy_name, 0.0)
                    weighted_strategy_score += (score * weight) / total_weight
            
            # 集成动态权重系统
            if self.dynamic_weight_system:
                try:
                    # 获取AI分析结果
                    ai_analysis = None
                    if self.enhanced_analyzer:
                        ai_analysis = self.enhanced_analyzer.analyze_stock_comprehensive(symbol)
                    
                    if ai_analysis:
                        # 记录信号到动态权重系统
                        current_price = current_price or df['Close'].iloc[-1]
                        self.dynamic_weight_system.record_signal(symbol, ai_analysis, strategy_signals, current_price)
                        
                        # 设置风险偏好（从配置中读取）
                        risk_tolerance = self.config.get('risk_tolerance', 'moderate')
                        self.dynamic_weight_system.set_risk_tolerance(risk_tolerance)
                        
                        # 计算动态权重
                        accuracy_comparison = self.dynamic_weight_system.calculate_accuracy_comparison(symbol)
                        if 'error' not in accuracy_comparison:
                            dynamic_weights = self.dynamic_weight_system.calculate_dynamic_weights(symbol, accuracy_comparison)
                            if 'error' not in dynamic_weights:
                                # 使用动态权重调整策略分数
                                new_weights = dynamic_weights['new_weights']
                                strategy_signals['dynamic_weights'] = new_weights
                                strategy_signals['weight_adjustment'] = dynamic_weights['adjustment']
                                strategy_signals['accuracy_comparison'] = accuracy_comparison
                                
                                logger.info(f"动态权重调整: {symbol} - AI权重: {new_weights['ai_weight']:.2f}, 策略权重: {new_weights['strategy_weight']:.2f}")
                                logger.info(f"准确性比较: AI={accuracy_comparison['ai_accuracy']:.3f}, 策略={accuracy_comparison['strategy_accuracy']:.3f}")
                                
                                # 使用动态权重重新计算加权分数
                                weighted_strategy_score = (weighted_strategy_score * new_weights['strategy_weight'])
                                
                                                        # 获取表现摘要
                        performance_summary = self.dynamic_weight_system.get_performance_summary(symbol)
                        if 'error' not in performance_summary:
                            strategy_signals['performance_summary'] = performance_summary
                        
                        # 集成AI策略优化分析（新增）
                        if self.ai_optimization_integration:
                            try:
                                integrated_result = self.ai_optimization_integration.integrate_with_ai_analysis(
                                    symbol, ai_analysis
                                )
                                strategy_signals['ai_optimization_integration'] = integrated_result
                                logger.info(f"🤖 AI优化集成完成: {symbol} - 最终信号: {integrated_result['final_signal']:.3f}")
                            except Exception as e:
                                logger.warning(f"AI优化集成失败: {e}")
                except Exception as e:
                    logger.warning(f"动态权重系统处理失败: {e}")
            
            return {
                'strategy_score': weighted_strategy_score,
                'strategy_signals': strategy_signals,
                'individual_scores': strategy_scores,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"{symbol} 策略信号分析失败: {e}")
            return {'strategy_score': 0.0, 'strategy_signals': {}, 'error': str(e)}
    
    def _convert_signal_to_score(self, signal_data: Dict, strategy_name: str) -> float:
        """
        将策略信号转换为分数 (0-1)
        
        Args:
            signal_data: 策略信号数据
            strategy_name: 策略名称
            
        Returns:
            信号分数 (0-1)
        """
        # 安全地获取信号值，处理不同类型的信号
        signal_raw = signal_data.get('signal', 'unknown')
        
        # 如果信号是数值类型，转换为字符串
        if isinstance(signal_raw, (int, float, np.integer, np.floating)):
            if signal_raw > 0:
                signal = 'buy'
            elif signal_raw < 0:
                signal = 'sell'
            else:
                signal = 'hold'
        else:
            # 如果是字符串，转换为小写
            signal = str(signal_raw).lower()
        
        strength = signal_data.get('strength', 0.0)
        confidence = signal_data.get('confidence', 0.0)
        
        # 基础信号分数映射
        signal_scores = {
            'buy': 1.0,
            'strong_buy': 1.0,
            'hold': 0.5,
            'sell': 0.0,
            'strong_sell': 0.0,
            'unknown': 0.5
        }
        
        # 获取基础分数
        base_score = signal_scores.get(signal, 0.5)
        
        # 根据强度和置信度调整分数
        adjusted_score = base_score * (0.7 + 0.3 * strength) * (0.8 + 0.2 * confidence)
        
        # 确保分数在0-1范围内
        return max(0.0, min(1.0, adjusted_score))
    
    def _fuse_enhanced_and_strategy_scores(self, enhanced_score: float, strategy_score: float) -> float:
        """
        融合AI增强分析分数和策略信号分数
        
        Args:
            enhanced_score: AI增强分析分数 (0-1)
            strategy_score: 策略信号分数 (0-1)
            
        Returns:
            融合后的综合分数 (0-1)
        """
        enhanced_weight = self.config['enhanced_analysis_weight']
        strategy_weight = self.config['strategy_signal_weight']
        
        # 加权融合
        fused_score = (enhanced_score * enhanced_weight) + (strategy_score * strategy_weight)
        
        # 确保分数在0-1范围内
        return max(0.0, min(1.0, fused_score))
    
    def _generate_personalized_report(self, results, report_type, 
                                    market_analysis=None, strategy_recommendations=None):
        """生成个性化投资报告"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # 根据报告类型生成不同内容
            if report_type == 'weekly':
                subject = f"📈 每周股票推荐 | {timestamp} | 个人投资策略"
                content = self._generate_weekly_content(results)
            elif report_type == 'monthly':
                subject = f"📊 每月深度分析 | {timestamp} | 投资组合优化"
                content = self._generate_monthly_content(results)
            elif report_type == 'quarterly':
                subject = f"🔄 季度策略调整 | {timestamp} | 市场环境分析"
                content = self._generate_quarterly_content(market_analysis, strategy_recommendations)
            
            # 发送邮件
            success = send_html(subject=subject, html_content=content)
            
            if success:
                logger.info(f"✅ {report_type}报告邮件发送成功")
            else:
                logger.error(f"❌ {report_type}报告邮件发送失败")
            
            return success
            
        except Exception as e:
            logger.error(f"生成{report_type}报告失败: {e}")
            return False
    
    def _generate_weekly_content(self, results):
        """生成每周报告内容"""
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; }}
                .stock-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .stock-table th, .stock-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .stock-table th {{ background-color: #f2f2f2; }}
                .high-quality {{ background-color: #e8f5e8; }}
                .investment-tips {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .risk-warning {{ background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 个人投资者每周股票推荐</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>风险偏好: {self.config['risk_tolerance']} | 最大仓位: {self.config['max_position_size']*100}%</p>
            </div>
            
            <h2>🏆 本周推荐股票</h2>
            <table class="stock-table">
                <tr>
                    <th>排名</th>
                    <th>股票代码</th>
                    <th>综合评分</th>
                    <th>质量因子</th>
                    <th>夏普比率</th>
                    <th>当前价格</th>
                    <th>建议买入</th>
                    <th>目标卖出</th>
                    <th>止损价格</th>
                    <th>增强评分</th>
                    <th>策略信号</th>
                    <th>动态权重</th>
                    <th>融合评分</th>
                    <th>成长性</th>
                    <th>行业表现</th>
                    <th>投资建议</th>
                </tr>
        """
        
        for i, stock in enumerate(results[:10], 1):
            quality_class = "high-quality" if stock['quality_factor'] > 0.8 else ""
            
            # 获取价格目标信息
            price_targets = stock.get('price_targets', {})
            buy_price = price_targets.get('buy_price', 'N/A')
            sell_price = price_targets.get('sell_price', 'N/A')
            stop_loss = price_targets.get('stop_loss', 'N/A')
            
            # 获取增强分析信息
            enhanced_analysis = stock.get('enhanced_analysis', {})
            enhanced_score = stock.get('enhanced_score', enhanced_analysis.get('overall_score', 0)) * 100
            growth_score = stock.get('growth_score', 0) * 100
            
            # 获取策略信号信息
            strategy_analysis = stock.get('strategy_analysis', {})
            strategy_score = strategy_analysis.get('strategy_score', 0) * 100
            fused_score = stock.get('fused_score', 0) * 100
            
            # 生成策略信号摘要
            strategy_signals = strategy_analysis.get('strategy_signals', {})
            strategy_summary = self._get_strategy_signal_summary(strategy_signals)
            
            # 获取动态权重信息
            dynamic_weights = strategy_signals.get('dynamic_weights', {})
            weight_summary = self._get_dynamic_weight_summary(dynamic_weights)
            
            # 获取行业表现信息
            industry_performance = self._get_industry_performance(stock)
            
            html_content += f"""
                <tr class="{quality_class}">
                    <td>{i}</td>
                    <td><strong>{stock['symbol']}</strong></td>
                    <td>{stock.get('multifactor_score', stock.get('enhanced_score', 0) * 100):.1f}</td>
                    <td>{stock['quality_factor']:.3f}</td>
                    <td>{stock.get('sharpe_ratio', 0.0):.2f}</td>
                    <td>${stock['current_price']:.2f}</td>
                    <td>${buy_price}</td>
                    <td>${sell_price}</td>
                    <td>${stop_loss}</td>
                    <td>{enhanced_score:.1f}</td>
                    <td>{strategy_summary}</td>
                    <td>{weight_summary}</td>
                    <td>{fused_score:.1f}</td>
                    <td>{growth_score:.1f}</td>
                    <td>{industry_performance}</td>
                    <td>{self._get_enhanced_investment_advice(stock)}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="investment-tips">
                <h3>💡 个人投资建议</h3>
                <ul>
                    <li><strong>🆕 智能价格指导</strong>: 基于增强评分和专家建议计算的买卖点</li>
                    <li><strong>建议买入价格</strong>: 考虑了风险和成长性的合理买入点</li>
                    <li><strong>目标卖出价格</strong>: 基于成长潜力和行业地位的目标价位</li>
                    <li><strong>止损价格</strong>: 风险控制价位，跌破建议考虑止损</li>
                    <li><strong>分批建仓</strong>: 建议分3-4次买入，每次25%仓位</li>
                    <li><strong>🆕 行业比较</strong>: 显示在同行业中的相对表现水平</li>
                    <li><strong>🆕 成长性分析</strong>: 评估了EPS增长率和自由现金流质量</li>
                    <li><strong>🆕 专家建议集成</strong>: FCF打分、成长性跟踪、预警提示全面集成</li>
                    <li><strong>🆕 策略信号融合</strong>: 集成TDI、NiuniuV3、CPGW策略信号，提升选股质量</li>
                    <li><strong>🆕 融合评分</strong>: AI增强分析(70%) + 策略信号(30%)的综合评分</li>
                    <li><strong>🆕 动态权重系统</strong>: 基于AI和策略准确性差异自动调整权重，实现自适应学习</li>
                    <li><strong>🆕 权重趋势</strong>: 显示AI和策略权重的调整方向和幅度，帮助理解系统学习过程</li>
                </ul>
            </div>
            
            <div class="risk-warning">
                <h3>⚠️ 风险提示</h3>
                <p>本推荐仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
                <p>请根据自身风险承受能力和投资目标做出投资决策。</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_monthly_content(self, results):
        """生成每月报告内容"""
        # 类似weekly但更详细的分析
        return self._generate_weekly_content(results) + "<h2>📊 月度深度分析</h2>"
    
    def _generate_quarterly_content(self, market_analysis, strategy_recommendations):
        """生成季度报告内容"""
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; }}
                .analysis-section {{ background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔄 季度策略调整报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="analysis-section">
                <h2>📊 市场环境分析</h2>
                <p>VIX指数: {market_analysis.get('vix_level', 'N/A')}</p>
                <p>市场情绪: {market_analysis.get('market_sentiment', {}).get('sentiment', 'N/A')}</p>
            </div>
            
            <div class="analysis-section">
                <h2>💡 策略调整建议</h2>
        """
        
        if strategy_recommendations:
            for rec in strategy_recommendations:
                html_content += f"""
                    <div style="margin: 10px 0; padding: 10px; border-left: 4px solid #007bff;">
                        <strong>{rec['action']}</strong><br>
                        <small>原因: {rec['reason']}</small>
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _get_investment_advice(self, stock):
        """获取投资建议"""
        quality = stock['quality_factor']
        score = stock.get('multifactor_score', stock.get('enhanced_score', 0) * 100)
        
        if quality > 0.8 and score > 70:
            return "强烈推荐，可考虑重仓"
        elif quality > 0.7 and score > 60:
            return "推荐买入，分批建仓"
        elif quality > 0.6 and score > 55:
            return "谨慎推荐，小仓位试仓"
        else:
            return "观望，等待更好机会"
    
    def _get_industry_performance(self, stock):
        """获取行业表现信息"""
        try:
            # 优先从增强分析中获取
            enhanced_analysis = stock.get('enhanced_analysis', {})
            financial_analysis = enhanced_analysis.get('enhanced_features', {}).get('financial_analysis', {})
            
            # 尝试从dimensions中获取行业比较信息
            dimensions = financial_analysis.get('dimensions', {})
            industry_comparison = dimensions.get('industry_comparison', {})
            industry_performance = industry_comparison.get('summary', 'N/A')
            
            # 如果没有获取到，尝试从其他字段获取
            if industry_performance == 'N/A':
                industry_performance = financial_analysis.get('industry_summary', 'N/A')
            
            # 如果还是没有，尝试从警告和建议中提取
            if industry_performance == 'N/A':
                warnings = enhanced_analysis.get('warnings', [])
                recommendations = enhanced_analysis.get('recommendations', [])
                for item in warnings + recommendations:
                    if '行业' in item:
                        if '优秀' in item:
                            industry_performance = '行业内优秀'
                        elif '良好' in item:
                            industry_performance = '行业内良好' 
                        elif '平均' in item:
                            industry_performance = '行业内平均'
                        elif '较差' in item or '落后' in item:
                            industry_performance = '行业内较差'
                        break
            
            return industry_performance if industry_performance != 'N/A' else '数据不可用'
            
        except Exception as e:
            return '数据获取失败'
    
    def _get_strategy_signal_summary(self, strategy_signals: Dict) -> str:
        """
        生成策略信号摘要
        
        Args:
            strategy_signals: 策略信号字典
            
        Returns:
            策略信号摘要字符串
        """
        if not strategy_signals:
            return "无信号"
        
        # 统计各策略信号
        buy_count = 0
        hold_count = 0
        sell_count = 0
        total_count = 0
        
        for strategy_name, signal_info in strategy_signals.items():
            # 安全地处理信号值，处理不同类型的信号
            signal_raw = signal_info.get('signal', 'unknown')
            
            # 如果信号是数值类型，转换为字符串
            if isinstance(signal_raw, (int, float, np.integer, np.floating)):
                if signal_raw > 0:
                    signal = 'buy'
                elif signal_raw < 0:
                    signal = 'sell'
                else:
                    signal = 'hold'
            else:
                # 如果是字符串，转换为小写
                signal = str(signal_raw).lower()
            
            if 'buy' in signal:
                buy_count += 1
            elif 'hold' in signal:
                hold_count += 1
            elif 'sell' in signal:
                sell_count += 1
            total_count += 1
        
        # 生成摘要
        if buy_count > hold_count and buy_count > sell_count:
            return f"🟢 买入({buy_count}/{total_count})"
        elif hold_count > sell_count:
            return f"🟡 持有({hold_count}/{total_count})"
        elif sell_count > 0:
            return f"🔴 卖出({sell_count}/{total_count})"
        else:
            return "⚪ 中性"
    
    def _get_dynamic_weight_summary(self, dynamic_weights: Dict) -> str:
        """
        生成动态权重摘要
        
        Args:
            dynamic_weights: 动态权重字典
            
        Returns:
            动态权重摘要字符串
        """
        if not dynamic_weights:
            return "默认权重"
        
        ai_weight = dynamic_weights.get('ai_weight', 0.7)
        strategy_weight = dynamic_weights.get('strategy_weight', 0.3)
        
        # 判断权重调整方向
        if ai_weight > 0.7:
            ai_trend = "↑"
        elif ai_weight < 0.7:
            ai_trend = "↓"
        else:
            ai_trend = "="
        
        if strategy_weight > 0.3:
            strategy_trend = "↑"
        elif strategy_weight < 0.3:
            strategy_trend = "↓"
        else:
            strategy_trend = "="
        
        return f"AI{ai_trend}{ai_weight:.1f} 策略{strategy_trend}{strategy_weight:.1f}"
    
    def _get_enhanced_investment_advice(self, stock, thresholds=None):
        """获取增强版投资建议（Phase 2新功能）
        thresholds: 可选，dict，支持动态调整各项阈值
        """
        # 默认阈值
        default_thresholds = {
            'strong_buy': {'enhanced_score': 0.7, 'quality': 0.7, 'score': 65},
            'buy':        {'enhanced_score': 0.6, 'quality': 0.6, 'score': 55},
            'hold':       {'enhanced_score': 0.5, 'quality': 0.5, 'score': 50},
            'watch':      {'enhanced_score': 0.3},
        }
        if thresholds is None:
            thresholds = default_thresholds
        else:
            # 合并用户自定义阈值和默认阈值
            for k, v in default_thresholds.items():
                if k not in thresholds:
                    thresholds[k] = v
                else:
                    for subk, subv in v.items():
                        if subk not in thresholds[k]:
                            thresholds[k][subk] = subv

        quality = stock['quality_factor']
        score = stock.get('multifactor_score', stock.get('enhanced_score', 0) * 100)
        enhanced_analysis = stock.get('enhanced_analysis', {})
        enhanced_score = enhanced_analysis.get('overall_score', 0)
        warnings = enhanced_analysis.get('warnings', [])
        recommendations = enhanced_analysis.get('recommendations', [])

        logger.debug(f"投资建议调试 - {stock.get('symbol', 'Unknown')}: "
                    f"enhanced_score={enhanced_score}, quality={quality}, score={score}")

        high_risk_warnings = [w for w in warnings if any(keyword in str(w).lower() 
                                for keyword in ['high risk', '高风险', 'overvalued', '估值过高', '严重'])]
        if high_risk_warnings:
            return f"⚠️ 谨慎：{high_risk_warnings[0][:20]}..."

        # 参数化阈值判断
        t = thresholds
        if (enhanced_score > t['strong_buy']['enhanced_score'] and 
            quality > t['strong_buy']['quality'] and 
            score > t['strong_buy']['score']):
            advice = "🟢 强烈推荐"
            if recommendations:
                advice += f"，{recommendations[0][:15]}..."
            return advice
        elif (enhanced_score > t['buy']['enhanced_score'] and 
              quality > t['buy']['quality'] and 
              score > t['buy']['score']):
            advice = "🔵 推荐买入"
            if recommendations:
                advice += f"，{recommendations[0][:15]}..."
            return advice
        elif (enhanced_score > t['hold']['enhanced_score'] and 
              quality > t['hold']['quality'] and 
              score > t['hold']['score']):
            return "🟡 小仓位试仓"
        elif enhanced_score > t['watch']['enhanced_score']:
            return "🟠 观望为主"
        else:
            return "🔴 暂时回避"
    
    def setup_schedule(self):
        """设置定时任务"""
        try:
            # 每周筛选 - 每周日20:00
            schedule.every().sunday.at("20:00").do(self.run_weekly_screening)
            
            # 每月深度分析 - 每月第一个周日 (改为每周检查，但只在月初执行)
            schedule.every().sunday.at("20:30").do(self._check_monthly_analysis)
            
            # 季度策略调整 - 每季度第一个周日 (改为每周检查，但只在季度初执行)
            schedule.every().sunday.at("21:00").do(self._check_quarterly_strategy)
            
            logger.info("✅ 定时任务设置完成")
            
        except Exception as e:
            logger.error(f"设置定时任务失败: {e}")
    
    def start_automation(self):
        """启动自动化系统"""
        try:
            print("=" * 80)
            print("🚀 个人投资者自动化股票推荐系统")
            print("=" * 80)
            print()
            print("📊 系统功能:")
            print("   ✓ 每周自动筛选优质股票")
            print("   ✓ 每月深度分析投资组合")
            print("   ✓ 季度策略调整建议")
            print("   ✓ 自动更新市场数据")
            print("   ✓ 个性化投资建议邮件")
            print()
            print("⏰ 定时安排:")
            print("   - 每周筛选: 每周日 20:00")
            print("   - 每月分析: 每月第一个周日 20:00")
            print("   - 季度调整: 每季度第一个周日 20:00")
            print()
            print(f"📧 邮件接收: {self.config['email']}")
            print(f"🎯 风险偏好: {self.config['risk_tolerance']}")
            print(f"💰 最大仓位: {self.config['max_position_size']*100}%")
            print()
            print("🛑 要停止服务，请按 Ctrl+C")
            print("=" * 80)
            print()
            
            # 设置定时任务
            self.setup_schedule()
            
            # 显示下次运行时间
            next_run = schedule.next_run()
            if next_run:
                print(f"⏰ 下次运行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # 可选：立即运行一次测试
            test_now = input("💡 是否立即运行一次测试？(y/N): ").strip().lower()
            if test_now == 'y':
                print("🧪 开始测试运行...")
                self.run_weekly_screening()
            
            print("⏳ 等待定时任务触发...")
            print("   (或按 Ctrl+C 停止服务)")
            
            # 保持运行
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            print("\n👋 自动化服务已停止")
        except Exception as e:
            logger.error(f"自动化服务运行出错: {e}")

def main():
    """主函数"""
    automation = PersonalInvestorAutomation()
    automation.start_automation()

if __name__ == "__main__":
    main() 
