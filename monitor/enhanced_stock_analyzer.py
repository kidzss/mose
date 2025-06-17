#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强股票分析器 - 统一集成新的专家建议功能
整合财务分析、行业比较、成长性分析和退出策略
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# 设置日志
logger = logging.getLogger(__name__)

class EnhancedStockAnalyzer:
    """增强股票分析器 - 集成所有新功能的统一接口"""
    
    def __init__(self):
        """初始化增强分析器"""
        self.financial_analyzer = None
        self.exit_strategy = None
        
        # 尝试初始化增强版财务分析器
        try:
            from monitor.financial_analyzer import FinancialAnalyzer
            self.financial_analyzer = FinancialAnalyzer()
            logger.info("✅ 增强版财务分析器初始化成功")
        except Exception as e:
            logger.warning(f"增强版财务分析器初始化失败: {e}")
            
        # 尝试初始化退出策略
        try:
            from strategy.enhanced_exit_strategy import EnhancedExitStrategy
            self.exit_strategy = EnhancedExitStrategy()
            logger.info("✅ 增强版退出策略初始化成功")
        except Exception as e:
            logger.warning(f"增强版退出策略初始化失败: {e}")
            
        logger.info("🚀 增强股票分析器初始化完成")
    
    def analyze_stock_enhanced(self, symbol: str, current_price: float = None) -> Dict[str, Any]:
        """增强分析股票 - 简化版本，与现有系统兼容"""
        return self.analyze_stock_comprehensive(symbol, current_price)
    
    def analyze_stock_comprehensive(self, symbol: str, current_price: float = None) -> Dict[str, Any]:
        """
        综合分析股票 - 包含所有新功能
        
        Args:
            symbol: 股票代码
            current_price: 当前价格（可选）
            
        Returns:
            Dict: 综合分析结果
        """
        result = {
            'symbol': symbol,
            'analysis_time': datetime.now().isoformat(),
            'enhanced_features': {},
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # 财务分析（包含行业比较和成长性分析）
            if self.financial_analyzer:
                try:
                    financial_analysis = self.financial_analyzer.analyze_stock(symbol)
                    if financial_analysis:
                        result['enhanced_features']['financial_analysis'] = financial_analysis
                        
                        # 提取关键信息
                        result['overall_score'] = financial_analysis.get('total_score', 0)
                        result['overall_rating'] = financial_analysis.get('overall_rating', 'N/A')
                        
                        # 从dimensions中提取成长性和行业评分
                        dimensions = financial_analysis.get('dimensions', {})
                        growth_data = dimensions.get('growth', {})
                        industry_data = dimensions.get('industry_comparison', {})
                        
                        result['growth_score'] = growth_data.get('score', 0)
                        result['industry_score'] = industry_data.get('industry_adjusted_score', 0)
                        
                        # 处理警告信息
                        warnings = financial_analysis.get('warnings', [])
                        if warnings:
                            result['warnings'].extend(warnings)
                            
                        logger.info(f"{symbol} 财务分析完成 - 总评分: {result['overall_score']:.2f}")
                    else:
                        logger.warning(f"{symbol} 财务分析数据不可用")
                        
                except Exception as e:
                    logger.error(f"{symbol} 财务分析失败: {e}")
                    result['warnings'].append(f"财务分析失败: {str(e)}")
            
            # 退出策略分析
            if self.exit_strategy and current_price:
                try:
                    # 构造模拟入场数据进行退出信号分析
                    entry_data = {
                        'entry_price': current_price * 0.9,  # 假设入场价格比当前价格低10%
                        'entry_date': datetime.now() - timedelta(days=30),
                        'position_size': 1000
                    }
                    
                    # 创建简单的市场数据（实际应用中需要真实数据）
                    import pandas as pd
                    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                        end=datetime.now(), freq='D')
                    market_data = pd.DataFrame({
                        'close': [current_price] * len(dates),
                        'volume': [1000000] * len(dates)
                    }, index=dates)
                    
                    exit_analysis = self.exit_strategy.calculate_exit_signals(
                        symbol, current_price, entry_data, market_data
                    )
                    
                    if exit_analysis:
                        result['enhanced_features']['exit_strategy'] = exit_analysis
                        
                        # 提取退出建议
                        if exit_analysis.get('should_exit', False):
                            exit_reason = exit_analysis.get('exit_reason', 'Unknown')
                            result['recommendations'].append(f"建议退出: {exit_reason}")
                            
                        logger.info(f"{symbol} 退出策略分析完成")
                    
                except Exception as e:
                    logger.error(f"{symbol} 退出策略分析失败: {e}")
                    result['warnings'].append(f"退出策略分析失败: {str(e)}")
            
            # 生成综合建议
            self._generate_comprehensive_recommendations(result)
            
            return result
            
        except Exception as e:
            logger.error(f"{symbol} 综合分析失败: {e}")
            result['error'] = str(e)
            return result
    
    def _generate_comprehensive_recommendations(self, analysis_result: Dict):
        """生成综合投资建议"""
        try:
            symbol = analysis_result['symbol']
            recommendations = []
            
            # 基于财务分析的建议
            if 'financial_analysis' in analysis_result.get('enhanced_features', {}):
                financial = analysis_result['enhanced_features']['financial_analysis']
                
                # 评级建议
                rating = financial.get('overall_rating', 'N/A')
                if rating == 'Excellent':
                    recommendations.append("📈 财务状况优秀，适合长期持有")
                elif rating == 'Good':
                    recommendations.append("✅ 财务状况良好，可考虑投资")
                elif rating == 'Fair':
                    recommendations.append("⚠️ 财务状况一般，需谨慎评估")
                elif rating == 'Poor':
                    recommendations.append("❌ 财务状况较差，建议避免")
                
                # 成长性建议
                growth_score = financial.get('growth_score', 0)
                if growth_score > 0.8:
                    recommendations.append("🚀 成长性优秀，具有较大上涨潜力")
                elif growth_score > 0.6:
                    recommendations.append("📊 成长性良好，值得关注")
                elif growth_score < 0.4:
                    recommendations.append("📉 成长性较弱，注意风险")
                
                # 行业比较建议
                industry_score = financial.get('industry_score', 0)
                if industry_score > 0.7:
                    recommendations.append("🏆 在同行业中表现优秀")
                elif industry_score < 0.4:
                    recommendations.append("⚠️ 在同行业中表现落后")
            
            # 基于退出策略的建议
            if 'exit_strategy' in analysis_result.get('enhanced_features', {}):
                exit_data = analysis_result['enhanced_features']['exit_strategy']
                if exit_data.get('should_exit', False):
                    exit_reason = exit_data.get('exit_reason', 'Unknown')
                    recommendations.append(f"🔄 退出信号: {exit_reason}")
            
            # 基于警告的建议
            warnings = analysis_result.get('warnings', [])
            if warnings:
                high_risk_warnings = [w for w in warnings if any(keyword in w.lower() 
                                    for keyword in ['high risk', '高风险', 'overvalued', '估值过高'])]
                if high_risk_warnings:
                    recommendations.append("⚠️ 检测到高风险警告，建议重新评估")
            
            # 更新建议列表
            analysis_result['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"生成综合建议失败: {e}")
    
    def is_available(self) -> bool:
        """检查增强功能是否可用"""
        return self.financial_analyzer is not None or self.exit_strategy is not None
    
    def get_feature_status(self) -> Dict[str, bool]:
        """获取各功能模块状态"""
        return {
            'financial_analyzer': self.financial_analyzer is not None,
            'exit_strategy': self.exit_strategy is not None
        } 