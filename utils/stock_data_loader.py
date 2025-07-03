#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据加载器
用于从JSON文件中加载股票分析数据，方便AI分析使用
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class StockDataLoader:
    """股票数据加载器 - 用于加载和分析JSON格式的股票数据"""
    
    def __init__(self, data_dir: str = "."):
        """
        初始化数据加载器
        
        Args:
            data_dir: JSON数据文件所在目录
        """
        self.data_dir = data_dir
        self.latest_data_file = None
        self.cached_data = None
    
    def get_latest_data_file(self) -> Optional[str]:
        """获取最新的JSON数据文件"""
        try:
            # 查找所有stock_analysis_data_开头的JSON文件
            pattern = os.path.join(self.data_dir, "stock_analysis_data_*.json")
            files = glob.glob(pattern)
            
            if not files:
                logger.warning("未找到股票分析数据文件")
                return None
            
            # 按修改时间排序，获取最新的文件
            latest_file = max(files, key=os.path.getmtime)
            self.latest_data_file = latest_file
            logger.info(f"找到最新数据文件: {latest_file}")
            return latest_file
            
        except Exception as e:
            logger.error(f"获取最新数据文件失败: {e}")
            return None
    
    def load_data(self, filename: str = None) -> Optional[Dict]:
        """
        加载JSON数据
        
        Args:
            filename: 指定文件名，如果为None则加载最新文件
            
        Returns:
            加载的数据字典
        """
        try:
            if filename is None:
                filename = self.get_latest_data_file()
                if filename is None:
                    return None
            
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.cached_data = data
            logger.info(f"成功加载数据文件: {filename}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误，文件可能损坏: {e}")
            logger.error(f"文件路径: {filename}")
            return None
        except FileNotFoundError:
            logger.error(f"文件不存在: {filename}")
            return None
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            return None
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        获取单个股票的数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票数据字典
        """
        if self.cached_data is None:
            self.load_data()
        
        if self.cached_data and 'stocks' in self.cached_data:
            return self.cached_data['stocks'].get(symbol)
        
        return None
    
    def get_all_stocks(self) -> List[str]:
        """获取所有股票代码列表"""
        if self.cached_data is None:
            self.load_data()
        
        if self.cached_data and 'stocks' in self.cached_data:
            return list(self.cached_data['stocks'].keys())
        
        return []
    
    def get_portfolio_summary(self) -> Optional[Dict]:
        """获取投资组合汇总信息"""
        if self.cached_data is None:
            self.load_data()
        
        return self.cached_data.get('portfolio_summary') if self.cached_data else None
    
    def get_macro_analysis(self) -> Optional[Dict]:
        """获取宏观分析数据"""
        if self.cached_data is None:
            self.load_data()
        
        return self.cached_data.get('macro_analysis') if self.cached_data else None
    
    def format_for_ai_input(self, symbols: List[str] = None) -> str:
        """
        格式化数据为AI输入格式
        
        Args:
            symbols: 指定股票代码列表，如果为None则包含所有股票
            
        Returns:
            格式化的AI输入字符串
        """
        if self.cached_data is None:
            self.load_data()
        
        if not self.cached_data:
            return "数据加载失败"
        
        # 确定要包含的股票
        if symbols is None:
            symbols = self.get_all_stocks()
        
        # 构建AI输入格式
        ai_input = []
        ai_input.append(f"📊 股票分析数据 - {self.cached_data.get('timestamp', '未知时间')}")
        ai_input.append("=" * 60)
        
        # 添加投资组合汇总
        portfolio_summary = self.get_portfolio_summary()
        if portfolio_summary:
            ai_input.append("💰 投资组合汇总:")
            ai_input.append(f"   总价值: ${portfolio_summary.get('total_value', 0):,.2f}")
            ai_input.append(f"   股票配置: {portfolio_summary.get('stock_allocation', 0):.2f}%")
            ai_input.append(f"   现金配置: {portfolio_summary.get('cash_allocation', 0):.2f}%")
            ai_input.append("")
        
        # 添加宏观分析
        macro_analysis = self.get_macro_analysis()
        if macro_analysis:
            ai_input.append("🌍 宏观环境分析:")
            ai_input.append(f"   宏观得分: {macro_analysis.get('macro_score', 0):.2f}/1.00")
            ai_input.append(f"   环境建议: {macro_analysis.get('recommendation', '无')}")
            ai_input.append("")
        
        # 添加个股数据
        ai_input.append("📈 个股详细分析:")
        ai_input.append("")
        
        for symbol in symbols:
            stock_data = self.get_stock_data(symbol)
            if not stock_data:
                continue
            
            ai_input.append(f"💼 {symbol}:")
            
            # 基本信息
            basic_info = stock_data.get('basic_info', {})
            if basic_info:
                ai_input.append(f"   当前价格: ${basic_info.get('current_price', 0):.2f}")
                ai_input.append(f"   涨跌幅: {basic_info.get('price_change_pct', 0):+.2f}%")
                ai_input.append(f"   RSI: {basic_info.get('rsi', 0):.1f}")
            
            # 市场环境
            market_env = stock_data.get('market_environment', {})
            if market_env:
                ai_input.append(f"   市场环境: {market_env.get('trend', 'unknown')}")
                ai_input.append(f"   置信度: {market_env.get('confidence', 0):.2f}")
            
            # 策略建议
            strategy = stock_data.get('strategy', {})
            if strategy:
                ai_input.append(f"   推荐策略: {strategy.get('recommended_strategy', 'unknown')}")
                ai_input.append(f"   信号质量: {strategy.get('signal_quality', 0):.2f}")
            
            # 持仓分析
            position = stock_data.get('position_analysis', {})
            if position:
                ai_input.append(f"   持仓成本: ${position.get('cost_price', 0):.2f}")
                ai_input.append(f"   持仓数量: {position.get('shares', 0):,.0f}")
                ai_input.append(f"   盈亏: {position.get('pnl_percent', 0):+.2f}%")
            
            # 财务分析
            financial = stock_data.get('financial_analysis', {})
            if financial:
                ai_input.append(f"   财务评分: {financial.get('total_score', 0):.1f}/100")
                ai_input.append(f"   评级: {financial.get('overall_rating', 'unknown')}")
            
            # 流动性分析
            liquidity = stock_data.get('liquidity_analysis', {})
            if liquidity:
                ai_input.append(f"   流动性评分: {liquidity.get('liquidity_score', 0):.1f}/100")
                ai_input.append(f"   风险等级: {liquidity.get('risk_level', 'unknown')}")
            
            ai_input.append("")
        
        return "\n".join(ai_input)
    
    def get_stock_summary(self, symbol: str) -> str:
        """
        获取单个股票的摘要信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票摘要字符串
        """
        stock_data = self.get_stock_data(symbol)
        if not stock_data:
            return f"未找到 {symbol} 的数据"
        
        summary = []
        summary.append(f"📊 {symbol} 股票摘要:")
        
        # 基本信息
        basic_info = stock_data.get('basic_info', {})
        if basic_info:
            summary.append(f"💰 价格: ${basic_info.get('current_price', 0):.2f} ({basic_info.get('price_change_pct', 0):+.2f}%)")
        
        # 市场环境
        market_env = stock_data.get('market_environment', {})
        if market_env:
            summary.append(f"🌍 环境: {market_env.get('trend', 'unknown')} (置信度: {market_env.get('confidence', 0):.2f})")
        
        # 策略建议
        strategy = stock_data.get('strategy', {})
        if strategy:
            summary.append(f"🎯 策略: {strategy.get('recommended_strategy', 'unknown')}")
        
        # 持仓信息
        position = stock_data.get('position_analysis', {})
        if position:
            summary.append(f"📈 持仓: {position.get('shares', 0):,.0f}股, 盈亏: {position.get('pnl_percent', 0):+.2f}%")
        
        return "\n".join(summary)
    
    def get_data_statistics(self) -> Dict:
        """获取数据统计信息"""
        if self.cached_data is None:
            self.load_data()
        
        if not self.cached_data:
            return {}
        
        stats = {
            'total_stocks': len(self.get_all_stocks()),
            'timestamp': self.cached_data.get('timestamp', 'unknown'),
            'data_version': self.cached_data.get('data_version', 'unknown'),
            'has_portfolio_summary': 'portfolio_summary' in self.cached_data,
            'has_macro_analysis': 'macro_analysis' in self.cached_data
        }
        
        # 统计各种分析类型
        stocks = self.cached_data.get('stocks', {})
        analysis_types = {
            'position_analysis': 0,
            'financial_analysis': 0,
            'liquidity_analysis': 0,
            'enhanced_analysis': 0,
            'wave_trading_analysis': 0
        }
        
        for stock_data in stocks.values():
            for analysis_type in analysis_types:
                if analysis_type in stock_data:
                    analysis_types[analysis_type] += 1
        
        stats['analysis_coverage'] = analysis_types
        
        return stats

def main():
    """测试函数"""
    loader = StockDataLoader()
    
    # 获取最新数据文件
    latest_file = loader.get_latest_data_file()
    if latest_file:
        print(f"最新数据文件: {latest_file}")
        
        # 加载数据
        data = loader.load_data()
        if data:
            print(f"数据加载成功，包含 {len(data.get('stocks', {}))} 只股票")
            
            # 获取统计信息
            stats = loader.get_data_statistics()
            print(f"数据统计: {stats}")
            
            # 获取所有股票代码
            symbols = loader.get_all_stocks()
            print(f"股票列表: {symbols}")
            
            # 获取单个股票数据
            if symbols:
                first_stock = symbols[0]
                stock_data = loader.get_stock_data(first_stock)
                print(f"\n{first_stock} 数据示例:")
                print(json.dumps(stock_data, indent=2, ensure_ascii=False)[:500] + "...")
                
                # 获取股票摘要
                summary = loader.get_stock_summary(first_stock)
                print(f"\n{first_stock} 摘要:")
                print(summary)
            
            # 格式化AI输入
            ai_input = loader.format_for_ai_input(symbols[:3])  # 只显示前3只股票
            print(f"\nAI输入格式示例:")
            print(ai_input)
    else:
        print("未找到数据文件")

if __name__ == "__main__":
    main() 