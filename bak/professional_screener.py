#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专业多策略量化股票筛选器
整合所有技术指标和策略，提供专业的股票筛选功能
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据接口
from data.data_interface import DataInterface

# 导入策略和指标
from strategy.strategy_factory import StrategyFactory
from strategy.indicators import TechnicalIndicators, calculate_indicators

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProfessionalScreener")


class ProfessionalMultiStrategyScreener:
    """专业多策略量化股票筛选器"""
    
    def __init__(self, min_market_cap: float = 1e9, min_avg_volume: float = 5e5):
        """
        初始化筛选器
        
        参数:
            min_market_cap: 最小市值门槛（默认10亿美元）
            min_avg_volume: 最小平均成交量门槛（默认50万股）
        """
        self.min_market_cap = min_market_cap
        self.min_avg_volume = min_avg_volume
        
        # 初始化组件
        self.data_interface = DataInterface()
        self.strategy_factory = StrategyFactory()
        self.technical_indicators = TechnicalIndicators()
        
        # 创建策略实例
        self.strategies = self._initialize_strategies()
        
        # 筛选权重配置
        self.weights = {
            'technical': 0.40,      # 技术指标权重
            'strategy': 0.35,       # 策略信号权重
            'liquidity': 0.15,      # 流动性权重
            'momentum': 0.10        # 动量权重
        }
        
        logger.info("🚀 专业多策略量化筛选器初始化完成")
    
    def _initialize_strategies(self) -> Dict[str, Any]:
        """初始化所有策略"""
        strategies = {}
        try:
            # 创建核心策略
            strategies['tdi'] = self.strategy_factory.create_strategy('TDI')
            strategies['niuniu'] = self.strategy_factory.create_strategy('NiuniuV3')
            strategies['cpgw'] = self.strategy_factory.create_strategy('CPGW')
            strategies['combined'] = self.strategy_factory.create_combined_strategy()
            
            logger.info(f"✅ 成功初始化 {len(strategies)} 个策略")
            return strategies
        except Exception as e:
            logger.error(f"❌ 策略初始化失败: {e}")
            return {}
    
    def get_stock_universe(self) -> List[str]:
        """获取股票池（带流动性筛选）"""
        try:
            # 获取所有股票
            all_stocks = self.data_interface.get_available_symbols()
            logger.info(f"📊 原始股票池: {len(all_stocks)} 只股票")
            
            # 流动性筛选
            qualified_stocks = []
            for symbol in all_stocks:
                try:
                    # 获取最近60天数据进行流动性筛选
                    hist = self.data_interface.get_data_for_strategy(symbol, lookback_days=60)
                    if hist is None or len(hist) < 30:
                        continue
                    
                    # 平均成交量筛选
                    avg_volume = hist['volume'].mean()
                    if avg_volume < self.min_avg_volume:
                        continue
                    
                    # 价格筛选（避免仙股）
                    current_price = hist['close'].iloc[-1]
                    if current_price < 2.0:  # 低于2美元的股票
                        continue
                    
                    qualified_stocks.append(symbol)
                    
                except Exception as e:
                    continue
            
            logger.info(f"✅ 流动性筛选后: {len(qualified_stocks)} 只股票")
            return qualified_stocks
            
        except Exception as e:
            logger.error(f"❌ 获取股票池失败: {e}")
            return []
    
    def calculate_technical_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标综合评分"""
        try:
            # 计算所有技术指标
            indicators = calculate_indicators(data, selected_indicators=[
                'sma', 'ema', 'bb', 'rsi', 'macd', 'adx'
            ])
            
            scores = {}
            
            # 1. 趋势指标评分
            trend_score = self._calculate_trend_score(data, indicators)
            scores['trend'] = trend_score
            
            # 2. 动量指标评分
            momentum_score = self._calculate_momentum_score(data, indicators)
            scores['momentum'] = momentum_score
            
            # 3. 超买超卖指标评分
            overbought_oversold_score = self._calculate_overbought_oversold_score(indicators)
            scores['overbought_oversold'] = overbought_oversold_score
            
            # 综合技术评分
            technical_score = (
                trend_score * 0.40 +
                momentum_score * 0.35 +
                overbought_oversold_score * 0.25
            )
            
            scores['technical_total'] = technical_score
            return scores
            
        except Exception as e:
            logger.error(f"技术指标计算失败: {e}")
            return {'technical_total': 0}
    
    def _calculate_trend_score(self, data: pd.DataFrame, indicators: Dict) -> float:
        """计算趋势评分"""
        try:
            close = data['close']
            score = 0
            
            # SMA趋势
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20 = indicators['sma_20'].iloc[-1]
                sma_50 = indicators['sma_50'].iloc[-1]
                current_price = close.iloc[-1]
                
                if current_price > sma_20 > sma_50:
                    score += 40  # 强势上升趋势
                elif current_price > sma_20:
                    score += 20  # 短期上升趋势
                elif current_price < sma_20 < sma_50:
                    score -= 40  # 强势下降趋势
                else:
                    score -= 20  # 短期下降趋势
            
            # ADX趋势强度
            if 'adx' in indicators:
                adx = indicators['adx'].iloc[-1]
                if adx > 25:
                    score += 30  # 强趋势
                elif adx > 20:
                    score += 15  # 中等趋势
            
            return max(-100, min(100, score))
            
        except Exception as e:
            return 0
    
    def _calculate_momentum_score(self, data: pd.DataFrame, indicators: Dict) -> float:
        """计算动量评分"""
        try:
            score = 0
            
            # MACD动量
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                macd_signal = indicators['macd_signal'].iloc[-1]
                
                if macd > macd_signal and macd > 0:
                    score += 35  # 强势上升动量
                elif macd > macd_signal:
                    score += 20  # 上升动量
                elif macd < macd_signal and macd < 0:
                    score -= 35  # 强势下降动量
                else:
                    score -= 20  # 下降动量
            
            # 价格动量
            close = data['close']
            if len(close) >= 20:
                momentum_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100
                if momentum_20 > 10:
                    score += 30
                elif momentum_20 > 5:
                    score += 15
                elif momentum_20 < -10:
                    score -= 30
                elif momentum_20 < -5:
                    score -= 15
            
            return max(-100, min(100, score))
            
        except Exception as e:
            return 0
    
    def _calculate_overbought_oversold_score(self, indicators: Dict) -> float:
        """计算超买超卖评分"""
        try:
            score = 0
            
            # RSI评分
            if 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                if 30 <= rsi <= 70:
                    score += 25  # 正常区间
                elif 20 <= rsi < 30:
                    score += 40  # 超卖，买入机会
                elif 70 < rsi <= 80:
                    score -= 25  # 超买警告
                elif rsi > 80:
                    score -= 50  # 严重超买
                elif rsi < 20:
                    score += 50  # 严重超卖，强买入机会
            
            return max(-100, min(100, score))
            
        except Exception as e:
            return 0
    
    def calculate_strategy_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算策略信号评分"""
        scores = {}
        
        for name, strategy in self.strategies.items():
            try:
                # 生成策略信号
                signals = strategy.generate_signals(data)
                
                if signals is not None and not signals.empty and 'signal' in signals.columns:
                    # 计算最近信号强度
                    recent_signals = signals['signal'].tail(10)
                    signal_strength = recent_signals.mean()
                    signal_consistency = 1 - recent_signals.std() if recent_signals.std() > 0 else 1
                    
                    # 策略评分
                    strategy_score = signal_strength * signal_consistency * 100
                    scores[name] = max(-100, min(100, strategy_score))
                else:
                    scores[name] = 0
                    
            except Exception as e:
                logger.warning(f"策略 {name} 计算失败: {e}")
                scores[name] = 0
        
        # 综合策略评分
        if scores:
            strategy_total = np.mean(list(scores.values()))
            scores['strategy_total'] = strategy_total
        else:
            scores['strategy_total'] = 0
        
        return scores
    
    def calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """计算流动性评分"""
        try:
            volume = data['volume']
            avg_volume = volume.mean()
            
            # 基于平均成交量的流动性评分
            if avg_volume > 10e6:
                return 100  # 极高流动性
            elif avg_volume > 5e6:
                return 80   # 高流动性
            elif avg_volume > 2e6:
                return 60   # 中高流动性
            elif avg_volume > 1e6:
                return 40   # 中等流动性
            else:
                return 20   # 低流动性
                
        except Exception as e:
            return 50
    
    def analyze_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """综合分析单只股票"""
        try:
            # 获取股票数据
            data = self.data_interface.get_data_for_strategy(symbol, lookback_days=180)
            if data is None or len(data) < 60:
                return None
            
            # 各维度评分
            technical_scores = self.calculate_technical_score(data)
            strategy_scores = self.calculate_strategy_score(data)
            liquidity_score = self.calculate_liquidity_score(data)
            
            # 综合评分
            total_score = (
                technical_scores.get('technical_total', 0) * self.weights['technical'] +
                strategy_scores.get('strategy_total', 0) * self.weights['strategy'] +
                liquidity_score * self.weights['liquidity'] +
                technical_scores.get('momentum', 0) * self.weights['momentum']
            )
            
            # 构建结果
            result = {
                'symbol': symbol,
                'total_score': total_score,
                'current_price': data['close'].iloc[-1],
                'avg_volume': data['volume'].mean(),
                
                # 技术指标评分
                'technical_total': technical_scores.get('technical_total', 0),
                'trend_score': technical_scores.get('trend', 0),
                'momentum_score': technical_scores.get('momentum', 0),
                'overbought_oversold_score': technical_scores.get('overbought_oversold', 0),
                
                # 策略评分
                'strategy_total': strategy_scores.get('strategy_total', 0),
                'tdi_score': strategy_scores.get('tdi', 0),
                'niuniu_score': strategy_scores.get('niuniu', 0),
                'cpgw_score': strategy_scores.get('cpgw', 0),
                'combined_score': strategy_scores.get('combined', 0),
                
                # 其他评分
                'liquidity_score': liquidity_score,
                
                # 分析时间
                'analysis_time': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {symbol} 失败: {e}")
            return None
    
    def screen_stocks(self, min_score: float = 50, max_results: int = 50) -> List[Dict[str, Any]]:
        """执行股票筛选"""
        logger.info("🚀 开始专业多策略股票筛选")
        logger.info("=" * 80)
        
        # 获取股票池
        stock_universe = self.get_stock_universe()
        if not stock_universe:
            logger.error("❌ 无法获取股票池")
            return []
        
        logger.info(f"📊 股票池规模: {len(stock_universe)} 只股票")
        logger.info(f"📈 筛选标准: 综合评分 >= {min_score}")
        logger.info(f"🎯 最大结果数: {max_results}")
        
        # 分析所有股票
        results = []
        total_count = len(stock_universe)
        
        logger.info(f"\n⏳ 开始分析 {total_count} 只股票...")
        
        for i, symbol in enumerate(stock_universe):
            if i % 50 == 0:
                logger.info(f"   进度: {i}/{total_count} ({i/total_count*100:.1f}%)")
            
            analysis = self.analyze_stock(symbol)
            if analysis and analysis['total_score'] >= min_score:
                results.append(analysis)
        
        # 按评分排序
        results.sort(key=lambda x: x['total_score'], reverse=True)
        results = results[:max_results]
        
        logger.info(f"\n🎯 筛选完成！发现 {len(results)} 只优质股票")
        
        return results


def main():
    """主函数"""
    print("🚀 专业多策略量化股票筛选器")
    print("=" * 80)
    
    # 创建筛选器
    screener = ProfessionalMultiStrategyScreener(
        min_market_cap=1e9,    # 10亿美元市值门槛
        min_avg_volume=2e5     # 20万股成交量门槛
    )
    
    # 执行筛选
    results = screener.screen_stocks(min_score=40, max_results=30)
    
    if results:
        print(f"\n🎯 发现 {len(results)} 只优质股票:")
        print("-" * 100)
        header = f"{'排名':^4} {'股票':^8} {'总分':^6} {'技术':^6} {'策略':^6} {'流动性':^6} {'价格':^8} {'成交量':^10}"
        print(header)
        print("-" * 100)
        
        for i, stock in enumerate(results[:20], 1):
            row = (f"{i:^4} "
                   f"{stock['symbol']:^8} "
                   f"{stock['total_score']:^6.1f} "
                   f"{stock['technical_total']:^6.1f} "
                   f"{stock['strategy_total']:^6.1f} "
                   f"{stock['liquidity_score']:^6.1f} "
                   f"${stock['current_price']:^7.2f} "
                   f"{stock['avg_volume']/1e6:^9.1f}M")
            print(row)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"professional_screening_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📝 详细结果已保存: {filename}")
        except Exception as e:
            print(f"⚠️ 保存失败: {e}")
        
        print(f"\n✅ 筛选完成！")
    else:
        print("❌ 未发现符合条件的股票")


if __name__ == "__main__":
    main() 