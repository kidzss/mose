#!/usr/bin/env python3
"""
增强版股票筛选器 - 使用data模块接口
结合技术面、基本面和动量分析进行股票筛选
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_interface import DataInterface
from strategy.tdi_strategy import TDIStrategy
import warnings
warnings.filterwarnings('ignore')

class EnhancedStockScreener:
    def __init__(self):
        """初始化股票筛选器"""
        try:
            self.data_interface = DataInterface()
            self.tdi_strategy = TDIStrategy()
            print("✅ 数据接口初始化成功")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            sys.exit(1)
    
    def get_stock_pool(self):
        """获取股票池 - 使用data模块接口"""
        try:
            # 使用data模块获取所有可用股票代码
            all_symbols = self.data_interface.get_available_symbols()
            print(f"📊 从数据库获取到 {len(all_symbols)} 只股票")
            
            # 过滤掉一些特殊代码（如果需要）
            filtered_symbols = []
            for symbol in all_symbols:
                # 过滤掉长度过短或包含特殊字符的代码
                if len(symbol) >= 1 and symbol.isalnum():
                    filtered_symbols.append(symbol)
            
            print(f"📈 过滤后可用股票: {len(filtered_symbols)} 只")
            return filtered_symbols
            
        except Exception as e:
            print(f"❌ 获取股票池失败: {e}")
            return []
    
    def _get_market_environment(self):
        """获取市场环境"""
        try:
            # 大盘指数
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            spy_data = self.data_interface.get_historical_data('SPY', start_date, end_date)
            qqq_data = self.data_interface.get_historical_data('QQQ', start_date, end_date)
            
            spy_momentum = 0
            qqq_momentum = 0
            
            if not spy_data.empty:
                spy_momentum = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[0] - 1) * 100
            
            if not qqq_data.empty:
                qqq_momentum = (qqq_data['close'].iloc[-1] / qqq_data['close'].iloc[0] - 1) * 100
            
            # VIX恐慌指数
            vix_data = self.data_interface.get_latest_data('VIX', n_bars=2)
            vix_current = vix_data['close'].iloc[-1] if not vix_data.empty else 20
            
            market_env = {
                'spy_momentum': spy_momentum,
                'qqq_momentum': qqq_momentum,
                'vix': vix_current,
                'market_strength': "strong" if spy_momentum > 0 and qqq_momentum > 0 else "weak"
            }
            
            return market_env
            
        except Exception as e:
            print(f"⚠️ 市场环境获取失败: {e}")
            return {'market_strength': 'neutral'}
    
    def _analyze_stock_comprehensive(self, symbol):
        """股票综合分析"""
        try:
            # 获取历史数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6个月
            
            hist = self.data_interface.get_historical_data(symbol, start_date, end_date)
            
            if hist.empty:
                return None
            
            # 1. 技术分析
            tech_score = self._calculate_technical_score(hist)
            
            # 2. 基本面分析 (简化版，因为没有info数据)
            fundamental_score = 50  # 默认中性评分，可以后续接入基本面数据源
            
            # 3. 市场表现
            momentum_score = self._calculate_momentum_score(hist)
            
            # 4. 流动性分析
            liquidity_score = self._calculate_liquidity_score(hist)
            
            # 综合评分
            total_score = (tech_score * 0.5 + 
                          fundamental_score * 0.3 + 
                          momentum_score * 0.2)
            
            return {
                'symbol': symbol,
                'total_score': total_score,
                'tech_score': tech_score,
                'fundamental_score': fundamental_score,
                'momentum_score': momentum_score,
                'liquidity_score': liquidity_score,
                'current_price': hist['close'].iloc[-1],
                'market_cap': 0,  # 待接入基本面数据
                'pe_ratio': 0,    # 待接入基本面数据
                'profit_margin': 0,  # 待接入基本面数据
                'revenue_growth': 0  # 待接入基本面数据
            }
            
        except Exception as e:
            return None
    
    def _calculate_technical_score(self, hist):
        """技术面评分"""
        try:
            # 使用TDI策略
            data = hist.copy()
            
            # 预处理数据
            if 'date' in data.columns:
                data = data.set_index('date')
            
            signals = self.tdi_strategy.generate_signals(data)
            
            if 'signal' in signals.columns and not signals.empty:
                recent_signals = signals['signal'].tail(10).mean()
                signal_strength = abs(recent_signals)
                signal_direction = 1 if recent_signals > 0 else -1
                
                # 技术评分 (-100 到 100)
                tech_score = signal_direction * signal_strength * 100
                return max(-100, min(100, tech_score))
            
            return 0
            
        except Exception as e:
            print(f"技术分析计算失败: {e}")
            return 0
    

    
    def _calculate_momentum_score(self, hist):
        """动量评分"""
        try:
            closes = hist['close']
            
            # 不同时间段的收益率
            returns_1w = (closes.iloc[-1] / closes.iloc[-5] - 1) * 100 if len(closes) >= 5 else 0
            returns_1m = (closes.iloc[-1] / closes.iloc[-20] - 1) * 100 if len(closes) >= 20 else 0
            returns_3m = (closes.iloc[-1] / closes.iloc[-60] - 1) * 100 if len(closes) >= 60 else 0
            
            # 加权评分
            momentum_score = (returns_1w * 0.5 + returns_1m * 0.3 + returns_3m * 0.2)
            
            # 标准化到 -100 到 100
            return max(-100, min(100, momentum_score))
            
        except:
            return 0
    
    def _calculate_liquidity_score(self, hist):
        """流动性评分"""
        try:
            # 平均交易量
            avg_volume = hist['volume'].mean()
            
            # 基于交易量的流动性评分
            if avg_volume > 5e6:  # 高交易量
                return 100
            elif avg_volume > 2e6:  # 中高交易量
                return 80
            elif avg_volume > 1e6:  # 中等交易量
                return 60
            elif avg_volume > 500000:  # 中低交易量
                return 40
            else:
                return 20
                
        except:
            return 50  # 默认中等流动性
    
    def screen_stocks(self, min_score=60, max_results=20):
        """筛选股票"""
        print(f"\n🔍 开始筛选股票池")
        print("="*50)
        
        # 获取股票池
        stock_list = self.get_stock_pool()
        if not stock_list:
            print(f"❌ 无法获取股票池")
            return []
        
        print(f"📊 股票池规模: {len(stock_list)}只股票")
        print(f"📈 筛选标准: 综合评分 >= {min_score}")
        
        # 获取市场环境
        market_env = self._get_market_environment()
        print(f"🌍 当前市场: {market_env.get('market_strength', 'neutral')}")
        
        # 分析所有股票
        results = []
        print(f"\n⏳ 正在分析股票...")
        
        # 限制分析数量避免太慢
        analyze_count = min(50, len(stock_list))
        for i, symbol in enumerate(stock_list[:analyze_count]):
            if i % 10 == 0:
                print(f"   进度: {i}/{analyze_count}")
            
            analysis = self._analyze_stock_comprehensive(symbol)
            if analysis and analysis['total_score'] >= min_score:
                results.append(analysis)
        
        # 按评分排序
        results.sort(key=lambda x: x['total_score'], reverse=True)
        results = results[:max_results]
        
        # 输出结果
        if results:
            print(f"\n🎯 发现 {len(results)} 只优质股票:")
            print("-" * 80)
            print(f"{'股票':^8} {'价格':^8} {'总分':^6} {'技术':^6} {'基本面':^6} {'动量':^6} {'PE':^6} {'利润率':^6}")
            print("-" * 80)
            
            for stock in results:
                print(f"{stock['symbol']:^8} "
                      f"${stock['current_price']:6.2f} "
                      f"{stock['total_score']:5.1f} "
                      f"{stock['tech_score']:5.1f} "
                      f"{stock['fundamental_score']:5.1f} "
                      f"{stock['momentum_score']:5.1f} "
                      f"{stock['pe_ratio']:5.1f} "
                      f"{stock['profit_margin']:5.1f}%")
        else:
            print(f"\n📊 未发现符合条件的股票（评分>={min_score}）")
        
        return results
    
    def generate_investment_report(self, results):
        """生成投资报告"""
        if not results:
            return
        
        print(f"\n📋 投资建议报告")
        print("="*50)
        
        # 选择前5只股票进行详细分析
        top_stocks = results[:5]
        
        for i, stock in enumerate(top_stocks, 1):
            print(f"\n{i}. {stock['symbol']} - 综合评分: {stock['total_score']:.1f}")
            print(f"   💰 当前价格: ${stock['current_price']:.2f}")
            print(f"   📊 市值: ${stock['market_cap']/1e9:.1f}B")
            print(f"   📈 PE比率: {stock['pe_ratio']:.2f}")
            print(f"   💡 利润率: {stock['profit_margin']:.1f}%")
            print(f"   🚀 营收增长: {stock['revenue_growth']:.1f}%")
            
            # 投资建议
            if stock['total_score'] >= 80:
                advice = "🟢 强烈推荐，可考虑10-15%仓位"
            elif stock['total_score'] >= 70:
                advice = "🟡 值得关注，可考虑5-10%仓位"
            else:
                advice = "🟠 谨慎观察，小仓位试水"
            
            print(f"   📝 建议: {advice}")
        
        # 保存到文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"stock_screening_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📝 详细报告已保存: {filename}")
        except Exception as e:
            print(f"⚠️ 报告保存失败: {e}")

def main():
    """主函数"""
    print("🚀 开始增强版股票筛选")
    print("="*60)
    
    screener = EnhancedStockScreener()
    
    # 运行股票筛选
    results = screener.screen_stocks(min_score=65, max_results=15)
    if results:
        screener.generate_investment_report(results)
    
    print(f"\n✅ 股票筛选完成！")

if __name__ == "__main__":
    main() 