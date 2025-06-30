#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI实时监控系统
使用真实的实时数据进行分析
"""

import asyncio
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入AI交易模块
from ai_trading_module import AITradingModule

# 导入数据接口
try:
    from data.data_interface import YahooFinanceRealTimeSource
    from data.data_loader import DataLoader
except ImportError:
    print("⚠️ 未找到数据接口模块，将使用yfinance直接获取数据")
    import yfinance as yf

class AIRealtimeMonitor:
    """AI实时监控系统"""
    
    def __init__(self, symbols: List[str] = None, config_file: str = "portfolio_config.json"):
        """初始化AI实时监控系统"""
        self.ai_module = AITradingModule()
        self.config_file = config_file
        
        # 从配置文件加载股票列表
        if symbols is None:
            self.symbols = self._load_symbols_from_config()
        else:
            self.symbols = symbols
        
        # 初始化数据源
        try:
            self.data_source = YahooFinanceRealTimeSource()
            print("✅ 使用YahooFinanceRealTimeSource数据源")
        except Exception as e:
            print(f"⚠️ 无法初始化数据源: {e}")
            self.data_source = None
        
        # 数据缓存
        self.data_cache = {}
        self.last_update = {}
        
        print(f"🤖 AI实时监控系统初始化完成，监控股票: {self.symbols}")
    
    def _load_symbols_from_config(self) -> List[str]:
        """从配置文件加载股票列表"""
        try:
            import json
            import os
            
            # 尝试多个配置文件路径
            config_paths = [
                self.config_file,
                f"config/{self.config_file}",
                "portfolio_config.json",
                "config/portfolio_config.json",
                "config/portfolio_config_latest.json"
            ]
            
            config_data = None
            for path in config_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            print(f"✅ 成功加载配置文件: {path}")
                            break
                    except Exception as e:
                        print(f"⚠️ 无法读取配置文件 {path}: {e}")
                        continue
            
            if config_data:
                symbols = []
                
                # 从持仓中获取股票
                if 'positions' in config_data:
                    position_symbols = list(config_data['positions'].keys())
                    symbols.extend(position_symbols)
                    print(f"📊 从持仓加载股票: {position_symbols}")
                
                # 从观察仓中获取股票
                if 'watchlist' in config_data:
                    watchlist_symbols = list(config_data['watchlist'].keys())
                    symbols.extend(watchlist_symbols)
                    print(f"👀 从观察仓加载股票: {watchlist_symbols}")
                
                # 去重
                symbols = list(set(symbols))
                
                if symbols:
                    return symbols
                else:
                    print("⚠️ 配置文件中未找到股票，使用默认股票列表")
            
        except Exception as e:
            print(f"⚠️ 加载配置文件失败: {e}")
        
        # 如果无法加载配置，使用默认股票列表
        default_symbols = ["NVDA", "AMD", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AMZN"]
        print(f"📋 使用默认股票列表: {default_symbols}")
        return default_symbols
    
    def update_symbols(self, symbols: List[str]):
        """更新监控的股票列表"""
        self.symbols = symbols
        print(f"🔄 更新监控股票列表: {self.symbols}")
    
    def add_symbol(self, symbol: str):
        """添加单个股票到监控列表"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            print(f"➕ 添加股票到监控列表: {symbol}")
    
    def remove_symbol(self, symbol: str):
        """从监控列表中移除股票"""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            print(f"➖ 从监控列表移除股票: {symbol}")
    
    def get_symbols(self) -> List[str]:
        """获取当前监控的股票列表"""
        return self.symbols.copy()
    
    async def get_realtime_data(self, symbol: str) -> Optional[Dict]:
        """获取单个股票的实时数据"""
        try:
            if self.data_source:
                # 使用数据接口
                df = await self.data_source.get_latest_data(symbol, n_bars=20, timeframe='1d')
            else:
                # 直接使用yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='20d', interval='1d')
            
            if df.empty or len(df) < 2:
                return None
            
            # 获取最新数据
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # 计算技术指标
            current_price = latest['close']
            prev_price = previous['close']
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            # 计算RSI
            rsi = self._calculate_rsi(df['close'])
            
            # 计算移动平均线
            ma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
            ma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else current_price
            
            # 计算成交量比率
            avg_volume = df['volume'].rolling(10).mean().iloc[-1] if len(df) >= 10 else latest['volume']
            volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1
            
            # 判断布林带位置
            bb_position = self._get_bollinger_position(df['close'])
            
            # 判断MACD
            macd_signal = self._get_macd_signal(df['close'])
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'change': change,
                'change_pct': change_pct,
                'volume': latest['volume'],
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'bollinger_position': bb_position,
                'macd': macd_signal,
                'high': latest['high'],
                'low': latest['low'],
                'open': latest['open'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ 获取 {symbol} 实时数据失败: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI指标"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50
    
    def _get_bollinger_position(self, prices: pd.Series, period: int = 20) -> str:
        """判断布林带位置"""
        try:
            ma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = ma + (2 * std)
            lower = ma - (2 * std)
            
            current_price = prices.iloc[-1]
            upper_band = upper.iloc[-1]
            lower_band = lower.iloc[-1]
            
            if current_price > upper_band:
                return "upper_band"
            elif current_price < lower_band:
                return "lower_band"
            else:
                return "middle_band"
        except:
            return "middle_band"
    
    def _get_macd_signal(self, prices: pd.Series) -> str:
        """判断MACD信号"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            prev_macd = macd.iloc[-2]
            prev_signal = signal.iloc[-2]
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                return "bullish"
            elif current_macd < current_signal and prev_macd >= prev_signal:
                return "bearish"
            else:
                return "neutral"
        except:
            return "neutral"
    
    async def analyze_all_stocks(self):
        """分析所有监控的股票"""
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - 开始AI实时分析...")
        
        analysis_results = []
        
        for symbol in self.symbols:
            try:
                # 获取实时数据
                stock_data = await self.get_realtime_data(symbol)
                
                if stock_data is None:
                    print(f"  ⚠️ {symbol}: 无法获取数据")
                    continue
                
                # 显示基础信息
                print(f"\n📊 {symbol}: ${stock_data['current_price']:.2f} ({stock_data['change_pct']:+.2f}%)")
                print(f"  RSI: {stock_data['rsi']:.1f}, 成交量比: {stock_data['volume_ratio']:.1f}x")
                
                # 构建AI分析数据
                ai_data = {
                    'current_price': stock_data['current_price'],
                    'change_pct': stock_data['change_pct'],
                    'volume': stock_data['volume'],
                    'volume_ratio': stock_data['volume_ratio'],
                    'rsi': stock_data['rsi'],
                    'macd': stock_data['macd'],
                    'bollinger_position': stock_data['bollinger_position'],
                    'ma_20': stock_data['ma_20'],
                    'ma_50': stock_data['ma_50']
                }
                
                # 根据数据特征选择分析类型
                if abs(stock_data['change_pct']) > 3 or stock_data['volume_ratio'] > 2:
                    analysis_type = "comprehensive"
                elif abs(stock_data['change_pct']) > 1:
                    analysis_type = "detailed"
                else:
                    analysis_type = "quick"
                
                # 调用AI分析
                result = await self.ai_module.analyze_stock_signal(
                    symbol, ai_data, analysis_type
                )
                
                if result.get('success'):
                    action_suggestion = result.get('action_suggestion', {})
                    action = action_suggestion.get('action', '不明确')
                    reason = action_suggestion.get('reason', '无')
                    risk = action_suggestion.get('risk_warning', '无')
                    
                    print(f"  🤖 AI建议: {action}")
                    print(f"  📝 理由: {reason}")
                    print(f"  ⚠️ 风险: {risk}")
                    
                    # 显示操作建议
                    if action in ['止损', '止盈', '减仓']:
                        print(f"  🚨 需要关注: {action}")
                    elif action == '加仓':
                        print(f"  📈 机会信号: {action}")
                    else:
                        print(f"  👀 保持观望")
                    
                    # 保存结果
                    analysis_results.append({
                        'symbol': symbol,
                        'data': stock_data,
                        'ai_result': result
                    })
                    
                else:
                    print(f"  ❌ AI分析失败: {result.get('error', '未知错误')}")
                    
            except Exception as e:
                print(f"  ❌ 分析 {symbol} 出错: {e}")
        
        return analysis_results
    
    async def run_continuous_monitor(self, interval: int = 300):
        """运行持续监控"""
        print(f"🚀 启动AI持续监控系统...")
        print(f"监控股票: {self.symbols}")
        print(f"更新间隔: {interval}秒")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                # 分析所有股票
                results = await self.analyze_all_stocks()
                
                # 显示摘要
                await self._show_summary(results)
                
                print(f"\n{'='*60}")
                print(f"⏰ 下次更新: {datetime.now() + timedelta(seconds=interval)}")
                
                # 等待下次更新
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户停止监控")
        except Exception as e:
            print(f"❌ 监控出错: {e}")
    
    async def _show_summary(self, results: List[Dict]):
        """显示分析摘要"""
        if not results:
            return
        
        print(f"\n📋 分析摘要:")
        print(f"  成功分析: {len(results)}/{len(self.symbols)} 只股票")
        
        # 统计操作建议
        actions = {}
        for result in results:
            action = result['ai_result'].get('action_suggestion', {}).get('action', '不明确')
            actions[action] = actions.get(action, 0) + 1
        
        print(f"  操作建议分布:")
        for action, count in actions.items():
            print(f"    {action}: {count} 只")
        
        # 显示高风险股票
        high_risk = []
        for result in results:
            risk = result['ai_result'].get('action_suggestion', {}).get('risk_warning', '')
            if '风险' in risk or '止损' in risk:
                high_risk.append(result['symbol'])
        
        if high_risk:
            print(f"  🚨 高风险股票: {', '.join(high_risk)}")
        
        # 显示AI模块统计
        ai_summary = self.ai_module.get_analysis_summary()
        print(f"  AI分析统计: 总次数 {ai_summary.get('total_analyses', 0)}, 成功率 {ai_summary.get('success_rate', 0):.1%}")

async def main():
    """主函数"""
    print("🤖 AI实时监控系统")
    print("=" * 60)
    
    # 可以自定义监控的股票
    symbols = ["NVDA", "AMD", "TSLA", "AAPL", "MSFT", "GOOGL"]
    
    # 创建监控系统
    monitor = AIRealtimeMonitor(symbols)
    
    # 运行持续监控（每5分钟更新一次）
    await monitor.run_continuous_monitor(interval=300)

if __name__ == "__main__":
    asyncio.run(main()) 