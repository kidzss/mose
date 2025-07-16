#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Forecast策略回归测试
使用真实数据测试策略效果，测试周期超过6个月
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------- 中文字体自动检测与设置 -----------
from matplotlib import font_manager

def set_chinese_font():
    """自动检测并设置可用的中文字体"""
    # 尝试多种方法设置中文字体
    font_candidates = [
        'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STHeiti', 'PingFang SC', 
        'Source Han Sans SC', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', '思源黑体',
        'SimSun', 'KaiTi', 'FangSong'
    ]
    
    # 方法1: 检查系统字体
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 检测到可用中文字体: {font}，已设置为matplotlib默认字体")
            return True
    
    # 方法2: 尝试直接设置常见中文字体路径
    import platform
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常见中文字体路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simsun.ttc'
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc'
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
        ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 成功加载中文字体: {font_path}")
                return True
            except Exception as e:
                print(f"⚠️ 加载字体失败: {font_path}, 错误: {e}")
    
    # 方法3: 使用matplotlib内置的中文字体支持
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # 测试中文字体是否可用
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文', fontsize=12)
        plt.close(fig)
        print("✅ 中文字体设置成功")
        return True
    except Exception as e:
        print(f"⚠️ 中文字体测试失败: {e}")
    
    # 降级为英文
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠️ 未检测到可用中文字体，图表将以英文显示。")
    return False

# 设置中文字体
chinese_font_available = set_chinese_font()
# ------------------------------------------

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MarketForecastRegressionTest:
    """Market Forecast策略回归测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        self.test_stocks = ['NVDA', 'AMD', 'GOOG', 'TSLA', 'MRK', 'BRK-B']
        self.start_date = '2024-01-01'
        self.end_date = '2025-07-10'
        self.results = {}
        
        # 导入策略
        from strategy.market_forecast_strategy import MarketForecastStrategy
        
        self.strategy = MarketForecastStrategy()
        
        print("🚀 Market Forecast策略回归测试初始化完成")
        print(f"📊 测试股票: {self.test_stocks}")
        print(f"📅 测试期间: {self.start_date} 至 {self.end_date}")
        print("=" * 80)
    
    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """获取股票数据"""
        try:
            print(f"📈 获取 {symbol} 历史数据...")
            
            # 直接使用yfinance获取数据
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start_date, end=self.end_date)
            
            if data.empty:
                print(f"❌ {symbol} 数据获取失败")
                return None
            
            # 标准化列名
            column_mapping = {
                'Close': 'close',
                'Open': 'open', 
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data[new_col] = data[old_col]
            
            print(f"✅ {symbol} 数据获取成功，共 {len(data)} 条记录")
            return data
            
        except Exception as e:
            print(f"❌ 获取 {symbol} 数据时出错: {e}")
            return None
    
    def run_strategy_backtest(self, symbol: str, data: pd.DataFrame) -> dict:
        """运行策略回测"""
        try:
            print(f"🔍 运行 {symbol} 策略回测...")
            
            # 计算技术指标
            df_with_indicators = self.strategy.calculate_indicators(data)
            
            # 生成交易信号
            df_with_signals = self.strategy.generate_signals(data)
            
            # 计算策略收益
            returns = self._calculate_strategy_returns(df_with_signals, symbol)
            
            # 计算性能指标
            performance = self._calculate_performance_metrics(returns, symbol)
            
            # 获取市场环境分析
            market_regime = self.strategy.get_market_regime(data)
            
            # 提取信号统计
            signal_stats = self._analyze_signals(df_with_signals)
            
            result = {
                'symbol': symbol,
                'data': df_with_signals,
                'returns': returns,
                'performance': performance,
                'market_regime': market_regime,
                'signal_stats': signal_stats,
                'indicators': df_with_indicators
            }
            
            print(f"✅ {symbol} 回测完成")
            return result
            
        except Exception as e:
            print(f"❌ {symbol} 回测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_strategy_returns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """计算策略收益"""
        try:
            # 确保使用正确的列名
            close_col = 'close' if 'close' in df.columns else 'Close'
            
            # 计算价格变化
            df['price_change'] = df[close_col].pct_change()
            
            # 计算策略收益（假设在信号生成时买入/卖出）
            df['strategy_return'] = 0.0
            
            # 买入信号：下一个交易日买入
            buy_signals = df['signal'] == 1
            if buy_signals.any():
                buy_indices = df[buy_signals].index
                for idx in buy_indices:
                    # 找到下一个交易日
                    next_idx = df.index[df.index.get_loc(idx) + 1] if df.index.get_loc(idx) + 1 < len(df) else idx
                    # 计算从买入到下一个交易日的收益
                    if next_idx != idx:
                        df.loc[next_idx, 'strategy_return'] = df.loc[next_idx, 'price_change']
            
            # 卖出信号：下一个交易日卖出
            sell_signals = df['signal'] == -1
            if sell_signals.any():
                sell_indices = df[sell_signals].index
                for idx in sell_indices:
                    # 找到下一个交易日
                    next_idx = df.index[df.index.get_loc(idx) + 1] if df.index.get_loc(idx) + 1 < len(df) else idx
                    # 计算从卖出到下一个交易日的收益（负收益）
                    if next_idx != idx:
                        df.loc[next_idx, 'strategy_return'] = -df.loc[next_idx, 'price_change']
            
            # 计算累积收益
            df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
            df['buy_hold_return'] = (1 + df['price_change']).cumprod()
            
            return df
            
        except Exception as e:
            print(f"❌ 计算策略收益时出错: {e}")
            return df
    
    def _calculate_performance_metrics(self, df: pd.DataFrame, symbol: str) -> dict:
        """计算性能指标"""
        try:
            # 移除NaN值
            df_clean = df.dropna()
            
            if df_clean.empty:
                return {}
            
            # 策略收益
            strategy_returns = df_clean['strategy_return'].dropna()
            buy_hold_returns = df_clean['price_change'].dropna()
            
            if strategy_returns.empty or buy_hold_returns.empty:
                return {}
            
            # 基础指标
            total_return = df_clean['cumulative_return'].iloc[-1] - 1
            buy_hold_total_return = df_clean['buy_hold_return'].iloc[-1] - 1
            
            # 年化收益率
            days = (df_clean.index[-1] - df_clean.index[0]).days
            annual_return = (1 + total_return) ** (365 / days) - 1
            buy_hold_annual_return = (1 + buy_hold_total_return) ** (365 / days) - 1
            
            # 波动率
            volatility = strategy_returns.std() * np.sqrt(252)
            buy_hold_volatility = buy_hold_returns.std() * np.sqrt(252)
            
            # 夏普比率
            risk_free_rate = 0.04  # 假设无风险利率4%
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            buy_hold_sharpe = (buy_hold_annual_return - risk_free_rate) / buy_hold_volatility if buy_hold_volatility > 0 else 0
            
            # 最大回撤
            cumulative = df_clean['cumulative_return']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 胜率
            winning_trades = (strategy_returns > 0).sum()
            total_trades = len(strategy_returns)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 平均收益
            avg_return = strategy_returns.mean()
            avg_win = strategy_returns[strategy_returns > 0].mean() if winning_trades > 0 else 0
            avg_loss = strategy_returns[strategy_returns < 0].mean() if (total_trades - winning_trades) > 0 else 0
            
            # 收益风险比
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            return {
                'symbol': symbol,
                'total_return': total_return,
                'buy_hold_return': buy_hold_total_return,
                'excess_return': total_return - buy_hold_total_return,
                'annual_return': annual_return,
                'buy_hold_annual_return': buy_hold_annual_return,
                'volatility': volatility,
                'buy_hold_volatility': buy_hold_volatility,
                'sharpe_ratio': sharpe_ratio,
                'buy_hold_sharpe': buy_hold_sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'avg_return': avg_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'days': days
            }
            
        except Exception as e:
            print(f"❌ 计算性能指标时出错: {e}")
            return {}
    
    def _analyze_signals(self, df: pd.DataFrame) -> dict:
        """分析信号统计"""
        try:
            signals = df['signal'].dropna()
            
            buy_signals = (signals == 1).sum()
            sell_signals = (signals == -1).sum()
            hold_signals = (signals == 0).sum()
            total_signals = len(signals)
            
            # 信号分布
            signal_distribution = {
                'buy': buy_signals / total_signals if total_signals > 0 else 0,
                'sell': sell_signals / total_signals if total_signals > 0 else 0,
                'hold': hold_signals / total_signals if total_signals > 0 else 0
            }
            
            # 信号强度分析
            if 'mf_indicator' in df.columns:
                buy_indicator_avg = df[df['signal'] == 1]['mf_indicator'].mean() if buy_signals > 0 else 0
                sell_indicator_avg = df[df['signal'] == -1]['mf_indicator'].mean() if sell_signals > 0 else 0
            else:
                buy_indicator_avg = sell_indicator_avg = 0
            
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'signal_distribution': signal_distribution,
                'buy_indicator_avg': buy_indicator_avg,
                'sell_indicator_avg': sell_indicator_avg
            }
            
        except Exception as e:
            print(f"❌ 分析信号统计时出错: {e}")
            return {}
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始运行Market Forecast策略回归测试...")
        print("=" * 80)
        
        for symbol in self.test_stocks:
            print(f"\n📊 测试股票: {symbol}")
            print("-" * 50)
            
            # 获取数据
            data = self.fetch_stock_data(symbol)
            if data is None:
                continue
            
            # 运行回测
            result = self.run_strategy_backtest(symbol, data)
            if result is None:
                continue
            
            # 保存结果
            self.results[symbol] = result
            
            # 显示简要结果
            self._display_brief_results(result)
        
        # 生成综合报告
        self._generate_comprehensive_report()
        
        # 生成可视化图表
        self._generate_visualizations()
    
    def _display_brief_results(self, result: dict):
        """显示简要结果"""
        performance = result['performance']
        signal_stats = result['signal_stats']
        
        if not performance:
            print("❌ 无法计算性能指标")
            return
        
        print(f"📈 策略总收益: {performance['total_return']:.2%}")
        print(f"📈 买入持有收益: {performance['buy_hold_return']:.2%}")
        print(f"📈 超额收益: {performance['excess_return']:.2%}")
        print(f"📊 年化收益率: {performance['annual_return']:.2%}")
        print(f"📊 夏普比率: {performance['sharpe_ratio']:.3f}")
        print(f"📉 最大回撤: {performance['max_drawdown']:.2%}")
        print(f"🎯 胜率: {performance['win_rate']:.2%}")
        print(f"🔄 总交易次数: {performance['total_trades']}")
        print(f"📊 市场环境: {result['market_regime'].value}")
    
    def _generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n" + "=" * 80)
        print("📋 MARKET FORECAST策略回归测试综合报告")
        print("=" * 80)
        
        if not self.results:
            print("❌ 没有可用的测试结果")
            return
        
        # 汇总所有股票的性能
        all_performance = []
        for symbol, result in self.results.items():
            if result['performance']:
                all_performance.append(result['performance'])
        
        if not all_performance:
            print("❌ 没有可用的性能数据")
            return
        
        # 计算平均性能
        df_performance = pd.DataFrame(all_performance)
        
        print(f"\n📊 测试概况:")
        print(f"   测试股票数量: {len(self.results)}")
        print(f"   测试期间: {self.start_date} 至 {self.end_date}")
        print(f"   平均测试天数: {df_performance['days'].mean():.0f} 天")
        
        print(f"\n📈 策略性能汇总:")
        print(f"   平均总收益: {df_performance['total_return'].mean():.2%}")
        print(f"   平均买入持有收益: {df_performance['buy_hold_return'].mean():.2%}")
        print(f"   平均超额收益: {df_performance['excess_return'].mean():.2%}")
        print(f"   平均年化收益率: {df_performance['annual_return'].mean():.2%}")
        print(f"   平均夏普比率: {df_performance['sharpe_ratio'].mean():.3f}")
        print(f"   平均最大回撤: {df_performance['max_drawdown'].mean():.2%}")
        print(f"   平均胜率: {df_performance['win_rate'].mean():.2%}")
        print(f"   平均交易次数: {df_performance['total_trades'].mean():.0f}")
        
        print(f"\n🏆 最佳表现股票:")
        best_return = df_performance.loc[df_performance['total_return'].idxmax()]
        best_sharpe = df_performance.loc[df_performance['sharpe_ratio'].idxmax()]
        best_winrate = df_performance.loc[df_performance['win_rate'].idxmax()]
        
        print(f"   最高收益: {best_return['symbol']} ({best_return['total_return']:.2%})")
        print(f"   最高夏普比率: {best_sharpe['symbol']} ({best_sharpe['sharpe_ratio']:.3f})")
        print(f"   最高胜率: {best_winrate['symbol']} ({best_winrate['win_rate']:.2%})")
        
        # 保存详细结果到CSV
        self._save_results_to_csv(df_performance)
    
    def _save_results_to_csv(self, df_performance: pd.DataFrame):
        """保存结果到CSV文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_forecast_regression_results_{timestamp}.csv"
            df_performance.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n💾 详细结果已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        print(f"\n📊 生成可视化图表...")
        
        try:
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 根据中文字体可用性设置标题
            if chinese_font_available:
                fig.suptitle('Market Forecast策略回归测试结果', fontsize=16, fontweight='bold')
            else:
                fig.suptitle('Market Forecast Strategy Regression Test Results', fontsize=16, fontweight='bold')
            
            # 1. 收益对比图
            ax1 = axes[0, 0]
            symbols = list(self.results.keys())
            strategy_returns = []
            buy_hold_returns = []
            
            for symbol in symbols:
                if self.results[symbol]['performance']:
                    strategy_returns.append(self.results[symbol]['performance']['total_return'])
                    buy_hold_returns.append(self.results[symbol]['performance']['buy_hold_return'])
                else:
                    strategy_returns.append(0)
                    buy_hold_returns.append(0)
            
            x = np.arange(len(symbols))
            width = 0.35
            
            if chinese_font_available:
                ax1.bar(x - width/2, strategy_returns, width, label='Market Forecast策略', alpha=0.8)
                ax1.bar(x + width/2, buy_hold_returns, width, label='买入持有', alpha=0.8)
            else:
                ax1.bar(x - width/2, strategy_returns, width, label='Market Forecast Strategy', alpha=0.8)
                ax1.bar(x + width/2, buy_hold_returns, width, label='Buy & Hold', alpha=0.8)
            
            if chinese_font_available:
                ax1.set_xlabel('股票代码')
                ax1.set_ylabel('总收益率')
                ax1.set_title('策略收益 vs 买入持有收益')
            else:
                ax1.set_xlabel('Stock Symbol')
                ax1.set_ylabel('Total Return')
                ax1.set_title('Strategy Return vs Buy & Hold Return')
            ax1.set_xticks(x)
            ax1.set_xticklabels(symbols, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 夏普比率对比
            ax2 = axes[0, 1]
            sharpe_ratios = []
            for symbol in symbols:
                if self.results[symbol]['performance']:
                    sharpe_ratios.append(self.results[symbol]['performance']['sharpe_ratio'])
                else:
                    sharpe_ratios.append(0)
            
            bars = ax2.bar(symbols, sharpe_ratios, alpha=0.8, color='skyblue')
            if chinese_font_available:
                ax2.set_xlabel('股票代码')
                ax2.set_ylabel('夏普比率')
                ax2.set_title('策略夏普比率')
            else:
                ax2.set_xlabel('Stock Symbol')
                ax2.set_ylabel('Sharpe Ratio')
                ax2.set_title('Strategy Sharpe Ratio')
            ax2.set_xticklabels(symbols, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, sharpe_ratios):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 3. 胜率分布
            ax3 = axes[1, 0]
            win_rates = []
            for symbol in symbols:
                if self.results[symbol]['performance']:
                    win_rates.append(self.results[symbol]['performance']['win_rate'])
                else:
                    win_rates.append(0)
            
            bars = ax3.bar(symbols, win_rates, alpha=0.8, color='lightgreen')
            if chinese_font_available:
                ax3.set_xlabel('股票代码')
                ax3.set_ylabel('胜率')
                ax3.set_title('策略胜率')
            else:
                ax3.set_xlabel('Stock Symbol')
                ax3.set_ylabel('Win Rate')
                ax3.set_title('Strategy Win Rate')
            ax3.set_xticklabels(symbols, rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, win_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2%}', ha='center', va='bottom')
            
            # 4. 最大回撤对比
            ax4 = axes[1, 1]
            max_drawdowns = []
            for symbol in symbols:
                if self.results[symbol]['performance']:
                    max_drawdowns.append(self.results[symbol]['performance']['max_drawdown'])
                else:
                    max_drawdowns.append(0)
            
            bars = ax4.bar(symbols, max_drawdowns, alpha=0.8, color='lightcoral')
            if chinese_font_available:
                ax4.set_xlabel('股票代码')
                ax4.set_ylabel('最大回撤')
                ax4.set_title('策略最大回撤')
            else:
                ax4.set_xlabel('Stock Symbol')
                ax4.set_ylabel('Max Drawdown')
                ax4.set_title('Strategy Max Drawdown')
            ax4.set_xticklabels(symbols, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, max_drawdowns):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.01,
                        f'{value:.2%}', ha='center', va='top')
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"market_forecast_regression_charts_{timestamp}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"📊 图表已保存到: {chart_filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 生成可视化图表失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("🚀 Market Forecast策略回归测试")
    print("=" * 80)
    
    # 创建测试实例
    tester = MarketForecastRegressionTest()
    
    # 运行测试
    tester.run_all_tests()
    
    print("\n🎉 回归测试完成！")
    print("📋 测试总结:")
    print("   ✅ 使用真实股票数据进行测试")
    print("   ✅ 测试期间超过6个月")
    print("   ✅ 包含完整的性能指标分析")
    print("   ✅ 生成可视化图表和详细报告")
    print("   ✅ 结果已保存到CSV文件")

if __name__ == "__main__":
    main() 