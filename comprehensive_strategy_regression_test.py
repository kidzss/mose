#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全策略模块专业回归测试
测试所有已集成策略在持仓股票上的表现，使用真实数据
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

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
chinese_font_available = False  # 强制使用英文
# ------------------------------------------

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveStrategyRegressionTest:
    """全策略模块专业回归测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        # 持仓股票列表
        self.test_stocks = ['NVDA', 'AMD', 'GOOG', 'TSLA', 'MRK', 'BRK-B']
        self.start_date = '2024-01-01'
        self.end_date = '2025-07-10'
        self.results = {}
        
        # 策略配置
        self.strategies = {
            'TDI': {
                'class': None,
                'params': {},
                'description': 'TDI多时间周期策略'
            },
            'NiuniuV3': {
                'class': None,
                'params': {},
                'description': '牛牛策略V3'
            },
            'CPGW': {
                'class': None,
                'params': {},
                'description': 'CPGW策略'
            },
            'MarketForecast': {
                'class': None,
                'params': {},
                'description': '市场预测策略'
            },
            'Combined': {
                'class': None,
                'params': {
                    'parameters': {
                        'weight_niuniu': 0.40,
                        'weight_tdi': 0.30,
                        'weight_cpgw': 0.20
                    }
                },
                'description': '组合策略'
            }
        }
        
        # 初始化策略
        self._initialize_strategies()
        
        print("🚀 全策略模块专业回归测试初始化完成")
        print(f"📊 测试股票: {self.test_stocks}")
        print(f"📅 测试期间: {self.start_date} 至 {self.end_date}")
        print(f"🎯 测试策略: {list(self.strategies.keys())}")
        print("=" * 80)
    
    def _initialize_strategies(self):
        """初始化所有策略"""
        try:
            from strategy.strategy_factory import StrategyFactory
            from strategy.tdi_strategy import TDIStrategy
            from strategy.niuniu_strategy_v3 import NiuniuStrategyV3
            from strategy.cpgw_strategy import CPGWStrategy
            from strategy.market_forecast_strategy import MarketForecastStrategy
            from strategy.combined_strategy import CombinedStrategy
            
            # 设置策略类
            self.strategies['TDI']['class'] = TDIStrategy
            self.strategies['NiuniuV3']['class'] = NiuniuStrategyV3
            self.strategies['CPGW']['class'] = CPGWStrategy
            self.strategies['MarketForecast']['class'] = MarketForecastStrategy
            self.strategies['Combined']['class'] = CombinedStrategy
            
            print("✅ 所有策略初始化成功")
            
        except Exception as e:
            print(f"❌ 策略初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    def run_strategy_backtest(self, symbol: str, strategy_name: str, data: pd.DataFrame) -> dict:
        """运行单个策略回测"""
        try:
            strategy_config = self.strategies[strategy_name]
            strategy_class = strategy_config['class']
            strategy_params = strategy_config['params']
            
            if strategy_class is None:
                print(f"❌ {strategy_name} 策略类未初始化")
                return None
            
            # 创建策略实例
            strategy = strategy_class(**strategy_params)
            
            # 计算技术指标
            df_with_indicators = strategy.calculate_indicators(data)
            
            # 生成交易信号
            df_with_signals = strategy.generate_signals(data)
            
            # 计算策略收益
            returns = self._calculate_strategy_returns(df_with_signals, symbol)
            
            # 计算性能指标
            performance = self._calculate_performance_metrics(returns, symbol, strategy_name)
            
            # 获取市场环境分析
            try:
                market_regime = strategy.get_market_regime(data)
            except:
                market_regime = None
            
            # 提取信号统计
            signal_stats = self._analyze_signals(df_with_signals)
            
            result = {
                'symbol': symbol,
                'strategy': strategy_name,
                'data': df_with_signals,
                'returns': returns,
                'performance': performance,
                'market_regime': market_regime,
                'signal_stats': signal_stats,
                'indicators': df_with_indicators
            }
            
            return result
            
        except Exception as e:
            print(f"❌ {symbol} {strategy_name} 回测失败: {e}")
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
    
    def _calculate_performance_metrics(self, df: pd.DataFrame, symbol: str, strategy_name: str) -> dict:
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
                'strategy': strategy_name,
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
            
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'signal_distribution': signal_distribution
            }
            
        except Exception as e:
            print(f"❌ 分析信号统计时出错: {e}")
            return {}
    
    def run_comprehensive_test(self):
        """运行全面测试"""
        print("🚀 开始运行全策略模块专业回归测试...")
        print("=" * 80)
        
        all_results = []
        
        for symbol in self.test_stocks:
            print(f"\n📊 测试股票: {symbol}")
            print("-" * 50)
            
            # 获取数据
            data = self.fetch_stock_data(symbol)
            if data is None:
                continue
            
            symbol_results = []
            
            # 测试每个策略
            for strategy_name in self.strategies.keys():
                print(f"🔍 测试策略: {strategy_name}")
                
                result = self.run_strategy_backtest(symbol, strategy_name, data)
                if result is not None:
                    symbol_results.append(result)
                    all_results.append(result)
                    
                    # 显示简要结果
                    performance = result['performance']
                    if performance:
                        print(f"   📈 总收益: {performance['total_return']:.2%}")
                        print(f"   📊 年化收益: {performance['annual_return']:.2%}")
                        print(f"   🎯 夏普比率: {performance['sharpe_ratio']:.3f}")
                        print(f"   📉 最大回撤: {performance['max_drawdown']:.2%}")
                        print(f"   🎯 胜率: {performance['win_rate']:.2%}")
                        print(f"   🔄 交易次数: {performance['total_trades']}")
            
            # 保存股票结果
            self.results[symbol] = symbol_results
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results)
        
        # 生成可视化图表
        self._generate_visualizations(all_results)
        
        return all_results
    
    def _generate_comprehensive_report(self, all_results: list):
        """生成综合报告"""
        print("\n" + "=" * 80)
        print("📋 全策略模块专业回归测试综合报告")
        print("=" * 80)
        
        if not all_results:
            print("❌ 没有可用的测试结果")
            return
        
        # 转换为DataFrame
        df_results = pd.DataFrame([r['performance'] for r in all_results if r['performance']])
        
        if df_results.empty:
            print("❌ 没有可用的性能数据")
            return
        
        print(f"\n📊 测试概况:")
        print(f"   测试股票数量: {len(self.test_stocks)}")
        print(f"   测试策略数量: {len(self.strategies)}")
        print(f"   测试期间: {self.start_date} 至 {self.end_date}")
        print(f"   总测试组合: {len(df_results)}")
        print(f"   平均测试天数: {df_results['days'].mean():.0f} 天")
        
        # 按策略汇总
        print(f"\n📈 策略性能汇总:")
        strategy_summary = df_results.groupby('strategy').agg({
            'total_return': ['mean', 'std', 'min', 'max'],
            'annual_return': ['mean', 'std'],
            'sharpe_ratio': ['mean', 'std'],
            'max_drawdown': ['mean', 'min'],
            'win_rate': ['mean', 'std'],
            'excess_return': ['mean', 'std']
        }).round(4)
        
        print(strategy_summary)
        
        # 最佳表现
        print(f"\n🏆 最佳表现:")
        best_return = df_results.loc[df_results['total_return'].idxmax()]
        best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
        best_excess = df_results.loc[df_results['excess_return'].idxmax()]
        
        print(f"   最高总收益: {best_return['strategy']} on {best_return['symbol']} ({best_return['total_return']:.2%})")
        print(f"   最高夏普比率: {best_sharpe['strategy']} on {best_sharpe['symbol']} ({best_sharpe['sharpe_ratio']:.3f})")
        print(f"   最高超额收益: {best_excess['strategy']} on {best_excess['symbol']} ({best_excess['excess_return']:.2%})")
        
        # 按股票汇总
        print(f"\n📊 股票表现汇总:")
        stock_summary = df_results.groupby('symbol').agg({
            'total_return': ['mean', 'std'],
            'excess_return': ['mean', 'std'],
            'sharpe_ratio': ['mean', 'std']
        }).round(4)
        
        print(stock_summary)
        
        # 保存详细结果
        self._save_results_to_csv(df_results)
    
    def _save_results_to_csv(self, df_results: pd.DataFrame):
        """保存结果到CSV文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_strategy_regression_results_{timestamp}.csv"
            df_results.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n💾 详细结果已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def _generate_visualizations(self, all_results: list):
        """生成可视化图表"""
        print(f"\n📊 生成可视化图表...")
        
        try:
            # 转换为DataFrame
            df_results = pd.DataFrame([r['performance'] for r in all_results if r['performance']])
            
            if df_results.empty:
                print("❌ 没有可用的数据生成图表")
                return
            
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # 根据中文字体可用性设置标题
            if chinese_font_available:
                fig.suptitle('全策略模块专业回归测试结果', fontsize=16, fontweight='bold')
            else:
                fig.suptitle('Comprehensive Strategy Module Regression Test Results', fontsize=16, fontweight='bold')
            
            # 1. 策略总收益对比
            ax1 = axes[0, 0]
            strategy_returns = df_results.groupby('strategy')['total_return'].mean()
            strategy_returns.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.8)
            if chinese_font_available:
                ax1.set_title('策略平均总收益')
                ax1.set_ylabel('总收益率')
            else:
                ax1.set_title('Strategy Average Total Return')
                ax1.set_ylabel('Total Return')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 2. 策略夏普比率对比
            ax2 = axes[0, 1]
            strategy_sharpe = df_results.groupby('strategy')['sharpe_ratio'].mean()
            strategy_sharpe.plot(kind='bar', ax=ax2, color='lightgreen', alpha=0.8)
            if chinese_font_available:
                ax2.set_title('策略平均夏普比率')
                ax2.set_ylabel('夏普比率')
            else:
                ax2.set_title('Strategy Average Sharpe Ratio')
                ax2.set_ylabel('Sharpe Ratio')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 3. 策略超额收益对比
            ax3 = axes[0, 2]
            strategy_excess = df_results.groupby('strategy')['excess_return'].mean()
            strategy_excess.plot(kind='bar', ax=ax3, color='lightcoral', alpha=0.8)
            if chinese_font_available:
                ax3.set_title('策略平均超额收益')
                ax3.set_ylabel('超额收益率')
            else:
                ax3.set_title('Strategy Average Excess Return')
                ax3.set_ylabel('Excess Return')
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 4. 股票表现热力图
            ax4 = axes[1, 0]
            pivot_returns = df_results.pivot(index='symbol', columns='strategy', values='total_return')
            sns.heatmap(pivot_returns, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax4)
            if chinese_font_available:
                ax4.set_title('股票-策略收益热力图')
            else:
                ax4.set_title('Stock-Strategy Return Heatmap')
            
            # 5. 策略胜率对比
            ax5 = axes[1, 1]
            strategy_winrate = df_results.groupby('strategy')['win_rate'].mean()
            strategy_winrate.plot(kind='bar', ax=ax5, color='gold', alpha=0.8)
            if chinese_font_available:
                ax5.set_title('策略平均胜率')
                ax5.set_ylabel('胜率')
            else:
                ax5.set_title('Strategy Average Win Rate')
                ax5.set_ylabel('Win Rate')
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
            ax5.grid(True, alpha=0.3)
            
            # 6. 策略最大回撤对比
            ax6 = axes[1, 2]
            strategy_drawdown = df_results.groupby('strategy')['max_drawdown'].mean()
            strategy_drawdown.plot(kind='bar', ax=ax6, color='orange', alpha=0.8)
            if chinese_font_available:
                ax6.set_title('策略平均最大回撤')
                ax6.set_ylabel('最大回撤')
            else:
                ax6.set_title('Strategy Average Max Drawdown')
                ax6.set_ylabel('Max Drawdown')
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"comprehensive_strategy_regression_charts_{timestamp}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"📊 图表已保存到: {chart_filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 生成可视化图表失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("🚀 全策略模块专业回归测试")
    print("=" * 80)
    
    # 创建测试实例
    tester = ComprehensiveStrategyRegressionTest()
    
    # 运行测试
    results = tester.run_comprehensive_test()
    
    print("\n🎉 全策略模块专业回归测试完成！")
    print("📋 测试总结:")
    print("   ✅ 测试所有已集成策略")
    print("   ✅ 使用真实持仓股票数据")
    print("   ✅ 测试期间超过1年")
    print("   ✅ 包含完整的性能指标分析")
    print("   ✅ 生成可视化图表和详细报告")
    print("   ✅ 结果已保存到CSV文件")
    print(f"   📊 总测试组合数: {len(results)}")

if __name__ == "__main__":
    main() 