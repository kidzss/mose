<<<<<<< HEAD
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 导入OpenBB
from openbb import obb

class SimpleOpenBBStrategy:
    """简单的OpenBB策略示例，展示如何集成OpenBB到您的项目中"""
    
    def __init__(self, symbols=None, lookback_days=60):
        """初始化策略"""
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        self.lookback_days = lookback_days
        self.data = {}
        self.signals = {}
        
    def fetch_data(self):
        """获取股票数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        print(f"获取{len(self.symbols)}只股票的历史数据...")
        
        for symbol in self.symbols:
            try:
                # 使用OpenBB获取历史价格数据
                stock_data = obb.equity.price.historical(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                ).to_df()
                
                if not stock_data.empty:
                    self.data[symbol] = stock_data
                    print(f"成功获取{symbol}数据: {len(stock_data)}行")
                else:
                    print(f"获取{symbol}数据失败或结果为空")
            except Exception as e:
                print(f"获取{symbol}数据时出错: {e}")
        
        return self.data
    
    def calculate_indicators(self):
        """计算技术指标"""
        for symbol, df in self.data.items():
            # 确保数据不为空
            if df.empty:
                continue
                
            # 添加技术指标
            # 1. 移动平均线 - 使用pandas自带的函数（更稳定）
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            # 2. RSI指标 - 使用pandas计算
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. MACD指标 - 使用pandas计算
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
            
            # 4. 布林带指标 - 使用pandas计算
            middle_band = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            df['BB_upper'] = middle_band + (std_dev * 2)
            df['BB_middle'] = middle_band
            df['BB_lower'] = middle_band - (std_dev * 2)
            
            # 5. ATR指标（真实波动幅度均值）- 使用pandas计算
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # 更新数据
            self.data[symbol] = df
            print(f"已为{symbol}计算技术指标")
        
        return self.data
    
    def generate_signals(self):
        """生成交易信号"""
        for symbol, df in self.data.items():
            # 确保数据不为空
            if df.empty:
                continue
            
            # 初始化信号列
            df['signal'] = 0
            
            # 均线交叉信号
            df.loc[(df['SMA20'] > df['SMA50']) & (df['SMA20'].shift(1) <= df['SMA50'].shift(1)), 'signal'] = 1  # 买入信号
            df.loc[(df['SMA20'] < df['SMA50']) & (df['SMA20'].shift(1) >= df['SMA50'].shift(1)), 'signal'] = -1  # 卖出信号
            
            # RSI超买超卖信号
            df.loc[(df['RSI'] < 30) & (df['RSI'].shift(1) >= 30), 'signal'] = 1  # 超卖买入
            df.loc[(df['RSI'] > 70) & (df['RSI'].shift(1) <= 70), 'signal'] = -1  # 超买卖出
            
            # MACD信号
            df.loc[(df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'signal'] = 1  # MACD金叉
            df.loc[(df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'signal'] = -1  # MACD死叉
            
            # 布林带信号（如果存在）
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                df.loc[(df['close'] < df['BB_lower']) & (df['close'].shift(1) >= df['BB_lower'].shift(1)), 'signal'] = 1  # 价格突破下轨买入
                df.loc[(df['close'] > df['BB_upper']) & (df['close'].shift(1) <= df['BB_upper'].shift(1)), 'signal'] = -1  # 价格突破上轨卖出
            
            # 保存信号
            self.signals[symbol] = df[df['signal'] != 0].copy()
        
        return self.signals
    
    def add_fundamental_data(self):
        """添加基本面数据（如果有API密钥）"""
        for symbol in self.symbols:
            try:
                # 尝试获取公司信息 - 使用profile而不是overview
                profile = obb.equity.profile(symbol=symbol).to_df()
                
                if not profile.empty:
                    print(f"\n{symbol}基本面数据:")
                    print(f"公司名称: {profile.get('name', ['未知'])[0] if 'name' in profile else '未知'}")
                    print(f"行业: {profile.get('industry', ['未知'])[0] if 'industry' in profile else '未知'}")
                    print(f"部门: {profile.get('sector', ['未知'])[0] if 'sector' in profile else '未知'}")
                    
                # 尝试获取财务摘要（如果API密钥可用）
                try:
                    financials = obb.equity.fundamental.overview(symbol=symbol).to_df()
                    if not financials.empty:
                        print(f"市值: {financials.get('MarketCapitalization', ['未知'])[0] if 'MarketCapitalization' in financials else '未知'}")
                        print(f"PE比率: {financials.get('PERatio', ['未知'])[0] if 'PERatio' in financials else '未知'}")
                except Exception as e:
                    pass  # 忽略错误，这个功能需要API密钥
            except Exception as e:
                print(f"获取{symbol}基本面数据时出错 (可能需要API密钥): {e}")
    
    def analyze_market_conditions(self):
        """分析市场状况"""
        try:
            # 使用OpenBB获取市场数据
            print("\n分析市场状况...")
            
            # 获取SPY数据作为市场指标
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            spy_data = obb.equity.price.historical(
                symbol="SPY",
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            ).to_df()
            
            if not spy_data.empty:
                # 计算市场技术指标 - 使用pandas
                spy_data['SMA20'] = spy_data['close'].rolling(window=20).mean()
                spy_data['SMA50'] = spy_data['close'].rolling(window=50).mean()
                spy_data['SMA200'] = spy_data['close'].rolling(window=200).mean()
                
                # 获取最新数据
                latest = spy_data.iloc[-1]
                
                # 判断市场趋势
                if latest['SMA20'] > latest['SMA50'] > latest['SMA200']:
                    market_trend = "牛市"
                elif latest['SMA20'] < latest['SMA50'] < latest['SMA200']:
                    market_trend = "熊市"
                else:
                    market_trend = "盘整"
                
                # 计算波动率
                volatility = spy_data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
                current_volatility = volatility.iloc[-1]
                
                print(f"市场趋势: {market_trend}")
                print(f"当前波动率: {current_volatility:.2%}")
                
                # 尝试获取恐惧与贪婪指数（如果有相应模块）
                try:
                    fear_greed = obb.economy.fear_and_greed_index().to_df()
                    if not fear_greed.empty:
                        latest_index = fear_greed.iloc[-1]['value']
                        latest_rating = fear_greed.iloc[-1]['rating']
                        print(f"恐惧与贪婪指数: {latest_index} ({latest_rating})")
                except Exception as e:
                    pass  # 忽略这个错误
                
                # 尝试获取经济指标
                try:
                    # 获取联邦基金利率
                    fed_rate = obb.economy.fred_series(series_id="FEDFUNDS").to_df()
                    latest_fed_rate = fed_rate.iloc[-1]['value'] if not fed_rate.empty else None
                    
                    if latest_fed_rate is not None:
                        print(f"当前联邦基金利率: {latest_fed_rate:.2f}%")
                except Exception as e:
                    pass  # 忽略这个错误，需要API密钥
                
                return {
                    "market_trend": market_trend,
                    "volatility": current_volatility
                }
            
            return None
        except Exception as e:
            print(f"分析市场状况时出错: {e}")
            return None
    
    def plot_results(self, symbol):
        """绘制分析结果"""
        if symbol not in self.data:
            print(f"没有{symbol}的数据可供绘制")
            return
        
        df = self.data[symbol]
        
        # 创建图表
        plt.figure(figsize=(12, 12))
        
        # 价格和均线
        plt.subplot(4, 1, 1)
        plt.plot(df.index, df['close'], label='收盘价')
        plt.plot(df.index, df['SMA20'], label='20日均线')
        plt.plot(df.index, df['SMA50'], label='50日均线')
        
        # 如果有布林带数据，添加到图表
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            plt.plot(df.index, df['BB_upper'], 'r--', label='布林带上轨')
            plt.plot(df.index, df['BB_middle'], 'g--', label='布林带中轨')
            plt.plot(df.index, df['BB_lower'], 'r--', label='布林带下轨')
        
        # 添加买入卖出信号
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', s=100, label='买入信号')
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', s=100, label='卖出信号')
        
        plt.title(f'{symbol}价格和交易信号')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        # RSI
        plt.subplot(4, 1, 2)
        plt.plot(df.index, df['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title('RSI指标')
        plt.ylabel('RSI值')
        plt.legend()
        plt.grid(True)
        
        # MACD
        plt.subplot(4, 1, 3)
        plt.plot(df.index, df['MACD'], label='MACD')
        plt.plot(df.index, df['MACD_signal'], label='信号线')
        plt.bar(df.index, df['MACD_histogram'], label='柱状图')
        plt.title('MACD指标')
        plt.ylabel('MACD值')
        plt.legend()
        plt.grid(True)
        
        # ATR - 如果有计算
        if 'ATR' in df.columns:
            plt.subplot(4, 1, 4)
            plt.plot(df.index, df['ATR'], label='ATR(14)')
            plt.title('ATR指标 - 真实波动幅度均值')
            plt.ylabel('ATR值')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = Path('strategy_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = output_dir / f'{symbol}_analysis_{timestamp}.png'
        plt.savefig(file_path)
        plt.close()
        
        print(f"分析图表已保存到: {file_path}")
    
    def run_strategy(self):
        """运行完整策略"""
        print("=== 运行OpenBB简单策略 ===")
        
        # 1. 获取数据
        self.fetch_data()
        
        # 2. 计算指标
        self.calculate_indicators()
        
        # 3. 生成信号
        self.generate_signals()
        
        # 4. 分析市场状况
        market_conditions = self.analyze_market_conditions()
        
        # 5. 获取基本面数据（如果有API密钥）
        self.add_fundamental_data()
        
        # 6. 输出结果
        print("\n=== 策略结果 ===")
        for symbol, signals_df in self.signals.items():
            if not signals_df.empty:
                print(f"\n{symbol}交易信号:")
                for idx, row in signals_df.iterrows():
                    signal_type = "买入" if row['signal'] == 1 else "卖出"
                    # 检查idx的类型，适当处理日期
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                    print(f"日期: {date_str}, 信号: {signal_type}, 价格: {row['close']:.2f}")
                
                # 绘制结果
                self.plot_results(symbol)
        
        print("\n策略运行完成!")

if __name__ == "__main__":
    # 运行策略
    strategy = SimpleOpenBBStrategy(symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
=======
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 导入OpenBB
from openbb import obb

class SimpleOpenBBStrategy:
    """简单的OpenBB策略示例，展示如何集成OpenBB到您的项目中"""
    
    def __init__(self, symbols=None, lookback_days=60):
        """初始化策略"""
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        self.lookback_days = lookback_days
        self.data = {}
        self.signals = {}
        
    def fetch_data(self):
        """获取股票数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        print(f"获取{len(self.symbols)}只股票的历史数据...")
        
        for symbol in self.symbols:
            try:
                # 使用OpenBB获取历史价格数据
                stock_data = obb.equity.price.historical(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                ).to_df()
                
                if not stock_data.empty:
                    self.data[symbol] = stock_data
                    print(f"成功获取{symbol}数据: {len(stock_data)}行")
                else:
                    print(f"获取{symbol}数据失败或结果为空")
            except Exception as e:
                print(f"获取{symbol}数据时出错: {e}")
        
        return self.data
    
    def calculate_indicators(self):
        """计算技术指标"""
        for symbol, df in self.data.items():
            # 确保数据不为空
            if df.empty:
                continue
                
            # 添加技术指标
            # 1. 移动平均线 - 使用pandas自带的函数（更稳定）
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            # 2. RSI指标 - 使用pandas计算
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. MACD指标 - 使用pandas计算
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
            
            # 4. 布林带指标 - 使用pandas计算
            middle_band = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            df['BB_upper'] = middle_band + (std_dev * 2)
            df['BB_middle'] = middle_band
            df['BB_lower'] = middle_band - (std_dev * 2)
            
            # 5. ATR指标（真实波动幅度均值）- 使用pandas计算
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # 更新数据
            self.data[symbol] = df
            print(f"已为{symbol}计算技术指标")
        
        return self.data
    
    def generate_signals(self):
        """生成交易信号"""
        for symbol, df in self.data.items():
            # 确保数据不为空
            if df.empty:
                continue
            
            # 初始化信号列
            df['signal'] = 0
            
            # 均线交叉信号
            df.loc[(df['SMA20'] > df['SMA50']) & (df['SMA20'].shift(1) <= df['SMA50'].shift(1)), 'signal'] = 1  # 买入信号
            df.loc[(df['SMA20'] < df['SMA50']) & (df['SMA20'].shift(1) >= df['SMA50'].shift(1)), 'signal'] = -1  # 卖出信号
            
            # RSI超买超卖信号
            df.loc[(df['RSI'] < 30) & (df['RSI'].shift(1) >= 30), 'signal'] = 1  # 超卖买入
            df.loc[(df['RSI'] > 70) & (df['RSI'].shift(1) <= 70), 'signal'] = -1  # 超买卖出
            
            # MACD信号
            df.loc[(df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'signal'] = 1  # MACD金叉
            df.loc[(df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'signal'] = -1  # MACD死叉
            
            # 布林带信号（如果存在）
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                df.loc[(df['close'] < df['BB_lower']) & (df['close'].shift(1) >= df['BB_lower'].shift(1)), 'signal'] = 1  # 价格突破下轨买入
                df.loc[(df['close'] > df['BB_upper']) & (df['close'].shift(1) <= df['BB_upper'].shift(1)), 'signal'] = -1  # 价格突破上轨卖出
            
            # 保存信号
            self.signals[symbol] = df[df['signal'] != 0].copy()
        
        return self.signals
    
    def add_fundamental_data(self):
        """添加基本面数据（如果有API密钥）"""
        for symbol in self.symbols:
            try:
                # 尝试获取公司信息 - 使用profile而不是overview
                profile = obb.equity.profile(symbol=symbol).to_df()
                
                if not profile.empty:
                    print(f"\n{symbol}基本面数据:")
                    print(f"公司名称: {profile.get('name', ['未知'])[0] if 'name' in profile else '未知'}")
                    print(f"行业: {profile.get('industry', ['未知'])[0] if 'industry' in profile else '未知'}")
                    print(f"部门: {profile.get('sector', ['未知'])[0] if 'sector' in profile else '未知'}")
                    
                # 尝试获取财务摘要（如果API密钥可用）
                try:
                    financials = obb.equity.fundamental.overview(symbol=symbol).to_df()
                    if not financials.empty:
                        print(f"市值: {financials.get('MarketCapitalization', ['未知'])[0] if 'MarketCapitalization' in financials else '未知'}")
                        print(f"PE比率: {financials.get('PERatio', ['未知'])[0] if 'PERatio' in financials else '未知'}")
                except Exception as e:
                    pass  # 忽略错误，这个功能需要API密钥
            except Exception as e:
                print(f"获取{symbol}基本面数据时出错 (可能需要API密钥): {e}")
    
    def analyze_market_conditions(self):
        """分析市场状况"""
        try:
            # 使用OpenBB获取市场数据
            print("\n分析市场状况...")
            
            # 获取SPY数据作为市场指标
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            spy_data = obb.equity.price.historical(
                symbol="SPY",
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            ).to_df()
            
            if not spy_data.empty:
                # 计算市场技术指标 - 使用pandas
                spy_data['SMA20'] = spy_data['close'].rolling(window=20).mean()
                spy_data['SMA50'] = spy_data['close'].rolling(window=50).mean()
                spy_data['SMA200'] = spy_data['close'].rolling(window=200).mean()
                
                # 获取最新数据
                latest = spy_data.iloc[-1]
                
                # 判断市场趋势
                if latest['SMA20'] > latest['SMA50'] > latest['SMA200']:
                    market_trend = "牛市"
                elif latest['SMA20'] < latest['SMA50'] < latest['SMA200']:
                    market_trend = "熊市"
                else:
                    market_trend = "盘整"
                
                # 计算波动率
                volatility = spy_data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
                current_volatility = volatility.iloc[-1]
                
                print(f"市场趋势: {market_trend}")
                print(f"当前波动率: {current_volatility:.2%}")
                
                # 尝试获取恐惧与贪婪指数（如果有相应模块）
                try:
                    fear_greed = obb.economy.fear_and_greed_index().to_df()
                    if not fear_greed.empty:
                        latest_index = fear_greed.iloc[-1]['value']
                        latest_rating = fear_greed.iloc[-1]['rating']
                        print(f"恐惧与贪婪指数: {latest_index} ({latest_rating})")
                except Exception as e:
                    pass  # 忽略这个错误
                
                # 尝试获取经济指标
                try:
                    # 获取联邦基金利率
                    fed_rate = obb.economy.fred_series(series_id="FEDFUNDS").to_df()
                    latest_fed_rate = fed_rate.iloc[-1]['value'] if not fed_rate.empty else None
                    
                    if latest_fed_rate is not None:
                        print(f"当前联邦基金利率: {latest_fed_rate:.2f}%")
                except Exception as e:
                    pass  # 忽略这个错误，需要API密钥
                
                return {
                    "market_trend": market_trend,
                    "volatility": current_volatility
                }
            
            return None
        except Exception as e:
            print(f"分析市场状况时出错: {e}")
            return None
    
    def plot_results(self, symbol):
        """绘制分析结果"""
        if symbol not in self.data:
            print(f"没有{symbol}的数据可供绘制")
            return
        
        df = self.data[symbol]
        
        # 创建图表
        plt.figure(figsize=(12, 12))
        
        # 价格和均线
        plt.subplot(4, 1, 1)
        plt.plot(df.index, df['close'], label='收盘价')
        plt.plot(df.index, df['SMA20'], label='20日均线')
        plt.plot(df.index, df['SMA50'], label='50日均线')
        
        # 如果有布林带数据，添加到图表
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            plt.plot(df.index, df['BB_upper'], 'r--', label='布林带上轨')
            plt.plot(df.index, df['BB_middle'], 'g--', label='布林带中轨')
            plt.plot(df.index, df['BB_lower'], 'r--', label='布林带下轨')
        
        # 添加买入卖出信号
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', s=100, label='买入信号')
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', s=100, label='卖出信号')
        
        plt.title(f'{symbol}价格和交易信号')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        # RSI
        plt.subplot(4, 1, 2)
        plt.plot(df.index, df['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title('RSI指标')
        plt.ylabel('RSI值')
        plt.legend()
        plt.grid(True)
        
        # MACD
        plt.subplot(4, 1, 3)
        plt.plot(df.index, df['MACD'], label='MACD')
        plt.plot(df.index, df['MACD_signal'], label='信号线')
        plt.bar(df.index, df['MACD_histogram'], label='柱状图')
        plt.title('MACD指标')
        plt.ylabel('MACD值')
        plt.legend()
        plt.grid(True)
        
        # ATR - 如果有计算
        if 'ATR' in df.columns:
            plt.subplot(4, 1, 4)
            plt.plot(df.index, df['ATR'], label='ATR(14)')
            plt.title('ATR指标 - 真实波动幅度均值')
            plt.ylabel('ATR值')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = Path('strategy_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = output_dir / f'{symbol}_analysis_{timestamp}.png'
        plt.savefig(file_path)
        plt.close()
        
        print(f"分析图表已保存到: {file_path}")
    
    def run_strategy(self):
        """运行完整策略"""
        print("=== 运行OpenBB简单策略 ===")
        
        # 1. 获取数据
        self.fetch_data()
        
        # 2. 计算指标
        self.calculate_indicators()
        
        # 3. 生成信号
        self.generate_signals()
        
        # 4. 分析市场状况
        market_conditions = self.analyze_market_conditions()
        
        # 5. 获取基本面数据（如果有API密钥）
        self.add_fundamental_data()
        
        # 6. 输出结果
        print("\n=== 策略结果 ===")
        for symbol, signals_df in self.signals.items():
            if not signals_df.empty:
                print(f"\n{symbol}交易信号:")
                for idx, row in signals_df.iterrows():
                    signal_type = "买入" if row['signal'] == 1 else "卖出"
                    # 检查idx的类型，适当处理日期
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                    print(f"日期: {date_str}, 信号: {signal_type}, 价格: {row['close']:.2f}")
                
                # 绘制结果
                self.plot_results(symbol)
        
        print("\n策略运行完成!")

if __name__ == "__main__":
    # 运行策略
    strategy = SimpleOpenBBStrategy(symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
>>>>>>> 3d7330be7ea0ecb409ac485e1c8391bc6d56a2de
    strategy.run_strategy() 