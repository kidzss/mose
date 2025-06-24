#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
import platform
import base64

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.market_environment_classifier import MarketEnvironmentClassifier
from strategy.dynamic_strategy_selector import DynamicStrategySelector
from strategy.signal_quality_evaluator import SignalQualityEvaluator
from monitor.alert_system import AlertSystem as AdvancedAlertSystem
from data.data_interface import DataInterface
from data.data_updater import MarketDataUpdater
from config.trading_config import default_config

# 添加宏观分析模块
try:
    from analysis.portfolio_macro_integration import PortfolioMacroIntegration
    MACRO_ANALYSIS_AVAILABLE = True
except ImportError:
    MACRO_ANALYSIS_AVAILABLE = False
    logger.warning("宏观分析模块不可用，将跳过宏观分析部分")

# 添加右侧交易系统模块
try:
    from right_side_trading_system import RightSideTradingSystem, generate_right_side_trading_alerts, format_right_side_trading_report
    RIGHT_SIDE_TRADING_AVAILABLE = True
except ImportError:
    RIGHT_SIDE_TRADING_AVAILABLE = False
    logger.warning("右侧交易系统模块不可用，将跳过右侧交易分析部分")

# 添加财务分析模块
try:
    from monitor.financial_analyzer import FinancialAnalyzer
    FINANCIAL_ANALYSIS_AVAILABLE = True
except ImportError:
    FINANCIAL_ANALYSIS_AVAILABLE = False
    logger.warning("财务分析模块不可用，将跳过财务分析部分")

# 添加流动性分析模块
try:
    from analysis.liquidity_analyzer import LiquidityAnalyzer
    LIQUIDITY_ANALYSIS_AVAILABLE = True
except ImportError:
    LIQUIDITY_ANALYSIS_AVAILABLE = False
    logger.warning("流动性分析模块不可用，将跳过流动性分析部分")

# 添加通胀-行业分析模块
try:
    from analysis.inflation_sector_analyzer import InflationSectorAnalyzer
    INFLATION_SECTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    INFLATION_SECTOR_ANALYSIS_AVAILABLE = False
    logger.warning("通胀-行业分析模块不可用，将跳过通胀行业分析部分")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartDailyReport")

class SmartDailyReportGenerator:
    """智能每日股票分析报告生成器 - 使用真实数据"""
    
    def __init__(self, watchlist=None, auto_update_data=True, portfolio=None, watch_targets=None):
        """
        初始化智能日报生成器
        
        Args:
            watchlist: 股票观察列表，默认为用户持仓股票+观察股票
            auto_update_data: 是否自动更新市场数据
            portfolio: 用户持仓信息
            watch_targets: 观察目标股票（准备买入的股票）
        """
        self.auto_update_data = auto_update_data
        self.data_source_type = None
        
        # 从统一配置文件加载持仓信息和观察列表 (更新日期: 2025-06-19)
        try:
            from utils.portfolio_config_loader import get_portfolio_config
            config_loader = get_portfolio_config()
            
            # 获取持仓股票列表
            portfolio_symbols = config_loader.get_portfolio_symbols()
            # 获取观察列表股票（排除港股和VIX指数）
            watchlist_symbols = [symbol for symbol in config_loader.get_watchlist_symbols() 
                               if not symbol.startswith('^') and not symbol.endswith('.HK')]
            
            # 合并持仓和观察列表作为监控列表
            if watchlist is None:
                self.watchlist = list(set(portfolio_symbols + watchlist_symbols))
                # 排除港股小米，避免数据获取问题
                self.watchlist = [symbol for symbol in self.watchlist if not symbol.endswith('.HK')]
                logger.info(f"✅ 从统一配置文件加载监控列表: {len(self.watchlist)}只股票")
                logger.info(f"   持仓股票: {len(portfolio_symbols)}只")
                logger.info(f"   观察列表: {len(watchlist_symbols)}只")
            else:
                self.watchlist = watchlist
            
            # 加载持仓信息
            if portfolio is None:
                self.portfolio = config_loader.to_smart_report_format()
                logger.info("✅ 从统一配置文件成功加载持仓信息")
            else:
                self.portfolio = portfolio
                
        except Exception as e:
            logger.warning(f"加载统一配置失败，使用默认配置: {e}")
            # 保留原有默认配置作为后备
            self.watchlist = watchlist or ['AMD', 'GOOGL', 'PFE', 'NVDA', 'TSLA', 'MSFT', 'PHM', 'CF', 'EOG']
            self.portfolio = portfolio or {
                'AMD': {'cost': 126.214, 'shares': 48, 'weight': 21.93, 'investment': 4788.89},
                'GOOGL': {'cost': 170.54, 'shares': 34, 'weight': 21.44, 'investment': 4715.83},
                'PFE': {'cost': 25.899, 'shares': 80, 'weight': 6.90, 'investment': 1526.65},
                'NVDA': {'cost': 138.843, 'shares': 40, 'weight': 20.91, 'investment': 4582.24},
                'TSLA': {'cost': 179.841, 'shares': 4, 'weight': 4.65, 'investment': 1038.22}
            }
        
        # 投资组合总价值计算 - 从统一配置文件获取
        try:
            if 'config_loader' in locals():
                portfolio_summary = config_loader.get_portfolio_summary()
                self.total_portfolio_value = portfolio_summary.get('total_value', 27884.87)
                self.total_stock_investment = portfolio_summary.get('stock_allocation', {}).get('total_amount', 21903.42)
                self.portfolio_allocation = portfolio_summary.get('stock_allocation', {}).get('percentage', 78.55)
                self.cash_allocation = portfolio_summary.get('cash_allocation', {}).get('percentage', 9.75)
                self.money_fund_allocation = portfolio_summary.get('money_fund_allocation', {}).get('percentage', 11.70)
                self.money_fund_value = portfolio_summary.get('money_fund_allocation', {}).get('amount', 3262.53)
            else:
                # 使用默认值 (美股部分，排除港股小米)
                self.total_portfolio_value = 27722.97
                self.total_stock_investment = 21121.29
                self.portfolio_allocation = 76.18
                self.cash_allocation = 2.20
                self.money_fund_allocation = 21.59
                self.money_fund_value = 5983.65
        except Exception as e:
            logger.warning(f"获取投资组合价值配置失败，使用默认值: {e}")
            self.total_portfolio_value = 27722.97
            self.total_stock_investment = 21121.29
            self.portfolio_allocation = 76.18
            self.cash_allocation = 2.20
            self.money_fund_allocation = 21.59
            self.money_fund_value = 5983.65
        
        # 观察目标股票（准备买入的股票）- 从统一配置文件加载
        if watch_targets is None:
            try:
                # 从统一配置文件加载观察列表详情
                watchlist_details = config_loader.get_watchlist()
                self.watch_targets = {}
                
                for symbol, details in watchlist_details.items():
                    # 跳过港股和VIX指数
                    if symbol.startswith('^') or symbol.endswith('.HK'):
                        continue
                        
                    self.watch_targets[symbol] = {
                        'previous_buy': details.get('previous_transactions', {}).get('last_buy'),
                        'previous_sell': details.get('previous_transactions', {}).get('last_sell'),
                        'previous_gain': details.get('previous_transactions', {}).get('profit_percentage'),
                        'target_buy_below': details.get('target_buy_price', details.get('target_level')),
                        'reason': details.get('reason', '无描述'),
                        'category': details.get('category', '观察股票')
                    }
                
                logger.info(f"✅ 从统一配置文件加载观察目标: {len(self.watch_targets)}只股票")
                
            except Exception as e:
                logger.warning(f"加载观察目标失败: {e}")
                # 使用默认观察目标
                self.watch_targets = {
                    'MSFT': {
                        'previous_buy': 370.95,
                        'previous_sell': 453.97,
                        'previous_gain': 22.4,
                        'target_buy_below': 420.0,
                        'reason': '准备再次买入，关注买入时机',
                        'category': '原有观察股'
                    }
                }
        else:
            self.watch_targets = watch_targets
        
        # 初始化数据接口 - 支持回退机制
        self._init_data_sources()
        
        # 初始化核心组件
        self.market_classifier = MarketEnvironmentClassifier()
        self.strategy_selector = DynamicStrategySelector()
        self.signal_evaluator = SignalQualityEvaluator()
        self.alert_system = AdvancedAlertSystem()
        
        # 初始化宏观分析器
        self.macro_integration = None
        if MACRO_ANALYSIS_AVAILABLE:
            try:
                self.macro_integration = PortfolioMacroIntegration()
                logger.info("宏观分析模块初始化成功")
            except Exception as e:
                logger.error(f"宏观分析模块初始化失败: {e}")
                self.macro_integration = None
        
        # 初始化流动性分析器
        self.liquidity_analyzer = None
        if LIQUIDITY_ANALYSIS_AVAILABLE:
            try:
                self.liquidity_analyzer = LiquidityAnalyzer()
                logger.info("流动性分析模块初始化成功")
            except Exception as e:
                logger.error(f"流动性分析模块初始化失败: {e}")
                self.liquidity_analyzer = None
        
        # 初始化通胀-行业分析器
        self.inflation_sector_analyzer = None
        if INFLATION_SECTOR_ANALYSIS_AVAILABLE:
            try:
                self.inflation_sector_analyzer = InflationSectorAnalyzer()
                logger.info("通胀-行业分析模块初始化成功")
            except Exception as e:
                logger.error(f"通胀-行业分析模块初始化失败: {e}")
                self.inflation_sector_analyzer = None
        
        # 初始化财务分析器
        self.financial_analyzer = None
        if FINANCIAL_ANALYSIS_AVAILABLE:
            try:
                self.financial_analyzer = FinancialAnalyzer()
                logger.info("财务分析模块初始化成功")
            except Exception as e:
                logger.error(f"财务分析模块初始化失败: {e}")
                self.financial_analyzer = None
        
        # 初始化增强分析器（新功能）
        self.enhanced_analyzer = None
        try:
            from monitor.enhanced_stock_analyzer import EnhancedStockAnalyzer
            self.enhanced_analyzer = EnhancedStockAnalyzer()
            if self.enhanced_analyzer.is_available():
                logger.info("增强分析器初始化成功")
            else:
                logger.warning("增强分析器无可用功能模块")
                self.enhanced_analyzer = None
        except Exception as e:
            logger.warning(f"增强分析器初始化失败: {e}")
            self.enhanced_analyzer = None
        
        # 初始化右侧交易系统（防抄底系统）
        self.right_side_trading_system = None
        if RIGHT_SIDE_TRADING_AVAILABLE:
            try:
                self.right_side_trading_system = RightSideTradingSystem()
                logger.info("右侧交易系统初始化成功")
            except Exception as e:
                logger.error(f"右侧交易系统初始化失败: {e}")
                self.right_side_trading_system = None
        
        # 设置中文字体支持
        self._setup_chinese_font()
        
        logger.info(f"智能日报生成器初始化完成，关注 {len(self.watchlist)} 只股票，数据源: {self.data_source_type}")
    
    def _init_data_sources(self):
        """初始化数据源，支持回退机制"""
        try:
            # 尝试连接MySQL数据库
            logger.info("尝试连接MySQL数据库...")
            self.data_interface = DataInterface(default_source='mysql')
            self.data_updater = MarketDataUpdater(
                db_config={
                    'host': default_config.database.host,
                    'port': default_config.database.port,
                    'user': default_config.database.user,
                    'password': default_config.database.password,
                    'database': default_config.database.database
                }
            )
            self.data_source_type = "MySQL数据库"
            logger.info("✅ MySQL数据库连接成功")
            
        except Exception as e:
            logger.warning(f"MySQL数据库连接失败: {e}")
            logger.info("🔄 切换到Yahoo Finance数据源...")
            
            try:
                # 回退到Yahoo Finance - 使用简单的初始化方式
                from data.data_interface import YahooFinanceDataSource
                yahoo_source = YahooFinanceDataSource()
                
                # 创建一个简单的数据接口包装器
                self.data_interface = yahoo_source
                self.data_updater = None  # Yahoo Finance不需要数据更新器
                self.auto_update_data = False  # 禁用自动更新
                self.data_source_type = "Yahoo Finance"
                logger.info("✅ Yahoo Finance数据源连接成功")
                
            except Exception as e2:
                logger.error(f"Yahoo Finance连接失败: {e2}")
                logger.info("🔄 使用模拟数据模式...")
                self._init_mock_data_interface()
                self.data_updater = None
                self.auto_update_data = False
                self.data_source_type = "模拟数据"
    
    def _init_mock_data_interface(self):
        """初始化模拟数据接口"""
        class MockDataInterface:
            def get_historical_data(self, symbol, start_date, end_date, timeframe='daily'):
                """返回模拟的历史数据"""
                import numpy as np
                
                # 生成日期范围
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                dates = dates[dates.weekday < 5]  # 只保留工作日
                
                if len(dates) == 0:
                    return pd.DataFrame()
                
                # 生成模拟价格数据
                np.random.seed(hash(symbol) % 1000)  # 为每个股票使用固定种子
                base_price = {"AMD": 126.50, "GOOGL": 176.80, "NVDA": 144.60, 
                             "PFE": 28.50, "TSLA": 250.00, "EOG": 125.00,
                             "MSFT": 430.00, "ADBE": 395.00, "PHM": 101.50, "CF": 99.90}.get(symbol, 100.0)
                
                prices = []
                current_price = base_price
                
                for i in range(len(dates)):
                    # 随机波动 -2% 到 +2%
                    change = np.random.normal(0, 0.02)
                    current_price *= (1 + change)
                    prices.append(current_price)
                
                prices = np.array(prices)
                
                # 创建OHLCV数据
                data = pd.DataFrame({
                    'date': dates,
                    'open': prices * (1 + np.random.normal(0, 0.005, len(prices))),
                    'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
                    'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
                    'close': prices,
                    'volume': np.random.randint(1000000, 10000000, len(prices)),
                    'adj_close': prices * (1 + np.random.normal(0, 0.001, len(prices)))
                })
                
                data.set_index('date', inplace=True)
                return data
        
        self.data_interface = MockDataInterface()
    
    def _setup_chinese_font(self):
        """设置中文字体支持"""
        try:
            system = platform.system()
            if system == "Windows":
                font_list = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi']
            elif system == "Darwin":  # macOS
                font_list = ['PingFang SC', 'Heiti SC', 'STHeiti']
            else:  # Linux
                font_list = ['WenQuanYi Micro Hei', 'DejaVu Sans']
            
            for font_name in font_list:
                try:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    logger.info(f"成功设置字体: {font_name}")
                    return
                except:
                    continue
                    
            logger.warning("无法设置中文字体，将使用默认字体")
        except Exception as e:
            logger.warning(f"设置中文字体失败: {e}")
    
    def _update_market_data(self, symbols: List[str]) -> bool:
        """更新市场数据，确保获取最新数据"""
        if not self.auto_update_data:
            logger.info("自动更新数据已禁用，跳过数据更新")
            return True
            
        try:
            logger.info("开始更新市场数据...")
            
            # 更新指定股票的数据
            update_result = self.data_updater.update_stock_data(
                symbols=symbols, 
                force_update=False  # 只更新需要更新的数据
            )
            
            if update_result['success']:
                logger.info(f"数据更新成功，更新了 {update_result['updated_count']} 只股票")
                return True
            else:
                logger.warning(f"数据更新部分失败：{update_result.get('errors', [])}")
                return True  # 即使部分失败也继续生成报告
                
        except Exception as e:
            logger.error(f"数据更新失败: {e}")
            logger.info("将使用现有数据生成报告")
            return True  # 即使更新失败也尝试生成报告
    
    def _get_stock_data(self, symbol: str, days: int = 400) -> Optional[pd.DataFrame]:
        """获取股票真实数据"""
        try:
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"获取 {symbol} 数据，时间范围: {start_date.date()} 到 {end_date.date()}")
            
            # 从数据库获取历史数据
            data = self.data_interface.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='daily'
            )
            
            if data is None or data.empty:
                logger.warning(f"{symbol} 没有可用数据")
                return None
            
            # 检查数据的新旧程度
            if isinstance(data.index, pd.DatetimeIndex):
                latest_date = data.index[-1]
            else:
                # 如果索引不是DatetimeIndex，尝试获取最大日期
                latest_date = pd.to_datetime(data.index).max()
            
            # 安全地获取日期 - 处理不同的日期类型
            if hasattr(latest_date, 'date'):
                latest_date_obj = latest_date.date()
            else:
                latest_date_obj = latest_date
            
            days_old = (datetime.now().date() - latest_date_obj).days
            
            if days_old > 7:
                logger.warning(f"{symbol} 数据较旧（{days_old}天前），可能需要更新")
            else:
                logger.info(f"{symbol} 数据较新（{days_old}天前）")
            
            # 添加技术指标
            return self._add_technical_indicators(data)
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        try:
            # 确保数据按日期排序
            if 'date' in data.columns:
                data = data.sort_values('date')
                data.set_index('date', inplace=True)
            else:
                data = data.sort_index()
            
            # 移动平均线
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['sma_200'] = data['close'].rolling(200).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # 布林带
            data['bb_middle'] = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            
            # 只删除关键技术指标列的NaN，保留其他数据
            # 不删除adj_close列的None值，因为这不影响技术分析
            essential_columns = ['close', 'sma_20', 'rsi', 'macd']
            return data.dropna(subset=[col for col in essential_columns if col in data.columns])
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return data
    
    def _check_data_quality(self, symbol: str, data: pd.DataFrame) -> Dict[str, any]:
        """检查数据质量"""
        # 安全地获取日期 - 处理不同的日期类型
        def safe_get_date(dt_obj):
            if hasattr(dt_obj, 'date'):
                return dt_obj.date()
            else:
                return dt_obj
        
        first_date = safe_get_date(data.index[0])
        last_date = safe_get_date(data.index[-1])
        
        quality_info = {
            'symbol': symbol,
            'total_records': len(data),
            'date_range': f"{first_date} 到 {last_date}",
            'latest_date': last_date,
            'days_old': (datetime.now().date() - last_date).days,
            'missing_data_pct': data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100,
            'has_sufficient_data': len(data) >= 60
        }
        
        # 数据质量评分
        score = 100
        if quality_info['days_old'] > 1:
            score -= min(quality_info['days_old'] * 5, 30)  # 每天扣5分，最多扣30分
        if quality_info['missing_data_pct'] > 5:
            score -= quality_info['missing_data_pct']  # 缺失数据扣分
        if not quality_info['has_sufficient_data']:
            score -= 20  # 数据不足扣20分
            
        quality_info['quality_score'] = max(score, 0)
        
        return quality_info
    
    def _analyze_stock(self, symbol: str) -> Dict:
        """分析单只股票"""
        logger.info(f"开始分析 {symbol}")
        
        data = self._get_stock_data(symbol)
        if data is None or len(data) < 60:
            logger.warning(f"{symbol} 数据不足或获取失败")
            return None
        
        # 检查数据质量
        quality_info = self._check_data_quality(symbol, data)
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
        price_change = ((current_price / prev_price) - 1) * 100 if prev_price != 0 else 0
        
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'price_change': price_change,
            'volume': data['volume'].iloc[-1],
            'rsi': data['rsi'].iloc[-1] if 'rsi' in data.columns else 50,
            'data_quality': quality_info
        }
        
        # 添加持仓分析
        if symbol in self.portfolio:
            portfolio_info = self.portfolio[symbol]
            cost_price = portfolio_info['cost']
            shares = portfolio_info['shares']
            position_weight = portfolio_info['weight']
            investment_amount = portfolio_info['investment']
            
            # 计算当前市值和盈亏
            current_value = current_price * shares
            # 正确的盈亏计算：当前市值 - 成本（成本价 * 股数）
            cost_basis = cost_price * shares
            pnl_amount = current_value - cost_basis
            pnl_percent = (pnl_amount / cost_basis) * 100 if cost_basis > 0 else 0
            
            result['portfolio'] = {
                'cost_price': cost_price,
                'shares': shares,
                'weight': position_weight,
                'investment_amount': investment_amount,
                'current_value': current_value,
                'pnl_amount': pnl_amount,
                'pnl_percent': pnl_percent,
                'is_profit': pnl_amount > 0
            }
        
        # 添加买入时机分析
        if symbol in self.watch_targets:
            target_info = self.watch_targets[symbol]
            target_price = target_info.get('target_buy_below', current_price)
            
            # 买入时机评估
            buy_signal_strength = 0
            buy_reasons = []
            
            # 价格分析
            if current_price <= target_price:
                buy_signal_strength += 3
                buy_reasons.append(f"价格${current_price:.2f}低于目标买入价${target_price:.2f}")
            
            # 技术指标分析
            if 'rsi' in data.columns:
                rsi_value = data['rsi'].iloc[-1]
                if rsi_value < 40:  # RSI超卖
                    buy_signal_strength += 2
                    buy_reasons.append(f"RSI({rsi_value:.1f})显示超卖状态")
                elif rsi_value < 50:
                    buy_signal_strength += 1
                    buy_reasons.append(f"RSI({rsi_value:.1f})处于中性偏低位置")
            
            # 移动平均线分析
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                sma_20 = data['sma_20'].iloc[-1]
                sma_50 = data['sma_50'].iloc[-1]
                
                if current_price < sma_20 < sma_50:  # 价格低于均线，可能是买入机会
                    buy_signal_strength += 1
                    buy_reasons.append("价格低于20日和50日均线，可能存在买入机会")
                elif sma_20 > sma_50 and current_price > sma_20:  # 上升趋势
                    buy_signal_strength += 1
                    buy_reasons.append("均线呈多头排列，趋势向上")
            
            # 成交量分析
            if len(data) > 20:
                avg_volume = data['volume'].iloc[-20:].mean()
                current_volume = data['volume'].iloc[-1]
                if current_volume > avg_volume * 1.5:  # 成交量放大
                    buy_signal_strength += 1
                    buy_reasons.append("成交量明显放大，市场关注度提升")
            
            # 买入时机评级
            if buy_signal_strength >= 5:
                buy_timing = "强烈建议买入"
                timing_color = "excellent"
            elif buy_signal_strength >= 3:
                buy_timing = "建议买入"
                timing_color = "good"
            elif buy_signal_strength >= 1:
                buy_timing = "谨慎观察"
                timing_color = "neutral"
            else:
                buy_timing = "暂不建议买入"
                timing_color = "poor"
            
            result['buy_timing'] = {
                'previous_buy': target_info['previous_buy'],
                'previous_sell': target_info['previous_sell'],
                'previous_gain': target_info['previous_gain'],
                'target_price': target_price,
                'current_price': current_price,
                'signal_strength': buy_signal_strength,
                'timing_rating': buy_timing,
                'timing_color': timing_color,
                'reasons': buy_reasons[:4],  # 最多显示4个原因
                'reason': target_info['reason']
            }
        
        try:
            # 市场环境分析
            env_result = self.market_classifier.classify_environment(data)
            result['environment'] = env_result['environment'].value
            result['confidence'] = env_result.get('confidence', 0)
            result['reasons'] = env_result.get('reasons', [])[:3]  # 只取前3个原因
            
            # 策略建议
            strategy_result = self.strategy_selector.get_best_strategy(data)
            result['strategy'] = strategy_result['primary_strategy']
            result['market_env'] = strategy_result['environment'].value
            
            # 信号评估
            signal_data = {
                'direction': 1,
                'entry_price': current_price,
                'stop_loss': current_price * 0.95,
                'target_price': current_price * 1.10,
                'indicator_signals': {
                    'macd': 1 if data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] else -1,
                    'rsi': 1 if 30 < data['rsi'].iloc[-1] < 70 else 0,
                    'sma_crossover': 1 if current_price > data['sma_20'].iloc[-1] else -1
                }
            }
            
            signal_eval = self.signal_evaluator.evaluate_signal(
                signal_data, data, env_result['environment']
            )
            result['signal_quality'] = signal_eval['quality_score']
            result['signal_strength'] = signal_eval['strength'].value
            
            # 财务分析
            if self.financial_analyzer:
                try:
                    financial_analysis = self.financial_analyzer.analyze_stock(symbol)
                    if financial_analysis:
                        result['financial_analysis'] = financial_analysis
                        logger.info(f"{symbol} 财务分析完成 - 综合评分: {financial_analysis['total_score']:.2f}, 等级: {financial_analysis['overall_rating']}")
                    else:
                        logger.warning(f"{symbol} 财务数据不可用")
                except Exception as e:
                    logger.error(f"{symbol} 财务分析失败: {e}")
            
            # 流动性分析（核心功能1：增强流动性评估）
            if self.liquidity_analyzer:
                try:
                    liquidity_metrics = self.liquidity_analyzer.analyze_stock_liquidity(symbol)
                    result['liquidity_analysis'] = {
                        'liquidity_score': liquidity_metrics.liquidity_score,
                        'risk_level': liquidity_metrics.risk_level,
                        'bid_ask_spread_pct': liquidity_metrics.bid_ask_spread_pct,
                        'market_cap_tier': liquidity_metrics.market_cap_tier,
                        'exit_difficulty': liquidity_metrics.exit_difficulty,
                        'risk_warning': liquidity_metrics.risk_warning,
                        'investment_suggestion': liquidity_metrics.investment_suggestion,
                        'spread_rating': liquidity_metrics.spread_rating,
                        'volume_consistency': liquidity_metrics.volume_consistency,
                        'market_depth_score': liquidity_metrics.market_depth_score,
                        'liquidity_reasons': liquidity_metrics.liquidity_reasons,
                        'avg_daily_volume': liquidity_metrics.avg_daily_volume
                    }
                    logger.info(f"{symbol} 流动性分析完成 - 评分: {liquidity_metrics.liquidity_score:.1f}, 风险: {liquidity_metrics.liquidity_risk_level}")
                except Exception as e:
                    logger.error(f"{symbol} 流动性分析失败: {e}")
                    result['liquidity_analysis'] = {
                        'liquidity_score': 0,
                        'risk_level': 'critical',
                        'warning_message': f"流动性分析失败: {str(e)}"
                    }
            
            # 增强分析（新功能集成）
            if self.enhanced_analyzer:
                try:
                    enhanced_analysis = self.enhanced_analyzer.analyze_stock_comprehensive(symbol, current_price)
                    if enhanced_analysis and not enhanced_analysis.get('error'):
                        result['enhanced_analysis'] = enhanced_analysis
                        
                        # 记录增强分析的关键信息
                        if 'overall_score' in enhanced_analysis:
                            logger.info(f"{symbol} 增强分析完成 - 总体评分: {enhanced_analysis['overall_score']:.3f}, 评级: {enhanced_analysis.get('overall_rating', 'N/A')}")
                        
                        # 将增强分析的建议合并到主要结果中
                        enhanced_recommendations = enhanced_analysis.get('recommendations', [])
                        if enhanced_recommendations:
                            if 'enhanced_recommendations' not in result:
                                result['enhanced_recommendations'] = []
                            result['enhanced_recommendations'].extend(enhanced_recommendations)
                        
                        # 将警告信息合并
                        enhanced_warnings = enhanced_analysis.get('warnings', [])
                        if enhanced_warnings:
                            if 'enhanced_warnings' not in result:
                                result['enhanced_warnings'] = []
                            result['enhanced_warnings'].extend(enhanced_warnings)
                            
                    else:
                        logger.warning(f"{symbol} 增强分析不可用或出错")
                except Exception as e:
                    logger.error(f"{symbol} 增强分析失败: {e}")
            
            # 右侧交易分析（防抄底系统）
            if self.right_side_trading_system:
                try:
                    right_side_analysis = self.right_side_trading_system.analyze_trend_confirmation(symbol)
                    if right_side_analysis and 'error' not in right_side_analysis:
                        result['right_side_trading'] = right_side_analysis
                        
                        # 记录右侧交易分析的关键信息
                        trend_status = right_side_analysis['trend_status']
                        entry_signals = right_side_analysis['entry_signals']
                        risk_warnings = right_side_analysis['risk_warnings']
                        
                        logger.info(f"{symbol} 右侧交易分析完成 - 趋势: {trend_status['direction']} ({trend_status['strength']['level']}), 确认: {trend_status['confirmed']}")
                        
                        # 将右侧交易建议合并到主要结果中
                        right_side_recommendations = []
                        
                        # 买入信号
                        if entry_signals['buy_signals']:
                            for signal in entry_signals['buy_signals']:
                                right_side_recommendations.append(f"✅ 右侧买入: {signal}")
                        
                        # 卖出信号
                        if entry_signals['sell_signals']:
                            for signal in entry_signals['sell_signals']:
                                right_side_recommendations.append(f"🔴 右侧卖出: {signal}")
                        
                        # 等待信号
                        if entry_signals['wait_signals']:
                            for signal in entry_signals['wait_signals']:
                                right_side_recommendations.append(f"⏳ 右侧等待: {signal}")
                        
                        if right_side_recommendations:
                            if 'right_side_recommendations' not in result:
                                result['right_side_recommendations'] = []
                            result['right_side_recommendations'].extend(right_side_recommendations)
                        
                        # 将风险警告合并
                        if risk_warnings:
                            if 'right_side_warnings' not in result:
                                result['right_side_warnings'] = []
                            result['right_side_warnings'].extend(risk_warnings)
                            
                    else:
                        logger.warning(f"{symbol} 右侧交易分析不可用或出错: {right_side_analysis.get('error', '未知错误')}")
                except Exception as e:
                    logger.error(f"{symbol} 右侧交易分析失败: {e}")

            # 创建图表
            chart_path = self._create_chart(symbol, data, env_result)
            if chart_path:
                result['chart_base64'] = self._image_to_base64(chart_path)
            
            # 警报系统分析
            try:
                alerts = self.alert_system.check_alerts(symbol, data)
                result['alerts'] = alerts
            except Exception as e:
                logger.warning(f"{symbol} 警报检查失败: {e}")
                result['alerts'] = []
            
            return result
            
        except Exception as e:
            logger.error(f"分析 {symbol} 时出错: {e}")
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change': price_change,
                'volume': data['volume'].iloc[-1],
                'error': str(e),
                'data_quality': quality_info
            }
    
    def _image_to_base64(self, image_path: str) -> str:
        """将图片文件转换为base64编码"""
        try:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    return img_data  # 只返回base64编码，不包含前缀
            return ""
        except Exception as e:
            logger.error(f"转换图片到base64失败: {e}")
            return ""
    
    def _create_chart(self, symbol: str, data: pd.DataFrame, env_result: Dict) -> str:
        """创建股票分析图表"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 价格走势图
            recent_data = data.iloc[-60:]  # 最近60天
            ax1.plot(recent_data.index, recent_data['close'], label='收盘价', linewidth=2, color='#1f77b4')
            
            if 'sma_20' in recent_data.columns:
                ax1.plot(recent_data.index, recent_data['sma_20'], label='20日均线', alpha=0.7, color='orange')
            if 'sma_50' in recent_data.columns:
                ax1.plot(recent_data.index, recent_data['sma_50'], label='50日均线', alpha=0.7, color='green')
            
            # 添加持仓成本线
            if symbol in self.portfolio:
                cost_price = self.portfolio[symbol]['cost']
                ax1.axhline(y=cost_price, color='red', linestyle=':', alpha=0.8, 
                           label=f'持仓成本: ${cost_price:.3f}', linewidth=2)
            
            # 添加目标买入价线
            if symbol in self.watch_targets:
                target_price = self.watch_targets[symbol].get('target_buy_below', 0)
                if target_price > 0:
                    ax1.axhline(y=target_price, color='purple', linestyle='--', alpha=0.8, 
                               label=f'目标买入价: ${target_price:.2f}', linewidth=2)
            
            env_name = env_result['environment'].value
            confidence = env_result.get('confidence', 0)
            
            if not np.isnan(confidence):
                title = f"{symbol} - 市场环境: {env_name} (置信度: {confidence:.2f})"
            else:
                title = f"{symbol} - 市场环境: {env_name}"
            
            # 添加数据日期信息
            latest_date = recent_data.index[-1].strftime('%Y-%m-%d')
            title += f"\n最新数据: {latest_date}"
            
            ax1.set_title(title, fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel('价格 ($)')
            
            # RSI图
            if 'rsi' in recent_data.columns:
                ax2.plot(recent_data.index, recent_data['rsi'], label='RSI', color='purple', linewidth=2)
                ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='超买线(70)')
                ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖线(30)')
                ax2.fill_between(recent_data.index, 30, 70, alpha=0.1, color='gray')
                
                ax2.set_title('相对强弱指数 (RSI)', fontsize=12)
                ax2.set_ylabel('RSI')
                ax2.set_ylim(0, 100)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 确保temp_pic目录存在
            os.makedirs('temp_pic', exist_ok=True)
            
            # 保存图表到temp_pic目录
            chart_filename = f"temp_pic/{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"已生成 {symbol} 图表: {chart_filename}")
            return chart_filename
            
        except Exception as e:
            logger.error(f"生成 {symbol} 图表失败: {e}")
            return ""
    
    def _generate_macro_analysis_html(self, macro_analysis: Optional[Dict]) -> str:
        """生成宏观分析HTML部分"""
        if not macro_analysis:
            return ""
            
        try:
            executive_summary = macro_analysis.get('executive_summary', {})
            macro_score = executive_summary.get('macro_score', 0)
            macro_recommendation = executive_summary.get('macro_recommendation', '暂无建议')
            portfolio_risk_level = executive_summary.get('portfolio_risk_level', 'medium')
            key_concerns = executive_summary.get('key_concerns', [])
            
            # 详细分析数据
            detailed_analysis = macro_analysis.get('detailed_analysis', {})
            sector_impact = detailed_analysis.get('sector_impact', {})
            portfolio_impact = detailed_analysis.get('portfolio_impact', {})
            
            # 行动计划
            action_plan = macro_analysis.get('action_plan', {})
            priority_1 = action_plan.get('priority_1', [])
            priority_2 = action_plan.get('priority_2', [])
            monitoring = action_plan.get('monitoring', [])
            
            # 确定风险等级的颜色
            risk_colors = {
                'low': '#28a745',
                'medium': '#ffc107', 
                'high': '#dc3545'
            }
            risk_color = risk_colors.get(portfolio_risk_level, '#6c757d')
            
            # 生成行业影响表格
            sector_rows = ""
            for sector, score in sector_impact.items():
                score_color = '#28a745' if score > 0.6 else '#ffc107' if score > 0.4 else '#dc3545'
                sector_rows += f"""
                <tr>
                    <td>{sector}</td>
                    <td style="color: {score_color}; font-weight: bold;">{score:.2f}</td>
                    <td>{'有利' if score > 0.6 else '中性' if score > 0.4 else '不利'}</td>
                </tr>
                """
            
            # 生成立即行动项目列表
            action_items = ""
            for i, action in enumerate(priority_1[:5], 1):  # 只显示前5项
                action_items += f"<li>{action}</li>"
                
            # 生成监控要点列表
            monitoring_items = ""
            for item in monitoring[:3]:  # 只显示前3项
                monitoring_items += f"<li>{item}</li>"
            
            # 构建基础宏观分析HTML
            html = f"""
                <div class="data-status" style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);">
                    <h3 style="margin-top: 0;">🌍 宏观环境分析</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px;">
                        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                            <div style="font-size: 0.9em; margin-bottom: 5px;">宏观得分</div>
                            <div style="font-size: 1.8em; font-weight: bold;">{macro_score:.2f}/1.0</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                            <div style="font-size: 0.9em; margin-bottom: 5px;">风险等级</div>
                            <div style="font-size: 1.3em; font-weight: bold; color: {risk_color};">{portfolio_risk_level.upper()}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                            <div style="font-size: 0.9em; margin-bottom: 5px;">重点关注</div>
                            <div style="font-size: 1.1em; font-weight: bold;">{len(key_concerns)} 只股票</div>
                        </div>
                    </div>
                    <p style="font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">💡 {macro_recommendation}</p>
                    {f'<p style="margin-top: 10px;"><strong>🚨 重点关注:</strong> {", ".join(key_concerns)}</p>' if key_concerns else ''}
                </div>
                
                <div class="summary" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3 style="margin-top: 0;">📊 行业影响分析</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                        <thead>
                            <tr style="background: rgba(255,255,255,0.1);">
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.3);">行业</th>
                                <th style="padding: 10px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.3);">影响得分</th>
                                <th style="padding: 10px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.3);">环境评估</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sector_rows}
                        </tbody>
                    </table>
                </div>
                
                <div class="summary" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333;">
                    <h3 style="margin-top: 0; color: #d35400;">⚡ 行动建议</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h4 style="color: #c0392b; margin-bottom: 10px;">🎯 立即行动 ({len(priority_1)} 项)</h4>
                            <ul style="margin: 0; padding-left: 20px;">
                                {action_items}
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: #8e44ad; margin-bottom: 10px;">👁️ 重点监控</h4>
                            <ul style="margin: 0; padding-left: 20px;">
                                {monitoring_items}
                            </ul>
                        </div>
                    </div>
                </div>
            """
            
            # 添加通胀-行业分析展示（核心功能2：完善宏观-行业影响分析）
            logger.info(f"🔍 检查通胀-行业分析: {'inflation_sector_analysis' in macro_analysis}")
            if 'inflation_sector_analysis' in macro_analysis:
                logger.info("✅ 进入通胀-行业分析HTML生成代码块")
                inflation_analysis = macro_analysis['inflation_sector_analysis']
                inflation_env = inflation_analysis.get('inflation_environment', {})
                sector_analysis = inflation_analysis.get('sector_analysis', {})
                recommendations = inflation_analysis.get('investment_recommendations', [])
                warnings = inflation_analysis.get('risk_warnings', [])
                
                # 通胀环境状态
                regime = inflation_env.get('regime', '未知')
                confidence = inflation_env.get('confidence', 0) * 100
                trend = inflation_env.get('trend', '稳定')
                risk_level = inflation_env.get('risk_level', 'medium')
                
                # 确定通胀环境的颜色
                inflation_colors = {
                    'low': '#28a745',
                    'medium': '#ffc107',
                    'medium_high': '#fd7e14',
                    'high': '#dc3545'
                }
                inflation_color = inflation_colors.get(risk_level, '#6c757d')
                
                # 生成行业通胀敏感性表格
                inflation_sector_rows = ""
                sorted_sectors = sorted(sector_analysis.items(), 
                                      key=lambda x: x[1]['overall_score'], reverse=True)
                
                for sector, data in sorted_sectors:
                    score = data['overall_score']
                    beta = data['inflation_beta']
                    pricing_power = data['pricing_power']
                    
                    score_color = '#28a745' if score > 0.6 else '#ffc107' if score > 0.4 else '#dc3545'
                    beta_display = f"+{beta:.2f}" if beta > 0 else f"{beta:.2f}"
                    beta_color = '#28a745' if beta > 0 else '#dc3545'
                    
                    sector_display = sector.replace('_', ' ').title()
                    
                    inflation_sector_rows += f"""
                    <tr>
                        <td>{sector_display}</td>
                        <td style="color: {score_color}; font-weight: bold;">{score:.2f}</td>
                        <td style="color: {beta_color}; font-weight: bold;">{beta_display}</td>
                        <td>{pricing_power:.2f}</td>
                        <td style="font-size: 0.9em;">{data['investment_suggestion'][:30]}...</td>
                    </tr>
                    """
                
                # 生成投资建议列表
                recommendation_items = ""
                for rec in recommendations[:6]:  # 显示前6项建议
                    recommendation_items += f"<li>{rec}</li>"
                
                # 生成风险警告列表
                warning_items = ""
                for warning in warnings[:4]:  # 显示前4项警告
                    warning_items += f"<li>{warning}</li>"
                
                # 添加通胀-行业分析HTML（使用更安全的字符串拼接）
                try:
                    inflation_html = f"""
<div class="summary" style="background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%); color: #333; margin-top: 20px;">
    <h3 style="margin-top: 0; color: #c0392b;">🔥 通胀-行业影响分析 <span style="font-size: 0.8em; color: #666;">(Enhanced!)</span></h3>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px;">
        <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.9em; margin-bottom: 5px;">通胀环境</div>
            <div style="font-size: 1.2em; font-weight: bold; color: {inflation_color};">{regime}</div>
        </div>
        <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.9em; margin-bottom: 5px;">分析信心度</div>
            <div style="font-size: 1.2em; font-weight: bold;">{confidence:.0f}%</div>
        </div>
        <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.9em; margin-bottom: 5px;">通胀趋势</div>
            <div style="font-size: 1.2em; font-weight: bold;">{trend}</div>
        </div>
        <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.9em; margin-bottom: 5px;">风险等级</div>
            <div style="font-size: 1.2em; font-weight: bold; color: {inflation_color};">{risk_level.upper()}</div>
        </div>
    </div>
    
    <h4 style="color: #8e44ad; margin-bottom: 15px;">📊 行业通胀敏感性分析</h4>
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.2); border-radius: 8px;">
            <thead>
                <tr style="background: rgba(255,255,255,0.1);">
                    <th style="padding: 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.3);">行业</th>
                    <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.3);">影响评分</th>
                    <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.3);">通胀敏感度</th>
                    <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.3);">定价能力</th>
                    <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.3);">投资建议</th>
                </tr>
            </thead>
            <tbody>
                {inflation_sector_rows}
            </tbody>
        </table>
    </div>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
    <div class="summary" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333;">
        <h4 style="margin-top: 0; color: #27ae60;">🎯 通胀环境投资建议</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.95em;">
            {recommendation_items}
        </ul>
    </div>
    
    <div class="summary" style="background: linear-gradient(135deg, #ffb347 0%, #ffcc33 100%); color: #333;">
        <h4 style="margin-top: 0; color: #d35400;">⚠️ 通胀风险警告</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.95em;">
            {warning_items}
        </ul>
    </div>
</div>
"""
                    
                    # 确保HTML拼接成功
                    html += inflation_html
                    logger.info(f"✅ 通胀-行业分析HTML已添加到报告中 ({len(inflation_html)} 字符)")
                    
                except Exception as inflation_html_error:
                    logger.error(f"❌ 通胀HTML生成失败: {inflation_html_error}")
                    # 添加简化版本确保有内容显示
                    html += f"""
<div class="summary" style="background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%); color: #333; margin-top: 20px;">
    <h3 style="margin-top: 0; color: #c0392b;">🔥 通胀-行业影响分析</h3>
    <p>通胀环境: {regime} (信心度: {confidence:.0f}%)</p>
    <p>风险等级: {risk_level}</p>
    <p>分析数据获取成功，详细显示遇到技术问题</p>
</div>
"""
            
            return html
            
        except Exception as e:
            logger.error(f"生成宏观分析HTML失败: {e}")
            return f'<div class="data-status" style="background: #f8d7da; color: #721c24;"><p>宏观分析数据加载失败: {e}</p></div>'

    def _generate_html_report(self, analysis_results: List[Dict], macro_analysis: Optional[Dict] = None) -> str:
        """生成HTML格式报告"""
        # 过滤有效结果
        valid_results = [r for r in analysis_results if r is not None]
        
        # 计算数据质量统计
        avg_quality = np.mean([r['data_quality']['quality_score'] for r in valid_results])
        oldest_data_days = max([r['data_quality']['days_old'] for r in valid_results]) if valid_results else 0
        
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>每日股票分析报告</title>
            <style>
                body {{ 
                    font-family: 'Microsoft YaHei', Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                    line-height: 1.6;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .header {{ 
                    text-align: center; 
                    border-bottom: 2px solid #007bff; 
                    padding-bottom: 20px; 
                    margin-bottom: 30px;
                }}
                .header h1 {{ 
                    color: #007bff; 
                    margin: 0;
                    font-size: 2.5em;
                }}
                .data-status {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    margin-bottom: 20px;
                }}
                .summary {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    margin-bottom: 30px;
                }}
                .stock-card {{ 
                    background-color: #fff; 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    margin-bottom: 25px; 
                    padding: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .stock-header {{ 
                    display: flex; 
                    justify-content: space-between; 
                    align-items: center; 
                    margin-bottom: 15px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                .stock-symbol {{ 
                    font-size: 1.8em; 
                    font-weight: bold; 
                    color: #333;
                }}
                .price-info {{ 
                    text-align: right;
                }}
                .current-price {{ 
                    font-size: 1.5em; 
                    font-weight: bold;
                }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                .metrics-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 15px; 
                    margin-bottom: 15px;
                }}
                .metric-item {{ 
                    background-color: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 5px; 
                    text-align: center;
                }}
                .metric-label {{ 
                    font-weight: bold; 
                    color: #666; 
                    font-size: 0.9em;
                }}
                .metric-value {{ 
                    font-size: 1.1em; 
                    margin-top: 5px;
                }}
                .analysis-section {{ 
                    margin-top: 15px;
                }}
                .reasons-list {{ 
                    background-color: #e7f3ff; 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin-top: 10px;
                }}
                .reasons-list ul {{ 
                    margin: 0; 
                    padding-left: 20px;
                }}
                .data-quality {{ 
                    background-color: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin-top: 10px; 
                    font-size: 0.9em;
                }}
                .quality-good {{ color: #28a745; }}
                .quality-warning {{ color: #ffc107; }}
                .quality-bad {{ color: #dc3545; }}
                .footer {{ 
                    text-align: center; 
                    margin-top: 40px; 
                    padding-top: 20px; 
                    border-top: 1px solid #ddd; 
                    color: #666; 
                    font-size: 0.9em;
                }}
                .chart-note {{ 
                    background-color: #fff3cd; 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin-top: 10px; 
                    font-size: 0.9em;
                }}
                .portfolio-info {{
                    background-color: #e8f5e8;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 15px;
                    border-left: 4px solid #28a745;
                }}
                .portfolio-profit {{
                    background-color: #d4edda;
                    border-left-color: #28a745;
                }}
                .portfolio-loss {{
                    background-color: #f8d7da;
                    border-left-color: #dc3545;
                }}
                .chart-image {{
                    width: 100%;
                    max-width: 800px;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    margin-top: 15px;
                }}
                .buy-timing-info {{
                    background-color: #f0f8ff;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 15px;
                    border-left: 4px solid #007bff;
                }}
                .timing-excellent {{
                    background-color: #d4edda;
                    border-left-color: #28a745;
                }}
                .timing-good {{
                    background-color: #d1ecf1;
                    border-left-color: #17a2b8;
                }}
                .timing-neutral {{
                    background-color: #fff3cd;
                    border-left-color: #ffc107;
                }}
                .timing-poor {{
                    background-color: #f8d7da;
                    border-left-color: #dc3545;
                }}
                .timing-rating {{
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .timing-excellent .timing-rating {{ color: #28a745; }}
                .timing-good .timing-rating {{ color: #17a2b8; }}
                .timing-neutral .timing-rating {{ color: #856404; }}
                .timing-poor .timing-rating {{ color: #721c24; }}
                .previous-trade {{
                    background-color: #e9ecef;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 智能股票分析日报</h1>
                    <p style="margin: 10px 0; font-size: 1.1em;">
                        生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')} | 
                        分析股票: {len(valid_results)} 只
                    </p>
                </div>
                
                <div class="data-status">
                    <h3 style="margin-top: 0;">📈 数据状态总览</h3>
                    <p>• 平均数据质量评分: {avg_quality:.1f} 分</p>
                    <p>• 最旧数据: {oldest_data_days} 天前</p>
                    <p>• 数据源: {self.data_source_type}</p>
                </div>
                
                {self._generate_macro_analysis_html(macro_analysis)}
                
                <div class="summary">
                    <h2 style="margin-top: 0;">🎯 今日市场概览</h2>
                    <p>基于高级市场环境分类器和动态策略选择器的智能分析</p>
                    <p>• 涵盖技术指标分析、市场环境识别、信号质量评估</p>
                    <p>• 提供个性化投资策略建议和风险提示</p>
                </div>
        """
        
        # 为每只股票生成分析卡片
        for result in valid_results:
            symbol = result['symbol']
            price = result['current_price']
            change = result['price_change']
            change_class = 'positive' if change >= 0 else 'negative'
            change_symbol = '+' if change >= 0 else ''
            
            # 数据质量状态
            quality_score = result['data_quality']['quality_score']
            quality_class = 'quality-good' if quality_score >= 80 else ('quality-warning' if quality_score >= 60 else 'quality-bad')
            
            html += f"""
                <div class="stock-card">
                    <div class="stock-header">
                        <div class="stock-symbol">{symbol}</div>
                        <div class="price-info">
                            <div class="current-price">${price:.2f}</div>
                            <div class="{change_class}">{change_symbol}{change:.2f}%</div>
                        </div>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-label">市场环境</div>
                            <div class="metric-value">{result['environment']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">推荐策略</div>
                            <div class="metric-value">{result['strategy']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">信号质量</div>
                            <div class="metric-value">{result.get('signal_quality', 0):.2f}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">信号强度</div>
                            <div class="metric-value">{result.get('signal_strength', '未知')}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">RSI指标</div>
                            <div class="metric-value">{result['rsi']:.1f}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">成交量</div>
                            <div class="metric-value">{result['volume']:,}</div>
                        </div>
                    </div>"""
            
            # 添加持仓信息
            if 'portfolio' in result:
                portfolio = result['portfolio']
                pnl_class = 'portfolio-profit' if portfolio['is_profit'] else 'portfolio-loss'
                pnl_symbol = '+' if portfolio['is_profit'] else ''
                profit_emoji = '📈' if portfolio['is_profit'] else '📉'
                
                html += f"""
                    <div class="portfolio-info {pnl_class}">
                        <h4>{profit_emoji} 持仓分析</h4>
                        <p><strong>持仓成本:</strong> ${portfolio['cost_price']:.3f}</p>
                        <p><strong>持仓股数:</strong> {portfolio['shares']:,}</p>
                        <p><strong>投资金额:</strong> ${portfolio['investment_amount']:.2f}</p>
                        <p><strong>当前市值:</strong> ${portfolio['current_value']:.2f}</p>
                        <p><strong>盈亏金额:</strong> <span class="{'positive' if portfolio['is_profit'] else 'negative'}">{pnl_symbol}${portfolio['pnl_amount']:.3f}</span></p>
                        <p><strong>盈亏比例:</strong> <span class="{'positive' if portfolio['is_profit'] else 'negative'}">{pnl_symbol}{portfolio['pnl_percent']:.2f}%</span></p>
                    </div>"""
            
            html += f"""
                    
                    <div class="data-quality">
                        <strong>📊 数据质量:</strong> 
                        <span class="{quality_class}">{quality_score:.0f}分</span> | 
                        数据范围: {result['data_quality']['date_range']} | 
                        最新数据: {result['data_quality']['days_old']}天前
                    </div>
                    
                    <div class="analysis-section">
                        <h4>📋 分析要点:</h4>
                        <div class="reasons-list">
                            <ul>
            """
            
            for reason in result['reasons']:
                html += f"<li>{reason}</li>"
            
            html += f"""
                            </ul>
                        </div>
                    </div>
                    
                    <div class="chart-note">
                        📈 技术分析图表
                    </div>"""
            
            # 添加买入时机分析
            if 'buy_timing' in result:
                buy_timing = result['buy_timing']
                timing_class = f"timing-{buy_timing['timing_color']}"
                
                html += f"""
                    <div class="buy-timing-info {timing_class}">
                        <h4>💰 买入时机分析</h4>
                        <div class="timing-rating">{buy_timing['timing_rating']}</div>
                        
                        <div class="previous-trade">
                            <strong>📊 历史交易记录:</strong><br>
                            {f"买入价: ${buy_timing['previous_buy']:.2f} | 卖出价: ${buy_timing['previous_sell']:.2f} | 收益: +{buy_timing['previous_gain']:.1f}%" if buy_timing['previous_buy'] is not None else "暂无历史交易记录"}
                        </div>
                        
                        <p><strong>🎯 目标买入价:</strong> ${buy_timing['target_price']:.2f}</p>
                        <p><strong>📈 当前价格:</strong> ${buy_timing['current_price']:.2f}</p>
                        <p><strong>🔍 信号强度:</strong> {buy_timing['signal_strength']}/7</p>
                        
                        <div style="margin-top: 10px;">
                            <strong>📋 买入分析要点:</strong>
                            <ul style="margin: 5px 0; padding-left: 20px;">"""
                
                for reason in buy_timing['reasons']:
                    html += f"<li>{reason}</li>"
                
                html += f"""
                            </ul>
                        </div>
                        
                        <p style="margin-top: 10px; font-style: italic;">
                            <strong>📝 备注:</strong> {buy_timing['reason']}
                        </p>
                    </div>"""
            
            # 添加财务分析
            if 'financial_analysis' in result:
                financial = result['financial_analysis']
                
                # 根据综合评分确定背景色
                total_score = financial['total_score']
                if total_score >= 0.8:
                    financial_class = "portfolio-profit"  # 绿色
                elif total_score >= 0.6:
                    financial_class = "timing-good"  # 蓝色
                elif total_score >= 0.4:
                    financial_class = "timing-neutral"  # 黄色
                else:
                    financial_class = "portfolio-loss"  # 红色
                
                basic_info = financial['basic_info']
                advice = financial['investment_advice']
                dimensions = financial['dimensions']
                
                html += f"""
                    <div class="buy-timing-info {financial_class}">
                        <h4>💼 财务基本面分析</h4>
                        <div class="timing-rating">综合评级: {financial['overall_rating']} ({total_score:.2f}/1.0)</div>
                        
                        <div class="previous-trade">
                            <strong>🏢 公司信息:</strong><br>
                            {basic_info['company_name']} | {basic_info['sector']} - {basic_info['industry']}<br>
                            市值: ${basic_info['market_cap']/1000000:.0f}M | 当前价格: ${basic_info['current_price']:.2f}
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 15px 0;">
                            <div>
                                <strong>📈 估值指标 ({dimensions['valuation']['summary']}):</strong>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">"""
                
                for key, detail in dimensions['valuation']['details'].items():
                    html += f"<li>{detail['comment']}</li>"
                
                html += f"""
                                </ul>
                            </div>
                            <div>
                                <strong>💰 盈利能力 ({dimensions['profitability']['summary']}):</strong>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">"""
                
                for key, detail in dimensions['profitability']['details'].items():
                    html += f"<li>{detail['comment']}</li>"
                
                html += f"""
                                </ul>
                            </div>
                            <div>
                                <strong>🚀 成长性 ({dimensions['growth']['summary']}):</strong>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">"""
                
                for key, detail in dimensions['growth']['details'].items():
                    html += f"<li>{detail['comment']}</li>"
                
                html += f"""
                                </ul>
                            </div>
                            <div>
                                <strong>🏦 财务健康 ({dimensions['financial_health']['summary']}):</strong>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">"""
                
                for key, detail in dimensions['financial_health']['details'].items():
                    html += f"<li>{detail['comment']}</li>"
                
                html += f"""
                                </ul>
                            </div>
                        </div>
                        
                        <div style="margin-top: 15px;">
                            <strong>📊 分析师观点 ({dimensions['analyst_sentiment']['summary']}):</strong>
                            <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">"""
                
                for key, detail in dimensions['analyst_sentiment']['details'].items():
                    html += f"<li>{detail['comment']}</li>"
                
                html += f"""
                            </ul>
                        </div>
                        
                        <div style="margin-top: 15px; padding: 10px; background-color: rgba(255,255,255,0.7); border-radius: 5px;">
                            <strong>🎯 投资建议: {advice['recommendation']} (信心度: {advice['confidence']}%)</strong>"""
                
                if advice['key_strengths']:
                    html += f"""
                            <div style="margin-top: 8px;">
                                <strong>✅ 主要优势:</strong>
                                <ul style="margin: 3px 0; padding-left: 20px;">"""
                    for strength in advice['key_strengths']:
                        html += f"<li>{strength}</li>"
                    html += "</ul></div>"
                
                if advice['key_concerns']:
                    html += f"""
                            <div style="margin-top: 8px;">
                                <strong>⚠️ 主要担忧:</strong>
                                <ul style="margin: 3px 0; padding-left: 20px;">"""
                    for concern in advice['key_concerns']:
                        html += f"<li>{concern}</li>"
                    html += "</ul></div>"
                
                if advice['action_items']:
                    html += f"""
                            <div style="margin-top: 8px;">
                                <strong>📋 行动建议:</strong>
                                <ul style="margin: 3px 0; padding-left: 20px;">"""
                    for action in advice['action_items']:
                        html += f"<li>{action}</li>"
                    html += "</ul></div>"
                
                html += """
                        </div>
                    </div>"""
            
            # 添加流动性分析（核心功能1：增强流动性评估）
            if 'liquidity_analysis' in result:
                liquidity = result['liquidity_analysis']
                
                # 根据流动性风险等级确定背景色
                risk_level = liquidity['risk_level']
                if risk_level == 'low':
                    liquidity_class = "portfolio-profit"  # 绿色
                elif risk_level == 'medium':
                    liquidity_class = "timing-neutral"  # 黄色
                elif risk_level == 'high':
                    liquidity_class = "timing-poor"  # 红色
                else:  # critical
                    liquidity_class = "portfolio-loss"  # 深红色
                
                html += f"""
                    <div class="buy-timing-info {liquidity_class}">
                        <h4>💧 流动性风险评估 <span style="font-size: 0.8em; color: #666;">(New!)</span></h4>
                        <div class="timing-rating">流动性评分: {liquidity['liquidity_score']:.1f}/10 | 风险等级: {liquidity['risk_level'].upper()}</div>
                        
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 15px 0;">
                            <div>
                                <strong>💱 价差分析:</strong>
                                <div style="margin: 5px 0;">
                                    <span style="font-size: 1.0em;">买卖价差: {liquidity['bid_ask_spread_pct']:.3f}%</span>
                                    <div style="margin-top: 3px; font-size: 0.9em; color: #666;">
                                        价差等级: {liquidity['spread_rating']}
                                    </div>
                                </div>
                            </div>
                            <div>
                                <strong>📊 成交量稳定性:</strong>
                                <div style="margin: 5px 0;">
                                    <span style="font-size: 1.0em;">一致性: {liquidity['volume_consistency']:.3f}</span>
                                    <div style="margin-top: 3px; font-size: 0.9em; color: #666;">
                                        平均日成交量: {liquidity['avg_daily_volume']:,.0f}
                                    </div>
                                </div>
                            </div>
                            <div>
                                <strong>🏢 市值等级:</strong>
                                <div style="margin: 5px 0;">
                                    <span style="font-size: 1.0em;">{liquidity['market_cap_tier'].upper()}</span>
                                    <div style="margin-top: 3px; font-size: 0.9em; color: #666;">
                                        市场深度: {liquidity['market_depth_score']:.3f}
                                    </div>
                                </div>
                            </div>
                            <div>
                                <strong>⚠️ 风险提示:</strong>
                                <div style="margin: 5px 0;">
                                    <span style="font-size: 1.0em;">{liquidity['risk_warning']}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 15px; padding: 10px; background-color: rgba(255,255,255,0.7); border-radius: 5px;">
                            <strong>📋 流动性分析要点:</strong>
                            <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">"""
                
                for reason in liquidity['liquidity_reasons']:
                    html += f"<li>{reason}</li>"
                
                html += f"""
                            </ul>
                            <div style="margin-top: 10px; font-style: italic;">
                                <strong>💡 投资建议:</strong> {liquidity['investment_suggestion']}
                            </div>
                        </div>
                    </div>"""
            
            # 添加增强分析显示（新功能）
            if 'enhanced_analysis' in result:
                enhanced = result['enhanced_analysis']
                enhanced_features = enhanced.get('enhanced_features', {})
                
                # 根据总体评分确定背景色
                overall_score = enhanced.get('overall_score', 0)
                if overall_score >= 0.8:
                    enhanced_class = "portfolio-profit"  # 绿色
                elif overall_score >= 0.6:
                    enhanced_class = "timing-good"  # 蓝色  
                elif overall_score >= 0.4:
                    enhanced_class = "timing-neutral"  # 黄色
                else:
                    enhanced_class = "portfolio-loss"  # 红色
                
                html += f"""
                    <div class="buy-timing-info {enhanced_class}">
                        <h4>🔧 智能增强分析 <span style="font-size: 0.8em; color: #666;">(New!)</span></h4>
                        <div class="timing-rating">综合评级: {enhanced.get('overall_rating', 'N/A')} ({overall_score:.3f}/1.0)</div>"""
                
                # 显示成长性和行业比较评分
                growth_score = enhanced.get('growth_score', 0)
                industry_score = enhanced.get('industry_score', 0)
                
                # 获取行业表现信息（从正确的数据结构中提取）
                financial_analysis = enhanced_features.get('financial_analysis', {})
                dimensions = financial_analysis.get('dimensions', {})
                industry_comparison = dimensions.get('industry_comparison', {})
                
                # 修复数据提取逻辑 - 从正确的数据结构中获取评分
                growth_data = dimensions.get('growth', {})
                actual_growth_score = growth_data.get('score', growth_score)  # 优先使用dimensions中的数据
                actual_industry_score = industry_comparison.get('industry_adjusted_score', industry_score)
                
                # 从行业比较数据中获取行业表现信息
                industry_performance = industry_comparison.get('summary', 'N/A')
                
                # 如果没有获取到行业表现信息，尝试从其他字段获取
                if industry_performance == 'N/A':
                    enhanced_warnings = result.get('enhanced_warnings', [])
                    enhanced_recommendations = result.get('enhanced_recommendations', [])
                    for item in enhanced_warnings + enhanced_recommendations:
                        if '行业' in item:
                            if '优秀' in item or '领先' in item:
                                industry_performance = '行业内优秀'
                            elif '良好' in item:
                                industry_performance = '行业内良好' 
                            elif '平均' in item:
                                industry_performance = '行业内平均'
                            elif '较差' in item or '落后' in item:
                                industry_performance = '行业内较差'
                            break
                    
                    # 如果仍然无法获取，基于数字评分生成描述
                    if industry_performance == 'N/A':
                        if actual_industry_score > 0.7:
                            industry_performance = '行业内表现优秀'
                        elif actual_industry_score > 0.5:
                            industry_performance = '行业内表现良好'
                        elif actual_industry_score > 0.3:
                            industry_performance = '行业内表现平均'
                        elif actual_industry_score > 0.1:
                            industry_performance = '行业内表现较差'
                        elif actual_industry_score > 0:
                            industry_performance = '行业内表现落后'
                        else:
                            industry_performance = '暂无行业对比数据'
                
                html += f"""
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 15px 0;">
                            <div>
                                <strong>📈 成长性分析:</strong>
                                <div style="margin: 5px 0;">
                                    <span style="font-size: 1.1em; font-weight: bold;">评分: {actual_growth_score:.3f}</span>
                                    <div style="margin-top: 3px;">"""
                
                # 使用实际的成长性评分进行判断
                if actual_growth_score > 0.8:
                    html += '<span style="color: #28a745;">🚀 成长性优秀</span>'
                elif actual_growth_score > 0.6:
                    html += '<span style="color: #007bff;">📊 成长性良好</span>'
                elif actual_growth_score > 0.4:
                    html += '<span style="color: #ffc107;">📈 成长性一般</span>'
                elif actual_growth_score > 0.2:
                    html += '<span style="color: #fd7e14;">📉 成长性较弱</span>'
                elif actual_growth_score > 0:
                    html += '<span style="color: #dc3545;">⚠️ 成长性偏弱</span>'
                else:
                    html += '<span style="color: #6c757d;">❓ 成长性数据不足</span>'
                
                html += f"""
                                    </div>
                                </div>
                            </div>
                            <div>
                                <strong>🏆 行业比较:</strong>
                                <div style="margin: 5px 0;">
                                    <span style="font-size: 1.1em; font-weight: bold;">{industry_performance}</span>
                                    <div style="margin-top: 3px;">"""
                
                # 基于行业表现文本设置颜色和图标
                if '优秀' in industry_performance or '领先' in industry_performance:
                    html += '<span style="color: #28a745;">🏆 同行业中表现突出</span>'
                elif '良好' in industry_performance:
                    html += '<span style="color: #007bff;">📊 同行业中表现良好</span>'
                elif '平均' in industry_performance:
                    html += '<span style="color: #ffc107;">⚖️ 同行业中表现平均</span>'
                elif '较差' in industry_performance or '落后' in industry_performance:
                    html += '<span style="color: #dc3545;">📉 同行业中表现落后</span>'
                else:
                    html += '<span style="color: #6c757d;">❓ 行业对比数据不足</span>'
                
                html += """
                                    </div>
                                </div>
                            </div>
                        </div>"""
                
                # 显示增强功能详情
                if enhanced_features:
                    html += """
                        <div style="margin: 15px 0;">
                            <strong>🔍 增强功能详情:</strong>
                            <div style="margin: 8px 0; display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">"""
                    
                    if 'financial_analysis' in enhanced_features:
                        fa_data = enhanced_features['financial_analysis']
                        warnings = fa_data.get('warnings', [])
                        warning_count = len(warnings)
                        
                        html += f"""
                                <div style="padding: 8px; background: rgba(255,255,255,0.5); border-radius: 5px;">
                                    <strong>💼 财务深度分析</strong><br>
                                    <small>包含行业基准对比</small>
                                    {f'<br><span style="color: #dc3545;">⚠️ {warning_count}个风险警告</span>' if warning_count > 0 else '<br><span style="color: #28a745;">✅ 无重大风险</span>'}
                                </div>"""
                    
                    if 'exit_strategy' in enhanced_features:
                        exit_data = enhanced_features['exit_strategy']
                        should_exit = exit_data.get('should_exit', False)
                        exit_reason = exit_data.get('exit_reason', 'N/A')
                        
                        html += f"""
                                <div style="padding: 8px; background: rgba(255,255,255,0.5); border-radius: 5px;">
                                    <strong>🔄 智能退出策略</strong><br>
                                    <small>动态止损止盈分析</small>
                                    {'<br><span style="color: #dc3545;">🚨 建议退出</span>' if should_exit else '<br><span style="color: #28a745;">✅ 持有信号</span>'}
                                </div>"""
                    
                    html += """
                            </div>
                        </div>"""
                
                # 显示增强建议
                enhanced_recommendations = result.get('enhanced_recommendations', [])
                if enhanced_recommendations:
                    html += """
                        <div style="margin: 15px 0;">
                            <strong>💡 智能投资建议:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px;">"""
                    
                    for rec in enhanced_recommendations[:4]:  # 最多显示4个建议
                        html += f"<li>{rec}</li>"
                    
                    html += "</ul></div>"
                
                # 显示增强警告
                enhanced_warnings = result.get('enhanced_warnings', [])
                if enhanced_warnings:
                    html += """
                        <div style="margin: 15px 0; padding: 10px; background: rgba(255,193,7,0.2); border-left: 4px solid #ffc107; border-radius: 0 5px 5px 0;">
                            <strong>⚠️ 风险提醒:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px;">"""
                    
                    for warning in enhanced_warnings[:3]:  # 最多显示3个警告
                        html += f"<li>{warning}</li>"
                    
                    html += "</ul></div>"
                
                html += """
                    </div>"""
            
            # 添加右侧交易分析显示（防抄底系统）
            if 'right_side_trading' in result:
                right_side = result['right_side_trading']
                trend_status = right_side['trend_status']
                entry_signals = right_side['entry_signals']
                risk_warnings = right_side['risk_warnings']
                
                # 根据趋势状态确定背景色
                if trend_status['direction'] == '上升' and trend_status['confirmed']:
                    right_side_class = "portfolio-profit"  # 绿色 - 上升趋势已确认
                elif trend_status['direction'] == '上升' and not trend_status['confirmed']:
                    right_side_class = "timing-good"  # 蓝色 - 上升趋势未确认
                elif trend_status['direction'] == '震荡':
                    right_side_class = "timing-neutral"  # 黄色 - 震荡
                elif trend_status['direction'] == '下跌' and not trend_status['confirmed']:
                    right_side_class = "timing-poor"  # 橙色 - 下跌趋势未确认
                else:
                    right_side_class = "portfolio-loss"  # 红色 - 下跌趋势已确认
                
                html += f"""
                    <div class="buy-timing-info {right_side_class}">
                        <h4>🎯 右侧交易分析 <span style="font-size: 0.8em; color: #666;">(防抄底系统)</span></h4>
                        <div class="timing-rating">
                            趋势状态: {trend_status['direction']} | 
                            强度: {trend_status['strength']['level']} | 
                            {'✅ 已确认' if trend_status['confirmed'] else '❌ 未确认'}
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 15px 0;">
                            <div>
                                <strong>📊 趋势分析:</strong>
                                <div style="margin: 5px 0; font-size: 0.9em;">
                                    <div>方向: <span style="font-weight: bold;">{trend_status['direction']}</span></div>
                                    <div>强度: <span style="font-weight: bold;">{trend_status['strength']['level']}</span></div>
                                    <div>持续: <span style="font-weight: bold;">{trend_status['trend_days']}天</span></div>
                                    <div>确认: <span style="font-weight: bold;">{'是' if trend_status['confirmed'] else '否'}</span></div>
                                </div>
                            </div>
                            <div>
                                <strong>🔍 技术指标:</strong>
                                <div style="margin: 5px 0; font-size: 0.9em;">
                                    <div>动量得分: {trend_status['strength']['momentum_score']}/3</div>
                                    <div>成交量: {'✅ 配合' if trend_status['strength']['volume_confirmed'] else '❌ 萎缩'}</div>
                                    <div>RSI趋势: {trend_status['strength']['rsi_trend']}</div>
                                    <div>MACD: {'✅ 金叉' if trend_status['strength']['macd_bullish'] else '❌ 死叉'}</div>
                                </div>
                            </div>
                        </div>"""
                
                # 显示买入信号
                if entry_signals['buy_signals']:
                    html += """
                        <div style="margin: 15px 0; padding: 10px; background: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745; border-radius: 0 5px 5px 0;">
                            <strong>🟢 右侧买入信号:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px;">"""
                    
                    for signal in entry_signals['buy_signals']:
                        html += f"<li>{signal}</li>"
                    
                    html += "</ul></div>"
                
                # 显示卖出信号
                if entry_signals['sell_signals']:
                    html += """
                        <div style="margin: 15px 0; padding: 10px; background: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; border-radius: 0 5px 5px 0;">
                            <strong>🔴 右侧卖出信号:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px;">"""
                    
                    for signal in entry_signals['sell_signals']:
                        html += f"<li>{signal}</li>"
                    
                    html += "</ul></div>"
                
                # 显示等待信号
                if entry_signals['wait_signals']:
                    html += """
                        <div style="margin: 15px 0; padding: 10px; background: rgba(255, 193, 7, 0.1); border-left: 4px solid #ffc107; border-radius: 0 5px 5px 0;">
                            <strong>🟡 右侧等待信号:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px;">"""
                    
                    for signal in entry_signals['wait_signals']:
                        html += f"<li>{signal}</li>"
                    
                    html += "</ul></div>"
                
                # 显示风险警告
                if risk_warnings:
                    html += """
                        <div style="margin: 15px 0; padding: 10px; background: rgba(255, 152, 0, 0.1); border-left: 4px solid #ff9800; border-radius: 0 5px 5px 0;">
                            <strong>⚠️ 左侧交易风险警告:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px;">"""
                    
                    for warning in risk_warnings:
                        html += f"<li>{warning}</li>"
                    
                    html += "</ul></div>"
                
                # 添加右侧交易核心原则提醒
                html += """
                        <div style="margin: 15px 0; padding: 10px; background: rgba(0, 123, 255, 0.1); border-left: 4px solid #007bff; border-radius: 0 5px 5px 0;">
                            <strong>💡 右侧交易核心原则:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px; font-size: 0.9em;">
                                <li>✅ 趋势确认后再进入，不抄底不摸顶</li>
                                <li>✅ 等待突破确认，避免假突破陷阱</li>
                                <li>✅ 成交量必须配合，无量上涨不追</li>
                                <li>✅ 设置止损位，严格执行纪律</li>
                            </ul>
                        </div>
                    </div>"""
            
            # 添加图表显示
            if result.get('chart_base64') and result['chart_base64']:
                    html += f"""
                    <div style="text-align: center; margin-top: 15px;">
                        <img src="data:image/png;base64,{result['chart_base64']}" alt="{symbol}技术分析图表" class="chart-image">
                    </div>"""
            
            html += f"""
                </div>
            """
        
        html += """
                <div class="footer">
                    <p><strong>⚠️ 重要提示:</strong> 本报告基于技术分析和历史数据，仅供参考，不构成投资建议。</p>
                    <p>投资有风险，入市需谨慎。请结合个人风险承受能力做出投资决策。</p>
                    <p style="margin-top: 15px; font-size: 0.8em;">
                        报告由智能股票分析系统自动生成 | © 2024 MOSE Trading System | 使用真实市场数据
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_macro_analysis(self) -> Optional[Dict]:
        """获取宏观分析结果"""
        try:
            macro_analysis = {}
            
            # 基础宏观分析
            if self.macro_integration:
                try:
                    logger.info("开始基础宏观因子分析...")
                    base_macro = self.macro_integration.generate_macro_report()
                    
                    if 'error' not in base_macro:
                        macro_analysis.update(base_macro)
                        logger.info("基础宏观分析完成")
                    else:
                        logger.error(f"基础宏观分析失败: {base_macro['error']}")
                except Exception as e:
                    logger.error(f"基础宏观分析失败: {e}")
            
            # 通胀-行业分析（核心功能2：完善宏观-行业影响分析）
            if self.inflation_sector_analyzer:
                try:
                    logger.info("开始通胀-行业影响分析...")
                    inflation_report = self.inflation_sector_analyzer.generate_inflation_sector_report()
                    if inflation_report:
                        macro_analysis['inflation_sector_analysis'] = inflation_report
                        logger.info("通胀-行业分析完成")
                except Exception as e:
                    logger.error(f"通胀-行业分析失败: {e}")
            
            return macro_analysis if macro_analysis else None
                
        except Exception as e:
            logger.error(f"获取宏观分析失败: {e}")
            return None

    def generate_report(self) -> str:
        """生成完整的日报"""
        logger.info("开始生成智能日报...")
        
        # 更新市场数据
        self._update_market_data(self.watchlist)
        
        # 获取宏观分析结果
        macro_analysis = self._get_macro_analysis()
        
        # 分析所有关注股票
        results = []
        for symbol in self.watchlist:
            try:
                result = self._analyze_stock(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"分析 {symbol} 失败: {e}")
                results.append(None)
        
        # 生成HTML报告（包含宏观分析）
        html_content = self._generate_html_report(results, macro_analysis)
        
        # 保存报告文件
        # 使用统一路径配置生成报告文件名
        try:
            from config.paths_config import get_report_path
            report_filename = get_report_path()
        except ImportError:
            # 后备方案
            report_filename = f"智能股票日报_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"日报生成完成: {report_filename}")
        return html_content  # 返回HTML内容而不是文件名

def main():
    """主函数 - 生成用户持仓股票日报"""
    # 创建报告生成器（使用默认的用户持仓股票）
    generator = SmartDailyReportGenerator(
        auto_update_data=True  # 启用自动数据更新
    )
    
    # 生成报告
    report_file = generator.generate_report()
    
    print(f"\n✅ 用户持仓股票日报已生成: {report_file}")
    print("📊 报告特性:")
    print("   ✓ 使用真实市场数据")
    print("   ✓ 包含持仓成本和盈亏分析")
    print("   ✓ 图表内嵌HTML显示")
    print("   ✓ 市场环境自动分类")
    print("   ✓ 动态策略推荐")
    print("   ✓ 信号质量评估")
    print("   ✓ 数据质量监控")
    print("   ✓ 技术指标分析")
    print("   ✓ 可视化图表")
    print("\n💼 分析的持仓股票:")
    for symbol, info in generator.portfolio.items():
        print(f"   • {symbol}: 成本${info['cost']:.3f}, 占比{info['weight']:.2f}%")
    
    print("\n👀 观察中的股票(准备买入):")
    for symbol, info in generator.watch_targets.items():
        if info['previous_buy'] is not None:
            print(f"   • {symbol}: 历史买入${info['previous_buy']:.2f}→卖出${info['previous_sell']:.2f} (+{info['previous_gain']:.1f}%)")
        else:
            print(f"   • {symbol}: 暂无历史交易记录")
        print(f"     目标买入价: <${info['target_buy_below']:.2f}")
        print(f"     买入理由: {info['reason']}")
    
    print("\n💡 数据更新建议:")
    print("   • 工作日收盘后自动运行")
    print("   • 确保数据库连接正常")
    print("   • 监控数据质量评分")

if __name__ == "__main__":
    main() 