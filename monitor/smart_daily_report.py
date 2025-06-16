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
        # 用户持仓股票列表 + 观察股票
        self.watchlist = watchlist or ['AMD', 'GOOGL', 'PFE', 'NVDA', 'TSLA', 'EOG', 'MSFT', 'PHM', 'CF']
        self.auto_update_data = auto_update_data
        self.data_source_type = None
        
        # 用户持仓信息 (更新日期: 2025-06-15, 基于精确金额数据)
        self.portfolio = portfolio or {
            'AMD': {'cost': 126.214, 'shares': 48, 'weight': 21.86, 'investment': 4788.89},   # $21,903.42 × 21.86%
            'GOOGL': {'cost': 170.54, 'shares': 34, 'weight': 21.53, 'investment': 4715.83}, # $21,903.42 × 21.53%
            'PFE': {'cost': 25.899, 'shares': 80, 'weight': 6.97, 'investment': 1526.65},    # $21,903.42 × 6.97%
            'NVDA': {'cost': 138.843, 'shares': 40, 'weight': 20.92, 'investment': 4582.24}, # $21,903.42 × 20.92%
            'TSLA': {'cost': 254.096, 'shares': 4, 'weight': 4.74, 'investment': 1038.22},   # $21,903.42 × 4.74%
            'EOG': {'cost': 122.119, 'shares': 5, 'weight': 2.20, 'investment': 481.88}      # $21,903.42 × 2.20%
        }
        
        # 投资组合总价值计算 (基于美元货币基金精确金额$3,262.53)
        self.total_portfolio_value = 27884.87  # 总资产价值 (重新计算)
        self.total_stock_investment = 21903.42  # 总股票投资金额 (78.55% × $27,884.87)
        self.portfolio_allocation = 78.55  # 股票占总投资组合的比例 (实际数据)
        self.cash_allocation = 9.75  # 现金占比 ($2,718.77)
        self.money_fund_allocation = 11.70  # 美元货币型基金占比
        self.money_fund_value = 3262.53  # 美元货币型基金精确金额
        
        # 观察目标股票（准备买入的股票）
        self.watch_targets = watch_targets or {
            'MSFT': {
                'previous_buy': 370.95,
                'previous_sell': 453.97,
                'previous_gain': 22.4,  # 约22.4%收益
                'target_buy_below': 420.0,  # 建议买入价格下方
                'reason': '准备再次买入，关注买入时机'
            },
            'ADBE': {
                'previous_buy': 346.896,  # 刚刚卖出
                'previous_sell': 398.2,
                'previous_gain': 14.8,  # 约14.8%收益
                'target_buy_below': 380.0,  # 建议回调后再次买入
                'reason': '刚刚获利了结，等待回调至$380以下再次买入机会'
            },
            'PHM': {
                'previous_buy': None,  # 从未购买过
                'previous_sell': None,
                'previous_gain': None,
                'target_buy_below': 98.00,  # 基于50日均线支撑位($100.16)下方2%
                'reason': '地产龙头，业绩稳健，接近买入区域，当前价格$101.61，关注50日均线支撑买入机会'
            },
            'CF': {
                'previous_buy': None,  # 从未购买过
                'previous_sell': None,
                'previous_gain': None,
                'target_buy_below': 84.00,  # 基于布林带下轨支撑位($85.69)下方2%
                'reason': '化肥龙头，周期回暖，当前价格$99.93，等待回调至布林带下轨支撑位附近买入'
            }
        }
        
        # 初始化数据接口 - 支持回退机制
        self._init_data_sources()
        
        # 初始化核心组件
        self.market_classifier = MarketEnvironmentClassifier()
        self.strategy_selector = DynamicStrategySelector()
        self.signal_evaluator = SignalQualityEvaluator()
        self.alert_system = AdvancedAlertSystem()
        
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
                # 回退到Yahoo Finance
                self.data_interface = DataInterface(default_source='yahoo')
                self.data_updater = None  # Yahoo Finance不需要数据更新器
                self.auto_update_data = False  # 禁用自动更新
                self.data_source_type = "Yahoo Finance"
                logger.info("✅ Yahoo Finance数据源连接成功")
                
            except Exception as e2:
                logger.error(f"Yahoo Finance连接也失败: {e2}")
                logger.info("🔄 使用模拟数据模式...")
                self.data_interface = None
                self.data_updater = None
                self.auto_update_data = False
                self.data_source_type = "模拟数据"
    
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
            
            days_old = (datetime.now().date() - latest_date.date()).days
            
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
        quality_info = {
            'symbol': symbol,
            'total_records': len(data),
            'date_range': f"{data.index[0].date()} 到 {data.index[-1].date()}",
            'latest_date': data.index[-1].date(),
            'days_old': (datetime.now().date() - data.index[-1].date()).days,
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
            pnl_amount = current_value - investment_amount
            pnl_percent = (pnl_amount / investment_amount) * 100
            
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
            
            # 生成图表
            chart_path = self._create_chart(symbol, data, env_result)
            result['chart_path'] = chart_path
            
            logger.info(f"{symbol} 分析完成 - 环境: {result['environment']}, 信号质量: {result['signal_quality']:.2f}, 数据质量: {quality_info['quality_score']:.0f}分")
            return result
            
        except Exception as e:
            logger.error(f"分析 {symbol} 出错: {e}")
            return result
    
    def _image_to_base64(self, image_path: str) -> str:
        """将图片文件转换为base64编码"""
        try:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    return f"data:image/png;base64,{img_data}"
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
            
            # 保存图表
            chart_filename = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"已生成 {symbol} 图表: {chart_filename}")
            return chart_filename
            
        except Exception as e:
            logger.error(f"生成 {symbol} 图表失败: {e}")
            return ""
    
    def _generate_html_report(self, analysis_results: List[Dict]) -> str:
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
            
            # 添加图表显示
            if result.get('chart_path') and os.path.exists(result['chart_path']):
                base64_image = self._image_to_base64(result['chart_path'])
                if base64_image:
                    html += f"""
                    <div style="text-align: center; margin-top: 15px;">
                        <img src="{base64_image}" alt="{symbol}技术分析图表" class="chart-image">
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
    
    def generate_report(self) -> str:
        """生成完整的日报"""
        logger.info("开始生成智能日报...")
        
        # 更新市场数据
        self._update_market_data(self.watchlist)
        
        # 分析所有关注股票
        results = []
        for symbol in self.watchlist:
            try:
                result = self._analyze_stock(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"分析 {symbol} 失败: {e}")
                results.append(None)
        
        # 生成HTML报告
        html_content = self._generate_html_report(results)
        
        # 保存报告文件
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