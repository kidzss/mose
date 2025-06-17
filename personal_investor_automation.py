#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个人投资者自动化股票推荐系统

功能：
1. 每周自动筛选优质股票
2. 自动更新股票数据和财务数据
3. 发送个性化投资建议邮件
4. 适合个人投资者的风险控制

推荐频率：
- 每周筛选：每周日20:00
- 每月深度分析：每月第一个周日
- 季度策略调整：每季度第一个周日
"""

import os
import sys
import schedule
import time
import logging
import json
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitor.phase2_professional_screener import Phase2ProfessionalScreener
from data.data_interface import DataInterface
from utils.unified_email_api import send_html

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('personal_investor_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PersonalInvestorAutomation")

class PersonalInvestorAutomation:
    """个人投资者自动化系统"""
    
    def __init__(self):
        self.screener = Phase2ProfessionalScreener()
        self.data_interface = DataInterface()
        
        # 个人投资者配置
        self.config = {
            'email': 'kidzss@gmail.com',
            'risk_tolerance': 'moderate',  # conservative, moderate, aggressive
            'max_position_size': 0.20,     # 单只股票最大仓位20%
            'min_quality_factor': 0.7,     # 最低质量因子
            'max_results': 15,             # 推荐股票数量
            'min_score': 60,               # 最低评分
        }
        
        # 加载个人配置
        self._load_personal_config()
        
        logger.info("🚀 个人投资者自动化系统初始化完成")
    
    def _load_personal_config(self):
        """加载个人配置文件"""
        config_file = 'personal_investor_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    personal_config = json.load(f)
                    self.config.update(personal_config)
                logger.info("✅ 已加载个人配置文件")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")
        else:
            # 创建默认配置文件
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            'email': 'kidzss@gmail.com',
            'risk_tolerance': 'moderate',
            'max_position_size': 0.20,
            'min_quality_factor': 0.7,
            'max_results': 15,
            'min_score': 60,
            'investment_goals': {
                'time_horizon': '3-5年',
                'risk_preference': '中等风险',
                'investment_amount': '可承受20%亏损'
            },
            'preferred_sectors': ['科技', '消费', '医疗', '金融'],
            'excluded_sectors': ['能源', '原材料']  # 可选排除
        }
        
        try:
            with open('personal_investor_config.json', 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info("✅ 已创建默认配置文件: personal_investor_config.json")
        except Exception as e:
            logger.error(f"创建配置文件失败: {e}")
    
    def update_market_data(self):
        """更新市场数据"""
        try:
            logger.info("📊 开始更新市场数据...")
            
            # 获取需要更新的股票列表
            watchlist = ['AMD', 'GOOGL', 'PFE', 'NVDA', 'TSLA', 'ADBE', 'MSFT', 'EOG', 'PHM', 'CF']
            
            # 更新股票数据
            updated_count = 0
            for symbol in watchlist:
                try:
                    # 获取最新数据
                    data = self.data_interface.get_data_for_strategy(symbol, lookback_days=60)
                    if data is not None and len(data) > 0:
                        updated_count += 1
                        logger.info(f"✅ {symbol} 数据已更新")
                except Exception as e:
                    logger.warning(f"更新 {symbol} 数据失败: {e}")
            
            logger.info(f"📊 数据更新完成，成功更新 {updated_count}/{len(watchlist)} 只股票")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"数据更新失败: {e}")
            return False
    
    def run_weekly_screening(self):
        """运行每周股票筛选"""
        try:
            logger.info("🎯 开始每周股票筛选...")
            
            # 更新数据
            self.update_market_data()
            
            # 根据风险偏好调整筛选参数
            if self.config['risk_tolerance'] == 'conservative':
                min_score = 65
                min_quality = 0.8
            elif self.config['risk_tolerance'] == 'aggressive':
                min_score = 55
                min_quality = 0.6
            else:  # moderate
                min_score = 60
                min_quality = 0.7
            
            # 执行筛选
            results = self.screener.screen_stocks_professional(
                min_score=min_score,
                max_results=self.config['max_results']
            )
            
            # 过滤高质量股票
            high_quality_results = [
                stock for stock in results 
                if stock['quality_factor'] >= min_quality
            ]
            
            if high_quality_results:
                # 生成个性化投资建议
                self._generate_personalized_report(high_quality_results, 'weekly')
                logger.info(f"✅ 每周筛选完成，推荐 {len(high_quality_results)} 只高质量股票")
            else:
                logger.warning("⚠️ 未找到符合条件的高质量股票")
            
            return high_quality_results
            
        except Exception as e:
            logger.error(f"每周筛选失败: {e}")
            return []
    
    def run_monthly_analysis(self):
        """运行每月深度分析"""
        try:
            logger.info("📈 开始每月深度分析...")
            
            # 更新所有数据
            self.update_market_data()
            
            # 更严格的筛选标准
            results = self.screener.screen_stocks_professional(
                min_score=70,  # 更高标准
                max_results=10  # 更少但更精
            )
            
            # 生成深度分析报告
            self._generate_personalized_report(results, 'monthly')
            
            logger.info("✅ 每月深度分析完成")
            return results
            
        except Exception as e:
            logger.error(f"每月分析失败: {e}")
            return []
    
    def run_quarterly_strategy(self):
        """运行季度策略调整"""
        try:
            logger.info("🔄 开始季度策略调整...")
            
            # 全面数据更新
            self.update_market_data()
            
            # 市场环境分析
            market_analysis = self._analyze_market_environment()
            
            # 策略调整建议
            strategy_recommendations = self._generate_strategy_recommendations(market_analysis)
            
            # 生成季度报告
            self._generate_personalized_report([], 'quarterly', 
                                             market_analysis=market_analysis,
                                             strategy_recommendations=strategy_recommendations)
            
            logger.info("✅ 季度策略调整完成")
            
        except Exception as e:
            logger.error(f"季度策略调整失败: {e}")
    
    def _check_monthly_analysis(self):
        """检查是否应该执行月度分析"""
        try:
            today = datetime.now()
            # 检查是否是当月第一个周日
            if today.day <= 7:  # 前7天内
                logger.info("📅 检测到月度分析时间")
                self.run_monthly_analysis()
            else:
                logger.info("📅 本周不是月度分析时间")
        except Exception as e:
            logger.error(f"月度分析检查失败: {e}")
    
    def _check_quarterly_strategy(self):
        """检查是否应该执行季度策略调整"""
        try:
            today = datetime.now()
            # 检查是否是季度第一个月的第一个周日
            if today.month in [1, 4, 7, 10] and today.day <= 7:
                logger.info("📅 检测到季度策略调整时间")
                self.run_quarterly_strategy()
            else:
                logger.info("📅 本周不是季度策略调整时间")
        except Exception as e:
            logger.error(f"季度策略调整检查失败: {e}")
    
    def _analyze_market_environment(self):
        """分析市场环境"""
        try:
            # 获取VIX数据
            vix_data = self._get_vix_data()
            
            # 市场情绪分析
            sentiment = self._analyze_market_sentiment()
            
            # 行业轮动分析
            sector_rotation = self._analyze_sector_rotation()
            
            return {
                'vix_level': vix_data.get('current', 20),
                'market_sentiment': sentiment,
                'sector_rotation': sector_rotation,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"市场环境分析失败: {e}")
            return {}
    
    def _get_vix_data(self):
        """获取VIX数据"""
        try:
            # 简化版VIX数据获取
            return {'current': 20, 'change': 0.5, 'level': 'normal'}
        except:
            return {'current': 20, 'change': 0, 'level': 'unknown'}
    
    def _analyze_market_sentiment(self):
        """分析市场情绪"""
        try:
            # 简化版情绪分析
            return {
                'fear_greed_index': 50,
                'market_breadth': 0.6,
                'sentiment': 'neutral'
            }
        except:
            return {'sentiment': 'unknown'}
    
    def _analyze_sector_rotation(self):
        """分析行业轮动"""
        try:
            # 简化版行业轮动分析
            return {
                'leading_sectors': ['科技', '消费'],
                'lagging_sectors': ['能源', '原材料'],
                'rotation_phase': 'growth_to_value'
            }
        except:
            return {'rotation_phase': 'unknown'}
    
    def _generate_strategy_recommendations(self, market_analysis):
        """生成策略调整建议"""
        recommendations = []
        
        vix_level = market_analysis.get('vix_level', 20)
        sentiment = market_analysis.get('market_sentiment', {}).get('sentiment', 'neutral')
        
        if vix_level > 30:
            recommendations.append({
                'type': 'risk_reduction',
                'action': '增加防御性股票配置',
                'reason': f'VIX指数较高({vix_level})，市场波动性增加'
            })
        elif vix_level < 15:
            recommendations.append({
                'type': 'opportunity_seeking',
                'action': '可以适当增加成长股配置',
                'reason': f'VIX指数较低({vix_level})，市场相对稳定'
            })
        
        if sentiment == 'fear':
            recommendations.append({
                'type': 'contrarian',
                'action': '考虑逆向投资机会',
                'reason': '市场恐慌情绪，可能存在超跌机会'
            })
        
        return recommendations
    
    def _generate_personalized_report(self, results, report_type, 
                                    market_analysis=None, strategy_recommendations=None):
        """生成个性化投资报告"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # 根据报告类型生成不同内容
            if report_type == 'weekly':
                subject = f"📈 每周股票推荐 | {timestamp} | 个人投资策略"
                content = self._generate_weekly_content(results)
            elif report_type == 'monthly':
                subject = f"📊 每月深度分析 | {timestamp} | 投资组合优化"
                content = self._generate_monthly_content(results)
            elif report_type == 'quarterly':
                subject = f"🔄 季度策略调整 | {timestamp} | 市场环境分析"
                content = self._generate_quarterly_content(market_analysis, strategy_recommendations)
            
            # 发送邮件
            success = send_html(subject=subject, html_content=content)
            
            if success:
                logger.info(f"✅ {report_type}报告邮件发送成功")
            else:
                logger.error(f"❌ {report_type}报告邮件发送失败")
            
            return success
            
        except Exception as e:
            logger.error(f"生成{report_type}报告失败: {e}")
            return False
    
    def _generate_weekly_content(self, results):
        """生成每周报告内容"""
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; }}
                .stock-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .stock-table th, .stock-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .stock-table th {{ background-color: #f2f2f2; }}
                .high-quality {{ background-color: #e8f5e8; }}
                .investment-tips {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .risk-warning {{ background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 个人投资者每周股票推荐</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>风险偏好: {self.config['risk_tolerance']} | 最大仓位: {self.config['max_position_size']*100}%</p>
            </div>
            
            <h2>🏆 本周推荐股票</h2>
            <table class="stock-table">
                <tr>
                    <th>排名</th>
                    <th>股票代码</th>
                    <th>综合评分</th>
                    <th>质量因子</th>
                    <th>夏普比率</th>
                    <th>当前价格</th>
                    <th>投资建议</th>
                </tr>
        """
        
        for i, stock in enumerate(results[:10], 1):
            quality_class = "high-quality" if stock['quality_factor'] > 0.8 else ""
            html_content += f"""
                <tr class="{quality_class}">
                    <td>{i}</td>
                    <td><strong>{stock['symbol']}</strong></td>
                    <td>{stock['multifactor_score']:.1f}</td>
                    <td>{stock['quality_factor']:.3f}</td>
                    <td>{stock['sharpe_ratio']:.2f}</td>
                    <td>${stock['current_price']:.2f}</td>
                    <td>{self._get_investment_advice(stock)}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="investment-tips">
                <h3>💡 个人投资建议</h3>
                <ul>
                    <li><strong>分批建仓</strong>: 建议分3-4次买入，每次25%仓位</li>
                    <li><strong>止损设置</strong>: 单只股票亏损超过15%时考虑减仓</li>
                    <li><strong>持有期限</strong>: 建议至少持有1-2年，给价值回归时间</li>
                    <li><strong>定期检查</strong>: 每月检查一次持仓，但不要频繁交易</li>
                </ul>
            </div>
            
            <div class="risk-warning">
                <h3>⚠️ 风险提示</h3>
                <p>本推荐仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
                <p>请根据自身风险承受能力和投资目标做出投资决策。</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_monthly_content(self, results):
        """生成每月报告内容"""
        # 类似weekly但更详细的分析
        return self._generate_weekly_content(results) + "<h2>📊 月度深度分析</h2>"
    
    def _generate_quarterly_content(self, market_analysis, strategy_recommendations):
        """生成季度报告内容"""
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; }}
                .analysis-section {{ background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔄 季度策略调整报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="analysis-section">
                <h2>📊 市场环境分析</h2>
                <p>VIX指数: {market_analysis.get('vix_level', 'N/A')}</p>
                <p>市场情绪: {market_analysis.get('market_sentiment', {}).get('sentiment', 'N/A')}</p>
            </div>
            
            <div class="analysis-section">
                <h2>💡 策略调整建议</h2>
        """
        
        if strategy_recommendations:
            for rec in strategy_recommendations:
                html_content += f"""
                    <div style="margin: 10px 0; padding: 10px; border-left: 4px solid #007bff;">
                        <strong>{rec['action']}</strong><br>
                        <small>原因: {rec['reason']}</small>
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _get_investment_advice(self, stock):
        """获取投资建议"""
        quality = stock['quality_factor']
        score = stock['multifactor_score']
        
        if quality > 0.8 and score > 70:
            return "强烈推荐，可考虑重仓"
        elif quality > 0.7 and score > 60:
            return "推荐买入，分批建仓"
        elif quality > 0.6 and score > 55:
            return "谨慎推荐，小仓位试仓"
        else:
            return "观望，等待更好机会"
    
    def setup_schedule(self):
        """设置定时任务"""
        try:
            # 每周筛选 - 每周日20:00
            schedule.every().sunday.at("20:00").do(self.run_weekly_screening)
            
            # 每月深度分析 - 每月第一个周日 (改为每周检查，但只在月初执行)
            schedule.every().sunday.at("20:30").do(self._check_monthly_analysis)
            
            # 季度策略调整 - 每季度第一个周日 (改为每周检查，但只在季度初执行)
            schedule.every().sunday.at("21:00").do(self._check_quarterly_strategy)
            
            logger.info("✅ 定时任务设置完成")
            
        except Exception as e:
            logger.error(f"设置定时任务失败: {e}")
    
    def start_automation(self):
        """启动自动化系统"""
        try:
            print("=" * 80)
            print("🚀 个人投资者自动化股票推荐系统")
            print("=" * 80)
            print()
            print("📊 系统功能:")
            print("   ✓ 每周自动筛选优质股票")
            print("   ✓ 每月深度分析投资组合")
            print("   ✓ 季度策略调整建议")
            print("   ✓ 自动更新市场数据")
            print("   ✓ 个性化投资建议邮件")
            print()
            print("⏰ 定时安排:")
            print("   - 每周筛选: 每周日 20:00")
            print("   - 每月分析: 每月第一个周日 20:00")
            print("   - 季度调整: 每季度第一个周日 20:00")
            print()
            print(f"📧 邮件接收: {self.config['email']}")
            print(f"🎯 风险偏好: {self.config['risk_tolerance']}")
            print(f"💰 最大仓位: {self.config['max_position_size']*100}%")
            print()
            print("🛑 要停止服务，请按 Ctrl+C")
            print("=" * 80)
            print()
            
            # 设置定时任务
            self.setup_schedule()
            
            # 显示下次运行时间
            next_run = schedule.next_run()
            if next_run:
                print(f"⏰ 下次运行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # 可选：立即运行一次测试
            test_now = input("💡 是否立即运行一次测试？(y/N): ").strip().lower()
            if test_now == 'y':
                print("🧪 开始测试运行...")
                self.run_weekly_screening()
            
            print("⏳ 等待定时任务触发...")
            print("   (或按 Ctrl+C 停止服务)")
            
            # 保持运行
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            print("\n👋 自动化服务已停止")
        except Exception as e:
            logger.error(f"自动化服务运行出错: {e}")

def main():
    """主函数"""
    automation = PersonalInvestorAutomation()
    automation.start_automation()

if __name__ == "__main__":
    main() 
