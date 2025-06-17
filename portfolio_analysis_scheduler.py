#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持股分析定时调度器

功能：
1. 每日持股分析报告（可选）
2. 每周持股深度分析
3. 每月持股组合优化建议
4. 自动发送邮件报告

推荐频率：
- 每日简报：每个交易日 16:30 (美股收盘后30分钟)
- 每周分析：每周日 20:00
- 每月深度分析：每月第一个周日 20:00
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

from monitor.smart_daily_report import SmartDailyReportGenerator
from utils.unified_email_api import send_html

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_analysis_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PortfolioAnalysisScheduler")

class PortfolioAnalysisScheduler:
    """持股分析定时调度器"""
    
    def __init__(self):
        self.report_generator = SmartDailyReportGenerator()
        
        # 调度配置
        self.config = {
            'email': 'kidzss@gmail.com',
            'daily_enabled': True,          # 是否启用每日报告
            'weekly_enabled': True,         # 是否启用每周分析
            'monthly_enabled': True,        # 是否启用每月分析
            'daily_time': '16:30',          # 每日报告时间（美股收盘后30分钟）
            'weekly_time': '20:00',         # 每周分析时间
            'monthly_time': '20:00',        # 每月分析时间
        }
        
        # 加载配置
        self._load_config()
        
        logger.info("🚀 持股分析定时调度器初始化完成")
    
    def _load_config(self):
        """加载配置文件"""
        config_file = 'portfolio_scheduler_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                logger.info("✅ 已加载调度器配置文件")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")
        else:
            # 创建默认配置文件
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            'email': 'kidzss@gmail.com',
            'daily_enabled': True,
            'weekly_enabled': True,
            'monthly_enabled': True,
            'daily_time': '16:30',
            'weekly_time': '20:00',
            'monthly_time': '20:00',
            'portfolio_settings': {
                'update_data_before_analysis': True,
                'include_market_sentiment': True,
                'include_technical_analysis': True,
                'risk_alerts_enabled': True
            },
            'email_settings': {
                'send_charts': True,
                'detailed_analysis': True,
                'include_recommendations': True
            }
        }
        
        try:
            with open('portfolio_scheduler_config.json', 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info("✅ 已创建默认调度器配置文件")
        except Exception as e:
            logger.error(f"创建配置文件失败: {e}")
    
    def run_daily_analysis(self):
        """运行每日持股分析"""
        try:
            if not self.config['daily_enabled']:
                logger.info("每日分析已禁用，跳过执行")
                return False
                
            logger.info("📊 开始每日持股分析...")
            
            # 生成每日报告
            html_report = self.report_generator.generate_report()
            
            if html_report:
                # 发送每日报告邮件
                subject = f"📈 每日持股分析简报 - {datetime.now().strftime('%Y-%m-%d')}"
                success = send_html(subject=subject, html_content=html_report)
                
                if success:
                    logger.info("✅ 每日持股分析报告发送成功")
                    return True
                else:
                    logger.error("❌ 每日报告邮件发送失败")
                    return False
            else:
                logger.error("❌ 每日报告生成失败")
                return False
                
        except Exception as e:
            logger.error(f"每日持股分析失败: {e}")
            return False
    
    def run_weekly_analysis(self):
        """运行每周持股深度分析"""
        try:
            if not self.config['weekly_enabled']:
                logger.info("每周分析已禁用，跳过执行")
                return False
                
            logger.info("📊 开始每周持股深度分析...")
            
            # 生成详细的每周分析报告
            html_report = self.report_generator.generate_report()
            
            # 添加每周特有的深度分析内容
            weekly_content = self._generate_weekly_analysis_content()
            
            # 合并报告内容
            if html_report and weekly_content:
                combined_report = self._combine_reports(html_report, weekly_content)
                
                # 发送每周分析邮件
                subject = f"📊 每周持股深度分析 - {datetime.now().strftime('%Y年第%U周')}"
                success = send_html(subject=subject, html_content=combined_report)
                
                if success:
                    logger.info("✅ 每周持股深度分析报告发送成功")
                    return True
                else:
                    logger.error("❌ 每周分析邮件发送失败")
                    return False
            else:
                logger.error("❌ 每周分析报告生成失败")
                return False
                
        except Exception as e:
            logger.error(f"每周持股分析失败: {e}")
            return False
    
    def run_monthly_analysis(self):
        """运行每月持股组合优化分析"""
        try:
            if not self.config['monthly_enabled']:
                logger.info("每月分析已禁用，跳过执行")
                return False
                
            logger.info("📊 开始每月持股组合优化分析...")
            
            # 生成基础报告
            html_report = self.report_generator.generate_report()
            
            # 生成月度特有的组合优化内容
            monthly_content = self._generate_monthly_optimization_content()
            
            # 合并报告内容
            if html_report and monthly_content:
                combined_report = self._combine_reports(html_report, monthly_content)
                
                # 发送月度分析邮件
                subject = f"📈 每月持股组合优化分析 - {datetime.now().strftime('%Y年%m月')}"
                success = send_html(subject=subject, html_content=combined_report)
                
                if success:
                    logger.info("✅ 每月持股组合优化分析报告发送成功")
                    return True
                else:
                    logger.error("❌ 每月分析邮件发送失败")
                    return False
            else:
                logger.error("❌ 每月分析报告生成失败")
                return False
                
        except Exception as e:
            logger.error(f"每月持股分析失败: {e}")
            return False
    
    def _generate_weekly_analysis_content(self):
        """生成每周分析特有内容"""
        try:
            weekly_html = f"""
            <div style="background-color: #e8f4f8; padding: 20px; margin: 20px 0; border-radius: 10px;">
                <h2>📊 每周深度分析</h2>
                <h3>📈 本周表现总结</h3>
                <ul>
                    <li>✅ 本周涨幅最大的持仓股票及原因分析</li>
                    <li>📉 本周跌幅较大的持仓股票及应对策略</li>
                    <li>🎯 重要财报/新闻事件对持仓的影响</li>
                    <li>💡 下周重点关注事项</li>
                </ul>
                
                <h3>🔍 风险监控警报</h3>
                <ul>
                    <li>⚠️ 单只股票仓位过重风险检查</li>
                    <li>📊 行业集中度风险评估</li>
                    <li>💰 止损线触发提醒</li>
                </ul>
                
                <h3>💡 下周投资建议</h3>
                <p>基于技术分析和基本面分析，提供具体的买入/卖出/持有建议。</p>
                
                <div style="margin-top: 15px; padding: 10px; background-color: #fff3cd; border-radius: 5px;">
                    <small>⏰ 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
            </div>
            """
            return weekly_html
        except Exception as e:
            logger.error(f"生成每周分析内容失败: {e}")
            return ""
    
    def _generate_monthly_optimization_content(self):
        """生成每月组合优化特有内容"""
        try:
            monthly_html = f"""
            <div style="background-color: #f0f8f0; padding: 20px; margin: 20px 0; border-radius: 10px;">
                <h2>🎯 每月组合优化分析</h2>
                
                <h3>📊 投资组合绩效评估</h3>
                <ul>
                    <li>📈 月度收益率 vs 基准指数</li>
                    <li>📉 最大回撤分析</li>
                    <li>🎯 夏普比率和风险调整收益</li>
                    <li>⚖️ 投资组合Beta系数</li>
                </ul>
                
                <h3>🔄 仓位调整建议</h3>
                <ul>
                    <li>➕ 建议增持的股票和理由</li>
                    <li>➖ 建议减持的股票和理由</li>
                    <li>🆕 新的投资机会推荐</li>
                    <li>⚠️ 风险资产配置优化</li>
                </ul>
                
                <h3>📅 下月投资日历</h3>
                <ul>
                    <li>📊 重要财报发布日期</li>
                    <li>💰 分红除权日提醒</li>
                    <li>📈 关键技术点位关注</li>
                    <li>🗞️ 重要经济数据发布</li>
                </ul>
                
                <div style="margin-top: 15px; padding: 10px; background-color: #d1ecf1; border-radius: 5px;">
                    <h4>💡 专业投资建议</h4>
                    <p>基于量化分析和专业研究，为您的投资组合提供个性化优化建议。</p>
                    <small>⏰ 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
            </div>
            """
            return monthly_html
        except Exception as e:
            logger.error(f"生成每月优化内容失败: {e}")
            return ""
    
    def _combine_reports(self, base_report, additional_content):
        """合并报告内容"""
        try:
            # 在</body>标签前插入额外内容
            if '</body>' in base_report:
                return base_report.replace('</body>', f"{additional_content}\n</body>")
            else:
                return base_report + additional_content
        except Exception as e:
            logger.error(f"合并报告失败: {e}")
            return base_report
    
    def setup_schedule(self):
        """设置定时任务"""
        try:
            # 每日持股分析 - 每个交易日16:30（美股收盘后30分钟）
            if self.config['daily_enabled']:
                schedule.every().monday.at(self.config['daily_time']).do(self.run_daily_analysis)
                schedule.every().tuesday.at(self.config['daily_time']).do(self.run_daily_analysis)
                schedule.every().wednesday.at(self.config['daily_time']).do(self.run_daily_analysis)
                schedule.every().thursday.at(self.config['daily_time']).do(self.run_daily_analysis)
                schedule.every().friday.at(self.config['daily_time']).do(self.run_daily_analysis)
            
            # 每周深度分析 - 每周日20:00
            if self.config['weekly_enabled']:
                schedule.every().sunday.at(self.config['weekly_time']).do(self.run_weekly_analysis)
            
            # 每月组合优化 - 每月第一个周日20:00
            if self.config['monthly_enabled']:
                schedule.every().month.at(self.config['monthly_time']).do(self.run_monthly_analysis)
            
            logger.info("✅ 持股分析定时任务设置完成")
            
        except Exception as e:
            logger.error(f"设置定时任务失败: {e}")
    
    def start_scheduler(self):
        """启动调度器"""
        try:
            print("=" * 80)
            print("📊 持股分析定时调度器")
            print("=" * 80)
            print()
            print("🎯 调度功能:")
            if self.config['daily_enabled']:
                print(f"   ✓ 每日分析: 交易日 {self.config['daily_time']} (美股收盘后30分钟)")
            if self.config['weekly_enabled']:
                print(f"   ✓ 每周深度分析: 每周日 {self.config['weekly_time']}")
            if self.config['monthly_enabled']:
                print(f"   ✓ 每月组合优化: 每月第一个周日 {self.config['monthly_time']}")
            print()
            
            print("📈 分析内容:")
            print("   • 持仓股票实时表现")
            print("   • 盈亏统计和风险评估")
            print("   • 技术指标和市场情绪")
            print("   • 投资建议和组合优化")
            print()
            
            print(f"📧 邮件接收: {self.config['email']}")
            print(f"🕐 服务启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
            
            # 可选：立即运行一次测试
            test_now = input("💡 是否立即运行一次每日分析测试？(y/N): ").strip().lower()
            if test_now == 'y':
                print("🧪 开始测试运行...")
                self.run_daily_analysis()
            
            print("⏳ 等待定时任务触发...")
            print("   (或按 Ctrl+C 停止服务)")
            
            # 保持运行
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            print("\n👋 持股分析调度服务已停止")
        except Exception as e:
            logger.error(f"调度服务运行出错: {e}")

def main():
    """主函数"""
    scheduler = PortfolioAnalysisScheduler()
    scheduler.start_scheduler()

if __name__ == "__main__":
    main() 