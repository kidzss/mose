#!/usr/bin/env python3
"""
自动化交易调度器
整合日常投资组合分析和股票筛选功能
"""

import os
import sys
import schedule
import time
import logging
from datetime import datetime, timedelta
import subprocess

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_trading.log'),
        logging.StreamHandler()
    ]
)

class TradingScheduler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = os.path.abspath(os.path.dirname(__file__))
        
    def run_daily_analysis(self):
        """运行每日交易分析"""
        try:
            self.logger.info("🚀 开始每日交易分析...")
            
            # 运行每日交易助手
            result = subprocess.run(
                [sys.executable, os.path.join(self.project_root, 'daily_trading_assistant.py')],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.logger.info("✅ 每日交易分析完成")
                # 记录关键输出
                if "总盈亏" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "总盈亏" in line or "投资建议" in line:
                            self.logger.info(f"📊 {line.strip()}")
            else:
                self.logger.error(f"❌ 每日分析失败: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"❌ 每日分析异常: {e}")
    
    def run_weekly_screening(self):
        """运行每周股票筛选"""
        try:
            self.logger.info("🔍 开始每周股票筛选...")
            
            # 运行增强版股票筛选器
            result = subprocess.run(
                [sys.executable, os.path.join(self.project_root, 'enhanced_stock_screener.py')],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.logger.info("✅ 每周股票筛选完成")
                # 记录关键输出
                if "发现" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "发现" in line or "股票" in line:
                            self.logger.info(f"📈 {line.strip()}")
            else:
                self.logger.error(f"❌ 股票筛选失败: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"❌ 股票筛选异常: {e}")
    
    def run_data_health_check(self):
        """数据健康检查"""
        try:
            self.logger.info("🔧 开始数据健康检查...")
            
            # 检查数据接口
            from data.data_interface import DataInterface
            data_interface = DataInterface()
            
            # 获取市场状态
            market_status = data_interface.get_market_status()
            self.logger.info(f"📊 可用股票数量: {market_status['total_symbols']}")
            self.logger.info(f"🎯 数据质量 - 有效: {market_status['data_quality']['valid']}")
            
            # 检查关键股票数据
            portfolio_symbols = ['AMD', 'GOOGL', 'PFE', 'NVDA', 'TSLA', 'ADBE']
            for symbol in portfolio_symbols:
                try:
                    latest_data = data_interface.get_latest_data(symbol, n_bars=1)
                    if not latest_data.empty:
                        price = latest_data['close'].iloc[-1]
                        self.logger.info(f"✅ {symbol}: ${price:.2f}")
                    else:
                        self.logger.warning(f"⚠️ {symbol}: 无数据")
                except Exception as e:
                    self.logger.error(f"❌ {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ 数据检查异常: {e}")
    
    def setup_schedule(self):
        """设置调度任务"""
        # 每个交易日早上9:00进行数据健康检查
        schedule.every().monday.at("09:00").do(self.run_data_health_check)
        schedule.every().tuesday.at("09:00").do(self.run_data_health_check)
        schedule.every().wednesday.at("09:00").do(self.run_data_health_check)
        schedule.every().thursday.at("09:00").do(self.run_data_health_check)
        schedule.every().friday.at("09:00").do(self.run_data_health_check)
        
        # 每个交易日下午4:30进行每日分析（美股收盘后）
        schedule.every().monday.at("16:30").do(self.run_daily_analysis)
        schedule.every().tuesday.at("16:30").do(self.run_daily_analysis)
        schedule.every().wednesday.at("16:30").do(self.run_daily_analysis)
        schedule.every().thursday.at("16:30").do(self.run_daily_analysis)
        schedule.every().friday.at("16:30").do(self.run_daily_analysis)
        
        # 每周日晚上8:00进行股票筛选
        schedule.every().sunday.at("20:00").do(self.run_weekly_screening)
        
        self.logger.info("⏰ 调度任务设置完成")
        self.logger.info("📅 每日数据检查: 交易日 09:00")
        self.logger.info("📊 每日分析: 交易日 16:30 (美股收盘后)")
        self.logger.info("🔍 每周筛选: 周日 20:00")
    
    def run_scheduler(self):
        """运行调度器"""
        self.setup_schedule()
        
        self.logger.info("🚀 自动化交易调度器启动")
        self.logger.info("按 Ctrl+C 停止调度器")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            self.logger.info("⏹️ 调度器已停止")
    
    def run_manual_test(self):
        """手动测试所有功能"""
        self.logger.info("🧪 开始手动测试...")
        
        print("\n" + "="*50)
        print("🔧 数据健康检查")
        print("="*50)
        self.run_data_health_check()
        
        print("\n" + "="*50)
        print("📊 每日交易分析")
        print("="*50)
        self.run_daily_analysis()
        
        print("\n" + "="*50)
        print("🔍 股票筛选测试")
        print("="*50)
        self.run_weekly_screening()
        
        self.logger.info("✅ 手动测试完成")

def main():
    """主函数"""
    scheduler = TradingScheduler()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # 手动测试模式
            scheduler.run_manual_test()
        elif sys.argv[1] == "--daily":
            # 仅运行每日分析
            scheduler.run_daily_analysis()
        elif sys.argv[1] == "--screen":
            # 仅运行股票筛选
            scheduler.run_weekly_screening()
        elif sys.argv[1] == "--check":
            # 仅运行数据检查
            scheduler.run_data_health_check()
        else:
            print("用法:")
            print("  python automated_trading_scheduler.py           # 启动自动化调度")
            print("  python automated_trading_scheduler.py --test    # 手动测试所有功能")
            print("  python automated_trading_scheduler.py --daily   # 仅运行每日分析")
            print("  python automated_trading_scheduler.py --screen  # 仅运行股票筛选")
            print("  python automated_trading_scheduler.py --check   # 仅运行数据检查")
    else:
        # 启动自动化调度
        scheduler.run_scheduler()

if __name__ == "__main__":
    main() 