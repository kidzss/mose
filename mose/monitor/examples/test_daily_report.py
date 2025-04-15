import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from monitor.daily_report_task import send_daily_report

def main():
    """测试每日报告功能"""
    print("开始测试每日报告生成和发送...")
    send_daily_report()
    print("测试完成，请检查邮箱")

if __name__ == "__main__":
    main() 