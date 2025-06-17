#!/usr/bin/env python3
"""
简化版每日持股分析脚本
直接运行，无复杂导入
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'monitor'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

# 直接执行monitor目录下的脚本
if __name__ == "__main__":
    print("🚀 启动每日持股分析系统...")
    print("📊 正在生成详细分析报告...")
    
    try:
        # 切换到monitor目录并执行
        os.chdir('monitor')
        os.system('python enhanced_daily_analysis.py')
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        # 备选方案：直接运行宏观分析
        print("🔄 尝试运行宏观分析...")
        os.chdir('..')
        os.system('python run_macro_analysis.py') 