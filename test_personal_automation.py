#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试个人投资者自动化系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personal_investor_automation import PersonalInvestorAutomation

def test_automation():
    """测试自动化系统"""
    print("🧪 测试个人投资者自动化系统")
    print("=" * 60)
    
    try:
        # 创建自动化实例
        automation = PersonalInvestorAutomation()
        
        print("✅ 系统初始化成功")
        print(f"📧 邮件地址: {automation.config['email']}")
        print(f"🎯 风险偏好: {automation.config['risk_tolerance']}")
        print(f"💰 最大仓位: {automation.config['max_position_size']*100}%")
        
        # 测试数据更新
        print("\n📊 测试数据更新...")
        update_success = automation.update_market_data()
        if update_success:
            print("✅ 数据更新测试成功")
        else:
            print("⚠️ 数据更新测试失败")
        
        # 测试每周筛选
        print("\n🎯 测试每周筛选...")
        results = automation.run_weekly_screening()
        if results:
            print(f"✅ 每周筛选测试成功，找到 {len(results)} 只股票")
            print("🏆 前3只推荐股票:")
            for i, stock in enumerate(results[:3], 1):
                print(f"   {i}. {stock['symbol']} - 评分: {stock['multifactor_score']:.1f}")
        else:
            print("⚠️ 每周筛选测试失败")
        
        print("\n🎉 测试完成！")
        print("💡 如果测试成功，您可以运行 start_personal_automation.bat 启动自动化服务")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_automation() 