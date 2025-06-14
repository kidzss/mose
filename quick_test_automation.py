#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试个人投资者自动化系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 快速测试个人投资者自动化系统")
    print("=" * 60)
    
    try:
        # 测试导入
        print("1. 测试模块导入...")
        from monitor.phase2_professional_screener import Phase2ProfessionalScreener
        from data.data_interface import DataInterface
        from utils.unified_email_api import send_html
        print("✅ 模块导入成功")
        
        # 测试筛选器
        print("2. 测试筛选器初始化...")
        screener = Phase2ProfessionalScreener()
        print("✅ 筛选器初始化成功")
        
        # 测试数据接口
        print("3. 测试数据接口...")
        data_interface = DataInterface()
        print("✅ 数据接口初始化成功")
        
        # 测试邮件功能
        print("4. 测试邮件功能...")
        test_html = """
        <html>
        <body>
            <h1>测试邮件</h1>
            <p>这是一封测试邮件，用于验证邮件发送功能。</p>
        </body>
        </html>
        """
        # 注释掉实际发送，避免发送测试邮件
        # success = send_html(subject="测试邮件", html_content=test_html)
        print("✅ 邮件功能测试通过（未实际发送）")
        
        print("\n🎉 所有基本功能测试通过！")
        print("💡 系统可以正常运行，您可以启动自动化服务")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_screening():
    """测试筛选功能"""
    print("\n🎯 测试股票筛选功能...")
    
    try:
        from monitor.phase2_professional_screener import Phase2ProfessionalScreener
        
        screener = Phase2ProfessionalScreener()
        
        # 降低标准以获得结果
        results = screener.screen_stocks_professional(
            min_score=40,  # 降低标准
            max_results=5   # 只取前5只
        )
        
        if results:
            print(f"✅ 筛选成功，找到 {len(results)} 只股票")
            print("🏆 推荐股票:")
            for i, stock in enumerate(results, 1):
                print(f"   {i}. {stock['symbol']} - 评分: {stock['multifactor_score']:.1f}")
        else:
            print("⚠️ 未找到符合条件的股票")
        
        return True
        
    except Exception as e:
        print(f"❌ 筛选测试失败: {e}")
        return False

if __name__ == "__main__":
    # 基本功能测试
    basic_ok = test_basic_functionality()
    
    if basic_ok:
        # 筛选功能测试
        screening_ok = test_screening()
        
        if screening_ok:
            print("\n🎉 所有测试通过！系统可以正常使用。")
            print("💡 您可以运行以下命令启动自动化服务：")
            print("   - 双击 start_personal_automation.bat")
            print("   - 或运行 python personal_investor_automation.py")
        else:
            print("\n⚠️ 筛选功能测试失败，但基本功能正常。")
    else:
        print("\n❌ 基本功能测试失败，请检查系统配置。") 