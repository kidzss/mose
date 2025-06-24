#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日报输出，验证通胀-行业分析是否正确显示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_daily_report_output():
    """测试日报输出，检查通胀-行业分析"""
    try:
        from monitor.smart_daily_report import SmartDailyReportGenerator
        
        print("🧪 测试日报生成器输出...")
        
        # 创建报告生成器（只测试少量股票）
        generator = SmartDailyReportGenerator(
            watchlist=['AAPL'],  # 只测试一只股票以加快速度
            auto_update_data=True
        )
        
        # 生成报告
        html_report = generator.generate_report()
        
        if html_report:
            # 检查是否包含通胀-行业分析
            inflation_keywords = [
                '🔥 通胀-行业影响分析',
                '通胀环境',
                '行业通胀敏感性分析',
                '通胀环境投资建议'
            ]
            
            found_keywords = []
            for keyword in inflation_keywords:
                if keyword in html_report:
                    found_keywords.append(keyword)
            
            print(f"✅ 报告生成成功，长度: {len(html_report)} 字符")
            print(f"✅ 找到通胀分析关键词: {len(found_keywords)}/{len(inflation_keywords)}")
            
            for keyword in found_keywords:
                print(f"   ✅ 包含: {keyword}")
            
            missing_keywords = set(inflation_keywords) - set(found_keywords)
            if missing_keywords:
                print(f"❌ 缺失关键词: {list(missing_keywords)}")
            
            # 保存报告文件以供检查
            output_file = "test_daily_report_output.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            print(f"💾 报告已保存到: {output_file}")
            
            # 检查是否包含基础的宏观分析
            basic_macro_keywords = [
                '🌍 宏观环境分析',
                '📊 行业影响分析'
            ]
            
            print(f"\n📊 基础宏观分析检查:")
            for keyword in basic_macro_keywords:
                if keyword in html_report:
                    print(f"   ✅ 包含: {keyword}")
                else:
                    print(f"   ❌ 缺失: {keyword}")
            
            return True
        else:
            print("❌ 报告生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_daily_report_output() 