#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_macro_analysis():
    """调试宏观分析数据结构"""
    try:
        from monitor.smart_daily_report import SmartDailyReportGenerator
        
        print("🧪 调试宏观分析数据结构...")
        
        generator = SmartDailyReportGenerator(
            watchlist=['AAPL'],
            auto_update_data=False
        )
        
        # 获取宏观分析数据
        macro_analysis = generator._get_macro_analysis()
        
        if macro_analysis:
            print(f"✅ 宏观分析数据获取成功")
            print(f"   数据结构键: {list(macro_analysis.keys())}")
            
            # 检查通胀-行业分析
            if 'inflation_sector_analysis' in macro_analysis:
                inflation_data = macro_analysis['inflation_sector_analysis']
                print(f"✅ 包含通胀-行业分析")
                print(f"   通胀分析键: {list(inflation_data.keys())}")
                
                if 'inflation_environment' in inflation_data:
                    env = inflation_data['inflation_environment']
                    print(f"   通胀环境: {env.get('regime', 'N/A')}")
                    print(f"   信心度: {env.get('confidence', 0):.1%}")
            else:
                print("❌ 未包含通胀-行业分析")
            
            # 测试HTML生成
            print("\n🧪 测试HTML生成...")
            html_content = generator._generate_macro_analysis_html(macro_analysis)
            
            print(f"✅ HTML生成成功，长度: {len(html_content)} 字符")
            
            # 检查关键内容
            inflation_keywords = ['🔥 通胀-行业影响分析', '通胀环境', '行业通胀敏感性分析']
            for keyword in inflation_keywords:
                if keyword in html_content:
                    print(f"   ✅ HTML包含: {keyword}")
                else:
                    print(f"   ❌ HTML缺失: {keyword}")
            
            # 保存调试HTML
            with open('debug_macro_html.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"💾 调试HTML已保存: debug_macro_html.html")
            
        else:
            print("❌ 宏观分析数据获取失败")
            
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_macro_analysis() 