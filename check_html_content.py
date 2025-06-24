#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    with open('test_daily_report_output.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"文件长度: {len(content)} 字符")
    print(f"包含'通胀': {'通胀' in content}")
    print(f"包含'🔥 通胀-行业影响分析': {'🔥 通胀-行业影响分析' in content}")
    print(f"包含'宏观环境分析': {'宏观环境分析' in content}")
    
    # 检查宏观分析相关部分
    macro_keywords = ['宏观环境分析', '行业影响分析', 'macro_analysis', 'inflation_sector_analysis']
    for keyword in macro_keywords:
        count = content.count(keyword)
        print(f"'{keyword}': {count} 次")
    
except Exception as e:
    print(f"错误: {e}") 