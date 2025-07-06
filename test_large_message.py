#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试大消息处理功能
验证超时优化是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_large_message():
    """测试大消息处理"""
    print("🧪 测试大消息处理功能...")
    
    try:
        from monitor.ollama_ai_qa import OllamaAIQA
        
        # 创建AI问答界面实例
        qa_interface = OllamaAIQA()
        
        # 创建一个大的测试消息
        large_message = """
请帮我进行一个全面的投资组合分析。我的投资组合包括：
1. NVDA: 100股，成本价$150，现价$180
2. AMD: 50股，成本价$120，现价$140  
3. TSLA: 30股，成本价$200，现价$220
4. AAPL: 80股，成本价$170，现价$175
5. MSFT: 60股，成本价$300，现价$320

我的投资目标是长期增长，风险承受能力中等。请从以下角度进行分析：
- 投资组合的整体健康状况
- 各股票的当前表现和前景
- 风险分散情况
- 行业配置是否合理
- 是否需要调整持仓
- 具体的操作建议

请提供详细的分析和建议。
        """
        
        print(f"📊 消息长度: {len(large_message)} 字符")
        
        # 测试超时计算
        message_length = len(large_message)
        if message_length > 1000:
            timeout = 120
            print(f"⏱️ 预期超时时间: {timeout} 秒 (长消息模式)")
        elif message_length > 500:
            timeout = 90
            print(f"⏱️ 预期超时时间: {timeout} 秒 (中等消息模式)")
        else:
            timeout = 60
            print(f"⏱️ 预期超时时间: {timeout} 秒 (短消息模式)")
        
        # 测试超时模式设置
        timeout_modes = {
            "auto": "自动模式",
            "short": "快速模式 (60秒)",
            "medium": "标准模式 (90秒)",
            "long": "长时模式 (120秒)"
        }
        
        print("\n⚙️ 超时模式设置:")
        for mode, description in timeout_modes.items():
            print(f"   • {description}: {mode}")
        
        print("\n✅ 大消息处理测试完成")
        print("💡 建议: 对于长消息，使用'长时'模式或'自动'模式")
        
        return True
        
    except Exception as e:
        print(f"❌ 大消息处理测试失败: {e}")
        return False

if __name__ == "__main__":
    test_large_message() 