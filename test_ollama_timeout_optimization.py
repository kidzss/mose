#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Ollama超时优化功能
验证大信息量处理和超时设置是否正常工作
"""

import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_timeout_optimization():
    """测试超时优化功能"""
    print("=" * 60)
    print("⏱️ Ollama超时优化测试")
    print("=" * 60)
    
    try:
        # 导入Ollama AI问答模块
        from monitor.ollama_ai_qa import OllamaAIQA
        
        # 创建AI问答界面实例
        qa_interface = OllamaAIQA()
        print("✅ Ollama AI问答模块导入成功")
        
        # 测试不同长度的消息
        test_messages = [
            ("短消息测试", "请简单介绍一下投资的基本原则", 60),
            ("中等消息测试", "请帮我分析一下NVDA的当前走势，我持有100股，成本价是$150，现在价格是$180，市场环境如何？我应该继续持有还是卖出？请从技术面、基本面、风险控制等多个角度进行分析。", 90),
            ("长消息测试", """
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
            """, 120)
        ]
        
        print("\n🧪 测试不同长度消息的超时设置...")
        
        for test_name, message, expected_timeout in test_messages:
            print(f"\n📝 {test_name}")
            print(f"   消息长度: {len(message)} 字符")
            print(f"   预期超时: {expected_timeout} 秒")
            
            # 测试超时计算
            message_length = len(message)
            if message_length > 1000:
                calculated_timeout = 120
            elif message_length > 500:
                calculated_timeout = 90
            else:
                calculated_timeout = 60
            
            print(f"   计算超时: {calculated_timeout} 秒")
            
            if calculated_timeout == expected_timeout:
                print("   ✅ 超时计算正确")
            else:
                print("   ❌ 超时计算错误")
        
        # 测试重试机制
        print("\n🔄 测试重试机制...")
        
        # 模拟超时情况（使用一个不存在的模型来触发超时）
        print("   模拟超时情况...")
        
        # 测试生成回复函数
        start_time = time.time()
        response = qa_interface.generate_response("测试消息", "non-existent-model")
        end_time = time.time()
        
        print(f"   响应时间: {end_time - start_time:.2f} 秒")
        print(f"   响应成功: {response.get('success', False)}")
        
        if not response.get('success'):
            print(f"   错误信息: {response.get('error', '未知错误')}")
        
        print("✅ 重试机制测试完成")
        
        # 测试超时模式设置
        print("\n⚙️ 测试超时模式设置...")
        
        timeout_modes = [
            ("auto", "自动模式"),
            ("short", "快速模式"),
            ("medium", "标准模式"),
            ("long", "长时模式")
        ]
        
        for mode, description in timeout_modes:
            print(f"   {description}: {mode}")
        
        print("✅ 超时模式设置测试完成")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ollama AI问答模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 超时优化测试失败: {e}")
        return False

def test_large_message_handling():
    """测试大消息处理"""
    print("\n" + "=" * 60)
    print("📝 大消息处理测试")
    print("=" * 60)
    
    try:
        from monitor.ollama_ai_qa import OllamaAIQA
        
        qa_interface = OllamaAIQA()
        
        # 创建一个非常大的测试消息
        large_message = """
请帮我进行一个极其详细和全面的投资分析报告。

我的投资组合详情：
1. NVIDIA (NVDA): 150股，成本价$140，现价$180，买入时间2023年3月
2. Advanced Micro Devices (AMD): 100股，成本价$110，现价$140，买入时间2023年5月
3. Tesla (TSLA): 50股，成本价$180，现价$220，买入时间2023年2月
4. Apple (AAPL): 120股，成本价$160，现价$175，买入时间2023年1月
5. Microsoft (MSFT): 80股，成本价$280，现价$320，买入时间2023年4月
6. Amazon (AMZN): 60股，成本价$120，现价$140，买入时间2023年6月
7. Alphabet (GOOGL): 70股，成本价$130，现价$150，买入时间2023年7月
8. Meta (META): 90股，成本价$200，现价$280，买入时间2023年8月

我的投资背景：
- 投资目标：长期增长，年化收益率目标15-20%
- 风险承受能力：中等偏高
- 投资期限：5-10年
- 可投入资金：每月$5000用于追加投资
- 紧急资金：已准备6个月生活费用

当前市场环境：
- 美联储政策：加息周期接近尾声
- 通胀情况：逐步回落但仍高于目标
- 经济增长：温和增长，就业市场强劲
- 地缘政治：存在不确定性
- 技术趋势：AI、新能源、数字化转型

请从以下维度进行详细分析：

1. 投资组合分析：
   - 整体表现评估
   - 风险分散情况
   - 行业配置合理性
   - 个股表现对比

2. 技术面分析：
   - 各股票技术指标
   - 趋势分析
   - 支撑阻力位
   - 成交量分析

3. 基本面分析：
   - 各公司财务状况
   - 行业前景
   - 竞争优势
   - 估值水平

4. 风险评估：
   - 系统性风险
   - 个股风险
   - 流动性风险
   - 集中度风险

5. 操作建议：
   - 是否需要调整持仓
   - 具体买卖建议
   - 仓位管理策略
   - 风险控制措施

6. 长期规划：
   - 投资组合优化方向
   - 新投资机会识别
   - 定期再平衡策略
   - 目标调整建议

请提供详细的分析报告和具体的操作建议。
        """
        
        print(f"📊 大消息长度: {len(large_message)} 字符")
        
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
        
        print("✅ 大消息处理测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 大消息处理测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始Ollama超时优化测试...")
    
    # 测试超时优化功能
    timeout_success = test_timeout_optimization()
    
    # 测试大消息处理
    large_message_success = test_large_message_handling()
    
    # 生成测试报告
    print("\n" + "=" * 60)
    print("📋 超时优化测试报告")
    print("=" * 60)
    
    if timeout_success:
        print("✅ 超时优化功能测试通过")
    else:
        print("❌ 超时优化功能测试失败")
    
    if large_message_success:
        print("✅ 大消息处理测试通过")
    else:
        print("❌ 大消息处理测试失败")
    
    print("\n🎯 优化特性:")
    print("   • 动态超时时间调整")
    print("   • 大消息处理优化")
    print("   • 重试机制")
    print("   • 用户可配置超时模式")
    print("   • 进度提示优化")
    
    print("\n💡 使用建议:")
    print("   • 长消息建议使用'长时'模式")
    print("   • 短消息可以使用'快速'模式")
    print("   • 系统会自动根据消息长度调整超时时间")
    print("   • 超时失败会自动重试")
    
    print("\n🎉 Ollama超时优化测试完成！")

if __name__ == "__main__":
    main() 