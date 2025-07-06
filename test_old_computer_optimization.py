#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
老电脑优化测试脚本
测试新的超时设置是否适合老电脑
"""

import requests
import time
import json
from datetime import datetime

class OldComputerOptimizationTest:
    """老电脑优化测试类"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.test_results = []
    
    def test_connection(self):
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Ollama连接正常")
                return True
            else:
                print(f"❌ Ollama连接失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Ollama连接错误: {e}")
            return False
    
    def test_timeout_settings(self):
        """测试不同超时设置"""
        print("\n🔍 测试超时设置...")
        
        # 测试消息
        test_messages = [
            ("短消息测试", "请简单介绍一下投资的基本原则", 120),  # 2分钟
            ("中等消息测试", "请分析一下当前科技股的投资机会，包括NVDA、AMD、TSLA等股票的技术面和基本面分析", 180),  # 3分钟
            ("长消息测试", "请详细分析我的投资组合：1. NVDA 100股，成本价$150，现价$180；2. AMD 50股，成本价$120，现价$140；3. TSLA 30股，成本价$200，现价$220。请从技术面、基本面、风险评估、操作建议等多个维度进行分析，并给出具体的投资建议。", 300)  # 5分钟
        ]
        
        for message_name, message_content, expected_timeout in test_messages:
            print(f"\n📝 测试: {message_name}")
            print(f"   消息长度: {len(message_content)} 字符")
            print(f"   预期超时: {expected_timeout} 秒")
            
            # 测试自动模式
            result = self._test_single_message(message_content, "auto")
            self.test_results.append({
                'test_name': message_name,
                'mode': 'auto',
                'message_length': len(message_content),
                'expected_timeout': expected_timeout,
                'result': result
            })
    
    def test_extended_timeout(self):
        """测试超长超时模式"""
        print("\n🔍 测试超长超时模式...")
        
        # 创建一个很长的测试消息
        long_message = """
        请详细分析我的完整投资组合和投资策略：

        当前持仓：
        1. NVDA: 100股，成本价$150，现价$180，持仓时间6个月
        2. AMD: 50股，成本价$120，现价$140，持仓时间3个月  
        3. TSLA: 30股，成本价$200，现价$220，持仓时间1年
        4. AAPL: 20股，成本价$170，现价$175，持仓时间2年
        5. MSFT: 15股，成本价$300，现价$320，持仓时间1.5年

        市场环境：
        - 当前市场处于震荡上行阶段
        - 科技股表现强劲，AI概念股领涨
        - 美联储政策相对宽松
        - 通胀数据有所回落

        个人情况：
        - 风险承受能力中等
        - 投资目标：长期稳健增长
        - 可投入资金：$50,000
        - 投资期限：3-5年

        请从以下维度进行全面分析：
        1. 投资组合健康状况评估
        2. 各股票的技术面和基本面分析
        3. 风险分散度评估
        4. 行业配置合理性
        5. 具体操作建议（买入、卖出、持有）
        6. 风险控制措施
        7. 未来投资策略建议
        """
        
        print(f"📝 测试: 超长消息测试")
        print(f"   消息长度: {len(long_message)} 字符")
        print(f"   预期超时: 600 秒 (10分钟)")
        
        result = self._test_single_message(long_message, "extended")
        self.test_results.append({
            'test_name': '超长消息测试',
            'mode': 'extended',
            'message_length': len(long_message),
            'expected_timeout': 600,
            'result': result
        })
    
    def _test_single_message(self, message: str, timeout_mode: str):
        """测试单个消息"""
        try:
            # 根据超时模式设置超时时间
            if timeout_mode == "auto":
                if len(message) > 1000:
                    timeout = 300
                elif len(message) > 500:
                    timeout = 180
                else:
                    timeout = 120
            elif timeout_mode == "extended":
                timeout = 600
            else:
                timeout = 180
            
            print(f"   ⏱️ 实际超时设置: {timeout} 秒")
            
            # 构建请求 - 使用正确的API格式
            request_data = {
                "model": "deepseek-r1:latest",
                "prompt": message,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 4000
                }
            }
            
            # 记录开始时间
            start_time = time.time()
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=timeout
            )
            
            # 计算实际耗时
            actual_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                content = result_data.get('response', '')
                
                print(f"   ✅ 成功 (耗时: {actual_time:.1f}秒)")
                print(f"   📄 回复长度: {len(content)} 字符")
                
                return {
                    'success': True,
                    'actual_time': actual_time,
                    'response_length': len(content),
                    'timeout_used': timeout
                }
            else:
                print(f"   ❌ API错误: {response.status_code}")
                return {
                    'success': False,
                    'error': f"API错误: {response.status_code}",
                    'timeout_used': timeout
                }
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ 超时 (超过{timeout}秒)")
            return {
                'success': False,
                'error': '请求超时',
                'timeout_used': timeout
            }
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'timeout_used': timeout
            }
    
    def test_model_performance(self):
        """测试不同模型的性能"""
        print("\n🔍 测试模型性能...")
        
        # 获取可用模型
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                print(f"📋 可用模型: {available_models}")
                
                # 测试每个模型的性能
                test_message = "请简单介绍一下投资的基本原则"
                
                for model in available_models[:3]:  # 只测试前3个模型
                    print(f"\n🤖 测试模型: {model}")
                    
                    start_time = time.time()
                    
                    try:
                        request_data = {
                            "model": model,
                            "prompt": test_message,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "num_predict": 2000
                            }
                        }
                        
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json=request_data,
                            timeout=300  # 5分钟超时
                        )
                        
                        actual_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result_data = response.json()
                            content = result_data.get('response', '')
                            
                            print(f"   ✅ 成功 (耗时: {actual_time:.1f}秒)")
                            print(f"   📄 回复长度: {len(content)} 字符")
                            
                            self.test_results.append({
                                'test_name': f'模型性能测试 - {model}',
                                'model': model,
                                'actual_time': actual_time,
                                'response_length': len(content),
                                'success': True
                            })
                        else:
                            print(f"   ❌ 失败: {response.status_code}")
                            
                    except Exception as e:
                        print(f"   ❌ 错误: {e}")
                        
            else:
                print("❌ 无法获取模型列表")
                
        except Exception as e:
            print(f"❌ 获取模型列表失败: {e}")
    
    def generate_report(self):
        """生成测试报告"""
        print("\n📊 生成测试报告...")
        
        report = {
            'test_time': datetime.now().isoformat(),
            'total_tests': len(self.test_results),
            'successful_tests': len([r for r in self.test_results if r.get('result', {}).get('success', False)]),
            'failed_tests': len([r for r in self.test_results if not r.get('result', {}).get('success', False)]),
            'results': self.test_results
        }
        
        # 保存报告
        filename = f"old_computer_optimization_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 测试报告已保存: {filename}")
        
        # 打印摘要
        print(f"\n📈 测试摘要:")
        print(f"   总测试数: {report['total_tests']}")
        print(f"   成功测试: {report['successful_tests']}")
        print(f"   失败测试: {report['failed_tests']}")
        print(f"   成功率: {report['successful_tests']/report['total_tests']*100:.1f}%")
        
        # 分析超时情况
        timeout_tests = [r for r in self.test_results if 'timeout_used' in r.get('result', {})]
        if timeout_tests:
            avg_timeout = sum(r['result']['timeout_used'] for r in timeout_tests) / len(timeout_tests)
            print(f"   平均超时设置: {avg_timeout:.0f}秒")
        
        return report
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始老电脑优化测试...")
        print("=" * 50)
        
        # 测试连接
        if not self.test_connection():
            print("❌ Ollama连接失败，无法进行测试")
            return
        
        # 运行各种测试
        self.test_timeout_settings()
        self.test_extended_timeout()
        self.test_model_performance()
        
        # 生成报告
        report = self.generate_report()
        
        print("\n✅ 测试完成!")
        return report


def main():
    """主函数"""
    print("💻 老电脑优化测试工具")
    print("=" * 50)
    
    # 创建测试实例
    tester = OldComputerOptimizationTest()
    
    # 运行测试
    report = tester.run_all_tests()
    
    if report:
        print(f"\n📋 测试结果已保存到: old_computer_optimization_test_*.json")
        print("💡 建议:")
        print("   1. 查看测试报告了解性能表现")
        print("   2. 根据测试结果调整超时设置")
        print("   3. 选择最适合的模型和超时模式")
        print("   4. 老电脑建议使用'超长'模式")


if __name__ == "__main__":
    main() 