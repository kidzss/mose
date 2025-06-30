#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI连接测试脚本
Test AI Connection Script
"""

import requests
import json
import asyncio
from datetime import datetime

def test_ai_connection():
    """测试AI连接"""
    print("🧪 测试AI连接...")
    print("=" * 40)
    
    # 测试基础连接
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            available_models = [model['name'] for model in models.get('models', [])]
            print(f"✅ Ollama服务正常")
            print(f"📊 可用模型: {available_models}")
        else:
            print(f"❌ Ollama服务异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到Ollama: {e}")
        return False
    
    # 测试简单AI请求
    print(f"\n🤖 测试简单AI请求...")
    
    test_prompt = "请简单回答：1+1等于几？"
    
    payload = {
        "model": "deepseek-r1:latest",
        "messages": [
            {
                "role": "user",
                "content": test_prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100
        }
    }
    
    try:
        print(f"📤 发送测试请求...")
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json=payload,
            timeout=30  # 30秒超时
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                ai_response = result['choices'][0]['message']['content']
                print(f"✅ AI响应成功!")
                print(f"📝 回答: {ai_response}")
                return True
            else:
                print(f"❌ AI响应格式异常: {result}")
                return False
        else:
            print(f"❌ AI请求失败: {response.status_code}")
            print(f"📄 响应内容: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ AI请求超时 (30秒)")
        return False
    except Exception as e:
        print(f"❌ AI请求异常: {e}")
        return False

def test_different_models():
    """测试不同模型"""
    print(f"\n🧪 测试不同模型...")
    
    models = ["deepseek-r1:latest", "qwen2.5-coder:14b"]
    
    for model in models:
        print(f"\n📊 测试模型: {model}")
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "你好，请简单介绍一下自己"
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 50
            }
        }
        
        try:
            response = requests.post(
                "http://localhost:11434/v1/chat/completions",
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    ai_response = result['choices'][0]['message']['content']
                    print(f"✅ {model} 响应成功")
                    print(f"📝 回答: {ai_response[:100]}...")
                else:
                    print(f"❌ {model} 响应格式异常")
            else:
                print(f"❌ {model} 请求失败: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"❌ {model} 请求超时")
        except Exception as e:
            print(f"❌ {model} 请求异常: {e}")

def main():
    """主函数"""
    print("🚀 AI连接测试")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 基础连接测试
    if test_ai_connection():
        print(f"\n✅ 基础连接测试通过")
        
        # 测试不同模型
        test_different_models()
        
        print(f"\n🎉 所有测试完成!")
    else:
        print(f"\n❌ 基础连接测试失败")
        print(f"💡 建议检查:")
        print(f"   1. Ollama服务是否正在运行")
        print(f"   2. 端口11434是否被占用")
        print(f"   3. 防火墙设置")

if __name__ == "__main__":
    main() 