#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek投资分析启动脚本
Start script for DeepSeek investment analysis
"""

import os
import sys
import subprocess
import time

def setup_environment():
    """设置环境变量"""
    print("🔧 设置DeepSeek环境变量...")
    
    # 设置环境变量
    os.environ['AI_API_KEY'] = ""
    os.environ['AI_API_ENDPOINT'] = "http://localhost:11434/v1/chat/completions"
    os.environ['AI_MODEL'] = "deepseek-r1"
    os.environ['OLLAMA_MODELS'] = "E:\\ollama_models"
    
    print("✅ 环境变量设置完成")

def check_ollama_service():
    """检查Ollama服务状态"""
    print("🔍 检查Ollama服务状态...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            print(f"✅ Ollama服务运行正常")
            print(f"📦 可用模型: {model_names}")
            
            # 检查是否有deepseek-r1模型（支持带版本号）
            deepseek_models = [name for name in model_names if 'deepseek-r1' in name]
            if deepseek_models:
                print(f"✅ DeepSeek-r1模型已加载: {deepseek_models[0]}")
                # 更新环境变量使用实际的模型名称
                os.environ['AI_MODEL'] = deepseek_models[0]
                return True
            else:
                print("⚠️ DeepSeek-r1模型未找到，请运行: ollama pull deepseek-r1")
                return False
        else:
            print(f"❌ Ollama服务响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到Ollama服务: {e}")
        return False

def start_ollama_service():
    """启动Ollama服务"""
    print("🚀 启动Ollama服务...")
    
    try:
        # 在后台启动Ollama服务
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # 等待服务启动
        print("⏳ 等待服务启动...")
        time.sleep(5)
        
        return check_ollama_service()
    except Exception as e:
        print(f"❌ 启动Ollama服务失败: {e}")
        return False

def start_investment_analysis():
    """启动投资分析系统"""
    print("🎯 启动投资分析系统...")
    
    try:
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "professional_trading_monitor.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出系统")
    except Exception as e:
        print(f"❌ 启动投资分析系统失败: {e}")

def main():
    """主函数"""
    print("🚀 DeepSeek投资分析系统启动")
    print("=" * 50)
    
    # 1. 设置环境
    setup_environment()
    
    # 2. 检查Ollama服务
    if not check_ollama_service():
        print("\n🔄 尝试启动Ollama服务...")
        if not start_ollama_service():
            print("\n❌ 无法启动Ollama服务")
            print("💡 请手动运行: ollama serve")
            return
    
    # 3. 启动投资分析系统
    print("\n🎉 所有服务就绪，启动投资分析系统...")
    start_investment_analysis()

if __name__ == "__main__":
    main() 