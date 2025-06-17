#!/usr/bin/env python3
"""
邮件配置设置工具
"""

import os
import json
from utils.unified_email_api import UnifiedEmailAPI

def setup_email_config():
    """设置邮件配置"""
    print("📧 邮件配置设置工具")
    print("=" * 50)
    
    # 获取当前配置
    api = UnifiedEmailAPI()
    
    print("当前配置:")
    print(f"  SMTP服务器: {api.smtp_server}")
    print(f"  SMTP端口: {api.smtp_port}")
    print(f"  发送邮箱: {api.sender_email or '未设置'}")
    print(f"  接收邮箱: {api.receiver_email or '未设置'}")
    
    print("\n请设置以下信息:")
    
    # 获取用户输入
    sender_email = input("发送邮箱 (Gmail推荐): ").strip()
    sender_password = input("应用专用密码: ").strip()
    receiver_email = input("接收邮箱: ").strip()
    
    if not all([sender_email, sender_password, receiver_email]):
        print("❌ 所有字段都必须填写")
        return False
    
    # 创建配置目录
    os.makedirs('configs', exist_ok=True)
    
    # 保存配置
    config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': sender_email,
        'sender_password': sender_password,
        'recipient_email': receiver_email
    }
    
    with open('configs/email_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n✅ 配置已保存到 configs/email_config.json")
    print("\n💡 提示:")
    print("1. 确保已开启Gmail的两步验证")
    print("2. 使用应用专用密码而不是普通密码")
    print("3. 可以设置环境变量来覆盖配置:")
    print("   export EMAIL_SENDER=your_email@gmail.com")
    print("   export EMAIL_PASSWORD=your_app_password")
    print("   export EMAIL_RECEIVER=receiver@email.com")
    
    # 测试配置
    print("\n🧪 测试邮件配置...")
    if api.test():
        print("✅ 邮件配置测试成功！")
        return True
    else:
        print("❌ 邮件配置测试失败，请检查设置")
        return False

def test_email_sending():
    """测试邮件发送功能"""
    print("\n📧 测试邮件发送功能...")
    
    email_sender = UnifiedEmailAPI()
    
    # 创建测试数据
    test_results = [
        {
            'symbol': 'AAPL',
            'multifactor_score': 85.5,
            'quality_factor': 0.75,
            'sharpe_ratio': 1.25,
            'max_drawdown': 0.15,
            'current_price': 150.25
        },
        {
            'symbol': 'MSFT',
            'multifactor_score': 82.3,
            'quality_factor': 0.68,
            'sharpe_ratio': 1.18,
            'max_drawdown': 0.18,
            'current_price': 280.50
        }
    ]
    
    test_summary = {
        'total_stocks_analyzed': 573,
        'qualified_stocks_found': 2,
        'high_quality_stocks': 2,
        'medium_quality_stocks': 0,
        'best_stock': 'AAPL',
        'best_score': 85.5
    }
    
    # 发送测试邮件
    success = email_sender.send_screening_results(
        results=test_results,
        summary=test_summary,
        subject="🧪 股票筛选系统 - 邮件功能测试"
    )
    
    if success:
        print("✅ 测试邮件发送成功！请检查您的邮箱")
    else:
        print("❌ 测试邮件发送失败")
    
    return success

def main():
    """主函数"""
    print("🚀 股票筛选系统 - 邮件功能设置")
    print("=" * 60)
    
    # 检查现有配置
    api = UnifiedEmailAPI()
    if api.test():
        print("✅ 检测到现有邮件配置")
        choice = input("是否重新配置? (y/N): ").strip().lower()
        if choice != 'y':
            print("📧 使用现有配置")
            test_email_sending()
            return
    
    # 设置新配置
    if setup_email_config():
        # 测试发送
        choice = input("\n是否发送测试邮件? (Y/n): ").strip().lower()
        if choice != 'n':
            test_email_sending()
    
    print("\n🎯 配置完成！现在可以使用邮件发送功能了")
    print("💡 使用方法:")
    print("   from monitor.phase2_professional_screener import Phase2ProfessionalScreener")
    print("   screener = Phase2ProfessionalScreener()")
    print("   screener.screen_and_email()  # 筛选并发送邮件")

if __name__ == "__main__":
    setup_email_config() 
