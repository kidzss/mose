"""
邮件配置设置脚本

帮助用户设置邮件发送功能的环境变量
"""

import os
import sys
from utils.email_sender import EmailSender

def setup_email_config():
    """设置邮件配置"""
    print("🔧 邮件功能配置向导")
    print("=" * 50)
    
    print("\n📧 Gmail配置说明:")
    print("1. 使用Gmail账户发送邮件")
    print("2. 需要开启两步验证并生成应用专用密码")
    print("3. 应用专用密码生成地址: https://myaccount.google.com/apppasswords")
    
    print("\n🔑 请输入邮件配置信息:")
    
    # 获取发送方邮箱
    sender_email = input("发送方邮箱 (Gmail): ").strip()
    if not sender_email.endswith('@gmail.com'):
        print("⚠️  建议使用Gmail邮箱以确保兼容性")
    
    # 获取应用专用密码
    print("\n🔐 应用专用密码获取步骤:")
    print("1. 登录 Google 账户")
    print("2. 进入 '安全性' 设置")
    print("3. 开启 '两步验证'")
    print("4. 生成 '应用专用密码'")
    print("5. 选择 '邮件' 应用类型")
    
    sender_password = input("应用专用密码 (16位): ").strip()
    
    # 获取接收方邮箱
    receiver_email = input("接收方邮箱: ").strip()
    
    # 设置环境变量
    os.environ['EMAIL_SENDER'] = sender_email
    os.environ['EMAIL_PASSWORD'] = sender_password
    os.environ['EMAIL_RECEIVER'] = receiver_email
    
    print("\n✅ 邮件配置已设置到当前会话")
    print("💡 如需永久保存，请将以下内容添加到系统环境变量:")
    print(f"EMAIL_SENDER={sender_email}")
    print(f"EMAIL_PASSWORD={sender_password}")
    print(f"EMAIL_RECEIVER={receiver_email}")
    
    # 测试配置
    print("\n🧪 测试邮件配置...")
    email_sender = EmailSender()
    if email_sender.test_email_config():
        print("🎉 邮件配置测试成功！")
        return True
    else:
        print("❌ 邮件配置测试失败，请检查配置信息")
        return False

def test_email_sending():
    """测试邮件发送功能"""
    print("\n📧 测试邮件发送功能...")
    
    email_sender = EmailSender()
    
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
    email_sender = EmailSender()
    if email_sender.test_email_config():
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
    main() 