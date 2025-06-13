"""
股票筛选器 - 带邮件发送功能

执行专业级股票筛选并自动发送HTML格式的邮件报告
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitor.phase2_professional_screener import Phase2ProfessionalScreener
from datetime import datetime
import argparse

def run_screening_with_email(min_score=50, max_results=25, send_email=True, 
                           custom_subject=None, send_report=False):
    """
    运行股票筛选并发送邮件
    
    Args:
        min_score: 最低评分
        max_results: 最大结果数
        send_email: 是否发送邮件
        custom_subject: 自定义邮件主题
        send_report: 是否同时发送执行报告
    """
    print("🚀 启动专业级股票筛选器 (带邮件功能)")
    print("=" * 80)
    
    # 初始化筛选器
    screener = Phase2ProfessionalScreener()
    
    # 显示配置信息
    print(f"📊 筛选配置:")
    print(f"   • 最低评分: {min_score}")
    print(f"   • 最大结果数: {max_results}")
    print(f"   • 质量因子权重: 30% (重点关注)")
    print(f"   • 邮件发送: {'✅ 启用' if send_email else '❌ 禁用'}")
    print("=" * 80)
    
    try:
        # 执行筛选和邮件发送
        start_time = datetime.now()
        
        if send_email:
            # 使用带邮件功能的筛选
            results = screener.screen_and_email(
                min_score=min_score,
                max_results=max_results,
                send_email=True,
                email_subject=custom_subject
            )
        else:
            # 仅筛选，不发送邮件
            results = screener.screen_stocks_professional(
                min_score=min_score,
                max_results=max_results
            )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 显示结果摘要
        if results:
            print(f"\n🎯 筛选完成！")
            print(f"⏱️  处理时间: {processing_time:.1f} 秒")
            print(f"📈 发现优质股票: {len(results)} 只")
            
            # 显示前5只股票
            print("\n🏆 TOP 5 优质股票:")
            print("-" * 80)
            print(f"{'排名':<4} {'股票':<8} {'评分':<8} {'质量因子':<10} {'夏普比率':<10} {'价格':<10}")
            print("-" * 80)
            
            for i, stock in enumerate(results[:5], 1):
                print(f"{i:<4} {stock['symbol']:<8} {stock['multifactor_score']:<8.1f} "
                      f"{stock['quality_factor']:<10.3f} {stock['sharpe_ratio']:<10.2f} "
                      f"${stock['current_price']:<9.2f}")
            
            # 质量分析
            high_quality = [s for s in results if s['quality_factor'] > 0.6]
            print(f"\n📊 质量分析:")
            print(f"   🏆 高质量股票 (>0.6): {len(high_quality)} 只")
            print(f"   🎯 最佳股票: {results[0]['symbol']} (评分: {results[0]['multifactor_score']:.1f})")
            
            if send_email:
                print(f"\n📧 邮件发送状态: 已发送HTML格式报告")
                print(f"   📨 邮件主题: {custom_subject or '🚀 股票筛选报告'}")
                print(f"   📊 包含内容: 筛选结果表格 + 投资建议 + JSON附件")
            
            # 发送执行报告
            if send_report and os.path.exists('PHASE2_EXECUTION_REPORT.md'):
                print(f"\n📄 发送执行报告...")
                report_success = screener.send_report_email(
                    'PHASE2_EXECUTION_REPORT.md',
                    f"📊 Phase 2 执行报告 | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                if report_success:
                    print("✅ 执行报告邮件发送成功")
                else:
                    print("❌ 执行报告邮件发送失败")
        
        else:
            print("❌ 未找到符合条件的股票")
            print("💡 建议降低最低评分标准或检查市场环境")
        
        return results
        
    except Exception as e:
        print(f"❌ 筛选过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='专业级股票筛选器 (带邮件功能)')
    parser.add_argument('--min-score', type=float, default=50, help='最低评分 (默认: 50)')
    parser.add_argument('--max-results', type=int, default=25, help='最大结果数 (默认: 25)')
    parser.add_argument('--no-email', action='store_true', help='禁用邮件发送')
    parser.add_argument('--subject', type=str, help='自定义邮件主题')
    parser.add_argument('--send-report', action='store_true', help='同时发送执行报告')
    
    args = parser.parse_args()
    
    # 运行筛选
    results = run_screening_with_email(
        min_score=args.min_score,
        max_results=args.max_results,
        send_email=not args.no_email,
        custom_subject=args.subject,
        send_report=args.send_report
    )
    
    print(f"\n✅ 任务完成！")
    if results and not args.no_email:
        print("📧 请检查您的邮箱查看详细的HTML格式报告")

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认配置运行
    if len(sys.argv) == 1:
        print("🎯 使用默认配置运行...")
        run_screening_with_email(
            min_score=50,
            max_results=25,
            send_email=True,
            custom_subject=f"🚀 股票筛选报告 | {datetime.now().strftime('%Y-%m-%d %H:%M')} | 质量导向策略"
        )
    else:
        main() 