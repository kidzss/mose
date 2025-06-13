"""
Phase 2 专业级多因子量化筛选器 - 正式运行脚本

重点关注质量因子，使用全量573只股票数据进行专业级分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
from monitor.phase2_professional_screener import Phase2ProfessionalScreener

def run_professional_screening():
    """运行专业级股票筛选"""
    print("🚀 Phase 2 专业级多因子量化筛选器")
    print("=" * 80)
    print("📊 特色功能:")
    print("   • Fama-French五因子模型")
    print("   • 质量因子权重提升至30% (重点关注)")
    print("   • 风险调整收益评估")
    print("   • 全量573只股票分析")
    print("=" * 80)
    
    # 初始化筛选器
    screener = Phase2ProfessionalScreener()
    
    # 设置筛选参数
    min_score = 50.0  # 专业级评分标准
    max_results = 25  # 最多返回25只优质股票
    
    print(f"📈 筛选参数:")
    print(f"   • 最低评分: {min_score}")
    print(f"   • 最大结果数: {max_results}")
    print(f"   • 质量因子权重: 30% (最高)")
    print("=" * 80)
    
    # 开始筛选
    start_time = datetime.now()
    results = screener.screen_stocks_professional(min_score=min_score, max_results=max_results)
    end_time = datetime.now()
    
    # 显示结果
    if results:
        print(f"\n🎯 筛选完成！发现 {len(results)} 只优质股票")
        print(f"⏱️  用时: {(end_time - start_time).total_seconds():.1f} 秒")
        print("=" * 120)
        print(f"{'排名':<4} {'股票':<8} {'多因子评分':<12} {'质量因子':<10} {'夏普比率':<10} {'动量因子':<10} {'最大回撤':<10} {'价格':<10}")
        print("=" * 120)
        
        for i, stock in enumerate(results, 1):
            print(f"{i:<4} {stock['symbol']:<8} {stock['multifactor_score']:<12.1f} "
                  f"{stock['quality_factor']:<10.2f} {stock['sharpe_ratio']:<10.2f} "
                  f"{stock['momentum_factor']:<10.2f} {stock['max_drawdown']:<10.2%} "
                  f"${stock['current_price']:<9.2f}")
        
        # 质量因子分析
        print("\n📊 质量因子分析 (权重30%):")
        print("=" * 80)
        high_quality_stocks = [s for s in results if s['quality_factor'] > 0.6]
        medium_quality_stocks = [s for s in results if 0.3 <= s['quality_factor'] <= 0.6]
        
        print(f"🏆 高质量股票 (质量因子>0.6): {len(high_quality_stocks)} 只")
        for stock in high_quality_stocks[:5]:  # 显示前5只
            print(f"   • {stock['symbol']}: 质量因子 {stock['quality_factor']:.2f}, 评分 {stock['multifactor_score']:.1f}")
        
        print(f"🔶 中等质量股票 (质量因子0.3-0.6): {len(medium_quality_stocks)} 只")
        
        # 最佳股票详细分析
        if results:
            best_stock = results[0]
            print(f"\n🏆 最佳投资标的: {best_stock['symbol']}")
            print("=" * 50)
            print(f"   多因子综合评分: {best_stock['multifactor_score']:.1f}/100")
            print(f"   质量因子评分: {best_stock['quality_factor']:.2f} (权重30%)")
            print(f"   动量因子评分: {best_stock['momentum_factor']:.2f} (权重20%)")
            print(f"   夏普比率: {best_stock['sharpe_ratio']:.2f}")
            print(f"   最大回撤: {best_stock['max_drawdown']:.2%}")
            print(f"   当前价格: ${best_stock['current_price']:.2f}")
            print(f"   平均成交量: {best_stock['avg_volume']:,.0f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase2_professional_screening_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'screening_time': start_time.isoformat(),
                'parameters': {
                    'min_score': min_score,
                    'max_results': max_results,
                    'quality_factor_weight': 0.30
                },
                'summary': {
                    'total_stocks_analyzed': 573,
                    'qualified_stocks_found': len(results),
                    'processing_time_seconds': (end_time - start_time).total_seconds(),
                    'high_quality_stocks': len(high_quality_stocks),
                    'medium_quality_stocks': len(medium_quality_stocks)
                },
                'results': results
            }, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存至: {filename}")
        
        # 投资建议
        print(f"\n💡 专业投资建议:")
        print("=" * 50)
        print("1. 🎯 重点关注质量因子>0.6的股票 (长期稳定)")
        print("2. 📈 结合动量因子选择入场时机")
        print("3. ⚠️  注意最大回撤<20%的股票 (风险控制)")
        print("4. 💰 建议分散投资前10只股票")
        print("5. 🔄 每周重新筛选更新投资组合")
        
    else:
        print("❌ 未找到符合专业标准的股票")
        print("💡 建议:")
        print("   • 降低最低评分标准")
        print("   • 检查市场环境是否适合投资")
        print("   • 等待更好的市场时机")
    
    return results

if __name__ == "__main__":
    try:
        results = run_professional_screening()
        print(f"\n✅ 筛选任务完成！")
    except Exception as e:
        print(f"\n❌ 筛选过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 