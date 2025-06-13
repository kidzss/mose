"""
第二阶段专业级筛选器演示测试

展示多因子模型的专业功能
"""

from monitor.phase2_professional_screener import Phase2ProfessionalScreener

def demo_phase2_screener():
    """演示第二阶段筛选器"""
    print("🚀 第二阶段专业级多因子量化筛选器演示")
    print("=" * 80)
    
    screener = Phase2ProfessionalScreener()
    
    # 降低评分标准以获得更多结果
    results = screener.screen_stocks_professional(min_score=40, max_results=15)
    
    if results:
        print(f"\n🎯 发现 {len(results)} 只优质股票:")
        print("=" * 100)
        print(f"{'排名':<4} {'股票':<8} {'多因子评分':<12} {'夏普比率':<10} {'质量因子':<10} {'动量因子':<10} {'价格':<10}")
        print("=" * 100)
        
        for i, stock in enumerate(results, 1):
            print(f"{i:<4} {stock['symbol']:<8} {stock['multifactor_score']:<12.1f} "
                  f"{stock['sharpe_ratio']:<10.2f} {stock['quality_factor']:<10.2f} "
                  f"{stock['momentum_factor']:<10.2f} ${stock['current_price']:<9.2f}")
        
        print("\n📊 评分体系说明:")
        print("• 多因子评分: 综合质量、动量、价值、低波动率等8个因子")
        print("• 夏普比率: 风险调整后收益指标")
        print("• 质量因子: 基于ROE、ROA、债务比率等财务指标")
        print("• 动量因子: 基于价格趋势的动量评分")
        
        # 分析最佳股票
        if results:
            best_stock = results[0]
            print(f"\n🏆 最佳股票分析: {best_stock['symbol']}")
            print(f"   多因子评分: {best_stock['multifactor_score']:.1f}/100")
            print(f"   夏普比率: {best_stock['sharpe_ratio']:.2f}")
            print(f"   最大回撤: {best_stock['max_drawdown']:.2%}")
            print(f"   当前价格: ${best_stock['current_price']:.2f}")
            print(f"   平均成交量: {best_stock['avg_volume']:,.0f}")
    else:
        print("❌ 未找到符合条件的股票，建议降低评分标准")
    
    return results

if __name__ == "__main__":
    demo_phase2_screener() 