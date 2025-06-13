"""
保存Phase 2专业级筛选结果
"""

import json
from datetime import datetime
from monitor.phase2_professional_screener import Phase2ProfessionalScreener

def save_results():
    """保存筛选结果"""
    print("💾 保存Phase 2专业级筛选结果...")
    
    screener = Phase2ProfessionalScreener()
    results = screener.screen_stocks_professional(min_score=50.0, max_results=25)
    
    if results:
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase2_professional_screening_{timestamp}.json"
        
        # 分析质量因子
        high_quality_stocks = [s for s in results if s['quality_factor'] > 0.6]
        medium_quality_stocks = [s for s in results if 0.3 <= s['quality_factor'] <= 0.6]
        
        result_data = {
            'screening_time': datetime.now().isoformat(),
            'parameters': {
                'min_score': 50.0,
                'max_results': 25,
                'quality_factor_weight': 0.30
            },
            'summary': {
                'total_stocks_analyzed': 573,
                'qualified_stocks_found': len(results),
                'high_quality_stocks': len(high_quality_stocks),
                'medium_quality_stocks': len(medium_quality_stocks),
                'best_stock': results[0]['symbol'] if results else None,
                'best_score': results[0]['multifactor_score'] if results else 0
            },
            'results': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存至: {filename}")
        print(f"📊 筛选摘要:")
        print(f"   • 发现优质股票: {len(results)} 只")
        print(f"   • 高质量股票: {len(high_quality_stocks)} 只")
        print(f"   • 最佳股票: {results[0]['symbol']} (评分: {results[0]['multifactor_score']:.1f})")
        
        return filename
    else:
        print("❌ 没有结果可保存")
        return None

if __name__ == "__main__":
    save_results() 