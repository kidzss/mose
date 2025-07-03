#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试AI日志记录功能
演示如何记录发送给AI的所有输入信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ai_logger import AILogger, log_ai_input, log_ai_output, log_ai_error
from utils.ai_logger_decorator import log_ai_interaction, AILoggerMixin
import time

def test_basic_logging():
    """测试基础日志记录功能"""
    print("=" * 60)
    print("测试基础AI日志记录功能")
    print("=" * 60)
    
    # 创建日志记录器
    logger = AILogger(log_dir="logs/test_ai_interactions")
    
    # 模拟发送给AI的输入
    prompt = """
    请分析以下股票的技术指标：
    
    股票代码：AAPL
    当前价格：$150.25
    成交量：45,678,901
    52周最高：$182.94
    52周最低：$124.17
    
    请提供：
    1. 技术分析
    2. 支撑阻力位
    3. 交易建议
    """
    
    context = {
        "symbol": "AAPL",
        "current_price": 150.25,
        "volume": 45678901,
        "high_52w": 182.94,
        "low_52w": 124.17,
        "analysis_type": "technical",
        "timeframe": "daily"
    }
    
    # 记录AI输入
    print("\n1. 记录AI输入...")
    interaction_id = logger.log_ai_input(
        prompt=prompt,
        context=context,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )
    print(f"   交互ID: {interaction_id}")
    
    # 模拟AI响应
    ai_response = """
    基于提供的技术指标，AAPL分析如下：
    
    1. 技术分析：
    - 当前价格处于52周区间的中上位置
    - 相对强弱指标(RSI)显示中性偏强
    - 移动平均线显示上升趋势
    
    2. 支撑阻力位：
    - 主要支撑位：$145.00
    - 次要支撑位：$140.00
    - 主要阻力位：$155.00
    - 次要阻力位：$160.00
    
    3. 交易建议：
    - 短期：谨慎持有，关注$155阻力位突破
    - 中期：如果突破$155，目标$160
    - 风险控制：止损设在$145以下
    """
    
    # 记录AI输出
    print("\n2. 记录AI输出...")
    logger.log_ai_output(
        interaction_id=interaction_id,
        response=ai_response,
        model="gpt-4",
        usage={
            "prompt_tokens": 150,
            "completion_tokens": 200,
            "total_tokens": 350
        }
    )
    
    # 获取会话摘要
    print("\n3. 会话摘要:")
    summary = logger.get_session_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

def test_decorator_logging():
    """测试装饰器日志记录功能"""
    print("\n" + "=" * 60)
    print("测试装饰器AI日志记录功能")
    print("=" * 60)
    
    # 使用装饰器记录AI函数调用
    @log_ai_interaction(model="gpt-4", temperature=0.8)
    def analyze_market_sentiment(market_data: dict, prompt: str, context: dict = None) -> str:
        """分析市场情绪的函数"""
        # 模拟AI分析
        sentiment = "bullish" if market_data.get("trend", "") == "up" else "bearish"
        return f"市场情绪分析：当前市场情绪为{sentiment}，建议相应调整策略。"
    
    # 调用函数
    print("\n1. 调用带装饰器的函数...")
    market_data = {"trend": "up", "volume": "high", "volatility": "medium"}
    context = {"analysis_type": "sentiment", "timeframe": "daily"}
    
    result = analyze_market_sentiment(
        market_data=market_data,
        prompt="请分析当前市场情绪并提供投资建议",
        context=context
    )
    print(f"   分析结果: {result}")

def test_mixin_logging():
    """测试混入类日志记录功能"""
    print("\n" + "=" * 60)
    print("测试混入类AI日志记录功能")
    print("=" * 60)
    
    # 使用混入类
    class PortfolioAnalyzer(AILoggerMixin):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
        
        def analyze_portfolio(self, portfolio_data: dict, prompt: str) -> str:
            """分析投资组合"""
            # 记录输入
            interaction_id = self.log_ai_input(
                prompt=prompt,
                context={
                    "portfolio_data": portfolio_data,
                    "analyzer": self.name,
                    "analysis_type": "portfolio"
                }
            )
            
            try:
                # 模拟AI分析
                total_value = sum(portfolio_data.get("positions", {}).values())
                result = f"投资组合分析：总价值${total_value:,.2f}，建议重新平衡配置。"
                
                # 记录输出
                self.log_ai_output(interaction_id, result)
                
                return result
                
            except Exception as e:
                # 记录错误
                self.log_ai_error(interaction_id, e, error_type="portfolio_analysis_error")
                raise
    
    # 使用混入类
    print("\n1. 使用混入类进行分析...")
    analyzer = PortfolioAnalyzer("PortfolioAI")
    
    portfolio_data = {
        "positions": {
            "AAPL": 1000,
            "MSFT": 800,
            "GOOGL": 500
        },
        "cash": 5000
    }
    
    result = analyzer.analyze_portfolio(
        portfolio_data=portfolio_data,
        prompt="请分析当前投资组合的风险和收益情况"
    )
    print(f"   分析结果: {result}")
    
    # 获取统计信息
    print("\n2. 获取统计信息:")
    stats = analyzer.get_ai_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

def test_error_logging():
    """测试错误日志记录功能"""
    print("\n" + "=" * 60)
    print("测试错误日志记录功能")
    print("=" * 60)
    
    logger = AILogger()
    
    # 记录AI输入
    interaction_id = logger.log_ai_input(
        prompt="请分析一个不存在的股票",
        context={"symbol": "INVALID_STOCK"}
    )
    
    # 模拟AI调用错误
    try:
        raise Exception("股票代码不存在，无法获取数据")
    except Exception as e:
        logger.log_ai_error(
            interaction_id=interaction_id,
            error=e,
            error_type="data_not_found"
        )
        print(f"   已记录错误: {e}")

def test_search_functionality():
    """测试搜索功能"""
    print("\n" + "=" * 60)
    print("测试搜索功能")
    print("=" * 60)
    
    logger = AILogger()
    
    # 搜索包含特定关键词的交互
    print("\n1. 搜索包含'AAPL'的交互:")
    results = logger.search_interactions(keyword="AAPL")
    print(f"   找到 {len(results)} 个相关交互")
    
    for result in results[:3]:  # 只显示前3个
        print(f"   - {result.get('timestamp', 'N/A')}: {result.get('prompt', 'N/A')[:50]}...")
    
    # 搜索特定模型的交互
    print("\n2. 搜索GPT-4模型的交互:")
    results = logger.search_interactions(model="gpt-4")
    print(f"   找到 {len(results)} 个GPT-4交互")

def main():
    """主测试函数"""
    print("🚀 开始测试AI日志记录功能")
    print("=" * 80)
    
    try:
        # 测试基础日志记录
        test_basic_logging()
        
        # 测试装饰器日志记录
        test_decorator_logging()
        
        # 测试混入类日志记录
        test_mixin_logging()
        
        # 测试错误日志记录
        test_error_logging()
        
        # 测试搜索功能
        test_search_functionality()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试完成！")
        print("\n📁 日志文件位置:")
        print("   - 基础日志: logs/test_ai_interactions/")
        print("   - 详细JSON: logs/test_ai_interactions/YYYYMMDD/")
        print("   - 控制台输出: 已显示在屏幕上")
        
        print("\n💡 使用说明:")
        print("1. 查看日志文件了解详细的AI交互记录")
        print("2. 使用search_interactions()函数搜索特定交互")
        print("3. 使用装饰器自动记录函数调用")
        print("4. 使用混入类为类添加日志功能")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 