import sys
from datetime import datetime, timedelta
import pandas as pd

# 导入OpenBB
try:
    from openbb import obb
    print("OpenBB导入成功!")
except ImportError as e:
    print(f"导入OpenBB失败: {e}")
    sys.exit(1)

def test_basic_functionality():
    """测试OpenBB的基本功能，不需要API密钥"""
    print("\n=== 测试基本功能 ===")
    
    try:
        # 获取标普500指数数据（不需要API密钥）
        print("获取SPY历史数据...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        spy_data = obb.equity.price.historical(
            symbol="SPY",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        ).to_df()
        
        if not spy_data.empty:
            print(f"成功获取SPY数据: {len(spy_data)}行")
            print(spy_data.head(3))
        else:
            print("获取SPY数据失败或结果为空")
    except Exception as e:
        print(f"测试基本功能时出错: {e}")

def test_api_keys():
    """测试需要API密钥的功能"""
    print("\n=== 测试API密钥 ===")
    
    # 测试FRED API (经济数据)
    try:
        print("测试FRED API...")
        gdp_data = obb.economy.gdp.real().to_df()
        if not gdp_data.empty:
            print("FRED API工作正常! 获取到GDP数据")
            print(gdp_data.tail(3))
        else:
            print("FRED API可能需要设置密钥")
    except Exception as e:
        print(f"测试FRED API时出错: {e}")
    
    # 测试Alpha Vantage API
    try:
        print("\n测试Alpha Vantage API...")
        # 获取公司概览数据
        overview = obb.equity.fundamental.overview(symbol="AAPL").to_df()
        if not overview.empty:
            print("Alpha Vantage API工作正常! 获取到AAPL公司概览")
            print(f"公司名称: {overview.get('Name', ['未知'])[0]}")
            print(f"行业: {overview.get('Industry', ['未知'])[0]}")
        else:
            print("Alpha Vantage API可能需要设置密钥")
    except Exception as e:
        print(f"测试Alpha Vantage API时出错: {e}")
    
    print("\n要使用更多功能，请在credentials.ini文件中添加相应的API密钥")
    print("OpenBB支持许多数据源，部分功能不需要API密钥也可使用")

if __name__ == "__main__":
    print(f"测试OpenBB - 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    test_basic_functionality()
    test_api_keys()
    print("\n测试完成!") 