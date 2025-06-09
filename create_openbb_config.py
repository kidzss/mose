import os
from pathlib import Path

# 创建OpenBB配置目录
config_dir = Path.home() / ".openbb"
config_dir.mkdir(exist_ok=True)
print(f"OpenBB配置目录: {config_dir}")

# 创建credentials.ini文件
credentials_file = config_dir / "credentials.ini"

# 检查文件是否存在
if not credentials_file.exists():
    # 创建带有示例API配置的credentials.ini文件
    credentials_content = """[fred]
fred_key = 

[alpha_vantage]
key = 

[finnhub]
key = 

[fmp]
api = 

[polygon]
key = 

[eodhd]
key = 

[intrinio]
key = 
"""
    
    with open(credentials_file, "w", encoding="utf-8") as f:
        f.write(credentials_content)
    
    print(f"已创建credentials.ini文件: {credentials_file}")
    print("请编辑此文件添加您的API密钥")
else:
    print(f"credentials.ini文件已存在: {credentials_file}")
    print("您可以编辑此文件添加或更新API密钥")

print("\n免费API密钥获取指南:")
print("1. FRED (Federal Reserve Economic Data): https://fred.stlouisfed.org/docs/api/api_key.html")
print("2. Alpha Vantage: https://www.alphavantage.co/support/#api-key")
print("3. Finnhub: https://finnhub.io/register")
print("4. FMP (Financial Modeling Prep): https://site.financialmodelingprep.com/developer/docs/")
print("5. Polygon.io: https://polygon.io/dashboard/signup")
print("6. EODHD: https://eodhistoricaldata.com/register")
print("7. Intrinio: https://intrinio.com/starter-plan") 