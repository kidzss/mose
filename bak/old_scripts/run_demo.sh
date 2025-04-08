#!/bin/bash

# 股票交易策略系统演示脚本
# 该脚本运行完整的演示流程

echo "开始运行股票交易策略系统演示..."
echo "正在运行演示脚本..."

# 清理旧的数据
rm -rf demo_data
rm -rf demo_results

# 运行主演示脚本
python stock_alert_system/demo.py

# 检查结果
echo "演示完成！结果保存在 demo_results 目录"
echo "生成的数据保存在 demo_data 目录"

# 列出生成的结果文件
echo ""
echo "生成的结果文件："
ls -la demo_results 