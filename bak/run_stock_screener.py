#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量化股票筛选器运行脚本
基于多因子模型的专业量化股票筛选工具
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 直接导入当前目录的模块
from quantitative_stock_screener import (
    QuantitativeStockScreener, 
    ScreeningCriteria, 
    RiskLevel
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StockScreenerRunner")

# 添加邮件发送功能类
class QuantitativeScreenerEmailSender:
    """量化筛选器邮件发送类"""
    
    def __init__(self):
        """初始化邮件发送器"""
        # 邮件配置（使用与智能日报相同的配置）
        self.email_config = {
            'sender_email': 'kidzss@gmail.com',
            'recipient_email': 'kidzss@gmail.com',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_password': 'wlkp dbbz xpgk rkhy'  # App Password
        }
    
    def send_screening_report_email(self, html_report: str, risk_level: str) -> bool:
        """发送量化筛选报告邮件"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # 创建邮件消息
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = f"🎯 量化股票筛选报告 - {risk_level} - {datetime.now().strftime('%Y年%m月%d日')}"
            
            # 添加邮件正文
            msg.attach(MIMEText(html_report, 'html', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                text = msg.as_string()
                server.sendmail(
                    self.email_config['sender_email'],
                    self.email_config['recipient_email'],
                    text
                )
            
            logger.info("✅ 量化筛选报告邮件发送成功！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 邮件发送失败: {e}")
            return False

def run_screening_analysis(
    risk_preference: RiskLevel = RiskLevel.MODERATE,
    custom_symbols: list = None,
    lookback_months: int = 6,
    top_n_results: int = 20,
    send_email: bool = True  # 新增邮件发送选项
):
    """
    运行股票筛选分析
    
    Args:
        risk_preference: 风险偏好 (CONSERVATIVE, MODERATE, AGGRESSIVE, SPECULATIVE)
        custom_symbols: 自定义股票池，None则使用默认股票池
        lookback_months: 回看月数 (3-12个月)
        top_n_results: 显示前N个结果
        send_email: 是否发送邮件报告
    """
    
    print("🚀 启动量化股票筛选器...")
    print(f"📊 风险偏好: {risk_preference.value}")
    print(f"📅 数据回看: {lookback_months}个月")
    print(f"🎯 显示结果: 前{top_n_results}只股票")
    print(f"📧 邮件发送: {'开启' if send_email else '关闭'}")
    print("-" * 60)
    
    try:
        # 创建筛选标准
        criteria = ScreeningCriteria(
            lookback_months=lookback_months,
            min_trading_days=max(120, lookback_months * 20),  # 至少120个交易日
            min_sharpe_ratio=0.8 if risk_preference == RiskLevel.CONSERVATIVE else 0.5,
            max_max_drawdown=0.15 if risk_preference == RiskLevel.CONSERVATIVE else 0.25
        )
        
        # 创建筛选器
        screener = QuantitativeStockScreener(criteria)
        
        # 执行筛选
        print("🔄 正在执行量化分析...")
        results = screener.screen_stocks(
            symbols=custom_symbols,
            risk_preference=risk_preference
        )
        
        if not results:
            print("❌ 未发现符合筛选条件的股票")
            return
        
        # 生成详细报告
        print("📋 正在生成分析报告...")
        report = screener.generate_screening_report(results, top_n=top_n_results)
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        report_filename = f"量化股票筛选报告_{risk_preference.value}_{timestamp}.html"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 发送邮件（如果启用）
        if send_email:
            print("📧 正在发送邮件报告...")
            email_sender = QuantitativeScreenerEmailSender()
            email_success = email_sender.send_screening_report_email(report, risk_preference.value)
            
            if email_success:
                print("✅ 邮件发送成功！请查收邮箱")
            else:
                print("❌ 邮件发送失败，但本地报告已保存")
        
        # 显示摘要结果
        print("✅ 筛选分析完成!")
        print(f"📄 详细报告已保存: {report_filename}")
        if send_email:
            print(f"📧 邮件报告已发送到: kidzss@gmail.com")
        print("-" * 60)
        print("📈 筛选结果摘要:")
        print(f"   总计筛选: {len(results)} 只股票")
        print(f"   推荐前 {min(len(results), 10)} 只:")
        
        for i, stock in enumerate(results[:10], 1):
            risk_emoji = {
                RiskLevel.CONSERVATIVE: "🟢",
                RiskLevel.MODERATE: "🟡", 
                RiskLevel.AGGRESSIVE: "🟠",
                RiskLevel.SPECULATIVE: "🔴"
            }.get(stock.risk_level, "⚪")
            
            print(f"   {i:2d}. {stock.symbol:6s} - "
                  f"{stock.total_score:.2f}分 "
                  f"{risk_emoji} {stock.risk_level.value:4s} "
                  f"夏普:{stock.sharpe_ratio:5.2f} "
                  f"回撤:{stock.max_drawdown:5.1%}")
        
        # 显示投资建议
        if results:
            best_stock = results[0]
            print("-" * 60)
            print("🎯 最佳投资机会:")
            print(f"   股票代码: {best_stock.symbol}")
            print(f"   综合评分: {best_stock.total_score:.2f}分")
            print(f"   风险等级: {best_stock.risk_level.value}")
            print(f"   买入建议: ${best_stock.buy_price:.2f}")
            print(f"   止损价格: ${best_stock.stop_loss:.2f}")
            print(f"   目标价格: ${best_stock.target_price:.2f}")
            print(f"   建议仓位: {best_stock.position_size:.1%}")
            print(f"   市场时机: {best_stock.market_timing}")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ 筛选分析失败: {e}")
        print(f"❌ 执行失败: {str(e)}")
        return None

def run_multi_strategy_screening(send_email: bool = True):
    """运行多策略筛选 - 不同风险偏好的对比分析"""
    
    print("🎯 多策略量化筛选分析")
    print("=" * 80)
    
    risk_levels = [
        RiskLevel.CONSERVATIVE,
        RiskLevel.MODERATE, 
        RiskLevel.AGGRESSIVE
    ]
    
    all_results = {}
    all_reports = []
    
    for risk_level in risk_levels:
        print(f"\n🔄 执行 {risk_level.value} 策略...")
        results = run_screening_analysis(
            risk_preference=risk_level,
            top_n_results=10,
            send_email=False  # 先不单独发送，最后统一发送
        )
        all_results[risk_level] = results
        print("-" * 40)
    
    # 综合分析
    print("\n📊 多策略对比分析:")
    for risk_level, results in all_results.items():
        if results:
            avg_score = sum(s.total_score for s in results[:5]) / 5
            avg_sharpe = sum(s.sharpe_ratio for s in results[:5]) / 5
            print(f"   {risk_level.value:6s}: "
                  f"前5只平均分={avg_score:.2f}, "
                  f"平均夏普={avg_sharpe:.2f}")
    
    # 生成综合报告并发送邮件
    if send_email and any(all_results.values()):
        print("\n📧 正在生成并发送综合邮件报告...")
        
        # 生成综合报告
        comprehensive_report = generate_comprehensive_multi_strategy_report(all_results)
        
        # 发送邮件
        email_sender = QuantitativeScreenerEmailSender()
        email_success = email_sender.send_screening_report_email(comprehensive_report, "多策略对比")
        
        if email_success:
            print("✅ 多策略综合邮件报告发送成功！")
        else:
            print("❌ 综合邮件发送失败")

def generate_comprehensive_multi_strategy_report(all_results: dict) -> str:
    """生成多策略综合报告"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>多策略量化股票筛选对比报告</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; text-align: center; }}
            .strategy-section {{ background: white; margin: 20px 0; padding: 25px; 
                               border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .stock-card {{ border: 1px solid #e0e0e0; margin: 15px 0; padding: 20px; 
                         border-radius: 8px; background: #fafafa; }}
            .score {{ font-size: 1.3em; font-weight: bold; }}
            .high-score {{ color: #28a745; }}
            .medium-score {{ color: #ffc107; }}
            .low-score {{ color: #dc3545; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
                       gap: 15px; margin: 15px 0; }}
            .metric {{ background: white; padding: 15px; border-radius: 8px; text-align: center; 
                     border: 1px solid #e0e0e0; }}
            .comparison-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .comparison-table th, .comparison-table td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
            .comparison-table th {{ background-color: #f8f9fa; font-weight: bold; }}
            .risk-badge {{ padding: 4px 8px; border-radius: 4px; color: white; font-size: 0.9em; }}
            .conservative {{ background-color: #28a745; }}
            .moderate {{ background-color: #ffc107; color: #000; }}
            .aggressive {{ background-color: #fd7e14; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 多策略量化股票筛选对比报告</h1>
            <p style="font-size: 1.1em;">生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
            <p>基于多因子模型的专业量化分析</p>
        </div>
        
        <div class="strategy-section">
            <h2>📊 策略表现对比总览</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>策略类型</th>
                        <th>发现股票数</th>
                        <th>前5只平均评分</th>
                        <th>前5只平均夏普比率</th>
                        <th>最佳股票</th>
                        <th>最佳评分</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # 添加策略对比数据
    for risk_level, results in all_results.items():
        if results and len(results) > 0:
            avg_score = sum(s.total_score for s in results[:5]) / min(5, len(results))
            avg_sharpe = sum(s.sharpe_ratio for s in results[:5]) / min(5, len(results))
            best_stock = results[0]
            
            risk_class = {
                RiskLevel.CONSERVATIVE: "conservative",
                RiskLevel.MODERATE: "moderate", 
                RiskLevel.AGGRESSIVE: "aggressive"
            }.get(risk_level, "moderate")
            
            html_content += f"""
                    <tr>
                        <td><span class="risk-badge {risk_class}">{risk_level.value}</span></td>
                        <td>{len(results)}</td>
                        <td>{avg_score:.2f}</td>
                        <td>{avg_sharpe:.2f}</td>
                        <td>{best_stock.symbol}</td>
                        <td>{best_stock.total_score:.2f}</td>
                    </tr>
            """
    
    html_content += """
                </tbody>
            </table>
        </div>
    """
    
    # 为每个策略添加详细结果
    for risk_level, results in all_results.items():
        if results and len(results) > 0:
            
            risk_class = {
                RiskLevel.CONSERVATIVE: "conservative",
                RiskLevel.MODERATE: "moderate", 
                RiskLevel.AGGRESSIVE: "aggressive"
            }.get(risk_level, "moderate")
            
            html_content += f"""
            <div class="strategy-section">
                <h2><span class="risk-badge {risk_class}">{risk_level.value}</span> 策略筛选结果</h2>
                <p>共发现 <strong>{len(results)}</strong> 只符合条件的股票，以下为前10只：</p>
            """
            
            for i, stock in enumerate(results[:10], 1):
                score_class = ("high-score" if stock.total_score >= 0.8 else 
                              "medium-score" if stock.total_score >= 0.6 else "low-score")
                
                html_content += f"""
                                 <div class="stock-card">
                     <h3>{i}. {stock.symbol} 
                         <span class="score {score_class}">{stock.total_score:.2f}分</span>
                         <span class="risk-badge {risk_class}">{stock.risk_level.value}</span>
                     </h3>
                     
                     <div class="metrics">
                         <div class="metric">
                             <strong>技术分析</strong><br>{stock.technical_score:.2f}
                         </div>
                         <div class="metric">
                             <strong>动量分析</strong><br>{stock.momentum_score:.2f}
                         </div>
                         <div class="metric">
                             <strong>夏普比率</strong><br>{stock.sharpe_ratio:.2f}
                         </div>
                         <div class="metric">
                             <strong>最大回撤</strong><br>{stock.max_drawdown:.1%}
                         </div>
                         <div class="metric">
                             <strong>预期收益</strong><br>{stock.expected_return:.1%}
                         </div>
                         <div class="metric">
                             <strong>胜率</strong><br>{stock.win_rate:.1%}
                         </div>
                     </div>
                     
                     <div style="background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                         <h4 style="margin-top: 0; color: #2c5282;">💰 交易建议</h4>
                         <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                             <div><strong>买入价格:</strong> ${stock.buy_price:.2f}</div>
                             <div><strong>止损价格:</strong> ${stock.stop_loss:.2f}</div>
                             <div><strong>目标价格:</strong> ${stock.target_price:.2f}</div>
                             <div><strong>建议仓位:</strong> {stock.position_size:.1%}</div>
                         </div>
                         <div style="margin-top: 10px;">
                             <strong>盈亏比:</strong> {((stock.target_price - stock.buy_price) / (stock.buy_price - stock.stop_loss)):.1f}:1 |
                             <strong>上涨空间:</strong> {((stock.target_price - stock.buy_price) / stock.buy_price * 100):.1f}% |
                             <strong>下跌风险:</strong> {((stock.buy_price - stock.stop_loss) / stock.buy_price * 100):.1f}%
                         </div>
                     </div>
                     
                     <p><strong>📊 市场环境:</strong> {stock.market_env.value}</p>
                     <p><strong>🎯 推荐策略:</strong> {stock.strategy_recommendation}</p>
                     <p><strong>⭐ 信号质量:</strong> {stock.signal_quality:.2f}</p>
                     <p><strong>⏰ 市场时机:</strong> {stock.market_timing}</p>
                     
                     <p><strong>✅ 优势:</strong> {', '.join(stock.strengths[:3])}</p>
                     <p><strong>⚠️ 风险:</strong> {', '.join(stock.risks[:3])}</p>
                 </div>
                """
            
            html_content += "</div>"
    
    # 添加投资建议
    html_content += f"""
        <div class="strategy-section">
            <h2>💡 综合投资建议</h2>
            
            <h3>🎯 策略选择建议：</h3>
            <ul>
                <li><strong>保守型投资者</strong>：选择保守型策略筛选出的股票，注重稳定性和风险控制</li>
                <li><strong>稳健型投资者</strong>：稳健型策略通常提供最佳的风险调整收益</li>
                <li><strong>激进型投资者</strong>：可选择激进型策略的高评分股票，但需注意风险管理</li>
            </ul>
            
            <h3>📊 组合构建建议：</h3>
            <ul>
                <li><strong>分散投资</strong>：从不同策略中选择股票，构建多元化投资组合</li>
                <li><strong>仓位管理</strong>：根据个人风险承受能力调整单只股票仓位</li>
                <li><strong>定期调整</strong>：建议每月重新运行筛选，动态调整投资组合</li>
            </ul>
            
            <h3>⚠️ 风险提示：</h3>
            <ul>
                <li>本报告仅供参考，不构成投资建议</li>
                <li>投资有风险，入市需谨慎</li>
                <li>建议结合基本面分析进行投资决策</li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
            <p><strong>🤖 AI量化股票筛选系统</strong></p>
            <p>基于多因子模型 | 风险调整收益优化 | 智能信号质量评估</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    """主函数"""
    print("=" * 80)
    print("🎯 量化股票筛选器 - 基于多因子模型的智能选股工具")
    print("=" * 80)
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 单策略筛选 (推荐)")
    print("2. 多策略对比分析")
    print("3. 自定义股票池筛选")
    
    try:
        choice = input("\n输入选择 (1-3, 默认1): ").strip() or "1"
        
        if choice == "1":
            # 单策略筛选
            print("\n请选择风险偏好:")
            print("1. 保守型 (低风险低收益)")
            print("2. 稳健型 (中等风险收益) [推荐]") 
            print("3. 激进型 (高风险高收益)")
            print("4. 投机型 (极高风险收益)")
            
            risk_choice = input("输入选择 (1-4, 默认2): ").strip() or "2"
            risk_mapping = {
                "1": RiskLevel.CONSERVATIVE,
                "2": RiskLevel.MODERATE,
                "3": RiskLevel.AGGRESSIVE, 
                "4": RiskLevel.SPECULATIVE
            }
            
            risk_preference = risk_mapping.get(risk_choice, RiskLevel.MODERATE)
            
            # 回看期选择
            months = input("数据回看月数 (3-12, 默认6): ").strip() or "6"
            try:
                lookback_months = max(3, min(12, int(months)))
            except ValueError:
                lookback_months = 6
            
            run_screening_analysis(
                risk_preference=risk_preference,
                lookback_months=lookback_months
            )
            
        elif choice == "2":
            # 多策略对比
            run_multi_strategy_screening()
            
        elif choice == "3":
            # 自定义股票池
            print("\n请输入股票代码 (用空格分隔, 如: AAPL MSFT GOOGL):")
            symbols_input = input("股票代码: ").strip()
            custom_symbols = symbols_input.split() if symbols_input else None
            
            if custom_symbols:
                print(f"✅ 将分析以下股票: {', '.join(custom_symbols)}")
                run_screening_analysis(
                    custom_symbols=custom_symbols,
                    risk_preference=RiskLevel.MODERATE
                )
            else:
                print("❌ 未输入有效股票代码，使用默认股票池")
                run_screening_analysis()
        
        else:
            print("❌ 无效选择，使用默认模式")
            run_screening_analysis()
            
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")

if __name__ == "__main__":
    main() 