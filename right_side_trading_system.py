#!/usr/bin/env python3
"""
右侧交易提醒系统设计
防止左侧抄底被套，强化趋势跟随思维
"""
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class RightSideTradingSystem:
    """右侧交易系统 - 趋势确认后再进入"""
    
    def __init__(self):
        self.trend_confirmation_days = 3  # 趋势确认天数
        self.volume_confirmation_factor = 1.2  # 成交量确认倍数
        self.momentum_threshold = 0.02  # 动量阈值
        self.ma_periods = [5, 10, 20, 50]  # 均线周期
        
    def analyze_trend_confirmation(self, symbol: str) -> Dict:
        """分析趋势确认状态"""
        try:
            # 获取股票数据
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            if data.empty:
                return {"error": "无法获取数据"}
            
            # 计算技术指标
            data = self._calculate_indicators(data)
            
            # 分析右侧交易信号
            trend_status = self._analyze_trend_status(data)
            entry_signals = self._check_entry_signals(data)
            risk_warnings = self._check_left_side_risks(data)
            
            return {
                "symbol": symbol,
                "current_price": data['Close'].iloc[-1],
                "trend_status": trend_status,
                "entry_signals": entry_signals,
                "risk_warnings": risk_warnings,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 移动平均线
        for period in self.ma_periods:
            data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
        
        # 成交量移动平均
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
        
        # 价格动量
        data['Momentum_3d'] = data['Close'].pct_change(periods=3)
        data['Momentum_5d'] = data['Close'].pct_change(periods=5)
        data['Momentum_10d'] = data['Close'].pct_change(periods=10)
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # 布林带
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        return data
    
    def _analyze_trend_status(self, data: pd.DataFrame) -> Dict:
        """分析趋势状态"""
        current = data.iloc[-1]
        prev_3d = data.iloc[-4] if len(data) >= 4 else data.iloc[0]
        
        # 均线排列分析
        ma_alignment = self._check_ma_alignment(current)
        
        # 趋势强度分析
        trend_strength = self._calculate_trend_strength(data)
        
        # 趋势确认状态
        trend_confirmed = self._is_trend_confirmed(data)
        
        return {
            "direction": ma_alignment["direction"],
            "strength": trend_strength,
            "confirmed": trend_confirmed,
            "ma_alignment": ma_alignment,
            "trend_days": self._count_trend_days(data)
        }
    
    def _check_ma_alignment(self, current_data) -> Dict:
        """检查均线排列"""
        mas = [current_data[f'MA_{period}'] for period in self.ma_periods]
        current_price = current_data['Close']
        
        # 多头排列：价格 > MA5 > MA10 > MA20 > MA50
        bullish_alignment = all(mas[i] > mas[i+1] for i in range(len(mas)-1))
        bullish_alignment = bullish_alignment and current_price > mas[0]
        
        # 空头排列：价格 < MA5 < MA10 < MA20 < MA50
        bearish_alignment = all(mas[i] < mas[i+1] for i in range(len(mas)-1))
        bearish_alignment = bearish_alignment and current_price < mas[0]
        
        if bullish_alignment:
            return {"direction": "上升", "quality": "强势", "score": 1.0}
        elif bearish_alignment:
            return {"direction": "下跌", "quality": "弱势", "score": -1.0}
        else:
            return {"direction": "震荡", "quality": "混乱", "score": 0.0}
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict:
        """计算趋势强度"""
        recent_data = data.tail(10)
        
        # 价格动量强度
        momentum_score = 0
        if recent_data['Momentum_3d'].iloc[-1] > self.momentum_threshold:
            momentum_score += 1
        if recent_data['Momentum_5d'].iloc[-1] > self.momentum_threshold:
            momentum_score += 1
        if recent_data['Momentum_10d'].iloc[-1] > self.momentum_threshold:
            momentum_score += 1
        
        # 成交量确认
        volume_confirmed = recent_data['Volume_Ratio'].iloc[-1] > self.volume_confirmation_factor
        
        # RSI趋势
        rsi_trend = "上升" if recent_data['RSI'].iloc[-1] > recent_data['RSI'].iloc[-5] else "下跌"
        
        # MACD趋势
        macd_bullish = recent_data['MACD'].iloc[-1] > recent_data['MACD_Signal'].iloc[-1]
        
        strength_score = momentum_score / 3.0
        if volume_confirmed:
            strength_score += 0.2
        if macd_bullish:
            strength_score += 0.2
        
        if strength_score >= 0.8:
            strength_level = "强"
        elif strength_score >= 0.5:
            strength_level = "中等"
        else:
            strength_level = "弱"
        
        return {
            "level": strength_level,
            "score": strength_score,
            "momentum_score": momentum_score,
            "volume_confirmed": volume_confirmed,
            "rsi_trend": rsi_trend,
            "macd_bullish": macd_bullish
        }
    
    def _is_trend_confirmed(self, data: pd.DataFrame) -> bool:
        """判断趋势是否已确认"""
        recent_data = data.tail(self.trend_confirmation_days + 1)
        
        # 检查连续上涨/下跌
        price_changes = recent_data['Close'].diff().dropna()
        
        # 连续3天同方向且有成交量配合
        consistent_direction = len(price_changes[price_changes > 0]) >= 2 or len(price_changes[price_changes < 0]) >= 2
        volume_support = recent_data['Volume_Ratio'].mean() > 1.0
        
        return consistent_direction and volume_support
    
    def _count_trend_days(self, data: pd.DataFrame) -> int:
        """计算当前趋势持续天数"""
        prices = data['Close'].tail(20)
        ma5 = data['MA_5'].tail(20)
        
        trend_days = 0
        current_trend = None
        
        for i in range(len(prices)):
            if prices.iloc[i] > ma5.iloc[i]:
                if current_trend == "up":
                    trend_days += 1
                else:
                    current_trend = "up"
                    trend_days = 1
            else:
                if current_trend == "down":
                    trend_days += 1
                else:
                    current_trend = "down"
                    trend_days = 1
        
        return trend_days
    
    def _check_entry_signals(self, data: pd.DataFrame) -> Dict:
        """检查右侧交易入场信号"""
        current = data.iloc[-1]
        
        signals = {
            "buy_signals": [],
            "sell_signals": [],
            "wait_signals": []
        }
        
        # 买入信号检查
        if self._check_breakout_signal(data):
            signals["buy_signals"].append("🚀 突破信号：价格突破关键阻力位且有成交量配合")
        
        if self._check_trend_continuation_signal(data):
            signals["buy_signals"].append("📈 趋势延续：上升趋势确认，可考虑加仓")
        
        if self._check_pullback_signal(data):
            signals["buy_signals"].append("🔄 回调买入：健康回调至支撑位，趋势仍完好")
        
        # 卖出信号检查
        if self._check_breakdown_signal(data):
            signals["sell_signals"].append("📉 跌破信号：价格跌破关键支撑位")
        
        if self._check_trend_reversal_signal(data):
            signals["sell_signals"].append("🔄 趋势反转：多重信号显示趋势可能反转")
        
        # 等待信号
        if not signals["buy_signals"] and not signals["sell_signals"]:
            signals["wait_signals"].append("⏳ 趋势不明确，建议等待更清晰的信号")
        
        return signals
    
    def _check_breakout_signal(self, data: pd.DataFrame) -> bool:
        """检查突破信号"""
        current = data.iloc[-1]
        prev_5d = data.tail(5)
        
        # 价格突破20日高点
        price_breakout = current['Close'] > prev_5d['High'].max()
        
        # 成交量放大
        volume_surge = current['Volume_Ratio'] > self.volume_confirmation_factor
        
        # 均线支撑
        ma_support = current['Close'] > current['MA_10']
        
        return price_breakout and volume_surge and ma_support
    
    def _check_trend_continuation_signal(self, data: pd.DataFrame) -> bool:
        """检查趋势延续信号"""
        current = data.iloc[-1]
        
        # 均线多头排列
        ma_bullish = (current['MA_5'] > current['MA_10'] > 
                     current['MA_20'] > current['MA_50'])
        
        # 价格在均线上方
        price_above_ma = current['Close'] > current['MA_5']
        
        # MACD金叉
        macd_bullish = current['MACD'] > current['MACD_Signal']
        
        return ma_bullish and price_above_ma and macd_bullish
    
    def _check_pullback_signal(self, data: pd.DataFrame) -> bool:
        """检查回调买入信号"""
        current = data.iloc[-1]
        recent = data.tail(10)
        
        # 整体趋势向上
        overall_uptrend = current['MA_20'] > data.iloc[-10]['MA_20']
        
        # 短期回调至支撑位
        pullback_to_support = (current['Close'] <= current['MA_10'] and 
                              current['Close'] > current['MA_20'])
        
        # RSI不过度超卖
        rsi_healthy = 35 < current['RSI'] < 65
        
        return overall_uptrend and pullback_to_support and rsi_healthy
    
    def _check_breakdown_signal(self, data: pd.DataFrame) -> bool:
        """检查跌破信号"""
        current = data.iloc[-1]
        
        # 跌破关键支撑
        breakdown = current['Close'] < current['MA_20']
        
        # 成交量放大
        volume_surge = current['Volume_Ratio'] > 1.5
        
        return breakdown and volume_surge
    
    def _check_trend_reversal_signal(self, data: pd.DataFrame) -> bool:
        """检查趋势反转信号"""
        current = data.iloc[-1]
        
        # MACD死叉
        macd_bearish = (current['MACD'] < current['MACD_Signal'] and 
                       data.iloc[-2]['MACD'] >= data.iloc[-2]['MACD_Signal'])
        
        # RSI超买后回落
        rsi_overbought = current['RSI'] > 70
        
        # 价格跌破短期均线
        price_breakdown = current['Close'] < current['MA_5']
        
        return macd_bearish or (rsi_overbought and price_breakdown)
    
    def _check_left_side_risks(self, data: pd.DataFrame) -> List[str]:
        """检查左侧交易风险警告"""
        current = data.iloc[-1]
        warnings = []
        
        # 抄底风险警告
        if current['Close'] < current['MA_50'] and current['RSI'] < 40:
            warnings.append("⚠️ 抄底风险：股价仍在长期均线下方，可能继续下跌")
        
        # 下跌趋势中的反弹
        if (current['MA_5'] < current['MA_20'] < current['MA_50'] and 
            current['Close'] > current['MA_5']):
            warnings.append("⚠️ 反弹陷阱：下跌趋势中的反弹，不建议追高")
        
        # 成交量萎缩的上涨
        if (current['Close'] > data.iloc[-2]['Close'] and 
            current['Volume_Ratio'] < 0.8):
            warnings.append("⚠️ 无量上涨：缺乏成交量支撑的上涨不可持续")
        
        # 高位震荡
        if current['RSI'] > 70 and abs(current['Momentum_5d']) < 0.01:
            warnings.append("⚠️ 高位震荡：RSI超买且动量不足，谨防回调")
        
        return warnings

def generate_right_side_trading_alerts(portfolio_positions: Dict, watchlist: Dict) -> Dict:
    """为投资组合生成右侧交易提醒"""
    
    system = RightSideTradingSystem()
    alerts = {
        "portfolio_alerts": {},
        "watchlist_alerts": {},
        "summary": {
            "total_analyzed": 0,
            "buy_opportunities": 0,
            "sell_warnings": 0,
            "wait_recommendations": 0
        }
    }
    
    # 分析持仓股票
    for symbol, position_info in portfolio_positions.items():
        if symbol == "9999.HK":  # 跳过港股
            continue
            
        analysis = system.analyze_trend_confirmation(symbol)
        if "error" not in analysis:
            alerts["portfolio_alerts"][symbol] = analysis
            alerts["summary"]["total_analyzed"] += 1
            
            # 统计信号类型
            if analysis["entry_signals"]["buy_signals"]:
                alerts["summary"]["buy_opportunities"] += 1
            if analysis["entry_signals"]["sell_signals"]:
                alerts["summary"]["sell_warnings"] += 1
            if analysis["entry_signals"]["wait_signals"]:
                alerts["summary"]["wait_recommendations"] += 1
    
    # 分析观察列表股票
    for symbol, watch_info in watchlist.items():
        analysis = system.analyze_trend_confirmation(symbol)
        if "error" not in analysis:
            alerts["watchlist_alerts"][symbol] = analysis
    
    return alerts

def format_right_side_trading_report(alerts: Dict) -> str:
    """格式化右侧交易报告"""
    
    report_lines = [
        "=" * 60,
        "🎯 右侧交易系统分析报告",
        "=" * 60,
        f"📊 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "💡 右侧交易核心原则:",
        "   • 趋势确认后再进入，不抄底不摸顶",
        "   • 等待突破确认，避免假突破陷阱", 
        "   • 成交量必须配合，无量上涨不追",
        "   • 设置止损位，严格执行纪律",
        "",
        f"📈 分析汇总:",
        f"   总分析股票: {alerts['summary']['total_analyzed']}只",
        f"   买入机会: {alerts['summary']['buy_opportunities']}只",
        f"   卖出警告: {alerts['summary']['sell_warnings']}只", 
        f"   等待建议: {alerts['summary']['wait_recommendations']}只",
        "",
        "📋 持仓股票分析:",
        "=" * 40
    ]
    
    # 持仓股票分析
    for symbol, analysis in alerts["portfolio_alerts"].items():
        trend_status = analysis["trend_status"]
        entry_signals = analysis["entry_signals"]
        risk_warnings = analysis["risk_warnings"]
        
        report_lines.extend([
            f"\n💼 {symbol} - ${analysis['current_price']:.2f}",
            f"   趋势状态: {trend_status['direction']} ({trend_status['strength']['level']})",
            f"   趋势确认: {'✅ 已确认' if trend_status['confirmed'] else '❌ 未确认'}",
            f"   持续天数: {trend_status['trend_days']}天"
        ])
        
        # 买入信号
        if entry_signals["buy_signals"]:
            report_lines.append("   🟢 买入信号:")
            for signal in entry_signals["buy_signals"]:
                report_lines.append(f"     {signal}")
        
        # 卖出信号
        if entry_signals["sell_signals"]:
            report_lines.append("   🔴 卖出信号:")
            for signal in entry_signals["sell_signals"]:
                report_lines.append(f"     {signal}")
        
        # 等待信号
        if entry_signals["wait_signals"]:
            report_lines.append("   🟡 等待信号:")
            for signal in entry_signals["wait_signals"]:
                report_lines.append(f"     {signal}")
        
        # 风险警告
        if risk_warnings:
            report_lines.append("   ⚠️ 风险警告:")
            for warning in risk_warnings:
                report_lines.append(f"     {warning}")
    
    # 观察列表分析
    if alerts["watchlist_alerts"]:
        report_lines.extend([
            "",
            "👀 观察列表分析:",
            "=" * 40
        ])
        
        for symbol, analysis in alerts["watchlist_alerts"].items():
            trend_status = analysis["trend_status"]
            entry_signals = analysis["entry_signals"]
            
            report_lines.extend([
                f"\n🔍 {symbol} - ${analysis['current_price']:.2f}",
                f"   趋势状态: {trend_status['direction']} ({trend_status['strength']['level']})",
                f"   买入时机: {'✅ 合适' if entry_signals['buy_signals'] else '❌ 等待'}"
            ])
            
            if entry_signals["buy_signals"]:
                for signal in entry_signals["buy_signals"][:2]:  # 只显示前2个信号
                    report_lines.append(f"     {signal}")
    
    # 总体建议
    report_lines.extend([
        "",
        "🎯 右侧交易操作建议:",
        "=" * 40,
        "",
        "✅ 执行原则:",
        "   1. 只在趋势确认后进入",
        "   2. 必须有成交量配合",
        "   3. 设置明确的止损位",
        "   4. 分批建仓，控制风险",
        "",
        "❌ 避免行为:",
        "   1. 不要试图抄底摸顶",
        "   2. 不要在下跌趋势中抢反弹",
        "   3. 不要追涨无量的股票",
        "   4. 不要忽视止损信号",
        "",
        "=" * 60
    ])
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    # 测试右侧交易系统
    system = RightSideTradingSystem()
    
    # 测试单只股票
    test_symbols = ["AAPL", "NVDA", "AMD"]
    
    for symbol in test_symbols:
        print(f"\n分析 {symbol}:")
        result = system.analyze_trend_confirmation(symbol)
        
        if "error" not in result:
            print(f"趋势方向: {result['trend_status']['direction']}")
            print(f"趋势强度: {result['trend_status']['strength']['level']}")
            print(f"趋势确认: {result['trend_status']['confirmed']}")
            
            if result['entry_signals']['buy_signals']:
                print("买入信号:")
                for signal in result['entry_signals']['buy_signals']:
                    print(f"  {signal}")
            
            if result['risk_warnings']:
                print("风险警告:")
                for warning in result['risk_warnings']:
                    print(f"  {warning}")
        else:
            print(f"分析失败: {result['error']}")