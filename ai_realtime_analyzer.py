#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI实时分析器
集成Ollama本地大模型到实时交易监控系统
增强版：使用每日持股分析结果
"""

import asyncio
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import re

# 导入每日持股分析器
from daily_holdings_analysis import DailyHoldingsAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIRealtimeAnalyzer:
    """AI实时分析器"""
    
    def __init__(self, 
                 api_endpoint: str = "http://localhost:11434/v1/chat/completions",
                 default_model: str = "deepseek-r1:latest",
                 timeout: int = 30,
                 use_daily_analysis: bool = True):
        """
        初始化AI实时分析器
        
        Args:
            api_endpoint: Ollama API端点
            default_model: 默认模型名称
            timeout: 请求超时时间
            use_daily_analysis: 是否使用每日持股分析结果
        """
        self.api_endpoint = api_endpoint
        self.default_model = default_model
        self.timeout = timeout
        self.use_daily_analysis = use_daily_analysis
        self.analysis_history = []
        
        # 初始化每日持股分析器
        if self.use_daily_analysis:
            try:
                self.daily_analyzer = DailyHoldingsAnalyzer()
                logger.info("✅ 启用每日持股分析功能")
            except Exception as e:
                logger.warning(f"⚠️ 每日持股分析器初始化失败: {e}")
                self.daily_analyzer = None
                self.use_daily_analysis = False
        else:
            self.daily_analyzer = None
            logger.info("ℹ️ 使用基础数据模式")
        
        # 测试连接
        self._test_connection()
    
    def _test_connection(self):
        """测试AI连接"""
        try:
            response = requests.get(
                f"{self.api_endpoint.replace('/v1/chat/completions', '')}/api/tags", 
                timeout=5
            )
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                logger.info(f"✅ AI连接成功，可用模型: {available_models}")
                
                # 检查默认模型是否可用
                if any(self.default_model in model for model in available_models):
                    logger.info(f"✅ 默认模型 {self.default_model} 可用")
                else:
                    logger.warning(f"⚠️ 默认模型 {self.default_model} 不可用，将使用第一个可用模型")
                    self.default_model = available_models[0] if available_models else "deepseek-r1:latest"
            else:
                logger.error(f"❌ AI连接失败: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ AI连接测试失败: {e}")
    
    async def analyze_market_event(self, 
                                 symbol: str, 
                                 event_type: str,
                                 market_data: Dict,
                                 analysis_type: str = "quick") -> Dict[str, Any]:
        """
        分析市场事件
        
        Args:
            symbol: 股票代码
            event_type: 事件类型 (price_alert, volume_spike, technical_signal, etc.)
            market_data: 市场数据
            analysis_type: 分析类型 (quick, detailed, comprehensive)
            
        Returns:
            AI分析结果
        """
        try:
            # 获取每日持股分析结果（如果启用）
            daily_analysis = {}
            if self.use_daily_analysis and self.daily_analyzer:
                try:
                    daily_analysis = self._get_daily_analysis_for_symbol(symbol)
                    logger.info(f"✅ 成功获取 {symbol} 的每日分析数据")
                except Exception as e:
                    logger.warning(f"⚠️ 获取每日分析数据失败: {e}")
            
            # 构建分析提示
            prompt = self._build_daily_analysis_prompt(symbol, event_type, market_data, daily_analysis, analysis_type)
            
            # 发送AI请求
            response = await self._send_ai_request(prompt)
            
            if response:
                # 解析结果
                result = {
                    "success": True,
                    "symbol": symbol,
                    "event_type": event_type,
                    "analysis_type": analysis_type,
                    "timestamp": datetime.now().isoformat(),
                    "ai_analysis": response,
                    "market_data": market_data,
                    "daily_analysis": daily_analysis,
                    "model_used": self.default_model,
                    "action_suggestion": self._extract_action_suggestion(response)
                }
                
                # 保存到历史记录
                self.analysis_history.append(result)
                
                # 保持最近100条记录
                if len(self.analysis_history) > 100:
                    self.analysis_history = self.analysis_history[-100:]
                
                return result
            else:
                return {
                    "success": False,
                    "symbol": symbol,
                    "event_type": event_type,
                    "error": "AI分析失败",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"分析市场事件失败: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "event_type": event_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_daily_analysis_for_symbol(self, symbol: str) -> Dict:
        """获取指定股票的每日分析数据"""
        try:
            # 获取所有需要的股票数据
            all_symbols = list(self.daily_analyzer.portfolio.keys()) + self.daily_analyzer.market_indices + self.daily_analyzer.watchlist
            all_symbols = list(set(all_symbols))  # 去重
            
            data = self.daily_analyzer.get_today_data(all_symbols)
            
            if not data:
                return {}
            
            # 分析投资组合表现
            portfolio_analysis = self.daily_analyzer.analyze_portfolio_performance(data)
            
            # 分析市场环境
            market_analysis = self._format_market_analysis(data)
            
            # 获取目标股票的详细分析
            symbol_analysis = {}
            if symbol in data:
                symbol_data = data[symbol]
                symbol_analysis = {
                    'price': symbol_data['price'],
                    'change_pct': symbol_data['change_pct'],
                    'rsi': symbol_data['rsi'],
                    'volume_ratio': symbol_data['volume_ratio'],
                    'position_52w': symbol_data['position_52w'],
                    'ma_5': symbol_data['ma_5'],
                    'ma_20': symbol_data['ma_20'],
                    'ma_50': symbol_data['ma_50']
                }
            
            return {
                'portfolio_analysis': portfolio_analysis,
                'market_analysis': market_analysis,
                'symbol_analysis': symbol_analysis,
                'current_data': data
            }
            
        except Exception as e:
            logger.error(f"获取每日分析数据失败: {e}")
            return {}
    
    def _format_market_analysis(self, data: Dict) -> Dict:
        """格式化市场分析"""
        market_summary = []
        
        # 主要指数表现
        indices = ['^GSPC', '^IXIC', '^DJI', '^VIX']
        for index in indices:
            if index in data:
                index_data = data[index]
                market_summary.append(f"{index}: {index_data['change_pct']:+.2f}%")
        
        # VIX恐慌指数分析
        vix_analysis = ""
        if '^VIX' in data:
            vix_value = data['^VIX']['price']
            if vix_value < 15:
                vix_analysis = "市场恐慌情绪低，风险偏好较高"
            elif vix_value < 25:
                vix_analysis = "市场恐慌情绪正常"
            else:
                vix_analysis = "市场恐慌情绪较高，需要谨慎"
        
        return {
            'indices_performance': market_summary,
            'vix_analysis': vix_analysis,
            'vix_value': data.get('^VIX', {}).get('price', 0)
        }
    
    def _build_daily_analysis_prompt(self, symbol: str, event_type: str, market_data: Dict, daily_analysis: Dict, analysis_type: str) -> str:
        """构建基于每日分析的分析提示"""
        
        # 基础提示模板
        base_prompt = f"""
# 专业股票投资分析请求 - 基于每日持股分析

## 基本信息
- 股票代码: {symbol}
- 事件类型: {event_type}
- 分析类型: {analysis_type}
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实时市场数据
"""
        
        # 添加基础市场数据
        if 'current_price' in market_data and market_data['current_price'] is not None:
            base_prompt += f"- 当前价格: ${market_data['current_price']:.2f}\n"
        if 'change_pct' in market_data and market_data['change_pct'] is not None:
            base_prompt += f"- 涨跌幅: {market_data['change_pct']:+.2f}%\n"
        if 'volume' in market_data and market_data['volume'] is not None:
            base_prompt += f"- 成交量: {market_data['volume']:,}\n"
        if 'rsi' in market_data and market_data['rsi'] is not None:
            base_prompt += f"- RSI: {market_data['rsi']:.1f}\n"
        if 'macd' in market_data and market_data['macd'] is not None:
            base_prompt += f"- MACD: {market_data['macd']}\n"
        if 'bollinger_position' in market_data and market_data['bollinger_position'] is not None:
            base_prompt += f"- 布林带位置: {market_data['bollinger_position']}\n"
        if 'volume_ratio' in market_data and market_data['volume_ratio'] is not None:
            base_prompt += f"- 成交量比率: {market_data['volume_ratio']:.1f}x\n"
        
        # 添加每日分析数据
        if daily_analysis:
            base_prompt += self._format_daily_analysis_for_prompt(daily_analysis, symbol)
        
        # 添加持仓信息（如果有）
        if 'position_info' in market_data:
            position_info = market_data['position_info']
            base_prompt += f"""
## 持仓信息
- 持股数量: {position_info.get('shares', 0)} 股
- 成本价格: ${position_info.get('cost_basis', 0):.2f}
- 仓位权重: {position_info.get('weight', 0):.2f}%
- 行业板块: {position_info.get('sector', 'Unknown')}
"""
        
        # 添加分析要求
        base_prompt += f"""
## 分析要求
请基于以上数据，对 {symbol} 进行专业的投资分析：

### 短期分析 (1-3天)
- 技术面评估
- 价格趋势判断
- 支撑阻力位

### 中期分析 (1-2周)
- 趋势持续性
- 风险因素

### 长期分析 (1-3个月)
- 基本面展望
- 投资价值

### 操作建议
请给出明确的操作建议：买入、卖出、持有、加仓、减仓
并说明理由

### 风险提示
- 市场风险
- 个股风险

请提供专业、客观的分析，字数控制在300-500字。
"""
        
        return base_prompt
    
    def _format_daily_analysis_for_prompt(self, daily_analysis: Dict, symbol: str) -> str:
        """格式化每日分析数据用于AI提示"""
        prompt_section = "\n## 每日持股分析结果\n"
        
        # 市场环境
        if 'market_analysis' in daily_analysis:
            market = daily_analysis['market_analysis']
            prompt_section += "### 市场环境\n"
            if 'indices_performance' in market:
                prompt_section += f"- 主要指数表现: {' | '.join(market['indices_performance'])}\n"
            if 'vix_analysis' in market:
                prompt_section += f"- VIX分析: {market['vix_analysis']}\n"
            if 'vix_value' in market:
                prompt_section += f"- VIX值: {market['vix_value']:.2f}\n"
        
        # 投资组合分析
        if 'portfolio_analysis' in daily_analysis:
            portfolio = daily_analysis['portfolio_analysis']
            prompt_section += "\n### 投资组合分析\n"
            if isinstance(portfolio, list):
                for item in portfolio:
                    if isinstance(item, dict) and 'symbol' in item:
                        prompt_section += f"- {item['symbol']}: 成本${item.get('cost_price', 0):.2f}, 现价${item.get('current_price', 0):.2f}, 盈亏{item.get('pnl_pct', 0):+.2f}%, RSI{item.get('rsi', 0):.1f}\n"
        
        # 目标股票详细分析
        if 'symbol_analysis' in daily_analysis:
            symbol_analysis = daily_analysis['symbol_analysis']
            if symbol_analysis:
                prompt_section += f"\n### {symbol} 详细分析\n"
                prompt_section += f"- 当前价格: ${symbol_analysis.get('price', 0):.2f}\n"
                prompt_section += f"- 涨跌幅: {symbol_analysis.get('change_pct', 0):+.2f}%\n"
                prompt_section += f"- RSI: {symbol_analysis.get('rsi', 0):.1f}\n"
                prompt_section += f"- 成交量比率: {symbol_analysis.get('volume_ratio', 0):.1f}x\n"
                prompt_section += f"- 52周位置: {symbol_analysis.get('position_52w', 0):.1f}%\n"
                prompt_section += f"- 均线: MA5=${symbol_analysis.get('ma_5', 0):.2f}, MA20=${symbol_analysis.get('ma_20', 0):.2f}, MA50=${symbol_analysis.get('ma_50', 0):.2f}\n"
        
        return prompt_section
    
    async def _send_ai_request(self, prompt: str) -> Optional[str]:
        """发送AI请求"""
        try:
            payload = {
                "model": self.default_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1500,  # 减少token数量
                    "num_predict": 1500  # 明确设置预测数量
                }
            }
            
            logger.info(f"发送AI请求到 {self.default_model}...")
            
            # 使用asyncio发送请求，增加超时时间
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=120  # 增加超时时间到120秒
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    logger.info(f"AI响应成功，长度: {len(content)} 字符")
                    return content
                else:
                    logger.error(f"AI响应格式异常: {result}")
                    return None
            else:
                logger.error(f"AI请求失败: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"AI请求超时 (120秒)")
            return None
        except Exception as e:
            logger.error(f"发送AI请求失败: {e}")
            return None
    
    async def analyze_price_alert(self, symbol: str, current_price: float, 
                                change_pct: float, volume: int = None) -> Dict[str, Any]:
        """分析价格警报"""
        market_data = {
            "current_price": current_price,
            "change_pct": change_pct,
            "volume": volume
        }
        
        event_type = "price_alert"
        if abs(change_pct) > 5:
            event_type = "price_spike"
        elif abs(change_pct) > 2:
            event_type = "price_movement"
        
        return await self.analyze_market_event(symbol, event_type, market_data, "quick")
    
    async def analyze_volume_spike(self, symbol: str, current_price: float, 
                                 volume: int, avg_volume: int) -> Dict[str, Any]:
        """分析成交量异常"""
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        market_data = {
            "current_price": current_price,
            "volume": volume,
            "volume_ratio": volume_ratio,
            "avg_volume": avg_volume
        }
        
        event_type = "volume_spike" if volume_ratio > 2 else "volume_increase"
        
        return await self.analyze_market_event(symbol, event_type, market_data, "detailed")
    
    async def analyze_technical_signal(self, symbol: str, signals: Dict) -> Dict[str, Any]:
        """分析技术信号"""
        market_data = {
            "current_price": signals.get('current_price', 0),
            "technical_signals": signals,
            "rsi": signals.get('rsi', 50),
            "change_pct": signals.get('change_pct', 0)
        }
        
        return await self.analyze_market_event(symbol, "technical_signal", market_data, "comprehensive")
    
    def get_analysis_history(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """获取分析历史"""
        history = self.analysis_history
        
        if symbol:
            history = [h for h in history if h.get('symbol') == symbol]
        
        return history[-limit:] if limit > 0 else history
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        if not self.analysis_history:
            return {"total_analyses": 0}
        
        total = len(self.analysis_history)
        success_count = len([h for h in self.analysis_history if h.get('success', False)])
        
        # 按股票统计
        symbol_stats = {}
        for analysis in self.analysis_history:
            symbol = analysis.get('symbol', 'Unknown')
            if symbol not in symbol_stats:
                symbol_stats[symbol] = 0
            symbol_stats[symbol] += 1
        
        return {
            "total_analyses": total,
            "success_rate": success_count / total if total > 0 else 0,
            "symbol_stats": symbol_stats,
            "last_analysis": self.analysis_history[-1] if self.analysis_history else None
        }

    def _extract_action_suggestion(self, ai_text: str) -> Dict[str, str]:
        """
        从AI回复中提取结构化操作建议
        
        Args:
            ai_text: AI返回的文本
            
        Returns:
            结构化的操作建议
        """
        try:
            # 查找【操作建议】部分
            if "【操作建议】" not in ai_text:
                return self._fallback_extraction(ai_text)
            
            # 提取【操作建议】部分
            start_idx = ai_text.find("【操作建议】")
            end_idx = ai_text.find("【", start_idx + 1)
            if end_idx == -1:
                end_idx = len(ai_text)
            
            suggestion_text = ai_text[start_idx:end_idx]
            
            # 解析各个字段
            result = {}
            
            # 建议操作
            action_match = re.search(r"建议操作[：:]\s*(加仓|减仓|观望|止损|止盈)", suggestion_text)
            if action_match:
                result["action"] = action_match.group(1)
            else:
                result["action"] = "不明确"
            
            # 简单理由
            reason_match = re.search(r"简单理由[：:]\s*(.+?)(?:\n|$)", suggestion_text)
            if reason_match:
                result["reason"] = reason_match.group(1).strip()
            else:
                result["reason"] = "无具体理由"
            
            # 风险提醒
            risk_match = re.search(r"风险提醒[：:]\s*(.+?)(?:\n|$)", suggestion_text)
            if risk_match:
                result["risk_warning"] = risk_match.group(1).strip()
            else:
                result["risk_warning"] = "注意风险"
            
            # 操作时机（详细分析才有）
            timing_match = re.search(r"操作时机[：:]\s*(.+?)(?:\n|$)", suggestion_text)
            if timing_match:
                result["timing"] = timing_match.group(1).strip()
            
            # 仓位建议（综合分析才有）
            position_match = re.search(r"仓位建议[：:]\s*(.+?)(?:\n|$)", suggestion_text)
            if position_match:
                result["position_suggestion"] = position_match.group(1).strip()
            
            return result
            
        except Exception as e:
            logger.error(f"解析操作建议失败: {e}")
            return self._fallback_extraction(ai_text)
    
    def _fallback_extraction(self, ai_text: str) -> Dict[str, str]:
        """兜底提取方法"""
        # 关键词匹配
        actions = ["加仓", "减仓", "观望", "止损", "止盈"]
        found_action = "不明确"
        
        for action in actions:
            if action in ai_text:
                found_action = action
                break
        
        # 简单理由提取
        reason = "AI分析完成"
        if "建议" in ai_text:
            # 提取包含"建议"的句子
            sentences = re.split(r'[。！？\n]', ai_text)
            for sentence in sentences:
                if "建议" in sentence and len(sentence) < 50:
                    reason = sentence.strip()
                    break
        
        return {
            "action": found_action,
            "reason": reason,
            "risk_warning": "注意风险控制"
        }

# 便捷函数
async def quick_analyze_price(symbol: str, price: float, change_pct: float) -> str:
    """快速分析价格变动"""
    analyzer = AIRealtimeAnalyzer()
    result = await analyzer.analyze_price_alert(symbol, price, change_pct)
    
    if result.get('success'):
        return result['ai_analysis']
    else:
        return f"分析失败: {result.get('error', '未知错误')}"

# 测试函数
async def test_ai_analyzer():
    """测试AI分析器"""
    print("🚀 测试AI实时分析器...")
    
    analyzer = AIRealtimeAnalyzer()
    
    # 测试价格警报分析
    print("\n📊 测试价格警报分析...")
    result1 = await analyzer.analyze_price_alert("NVDA", 155.02, 2.5)
    if result1.get('success'):
        print("✅ 价格警报分析成功")
        print(f"AI分析: {result1['ai_analysis'][:200]}...")
        
        # 显示结构化建议
        action_suggestion = result1.get('action_suggestion', {})
        print(f"\n🎯 操作建议:")
        print(f"  建议操作: {action_suggestion.get('action', '不明确')}")
        print(f"  简单理由: {action_suggestion.get('reason', '无')}")
        print(f"  风险提醒: {action_suggestion.get('risk_warning', '无')}")
    else:
        print(f"❌ 价格警报分析失败: {result1.get('error')}")
    
    # 测试成交量异常分析
    print("\n📈 测试成交量异常分析...")
    result2 = await analyzer.analyze_volume_spike("AMD", 59.19, 15000000, 8000000)
    if result2.get('success'):
        print("✅ 成交量异常分析成功")
        print(f"AI分析: {result2['ai_analysis'][:200]}...")
        
        # 显示结构化建议
        action_suggestion = result2.get('action_suggestion', {})
        print(f"\n🎯 操作建议:")
        print(f"  建议操作: {action_suggestion.get('action', '不明确')}")
        print(f"  简单理由: {action_suggestion.get('reason', '无')}")
        print(f"  风险提醒: {action_suggestion.get('risk_warning', '无')}")
        if 'timing' in action_suggestion:
            print(f"  操作时机: {action_suggestion.get('timing', '无')}")
    else:
        print(f"❌ 成交量异常分析失败: {result2.get('error')}")
    
    # 测试技术信号分析
    print("\n🎯 测试技术信号分析...")
    signals = {
        "current_price": 155.02,
        "rsi": 65,
        "change_pct": 2.5,
        "macd": "bullish",
        "bollinger_position": "middle"
    }
    result3 = await analyzer.analyze_technical_signal("TSLA", signals)
    if result3.get('success'):
        print("✅ 技术信号分析成功")
        print(f"AI分析: {result3['ai_analysis'][:200]}...")
        
        # 显示结构化建议
        action_suggestion = result3.get('action_suggestion', {})
        print(f"\n🎯 操作建议:")
        print(f"  建议操作: {action_suggestion.get('action', '不明确')}")
        print(f"  简单理由: {action_suggestion.get('reason', '无')}")
        print(f"  风险提醒: {action_suggestion.get('risk_warning', '无')}")
        if 'timing' in action_suggestion:
            print(f"  操作时机: {action_suggestion.get('timing', '无')}")
        if 'position_suggestion' in action_suggestion:
            print(f"  仓位建议: {action_suggestion.get('position_suggestion', '无')}")
    else:
        print(f"❌ 技术信号分析失败: {result3.get('error')}")
    
    # 显示分析摘要
    summary = analyzer.get_analysis_summary()
    print(f"\n分析摘要: {summary}")

if __name__ == "__main__":
    asyncio.run(test_ai_analyzer()) 