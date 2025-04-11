import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import talib
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    """市场状态"""
    trend: str  # 上升/下降/震荡
    strength: float  # 趋势强度 0-1
    volatility: float  # 波动率
    risk_level: str  # 低/中/高
    opportunity_sectors: List[str]  # 机会行业
    timestamp: datetime

@dataclass
class MarketCycle:
    """市场周期"""
    current_phase: str  # 积累期/上升期/分配期/下降期
    duration: int  # 当前阶段持续天数
    confidence: float  # 判断的置信度 0-1

class MarketAnalyzer:
    """市场分析器"""
    
    def __init__(self, data_manager=None):
        self.data_manager = data_manager
        self.market_indicators = {}
        self.sector_performance = {}
        self.last_update = None
        
    def analyze_market(self, index_data: pd.DataFrame) -> MarketState:
        """分析市场状态"""
        try:
            # 1. 趋势分析
            trend, strength = self._analyze_trend(index_data)
            
            # 2. 波动率分析
            volatility = self._calculate_volatility(index_data)
            
            # 3. 风险水平评估
            risk_level = self._evaluate_risk(index_data, volatility)
            
            # 4. 行业机会分析
            opportunities = self._analyze_sector_opportunities()
            
            market_state = MarketState(
                trend=trend,
                strength=strength,
                volatility=volatility,
                risk_level=risk_level,
                opportunity_sectors=opportunities,
                timestamp=datetime.now()
            )
            
            return market_state
            
        except Exception as e:
            logger.error(f"分析市场状态时出错: {e}")
            return None
            
    def identify_market_cycle(self, index_data: pd.DataFrame) -> MarketCycle:
        """识别市场周期"""
        try:
            # 1. 计算技术指标
            ma_short = talib.SMA(index_data['Close'], timeperiod=20)
            ma_long = talib.SMA(index_data['Close'], timeperiod=60)
            rsi = talib.RSI(index_data['Close'], timeperiod=14)
            
            # 2. 判断周期阶段
            if ma_short.iloc[-1] > ma_long.iloc[-1] and rsi.iloc[-1] < 70:
                phase = "上升期"
            elif ma_short.iloc[-1] > ma_long.iloc[-1] and rsi.iloc[-1] >= 70:
                phase = "分配期"
            elif ma_short.iloc[-1] < ma_long.iloc[-1] and rsi.iloc[-1] > 30:
                phase = "下降期"
            else:
                phase = "积累期"
                
            # 3. 计算持续时间
            duration = self._calculate_phase_duration(index_data, phase)
            
            # 4. 计算置信度
            confidence = self._calculate_cycle_confidence(index_data, phase)
            
            return MarketCycle(
                current_phase=phase,
                duration=duration,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"识别市场周期时出错: {e}")
            return None
            
    def _analyze_trend(self, data: pd.DataFrame) -> Tuple[str, float]:
        """分析趋势和强度"""
        try:
            # 计算多个时间周期的均线
            ma_periods = [5, 10, 20, 60]
            mas = {p: talib.SMA(data['Close'], timeperiod=p) for p in ma_periods}
            
            # 判断趋势
            last_close = data['Close'].iloc[-1]
            ma_alignment = [last_close > ma.iloc[-1] for ma in mas.values()]
            
            if all(ma_alignment):
                trend = "上升"
                strength = 1.0
            elif not any(ma_alignment):
                trend = "下降"
                strength = 1.0
            else:
                trend = "震荡"
                strength = sum(ma_alignment) / len(ma_alignment)
                
            return trend, strength
            
        except Exception as e:
            logger.error(f"分析趋势时出错: {e}")
            return "未知", 0.0
            
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """计算波动率"""
        try:
            returns = data['Close'].pct_change()
            return returns.std() * np.sqrt(252)  # 年化波动率
        except Exception as e:
            logger.error(f"计算波动率时出错: {e}")
            return 0.0
            
    def _evaluate_risk(self, data: pd.DataFrame, volatility: float) -> str:
        """评估风险水平"""
        try:
            # 1. 波动率评分
            vol_score = 1 if volatility > 0.3 else (2 if volatility > 0.2 else 3)
            
            # 2. 趋势强度评分
            trend_score = self._calculate_trend_strength(data)
            
            # 3. 市场宽度评分
            breadth_score = self._calculate_market_breadth()
            
            # 综合评分
            total_score = (vol_score + trend_score + breadth_score) / 3
            
            if total_score < 1.5:
                return "高"
            elif total_score < 2.5:
                return "中"
            else:
                return "低"
                
        except Exception as e:
            logger.error(f"评估风险水平时出错: {e}")
            return "未知"
            
    def _analyze_sector_opportunities(self) -> List[str]:
        """分析行业机会"""
        try:
            if not self.sector_performance:
                return []
                
            # 计算行业相对强度
            sector_rs = {}
            for sector, perf in self.sector_performance.items():
                # 计算相对大盘的强度
                rs = perf['return'] - self.market_indicators.get('market_return', 0)
                sector_rs[sector] = rs
                
            # 选择相对强度最高的3个行业
            top_sectors = sorted(
                sector_rs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            return [sector for sector, _ in top_sectors]
            
        except Exception as e:
            logger.error(f"分析行业机会时出错: {e}")
            return []
            
    def _calculate_phase_duration(self, data: pd.DataFrame, current_phase: str) -> int:
        """计算当前阶段持续时间"""
        try:
            # 简单实现：计算均线交叉后的持续天数
            ma_short = talib.SMA(data['Close'], timeperiod=20)
            ma_long = talib.SMA(data['Close'], timeperiod=60)
            
            cross_idx = None
            for i in range(len(data)-1, 0, -1):
                if current_phase in ["上升期", "分配期"]:
                    if ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i-1] <= ma_long.iloc[i-1]:
                        cross_idx = i
                        break
                else:
                    if ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i-1] >= ma_long.iloc[i-1]:
                        cross_idx = i
                        break
                        
            return len(data) - cross_idx if cross_idx else 0
            
        except Exception as e:
            logger.error(f"计算阶段持续时间时出错: {e}")
            return 0
            
    def _calculate_cycle_confidence(self, data: pd.DataFrame, phase: str) -> float:
        """计算周期判断的置信度"""
        try:
            # 1. 趋势确认度 (0.4)
            trend_conf = self._calculate_trend_confidence(data)
            
            # 2. 成交量确认度 (0.3)
            volume_conf = self._calculate_volume_confidence(data)
            
            # 3. 市场宽度确认度 (0.3)
            breadth_conf = self._calculate_breadth_confidence()
            
            # 加权平均
            confidence = (
                0.4 * trend_conf +
                0.3 * volume_conf +
                0.3 * breadth_conf
            )
            
            return min(max(confidence, 0), 1)
            
        except Exception as e:
            logger.error(f"计算周期置信度时出错: {e}")
            return 0.5
            
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """计算趋势强度"""
        try:
            # 使用ADX指标衡量趋势强度
            adx = talib.ADX(
                data['High'],
                data['Low'],
                data['Close'],
                timeperiod=14
            )
            
            last_adx = adx.iloc[-1]
            if last_adx > 40:
                return 3  # 强趋势
            elif last_adx > 20:
                return 2  # 中等趋势
            else:
                return 1  # 弱趋势
                
        except Exception as e:
            logger.error(f"计算趋势强度时出错: {e}")
            return 2
            
    def _calculate_market_breadth(self) -> float:
        """计算市场宽度"""
        try:
            if not self.market_indicators:
                return 2
                
            advance_decline = self.market_indicators.get('advance_decline', 1)
            new_highs_lows = self.market_indicators.get('new_highs_lows', 1)
            
            if advance_decline > 2 and new_highs_lows > 2:
                return 3  # 市场很健康
            elif advance_decline < 0.5 and new_highs_lows < 0.5:
                return 1  # 市场不健康
            else:
                return 2  # 市场一般
                
        except Exception as e:
            logger.error(f"计算市场宽度时出错: {e}")
            return 2
            
    def update_market_indicators(self, indicators: Dict):
        """更新市场指标"""
        self.market_indicators = indicators
        self.last_update = datetime.now()
        
    def update_sector_performance(self, performance: Dict):
        """更新行业表现"""
        self.sector_performance = performance
        
    def get_market_summary(self) -> Dict:
        """获取市场综述"""
        if not self.last_update or \
           datetime.now() - self.last_update > timedelta(minutes=5):
            return None
            
        return {
            'indicators': self.market_indicators,
            'sectors': self.sector_performance,
            'last_update': self.last_update
        }

    def analyze_sector_correlation(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """分析板块联动性"""
        try:
            # 按行业分组
            tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META']
            semi_stocks = ['NVDA', 'AMD', 'ASML']
            consumer_stocks = ['TSLA', 'AMZN']
            
            # 计算各组收益率
            returns = {}
            for symbol, df in data.items():
                if 'Close' in df.columns:
                    returns[symbol] = df['Close'].pct_change()
            
            # 计算相关性
            correlations = {}
            
            # 科技股相关性
            tech_returns = pd.DataFrame({s: returns[s] for s in tech_stocks if s in returns})
            if not tech_returns.empty:
                tech_corr = tech_returns.corr().mean().mean()
                correlations['tech'] = tech_corr
            
            # 半导体股相关性
            semi_returns = pd.DataFrame({s: returns[s] for s in semi_stocks if s in returns})
            if not semi_returns.empty:
                semi_corr = semi_returns.corr().mean().mean()
                correlations['semi'] = semi_corr
            
            # 消费股相关性
            consumer_returns = pd.DataFrame({s: returns[s] for s in consumer_stocks if s in returns})
            if not consumer_returns.empty:
                consumer_corr = consumer_returns.corr().mean().mean()
                correlations['consumer'] = consumer_corr
            
            # 分析板块轮动
            sector_strength = {}
            for sector, stocks in {
                'tech': tech_stocks,
                'semi': semi_stocks,
                'consumer': consumer_stocks
            }.items():
                sector_returns = []
                for stock in stocks:
                    if stock in returns:
                        # 计算5日收益率
                        ret = returns[stock].iloc[-5:].mean() * 100
                        sector_returns.append(ret)
                if sector_returns:
                    sector_strength[sector] = np.mean(sector_returns)
            
            # 生成分析结果
            result = {
                'correlations': correlations,
                'sector_strength': sector_strength,
                'leading_sector': max(sector_strength.items(), key=lambda x: x[1])[0],
                'rotation_signal': None
            }
            
            # 判断是否存在板块轮动
            if len(sector_strength) > 1:
                strength_diff = max(sector_strength.values()) - min(sector_strength.values())
                if strength_diff > 1.0:  # 超过1%的差异视为显著
                    result['rotation_signal'] = {
                        'from': min(sector_strength.items(), key=lambda x: x[1])[0],
                        'to': max(sector_strength.items(), key=lambda x: x[1])[0],
                        'strength': strength_diff
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"分析板块联动性时出错: {e}")
            return {} 