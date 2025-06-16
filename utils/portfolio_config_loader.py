#!/usr/bin/env python3
"""
持仓配置加载器
提供统一的持仓配置加载接口，确保所有模块使用相同的配置数据
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class PortfolioConfigLoader:
    """持仓配置加载器"""
    
    def __init__(self, config_path: str = "portfolio_config.json"):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            # 尝试从根目录加载
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                logger.info(f"成功加载持仓配置: {self.config_path}")
            else:
                # 尝试相对路径
                root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config_path)
                if os.path.exists(root_path):
                    with open(root_path, 'r', encoding='utf-8') as f:
                        self._config = json.load(f)
                    logger.info(f"成功加载持仓配置: {root_path}")
                else:
                    logger.error(f"配置文件不存在: {self.config_path}")
                    self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "meta": {
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "total_assets": 0,
                "currency": "USD",
                "description": "默认配置 - 请更新实际持仓信息"
            },
            "portfolio": {
                "total_value": 0,
                "stock_allocation": {"total_amount": 0, "percentage": 0},
                "cash_allocation": {"amount": 0, "percentage": 0},
                "money_fund_allocation": {"amount": 0, "percentage": 0}
            },
            "positions": {},
            "watchlist": {},
            "monitoring_config": {
                "price_alert_threshold": 0.05,
                "loss_alert_threshold": 0.05,
                "profit_target": 0.25,
                "check_interval": 60,
                "email_notifications": True
            }
        }
    
    def get_positions(self) -> Dict[str, Dict]:
        """获取持仓信息"""
        return self._config.get("positions", {})
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """获取特定股票持仓信息"""
        return self._config.get("positions", {}).get(symbol)
    
    def get_watchlist(self) -> Dict[str, Dict]:
        """获取观察列表"""
        return self._config.get("watchlist", {})
    
    def get_portfolio_summary(self) -> Dict:
        """获取投资组合概览"""
        return self._config.get("portfolio", {})
    
    def get_monitoring_config(self) -> Dict:
        """获取监控配置"""
        return self._config.get("monitoring_config", {})
    
    def get_portfolio_symbols(self) -> List[str]:
        """获取持仓股票代码列表"""
        return list(self._config.get("positions", {}).keys())
    
    def get_watchlist_symbols(self) -> List[str]:
        """获取观察列表股票代码"""
        return list(self._config.get("watchlist", {}).keys())
    
    def get_all_symbols(self) -> List[str]:
        """获取所有需要监控的股票代码（持仓+观察列表）"""
        positions = self.get_portfolio_symbols()
        watchlist = self.get_watchlist_symbols()
        return list(set(positions + watchlist))
    
    def get_total_portfolio_value(self) -> float:
        """获取投资组合总价值"""
        return self._config.get("portfolio", {}).get("total_value", 0)
    
    def get_meta_info(self) -> Dict:
        """获取元信息"""
        return self._config.get("meta", {})
    
    def is_position_held(self, symbol: str) -> bool:
        """检查是否持有某只股票"""
        return symbol in self._config.get("positions", {})
    
    def is_in_watchlist(self, symbol: str) -> bool:
        """检查股票是否在观察列表中"""
        return symbol in self._config.get("watchlist", {})
    
    def get_position_weight(self, symbol: str) -> float:
        """获取持仓权重"""
        position = self.get_position(symbol)
        return position.get("weight", 0) / 100 if position else 0
    
    def get_stop_loss_threshold(self, symbol: str) -> float:
        """获取止损阈值"""
        position = self.get_position(symbol)
        return position.get("stop_loss_threshold", 0.1) if position else 0.1
    
    def to_legacy_format(self) -> Dict:
        """转换为旧格式，兼容现有代码"""
        positions = {}
        for symbol, info in self.get_positions().items():
            positions[symbol] = {
                'cost_basis': info.get('cost_basis', 0),
                'weight': info.get('weight', 0) / 100,  # 转换为百分比
                'shares': info.get('shares', 0),
                'stop_loss': info.get('stop_loss_threshold', 0.1)
            }
        
        return {
            'positions': positions,
            'monitor_config': self.get_monitoring_config()
        }
    
    def to_smart_report_format(self) -> Dict:
        """转换为智能日报格式"""
        portfolio = {}
        for symbol, info in self.get_positions().items():
            portfolio[symbol] = {
                'cost': info.get('cost_basis', 0),
                'shares': info.get('shares', 0),
                'weight': info.get('weight', 0),
                'investment': info.get('investment_amount', 0)
            }
        return portfolio
    
    def reload(self):
        """重新加载配置文件"""
        self._load_config()


# 全局实例
_portfolio_config = None


def get_portfolio_config(config_path: str = "portfolio_config.json") -> PortfolioConfigLoader:
    """获取全局持仓配置实例"""
    global _portfolio_config
    if _portfolio_config is None:
        _portfolio_config = PortfolioConfigLoader(config_path)
    return _portfolio_config


def reload_portfolio_config():
    """重新加载持仓配置"""
    global _portfolio_config
    if _portfolio_config:
        _portfolio_config.reload()


if __name__ == "__main__":
    # 测试功能
    loader = get_portfolio_config()
    
    print("=== 持仓配置测试 ===")
    print(f"持仓股票: {loader.get_portfolio_symbols()}")
    print(f"观察列表: {loader.get_watchlist_symbols()}")
    print(f"总资产: ${loader.get_total_portfolio_value():,.2f}")
    
    print("\n=== 持仓详情 ===")
    for symbol in loader.get_portfolio_symbols():
        pos = loader.get_position(symbol)
        print(f"{symbol}: {pos['shares']}股 @ ${pos['cost_basis']:.3f} (权重: {pos['weight']:.2f}%)")
    
    print("\n=== 观察列表 ===")
    for symbol in loader.get_watchlist_symbols():
        watch = loader.get_watchlist().get(symbol, {})
        print(f"{symbol}: 目标价格 ${watch.get('target_buy_price', 0):.2f}") 