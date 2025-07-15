#!/usr/bin/env python3
"""
投资组合配置管理器
提供统一的配置更新功能
"""

import json
import streamlit as st
from datetime import datetime
import os

class PortfolioConfigManager:
    def __init__(self):
        self.config_file = 'portfolio_config.json'
        self.personal_config_file = 'personal_investor_config.json'
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"加载配置文件失败: {e}")
            return None
    
    def save_config(self, config):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            st.error(f"保存配置文件失败: {e}")
            return False
    
    def sync_with_personal_config(self):
        """与personal_investor_config.json同步"""
        try:
            with open(self.personal_config_file, 'r', encoding='utf-8') as f:
                personal_config = json.load(f)
            
            portfolio_config = self.load_config()
            if not portfolio_config:
                return False
            
            # 同步基本信息
            personal_portfolio = personal_config.get('current_portfolio', {})
            portfolio_config['meta']['total_assets'] = personal_portfolio.get('total_assets', 0)
            portfolio_config['meta']['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            portfolio_config['meta']['description'] = f"与personal_investor_config.json同步 - {datetime.now().strftime('%Y-%m-%d')}"
            
            # 同步资产配置
            portfolio_config['meta']['asset_allocation']['stocks']['percentage'] = personal_portfolio.get('stock_allocation', 0)
            portfolio_config['meta']['asset_allocation']['money_fund']['percentage'] = personal_portfolio.get('fund_allocation', 0)
            portfolio_config['meta']['asset_allocation']['cash']['percentage'] = personal_portfolio.get('cash_allocation', 0)
            
            # 同步持仓信息
            detailed_holdings = personal_portfolio.get('detailed_holdings', {})
            positions = {}
            
            for symbol, holding in detailed_holdings.items():
                positions[symbol] = {
                    "shares": holding.get('shares', 0),
                    "cost_basis": holding.get('cost_basis', 0),
                    "weight": holding.get('allocation', 0),
                    "investment_amount": holding.get('shares', 0) * holding.get('cost_basis', 0),
                    "sector": "Technology" if symbol in ['GOOG', 'NVDA', 'AMD', 'TSLA'] else "Healthcare" if symbol == 'MRK' else "Financial",
                    "stop_loss_threshold": 0.10,
                    "transaction_note": f"{symbol}持仓"
                }
            
            portfolio_config['positions'] = positions
            
            return self.save_config(portfolio_config)
            
        except Exception as e:
            st.error(f"同步配置失败: {e}")
            return False

def render_portfolio_config_manager():
    """渲染投资组合配置管理器界面"""
    st.markdown("## 📊 投资组合配置管理器")
    st.markdown("---")
    
    manager = PortfolioConfigManager()
    
    # 加载当前配置
    config = manager.load_config()
    if not config:
        st.error("无法加载配置文件")
        return
    
    # 显示当前配置
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 基本信息")
        meta = config.get('meta', {})
        st.write(f"**总资产**: ${meta.get('total_assets', 0):,.2f}")
        st.write(f"**最后更新**: {meta.get('last_updated', 'Unknown')}")
        st.write(f"**描述**: {meta.get('description', 'No description')}")
        
        st.subheader("📈 资产配置")
        asset_allocation = meta.get('asset_allocation', {})
        for asset_type, info in asset_allocation.items():
            if isinstance(info, dict):
                percentage = info.get('percentage', 0)
                amount = info.get('amount', 0)
                st.write(f"**{asset_type}**: {percentage:.2f}% (${amount:,.2f})")
    
    with col2:
        st.subheader("📋 当前持仓")
        positions = config.get('positions', {})
        if positions:
            for symbol, position in positions.items():
                shares = position.get('shares', 0)
                weight = position.get('weight', 0)
                cost_basis = position.get('cost_basis', 0)
                st.write(f"**{symbol}**: {shares}股 ({weight:.2f}%) @ ${cost_basis:.2f}")
        else:
            st.write("无持仓信息")
    
    st.markdown("---")
    
    # 配置更新选项
    st.subheader("🔄 配置更新选项")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 同步配置", help="与personal_investor_config.json同步"):
            if manager.sync_with_personal_config():
                st.success("✅ 配置同步成功！")
                st.rerun()
            else:
                st.error("❌ 配置同步失败")
    
    with col2:
        if st.button("📊 刷新显示", help="刷新当前配置显示"):
            st.rerun()
    
    with col3:
        if st.button("🗑️ 清理缓存", help="清理Streamlit缓存"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ 缓存已清理")
    
    # 手动编辑配置
    st.markdown("---")
    st.subheader("✏️ 手动编辑配置")
    
    # 添加新持仓
    with st.expander("➕ 添加新持仓"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_symbol = st.text_input("股票代码", key="new_symbol")
        with col2:
            new_shares = st.number_input("股数", min_value=0, key="new_shares")
        with col3:
            new_cost_basis = st.number_input("成本价", min_value=0.0, key="new_cost_basis")
        with col4:
            new_sector = st.selectbox("行业", ["Technology", "Healthcare", "Financial", "Consumer", "Industrial"], key="new_sector")
        
        if st.button("添加持仓", key="add_position"):
            if new_symbol and new_shares > 0 and new_cost_basis > 0:
                # 计算权重
                total_assets = config['meta']['total_assets']
                investment_amount = new_shares * new_cost_basis
                weight = (investment_amount / total_assets) * 100
                
                config['positions'][new_symbol] = {
                    "shares": new_shares,
                    "cost_basis": new_cost_basis,
                    "weight": weight,
                    "investment_amount": investment_amount,
                    "sector": new_sector,
                    "stop_loss_threshold": 0.10,
                    "transaction_note": f"新增{new_symbol}持仓"
                }
                
                if manager.save_config(config):
                    st.success(f"✅ 成功添加{new_symbol}持仓")
                    st.rerun()
                else:
                    st.error("❌ 保存失败")
            else:
                st.error("请填写完整的持仓信息")
    
    # 删除持仓
    with st.expander("🗑️ 删除持仓"):
        if positions:
            symbol_to_delete = st.selectbox("选择要删除的股票", list(positions.keys()), key="delete_symbol")
            
            if st.button("删除持仓", key="delete_position"):
                if symbol_to_delete in config['positions']:
                    del config['positions'][symbol_to_delete]
                    
                    if manager.save_config(config):
                        st.success(f"✅ 成功删除{symbol_to_delete}持仓")
                        st.rerun()
                    else:
                        st.error("❌ 保存失败")
        else:
            st.write("当前无持仓可删除")
    
    # 编辑持仓
    with st.expander("✏️ 编辑持仓"):
        if positions:
            symbol_to_edit = st.selectbox("选择要编辑的股票", list(positions.keys()), key="edit_symbol")
            
            if symbol_to_edit in positions:
                position = positions[symbol_to_edit]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    new_shares = st.number_input("股数", value=position.get('shares', 0), key="edit_shares")
                    new_cost_basis = st.number_input("成本价", value=position.get('cost_basis', 0.0), key="edit_cost_basis")
                
                with col2:
                    new_sector = st.selectbox("行业", ["Technology", "Healthcare", "Financial", "Consumer", "Industrial"], 
                                            index=["Technology", "Healthcare", "Financial", "Consumer", "Industrial"].index(position.get('sector', 'Technology')), 
                                            key="edit_sector")
                    new_note = st.text_input("备注", value=position.get('transaction_note', ''), key="edit_note")
                
                if st.button("更新持仓", key="update_position"):
                    # 更新持仓信息
                    config['positions'][symbol_to_edit].update({
                        "shares": new_shares,
                        "cost_basis": new_cost_basis,
                        "sector": new_sector,
                        "transaction_note": new_note,
                        "investment_amount": new_shares * new_cost_basis
                    })
                    
                    # 重新计算权重
                    total_assets = config['meta']['total_assets']
                    weight = ((new_shares * new_cost_basis) / total_assets) * 100
                    config['positions'][symbol_to_edit]['weight'] = weight
                    
                    if manager.save_config(config):
                        st.success(f"✅ 成功更新{symbol_to_edit}持仓")
                        st.rerun()
                    else:
                        st.error("❌ 保存失败")
        else:
            st.write("当前无持仓可编辑")

if __name__ == "__main__":
    render_portfolio_config_manager() 