import pandas as pd
import logging
from typing import List, Optional, Dict
from sqlalchemy import create_engine, text
import datetime as dt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_monitor_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StockMonitorManager")

class StockMonitorManager:
    """监控股票管理类"""
    
    def __init__(self, db_config: Dict):
        """
        初始化监控股票管理器
        
        参数:
            db_config: 数据库配置信息
        """
        self.db_config = db_config
        self.engine = self._create_db_engine()
        logger.info("StockMonitorManager初始化完成")
        
    def _create_db_engine(self):
        """创建数据库引擎"""
        try:
            engine = create_engine(
                f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
                f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            return engine
        except Exception as e:
            logger.error(f"创建数据库引擎失败: {e}")
            return None
            
    def add_stock(self, symbol: str, name: str = None, sector: str = None, industry: str = None) -> bool:
        """
        添加股票到监控列表
        
        参数:
            symbol: 股票代码
            name: 股票名称
            sector: 行业分类
            industry: 具体行业
            
        返回:
            是否添加成功
        """
        try:
            query = """
            INSERT INTO monitored_stocks (symbol, name, sector, industry)
            VALUES (:symbol, :name, :sector, :industry)
            ON DUPLICATE KEY UPDATE
                name = COALESCE(:name, name),
                sector = COALESCE(:sector, sector),
                industry = COALESCE(:industry, industry),
                is_active = TRUE,
                last_updated = CURRENT_TIMESTAMP
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(query), {
                    "symbol": symbol,
                    "name": name,
                    "sector": sector,
                    "industry": industry
                })
                conn.commit()
                
            logger.info(f"成功添加股票到监控列表: {symbol}")
            return True
        except Exception as e:
            logger.error(f"添加股票 {symbol} 到监控列表失败: {e}")
            return False
            
    def remove_stock(self, symbol: str, hard_delete: bool = False) -> bool:
        """
        从监控列表中移除股票
        
        参数:
            symbol: 股票代码
            hard_delete: 是否物理删除，默认为逻辑删除
            
        返回:
            是否移除成功
        """
        try:
            if hard_delete:
                query = "DELETE FROM monitored_stocks WHERE symbol = :symbol"
            else:
                query = """
                UPDATE monitored_stocks
                SET is_active = FALSE, last_updated = CURRENT_TIMESTAMP
                WHERE symbol = :symbol
                """
                
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"symbol": symbol})
                conn.commit()
                
            if result.rowcount > 0:
                action = "删除" if hard_delete else "停用"
                logger.info(f"成功{action}股票: {symbol}")
                return True
            else:
                logger.warning(f"股票 {symbol} 不在监控列表中")
                return False
        except Exception as e:
            logger.error(f"移除股票 {symbol} 失败: {e}")
            return False
            
    def get_monitored_stocks(self, include_inactive: bool = False) -> pd.DataFrame:
        """
        获取监控的股票列表
        
        参数:
            include_inactive: 是否包含未激活的股票
            
        返回:
            监控股票DataFrame
        """
        try:
            query = """
            SELECT * FROM monitored_stocks
            WHERE 1=1
            """
            if not include_inactive:
                query += " AND is_active = TRUE"
            query += " ORDER BY symbol"
            
            stocks = pd.read_sql(query, self.engine)
            logger.info(f"获取监控股票列表成功，共 {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            logger.error(f"获取监控股票列表失败: {e}")
            return pd.DataFrame()
            
    def update_stock_info(
        self,
        symbol: str,
        name: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> bool:
        """
        更新股票信息
        
        参数:
            symbol: 股票代码
            name: 新的股票名称
            sector: 新的行业分类
            industry: 新的具体行业
            is_active: 是否激活
            
        返回:
            是否更新成功
        """
        try:
            update_fields = []
            params = {"symbol": symbol}
            
            if name is not None:
                update_fields.append("name = :name")
                params["name"] = name
            if sector is not None:
                update_fields.append("sector = :sector")
                params["sector"] = sector
            if industry is not None:
                update_fields.append("industry = :industry")
                params["industry"] = industry
            if is_active is not None:
                update_fields.append("is_active = :is_active")
                params["is_active"] = is_active
                
            if not update_fields:
                logger.warning("没有提供需要更新的字段")
                return False
                
            query = f"""
            UPDATE monitored_stocks
            SET {", ".join(update_fields)}, last_updated = CURRENT_TIMESTAMP
            WHERE symbol = :symbol
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                conn.commit()
                
            if result.rowcount > 0:
                logger.info(f"成功更新股票信息: {symbol}")
                return True
            else:
                logger.warning(f"股票 {symbol} 不在监控列表中")
                return False
        except Exception as e:
            logger.error(f"更新股票 {symbol} 信息失败: {e}")
            return False
            
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        获取单个股票的详细信息
        
        参数:
            symbol: 股票代码
            
        返回:
            股票信息字典
        """
        try:
            query = "SELECT * FROM monitored_stocks WHERE symbol = :symbol"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"symbol": symbol})
                row = result.fetchone()
                
            if row:
                # 将Row对象转换为字典
                info = dict(row._mapping)
                # 转换时间戳为datetime对象
                info['added_date'] = pd.to_datetime(info['added_date'])
                info['last_updated'] = pd.to_datetime(info['last_updated'])
                return info
            else:
                logger.warning(f"未找到股票 {symbol} 的信息")
                return None
        except Exception as e:
            logger.error(f"获取股票 {symbol} 信息失败: {e}")
            return None
            
    def search_stocks(
        self,
        keyword: str = None,
        sector: str = None,
        industry: str = None,
        is_active: bool = None
    ) -> pd.DataFrame:
        """
        搜索股票
        
        参数:
            keyword: 关键词（匹配代码或名称）
            sector: 行业分类
            industry: 具体行业
            is_active: 是否激活
            
        返回:
            匹配的股票DataFrame
        """
        try:
            conditions = []
            params = {}
            
            if keyword:
                conditions.append("(symbol LIKE :keyword OR name LIKE :keyword)")
                params["keyword"] = f"%{keyword}%"
            if sector:
                conditions.append("sector = :sector")
                params["sector"] = sector
            if industry:
                conditions.append("industry = :industry")
                params["industry"] = industry
            if is_active is not None:
                conditions.append("is_active = :is_active")
                params["is_active"] = is_active
                
            query = "SELECT * FROM monitored_stocks"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY symbol"
            
            stocks = pd.read_sql(text(query), self.engine, params=params)
            logger.info(f"搜索到 {len(stocks)} 只匹配的股票")
            return stocks
        except Exception as e:
            logger.error(f"搜索股票失败: {e}")
            return pd.DataFrame()
            
    def get_stocks_by_sector(self) -> Dict[str, pd.DataFrame]:
        """
        按行业分类获取股票列表
        
        返回:
            行业分类到股票DataFrame的映射
        """
        try:
            stocks = self.get_monitored_stocks()
            if stocks.empty:
                return {}
                
            # 按sector分组
            grouped = stocks.groupby('sector')
            result = {sector: group for sector, group in grouped}
            
            logger.info(f"成功按行业分类获取股票列表，共 {len(result)} 个行业")
            return result
        except Exception as e:
            logger.error(f"按行业分类获取股票列表失败: {e}")
            return {}
            
    def get_inactive_stocks(self) -> pd.DataFrame:
        """
        获取未激活的股票列表
        
        返回:
            未激活的股票DataFrame
        """
        try:
            query = """
            SELECT * FROM monitored_stocks
            WHERE is_active = FALSE
            ORDER BY last_updated DESC
            """
            
            stocks = pd.read_sql(query, self.engine)
            logger.info(f"获取未激活股票列表成功，共 {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            logger.error(f"获取未激活股票列表失败: {e}")
            return pd.DataFrame()
            
    def activate_stock(self, symbol: str) -> bool:
        """
        激活股票
        
        参数:
            symbol: 股票代码
            
        返回:
            是否激活成功
        """
        return self.update_stock_info(symbol, is_active=True)
        
    def deactivate_stock(self, symbol: str) -> bool:
        """
        停用股票
        
        参数:
            symbol: 股票代码
            
        返回:
            是否停用成功
        """
        return self.update_stock_info(symbol, is_active=False)
        
    def get_recently_updated(self, days: int = 7) -> pd.DataFrame:
        """
        获取最近更新的股票列表
        
        参数:
            days: 最近的天数
            
        返回:
            最近更新的股票DataFrame
        """
        try:
            query = f"""
            SELECT * FROM monitored_stocks
            WHERE last_updated >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL {days} DAY)
            ORDER BY last_updated DESC
            """
            
            stocks = pd.read_sql(query, self.engine)
            logger.info(f"获取最近 {days} 天更新的股票列表成功，共 {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            logger.error(f"获取最近更新的股票列表失败: {e}")
            return pd.DataFrame() 