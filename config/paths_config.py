#!/usr/bin/env python3
"""
路径配置管理
管理项目中各种文件的存储路径，避免文件堆积在根目录
"""

import os
from pathlib import Path
from datetime import datetime

class PathsConfig:
    """路径配置类"""
    
    def __init__(self, base_dir: str = None):
        """
        初始化路径配置
        
        Args:
            base_dir: 项目根目录，默认为当前文件的上级目录
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # 临时文件目录
        self.temp_dir = self.base_dir / "temp"
        self.reports_dir = self.temp_dir / "reports"
        self.charts_dir = self.temp_dir / "charts"
        self.logs_dir = self.temp_dir / "logs"
        
        # 文档目录
        self.docs_dir = self.base_dir / "docs"
        self.md_dir = self.docs_dir / "md"
        
        # 备份目录
        self.backup_dir = self.base_dir / "bak"
        
        # 配置目录
        self.config_dir = self.base_dir / "config"
        
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有目录存在"""
        directories = [
            self.temp_dir,
            self.reports_dir,
            self.charts_dir,
            self.logs_dir,
            self.docs_dir,
            self.md_dir,
            self.backup_dir,
            self.config_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_report_path(self, filename: str = None) -> Path:
        """
        获取报告文件路径
        
        Args:
            filename: 文件名，如果为None则生成默认文件名
            
        Returns:
            Path: 报告文件完整路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"智能股票日报_{timestamp}.html"
        
        return self.reports_dir / filename
    
    def get_chart_path(self, symbol: str, date: str = None) -> Path:
        """
        获取图表文件路径
        
        Args:
            symbol: 股票代码
            date: 日期，格式为YYYYMMDD，如果为None则使用当前日期
            
        Returns:
            Path: 图表文件完整路径
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        filename = f"{symbol}_analysis_{date}.png"
        return self.charts_dir / filename
    
    def get_log_path(self, log_name: str) -> Path:
        """
        获取日志文件路径
        
        Args:
            log_name: 日志名称（不包含.log扩展名）
            
        Returns:
            Path: 日志文件完整路径
        """
        if not log_name.endswith('.log'):
            log_name += '.log'
        
        return self.logs_dir / log_name
    
    def get_md_path(self, filename: str) -> Path:
        """
        获取Markdown文件路径
        
        Args:
            filename: 文件名
            
        Returns:
            Path: Markdown文件完整路径
        """
        if not filename.endswith('.md'):
            filename += '.md'
        
        return self.md_dir / filename
    
    def get_backup_path(self, filename: str) -> Path:
        """
        获取备份文件路径
        
        Args:
            filename: 文件名
            
        Returns:
            Path: 备份文件完整路径
        """
        return self.backup_dir / filename
    
    def clean_old_files(self, days: int = 30):
        """
        清理旧文件
        
        Args:
            days: 保留天数，超过此天数的文件将被删除
        """
        import time
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # 清理临时目录中的旧文件
        for directory in [self.reports_dir, self.charts_dir, self.logs_dir]:
            for file_path in directory.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        print(f"已删除旧文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件失败 {file_path}: {e}")
    
    def get_directory_info(self) -> dict:
        """
        获取目录信息
        
        Returns:
            dict: 包含各目录路径和文件数量的信息
        """
        info = {}
        
        directories = {
            'reports': self.reports_dir,
            'charts': self.charts_dir,
            'logs': self.logs_dir,
            'md': self.md_dir,
            'backup': self.backup_dir
        }
        
        for name, path in directories.items():
            if path.exists():
                files = list(path.glob("*"))
                file_count = len([f for f in files if f.is_file()])
                info[name] = {
                    'path': str(path),
                    'file_count': file_count,
                    'exists': True
                }
            else:
                info[name] = {
                    'path': str(path),
                    'file_count': 0,
                    'exists': False
                }
        
        return info


# 全局路径配置实例
_paths_config = None

def get_paths_config() -> PathsConfig:
    """获取全局路径配置实例"""
    global _paths_config
    if _paths_config is None:
        _paths_config = PathsConfig()
    return _paths_config

def get_report_path(filename: str = None) -> str:
    """快捷方法：获取报告文件路径"""
    return str(get_paths_config().get_report_path(filename))

def get_chart_path(symbol: str, date: str = None) -> str:
    """快捷方法：获取图表文件路径"""
    return str(get_paths_config().get_chart_path(symbol, date))

def get_log_path(log_name: str) -> str:
    """快捷方法：获取日志文件路径"""
    return str(get_paths_config().get_log_path(log_name))

def get_md_path(filename: str) -> str:
    """快捷方法：获取Markdown文件路径"""
    return str(get_paths_config().get_md_path(filename))


if __name__ == "__main__":
    # 测试路径配置
    paths = get_paths_config()
    
    print("=== 路径配置测试 ===")
    print(f"项目根目录: {paths.base_dir}")
    print(f"报告目录: {paths.reports_dir}")
    print(f"图表目录: {paths.charts_dir}")
    print(f"日志目录: {paths.logs_dir}")
    print(f"文档目录: {paths.md_dir}")
    print(f"备份目录: {paths.backup_dir}")
    
    print("\n=== 目录信息 ===")
    info = paths.get_directory_info()
    for name, details in info.items():
        status = "✅" if details['exists'] else "❌"
        print(f"{status} {name}: {details['file_count']} 个文件 - {details['path']}")
    
    print("\n=== 路径生成测试 ===")
    print(f"报告路径: {get_report_path()}")
    print(f"AMD图表路径: {get_chart_path('AMD')}")
    print(f"日志路径: {get_log_path('portfolio_analysis')}")
    print(f"MD文档路径: {get_md_path('test_document')}") 