import argparse
import json
import logging
import os
from pathlib import Path
from config.trading_config import DatabaseConfig

from trading_system.core.system_manager import SystemManager


def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "trading_system.log"),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = "config/system_config.json"):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        # 如果配置文件不存在，创建默认配置
        config = {
            "system": {
                "name": "TradingSystem",
                "version": "1.0.0",
                "mode": "development"
            },
            "monitoring": {
                "market_indices": ["SPY", "QQQ", "IWM"],
                "check_interval": 60,
                "alert_thresholds": {
                    "price_change": 0.02,
                    "volume_ratio": 2.0,
                    "volatility_ratio": 1.5
                }
            },
            "trading": {
                "max_positions": 10,
                "position_size": 0.1,
                "risk_per_trade": 0.02
            }
        }

        # 添加数据库配置
        config["database"] = {
            "host": DatabaseConfig.host,
            "port": DatabaseConfig.port,
            "user": DatabaseConfig.user,
            "password": DatabaseConfig.password,
            "database": DatabaseConfig.database
        }

        # 创建配置目录
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        # 保存默认配置
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    return config


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="交易系统启动脚本")
    parser.add_argument(
        "--config",
        default="config/system_config.json",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mode",
        choices=["production", "development"],
        default="development",
        help="运行模式"
    )
    args = parser.parse_args()

    # 设置日志
    setup_logging()
    logger = logging.getLogger("SystemLauncher")

    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)

        # 更新运行模式
        config["system"]["mode"] = args.mode

        # 创建并启动系统
        logger.info("初始化交易系统...")
        system = SystemManager(args.config)

        logger.info(f"以 {args.mode} 模式启动系统")
        system.start()

        # 保持程序运行
        try:
            while True:
                # 每60秒打印一次系统状态
                import time
                time.sleep(60)
                status = system.get_system_status()
                logger.info(f"系统状态: {json.dumps(status, indent=2)}")
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭系统...")
            system.stop()
            logger.info("系统已关闭")

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        raise


if __name__ == "__main__":
    main()
