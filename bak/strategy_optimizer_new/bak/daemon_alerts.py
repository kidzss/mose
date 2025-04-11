#!/usr/bin/env python3
import os
import sys
import time
import signal
import logging
import subprocess
import argparse
from datetime import datetime
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("daemon_alerts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertDaemon:
    """警报系统守护进程"""
    
    def __init__(self):
        self.pid_file = "alert_daemon.pid"
        self.scheduler_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "schedule_alerts.py"
        )
        
    def _write_pid(self, pid):
        """写入PID文件"""
        with open(self.pid_file, 'w') as f:
            f.write(str(pid))
            
    def _read_pid(self):
        """读取PID文件"""
        try:
            with open(self.pid_file, 'r') as f:
                return int(f.read().strip())
        except:
            return None
            
    def _is_running(self):
        """检查守护进程是否在运行"""
        pid = self._read_pid()
        if pid is None:
            return False
            
        try:
            process = psutil.Process(pid)
            return process.is_running() and "python" in process.name().lower()
        except:
            return False
            
    def start(self):
        """启动守护进程"""
        if self._is_running():
            logger.info("警报系统已经在运行中")
            return
            
        logger.info("启动警报系统守护进程")
        
        try:
            # 启动调度脚本
            process = subprocess.Popen(
                ["python", self.scheduler_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # 创建新的会话
            )
            
            # 写入PID文件
            self._write_pid(process.pid)
            
            logger.info(f"警报系统守护进程已启动，PID: {process.pid}")
            
        except Exception as e:
            logger.error(f"启动守护进程时出错: {str(e)}")
            sys.exit(1)
            
    def stop(self):
        """停止守护进程"""
        if not self._is_running():
            logger.info("警报系统未在运行")
            return
            
        pid = self._read_pid()
        logger.info(f"停止警报系统守护进程 (PID: {pid})")
        
        try:
            # 获取进程及其子进程
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # 停止所有子进程
            for child in children:
                child.terminate()
                
            # 停止主进程
            parent.terminate()
            
            # 等待进程结束
            gone, alive = psutil.wait_procs([parent] + children, timeout=3)
            
            # 如果有进程仍在运行，强制结束
            for p in alive:
                p.kill()
                
            # 删除PID文件
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
                
            logger.info("警报系统守护进程已停止")
            
        except Exception as e:
            logger.error(f"停止守护进程时出错: {str(e)}")
            sys.exit(1)
            
    def restart(self):
        """重启守护进程"""
        self.stop()
        time.sleep(2)  # 等待进程完全停止
        self.start()
        
    def status(self):
        """查看守护进程状态"""
        if not self._is_running():
            logger.info("警报系统未在运行")
            return
            
        pid = self._read_pid()
        try:
            process = psutil.Process(pid)
            create_time = datetime.fromtimestamp(process.create_time())
            running_time = datetime.now() - create_time
            
            logger.info(f"警报系统正在运行")
            logger.info(f"PID: {pid}")
            logger.info(f"启动时间: {create_time}")
            logger.info(f"运行时长: {running_time}")
            logger.info(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            logger.info(f"CPU使用率: {process.cpu_percent()}%")
            
        except Exception as e:
            logger.error(f"获取进程状态时出错: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="警报系统守护进程控制")
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'],
                      help="执行的操作：启动、停止、重启或查看状态")
    
    args = parser.parse_args()
    daemon = AlertDaemon()
    
    if args.action == 'start':
        daemon.start()
    elif args.action == 'stop':
        daemon.stop()
    elif args.action == 'restart':
        daemon.restart()
    elif args.action == 'status':
        daemon.status()

if __name__ == "__main__":
    main() 