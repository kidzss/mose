import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.trading_monitor import test_strategy, optimize_parameters
from strategy.uss_cpgw import calculate_indicators, check_last_day_signal

def main():
    print("开始CPGW策略回测...")
    print("\n1. 运行策略回测")
    print("=" * 50)
    test_strategy()
    
    print("\n2. 运行参数优化")
    print("=" * 50)
    optimize_parameters()

if __name__ == "__main__":
    main() 