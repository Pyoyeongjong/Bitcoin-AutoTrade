# 바이낸스 API
from binance.client import Client

# 시간 동기화
import win32api
import time
from datetime import datetime

binance_access_key = "CfczETU6grhIBePjrVXclSLPAxtNWmoT7QycenRbPQKLXRzKtCuZ8O6Xq58n8Kpz"
binance_secret_key = "YUrtnHnHAMVKRgB5rR6ySybREx6MpnabT1tlPZiWSBhQN5cQH9RMQcjT7BIpuupB"


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('Binance')
    client = Client(binance_access_key, binance_secret_key)

    print(client.get_symbol_info('BTCUSDT'))
