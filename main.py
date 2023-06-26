# 바이낸스 API
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *

# 시간 동기화
import win32api
import time
from datetime import datetime

# Dict 깔끔한 출력
import pprint


client = None
binance_access_key = "CfczETU6grhIBePjrVXclSLPAxtNWmoT7QycenRbPQKLXRzKtCuZ8O6Xq58n8Kpz"
binance_secret_key = "YUrtnHnHAMVKRgB5rR6ySybREx6MpnabT1tlPZiWSBhQN5cQH9RMQcjT7BIpuupB"


# 디버그 프린트용
def print_hi(name):
    print(f'Hi, {name}')


# 시스템 시간 동기화
def set_system_time(serv_time):
    gmtime = time.gmtime(int((serv_time["serverTime"])/1000))
    win32api.SetSystemTime(gmtime[0],
                           gmtime[1],
                           0,
                           gmtime[2],
                           gmtime[3],
                           gmtime[4],
                           gmtime[5],
                           0)


# 선물 거래 ( Not Margin!! )
def future_order(side, amount):
    f_order = client.futures_create_order(
        symbol='BTCUSDT',
        side=side,
        type=ORDER_TYPE_MARKET,
        quantity=amount
    )
    return f_order


# 메인 함수
if __name__ == '__main__':

    # 계좌 연결
    try:
        client = Client(binance_access_key, binance_secret_key)
        server_time = client.get_server_time()
        set_system_time(server_time)

    except BinanceAPIException as e:
        print(e)
        exit()

    print_hi("User")

    # 선물 계좌 잔고 출력
    usdt_balance = None
    futures_account = client.futures_account_balance()

    for asset in futures_account:
        if asset['asset'] == 'USDT':
            usdt_balance = float(asset['balance'])
            break

    if usdt_balance is not None:
        print(f"USDT 잔고: {usdt_balance}")
    else:
        print("USDT 잔고를 찾을 수 없습니다.")

    info = client.get_margin_account()
    print(info)

    while True:
        inputC = input("사기:B, 팔기:S ")
        if inputC == 'B':
            num = input("수량을 입력하세요: ")
            future_order(SIDE_BUY, num)
            print("매수 완료")
        elif inputC == 'S':
            num = input("수량을 입력하세요: ")
            future_order(SIDE_SELL, num)
            print("매도 완료")



