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

# 보조지표 계산 라이브러리
import talib

# Numpy / pandas
import numpy as np
import pandas as pd


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

    # 데이터 매개변수
    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_1MINUTE  # 1분 분봉 데이터
    limit = 100  # 가져올 분봉 데이터의 개수 (최대 500개까지 가능)

    # klines 데이터 형태
    # 0=Open time(ms), 1=Open, 2=High, 3=Low, 4=Close, 5=Voume,
    # 6=Close time, 7=Quote asset vloume, 8=Number of trades
    # 9=Taker buy base asset volume 10=Take buy quote asset volume [2차원 list]
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    col_name=['Time','Open','High','Low','Close','Volume','Close Time','Quote','TradeNum','Taker buy base',' Taker buy quote', 'ignored']
    candles = pd.DataFrame(klines, columns=col_name)

    # 문자열 -> 숫자 변환 && Pd Series
    close_1m = candles['Close'].apply(pd.to_numeric)

    # Numpy밖에 못 쓴다. .to_numpy()
    sma_1m = pd.Series(talib.SMA(close_1m.to_numpy(),timeperiod=20),name="SMA")
    rsi_1m = pd.Series(talib.RSI(close_1m.to_numpy(),timeperiod=14),name="RSI")
    volume_1m = candles['Volume'].apply(pd.to_numeric)
    volume_sma_1m = pd.Series(talib.SMA(volume_1m.to_numpy(),timeperiod=20),name="Vol_SMA")

    datetime = pd.to_datetime(candles['Time'], unit='ms')

    info = pd.concat([datetime,sma_1m,rsi_1m,volume_1m,volume_sma_1m],axis=1)

    # row 생략 없이 출력
    pd.set_option('display.max_rows', None)
    # col 생략 없이 출력
    pd.set_option('display.max_columns', None)
    print(info)

    # Numpy

    #print(sma_1m)
    #print(rsi_1m)
    # while True:
    #     inputC = input("사기:B, 팔기:S ")
    #     if inputC == 'B':
    #         num = input("수량을 입력하세요: ")
    #         future_order(SIDE_BUY, num)
    #         print("매수 완료")
    #     elif inputC == 'S':
    #         num = input("수량을 입력하세요: ")
    #         future_order(SIDE_SELL, num)
    #         print("매도 완료")



