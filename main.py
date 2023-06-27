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
import math

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


# 선물 계좌 잔고/레버리지 출력
def set_future_client_info(client, symbol, lev):
    # USDT 잔고 출력
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
    # 레버리지 변경
    leverage_info = client.futures_change_leverage(symbol=symbol, leverage=lev)
    leverage = leverage_info['leverage']
    print(f"레버리지: {leverage}")

    # 최대 몇사토시까지 살 수 있는가?
    # 비트코인 현재 가격
    ticker = client.get_ticker(symbol=symbol)
    current_price = ticker['lastPrice']
    # 형 변환 / 최대 매수 사토시 계산
    satoshi = math.floor(float(usdt_balance)*float(leverage)/float(current_price) * 1000) / 1000
    print(f"최대 매수 가능 BTC: {satoshi}")
    return leverage, satoshi



def get_klines(client, symbol, limit, interval):
    # klines 데이터 형태
    # 0=Open time(ms), 1=Open, 2=High, 3=Low, 4=Close, 5=Voume,
    # 6=Close time, 7=Quote asset vloume, 8=Number of trades
    # 9=Taker buy base asset volume 10=Take buy quote asset volume [2차원 list]
    klines_1m = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    col_name = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote', 'TradeNum', 'Taker buy base',
                ' Taker buy quote', 'ignored']
    return pd.DataFrame(klines_1m, columns=col_name)

# 캔들 데이터 가져오기
def get_candles(client, sym, limit):

    candles_1m = get_klines(client, sym, limit, Client.KLINE_INTERVAL_1MINUTE)
    candles_5m = get_klines(client, sym, limit, Client.KLINE_INTERVAL_5MINUTE)
    candles_15m = get_klines(client, sym, limit, Client.KLINE_INTERVAL_15MINUTE)
    candles_1h = get_klines(client, sym, limit, Client.KLINE_INTERVAL_1HOUR)
    candles_4h = get_klines(client, sym, limit, Client.KLINE_INTERVAL_4HOUR)
    candles_1d = get_klines(client, sym, limit, Client.KLINE_INTERVAL_1DAY)
    candles_1w = get_klines(client, sym, limit, Client.KLINE_INTERVAL_1WEEK)

    return candles_1m, candles_5m, candles_15m, candles_1h, candles_4h, candles_1d, candles_1w

# SMA, RSI, VOL, VOL_SMA
def get_candle_subdatas(candles):
    ### 데이터 분석
    # 문자열 -> 숫자 변환 && Pd Series
    close = candles['Close'].apply(pd.to_numeric)  # 종가 값 활용
    # Numpy밖에 못 쓴다 -> .to_numpy()
    sma = pd.Series(talib.SMA(close.to_numpy(), timeperiod=20), name="SMA")
    rsi = pd.Series(talib.RSI(close.to_numpy(), timeperiod=14), name="RSI")
    volume = candles['Volume'].apply(pd.to_numeric)
    volume_sma = pd.Series(talib.SMA(volume.to_numpy(), timeperiod=20), name="Vol_SMA")
    datetime = pd.to_datetime(candles_1m['Time'], unit='ms')
    datas = pd.concat([datetime, sma, rsi, volume, volume_sma], axis=1)

    return datas


# 메인 함수
if __name__ == '__main__':

    ### Initiation
    # row 생략 없이 출력
    pd.set_option('display.max_rows', 10)
    # col 생략 없이 출력
    pd.set_option('display.max_columns', None)
    # 캔들 데이터 가져오기
    symbol = "BTCUSDT"
    limit = 500  # 가져올 분봉 데이터의 개수 (최대 500개까지 가능)
    # 최대 매수 BTC / 레버리지
    satoshi = None
    leverage = None

    # 계좌 연결
    try:
        client = Client(binance_access_key, binance_secret_key)
        server_time = client.get_server_time()
        set_system_time(server_time)

    except BinanceAPIException as e:
        print(e)
        exit()

    # 선물 계좌 잔고 출력
    leverage, satoshi = set_future_client_info(client, symbol, 5)

    candles_1m, candles_5m, candles_15m, candles_1h, candles_4h, candles_ld, candles_lw = get_candles(client, symbol, limit)

    ### 보조지표 추출
    info_1m = get_candle_subdatas(candles_1m)
    info_5m = get_candle_subdatas(candles_5m)
    info_15m = get_candle_subdatas(candles_15m)
    info_1h = get_candle_subdatas(candles_1h)
    print(info_15m)

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



