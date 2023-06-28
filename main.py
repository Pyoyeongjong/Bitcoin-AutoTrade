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
import pytz


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

# USDT 잔고 출력
def get_usdt_balance(client):

    usdt_balance = None
    futures_account = client.futures_account_balance()
    for asset in futures_account:
        if asset['asset'] == "USDT":
            usdt_balance = float(asset['balance'])
            break
    if usdt_balance is not None:
        print(f"USDT 잔고: {usdt_balance}")
    else:
        print("USDT 잔고를 찾을 수 없습니다.")
    return usdt_balance

# 선물 계좌 잔고/레버리지 출력
def set_future_client_info(client, symbol, lev):

    usdt_balance = get_usdt_balance(client)

    # 레버리지 변경
    leverage_info = client.futures_change_leverage(symbol=symbol, leverage=lev)
    leverage = leverage_info['leverage']
    print(f"레버리지: {leverage}")

    # 최대 몇사토시까지 살 수 있는가?
    # 비트코인 현재 가격
    ticker = client.get_ticker(symbol=symbol)
    current_price = ticker['lastPrice']
    # 형 변환 / 최대 가용 사토시 계산
    satoshi = math.floor(float(usdt_balance)*float(leverage)/float(current_price) * 1000) / 1000
    print(f"최대 매수(매도) 가능 BTC: {satoshi}")
    return leverage, satoshi


# 캔들 기본 데이터
def get_klines(client, symbol, limit, interval):
    # klines 데이터 형태
    # 0=Open time(ms), 1=Open, 2=High, 3=Low, 4=Close, 5=Voume,
    # 6=Close time, 7=Quote asset vloume, 8=Number of trades
    # 9=Taker buy base asset volume 10=Take buy quote asset volume [2차원 list]
    klines_1m = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    col_name = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote', 'TradeNum', 'Taker buy base',
                ' Taker buy quote', 'ignored']
    return pd.DataFrame(klines_1m, columns=col_name)

def get_klines_by_date(client, symbol, limit, interval, start_time, end_time):

    start_timestamp = int(start_time.timestamp() * 1000)  # 밀리초 단위로 변환
    end_timestamp = int(end_time.timestamp() * 1000)  # 밀리초 단위로 변환

    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit,
                                   startTime=start_timestamp, endTime=end_timestamp)
    col_name = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote', 'TradeNum', 'Taker buy base',
                ' Taker buy quote', 'ignored']
    return pd.DataFrame(candles, columns=col_name)

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
    # 한국 시간으로 맞춰주기 + DateTime으로 변환
    korea_tz = pytz.timezone('Asia/Seoul')
    datetime = pd.to_datetime(candles['Time'], unit='ms')
    datetime = datetime.dt.tz_localize(pytz.utc).dt.tz_convert(korea_tz)
    # 연결
    datas = pd.concat([datetime, sma, rsi, volume, volume_sma], axis=1)

    return datas


# 저점 구하는 함수
# Low값을 활용 + Array 숫자 변환 후 사용
def is_low_point(point, candles):
    count = 0
    temp_low = candles[point]
    for i in range(-8,9): # 좌우 8개의 값을 비교
        if candles[point+i]<temp_low:
            count+=1 # 꼭 저점 아니어도 저점 부근이면 OK
    if count>0:
        return False
    else:
        return True


# 고점 구하는 함수
# High값을 활용 + Array 숫자 변환 후 사용
def is_high_point(point, candles):
    count = 0
    temp_low = candles[point]
    for i in range(-8,9): # 좌우 8개의 값을 비교
        if candles[point+i]>temp_low:
            count+=1 # 꼭 고점 아니어도 고점 부근이면 OK
    if count>0:
        return False
    else:
        return True

# 하락 다이버전스 발견 (과거부터 탐색 -> 처움 만나는 다이버전스 Return) = 과거 Test 용
def detect_bearish_divergence(candles, candles_info, bottom):
    frlp = None # First RSI Low Point
    srlp = None # Second RSI Low Point
    print(f"[Start - BEAR]")
    print("[First]")
    for i, e in enumerate(candles_info['RSI']):
        if e <= bottom:
            print(candles_info["Time"][i], e)
            if is_low_point(i, candles['Low'].apply(pd.to_numeric).to_list()):
                frlp = i
                break

    if frlp is None:
        return 0

    while 1:
        print("[Second]")
        for i, e in enumerate(candles_info['RSI'][frlp+1:], start=frlp+1):
            if e <= bottom:
                print(candles_info["Time"][i], e)
                if is_low_point(i, candles["Low"].apply(pd.to_numeric).to_list()):
                    srlp = i
                    break

        if srlp is None:
            return 0

        print(frlp, srlp)
        if candles['Low'][frlp] < candles['Low'][srlp] or candles_info['RSI'][frlp] > candles_info['RSI'][srlp]:
            frlp = srlp
        else:
            return frlp, srlp

# 상승 다이버전스 발견 (과거부터 탐색 -> 처움 만나는 다이버전스 Return) = 과거 Test 용
def detect_bullish_divergence(candles, candles_info, top):
    frhp = None # First RSI Low Point
    srhp = None # Second RSI Low Point
    print(f"[Start - BULL]")
    print("[First]")
    for i, e in enumerate(candles_info['RSI']):
        if e >= top:
            print(candles_info["Time"][i], e)
            if is_high_point(i, candles['High'].apply(pd.to_numeric).to_list()):
                frhp = i
                break

    if frhp is None:
        return 0

    while 1:
        print("[Second]")
        for i, e in enumerate(candles_info['RSI'][frhp+1:], start=frhp+1):
            if e >= top:
                print(candles_info["Time"][i], e)
                if is_high_point(i, candles["High"].apply(pd.to_numeric).to_list()):
                    srhp = i
                    break

        if srhp is None:
            return 0

        print(frhp, srhp)
        if candles['High'][frhp] > candles['High'][srhp] or candles_info['RSI'][frhp] < candles_info['RSI'][srhp]:
            frhp = srhp
        else:
            return frhp, srhp

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

    # Client 정보 설정 및 잔고 출력
    get_usdt_balance(client)
    # leverage, satoshi = set_future_client_info(client, symbol, 5) // 현재 거래 중일 시 레버리지 움직이면 오류.
    # 캔들 정보 가져오기

    # candles_1m, candles_5m, candles_15m, candles_1h, candles_4h, candles_ld, candles_lw = get_candles(client, symbol, limit)
    # 캔들 정보 가져오기 (특정 시각)
    start_time = datetime(2023, 5, 20)
    end_time = datetime(2023, 6, 26)
    candles_15m = get_klines_by_date(client, symbol, limit, Client.KLINE_INTERVAL_15MINUTE, start_time, end_time)
    ### 보조지표 추출
    candles_info_15m = get_candle_subdatas(candles_15m)
    print(candles_info_15m)


    # 하락 다이버전스
    print(detect_bearish_divergence(candles_15m, candles_info_15m, 30))
    print(detect_bullish_divergence(candles_15m, candles_info_15m, 70))


