# 바이낸스 API
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *

# 시간 동기화
import win32api
import time
from datetime import datetime
from datetime import timedelta

# Dict 깔끔한 출력
import pprint

# 보조지표 계산/출력 라이브러리
import talib
import math
import matplotlib.pyplot as plt

# Numpy / pandas
import numpy as np
import pandas as pd
import pytz

# CSV파일
import os
import csv

# 클라이언트 변수
client = None
# 현재 매수 중인지 확인하는 변수
isOrdered = False
is_Long = False
is_Short = False

# API 파일 경로
api_key_file_path = "api.txt"

# API 키를 읽어오는 함수
def read_api_keys(file_path):
    with open(file_path, "r") as file:
        api_key = file.readline().strip()
        api_secret = file.readline().strip()
    return api_key, api_secret

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
def future_order(side, amount, sonjul):

    global isOrdered

    f_order = client.futures_create_order(
        symbol='BTCUSDT',
        side=side,
        type=FUTURE_ORDER_TYPE_MARKET,
        quantity=0.001
    )

    f_order = client.futures_create_order(
        symbol='BTCUSDT',
        side=SIDE_SELL,
        type=FUTURE_ORDER_TYPE_STOP_MARKET,
        stopPrice = sonjul,
        quantity=0.001
    )
    if isOrdered is True:
        isOrdered = False
    else:
        isOrdered = True
    return f_order

def future_order_close(side, amount):
    f_order = client.futures_create_order(
        symbol='BTCUSDT',
        side=side,
        type=ORDER_TYPE_MARKET,
        quantity=amount
    )
    if isOrdered is True:
        isOrdered = False
    else:
        isOrdered = True
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
def get_future_client_info(client, symbol, lev):

    global isOrdered
    leverage = None
    satoshi = None

    usdt_balance = get_usdt_balance(client)
    # 레버리지 변경
    if isOrdered is False:
        #print(isOrdered)
        try:
            leverage_info = client.futures_change_leverage(symbol=symbol, leverage=lev)
            leverage = leverage_info['leverage']
            #print(f"레버리지: {leverage}")
        except BinanceAPIException as e:
            print(e)
    else:
        print("현재 포지션을 가지고 있어 레버리지가 변경되지 않습니다.")

    # 최대 몇사토시까지 살 수 있는가?
    # 비트코인 현재 가격
    ticker = client.get_ticker(symbol=symbol)
    current_price = ticker['lastPrice']
    # 형 변환 / 최대 가용 사토시 계산
    if leverage is not None:
        satoshi = math.floor(float(usdt_balance)*float(leverage)/float(current_price) * 1000) / 1000
    #print(f"최대 매수(매도) 가능 BTC: {satoshi}")
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
    sma7 = pd.Series(talib.SMA(close.to_numpy(), timeperiod=7), name="SMA7")
    sma20 = pd.Series(talib.SMA(close.to_numpy(), timeperiod=20), name="SMA20")
    sma60 = pd.Series(talib.SMA(close.to_numpy(), timeperiod=60), name="SMA60")
    sma120 = pd.Series(talib.SMA(close.to_numpy(), timeperiod=120), name="SMA120")

    rsi = pd.Series(talib.RSI(close.to_numpy(), timeperiod=14), name="RSI")
    volume = candles['Volume'].apply(pd.to_numeric)
    volume_sma = pd.Series(talib.SMA(volume.to_numpy(), timeperiod=20), name="Vol_SMA")
    ### 한국 시간으로 맞춰주기 + DateTime으로 변환
    korea_tz = pytz.timezone('Asia/Seoul')
    datetime = pd.to_datetime(candles['Time'], unit='ms')
    candles['Time'] = datetime.dt.tz_localize(pytz.utc).dt.tz_convert(korea_tz)
    # 볼린저 밴드
    upperband, middleband, lowerband = talib.BBANDS(candles['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    upperband.name = "UpperBand"
    lowerband.name = "LowerBand"
    # 트렌드
    inclination = calculate_trends(candles, 0)
    # 연결
    data = pd.concat([candles, sma7, sma20, sma60, sma120, rsi, volume, volume_sma, upperband, lowerband, inclination], axis=1)
    return data

# 저점 구하는 함수
# Low값을 활용 + Array 숫자 변환 후 사용
def is_low_point(point, candles):
    count = 0
    temp_low = candles[point]
    for i in range(-8,9): # 좌우 8개의 값을 비교
        if point+i >= len(candles):
            break
        if candles[point+i]<temp_low:
            count+=1 #
    if count>2: # 꼭 저점 아니어도 저점 부근이면 OK
        return False
    else:
        return True

# 고점 구하는 함수
# High값을 활용 + Array 숫자 변환 후 사용
def is_high_point(point, candles):
    count = 0
    temp_low = candles[point]
    for i in range(-8,9): # 좌우 8개의 값을 비교
        if point+i >= len(candles):
            break
        if candles[point+i]>temp_low:
            count+=1
    if count>2: # 꼭 고점 아니어도 고점 부근이면 OK
        return False
    else:
        return True

# 하락 다이버전스 발견 (과거부터 탐색 -> 처움 만나는 다이버전스 Return) = 과거 Test 용
def detect_bearish_divergence(candles, candles_info, bottom, k):
    frlp = None # First RSI Low Point
    srlp = None # Second RSI Low Point

    candles_low = candles['Low'].apply(pd.to_numeric).to_list()

    for i, e in enumerate(candles_info['RSI'][k:], start=k):
        if e <= bottom:
            if is_low_point(i, candles_low):
                frlp = i
                break

    if frlp is None:
        return 0

    while 1:
        for i, e in enumerate(candles_info['RSI'][frlp+1:], start=frlp+1):
            if e <= bottom:
                if is_low_point(i, candles_low):
                    srlp = i
                    break

        if srlp is None:
            return 0

        if candles['Low'][frlp] < candles['Low'][srlp] or candles_info['RSI'][frlp] > candles_info['RSI'][srlp]:
            frlp = srlp
        else:
            return candles_info['Time'][frlp], candles_info['Time'][srlp], srlp, srlp-frlp

def detect_bearish_divergences(candles, candles_info, bottom):
    bear_div_list = []
    k = 0

    while 1:
        result = detect_bearish_divergence(candles, candles_info, bottom, k)
        if result==0:
            return bear_div_list
        else:
            time1, time2, next, term = result
            if 5 < term < 60: # 봉 간 60개 이상 차이가 안 나야 한다.
                bear_div_list.append([time1, time2])
            k = next

    # bear_div_list.append((time1, time2))



# 하락 다이버전스 감시 (현재 데이터에서 다이버전스가 일어났나?), 일반 다이버전스만 구현.
def spectate_bearish_divergence(candles, candles_info, bottom):

    rlp_1 = None # First RSI Low Point // 최근에 가까운 기준
    rlp_2 = None # Second RSI Low Point # 삼중, 사중 다이버전스 후보군
    rlp_3 = None # Third RSI Low Point

    candles_low = candles['Low'].apply(pd.to_numeric).to_list()

    # 가장 최근 RSI 저점 구하기
    for i, v in reversed(list(enumerate(candles_info['RSI']))):
        if v <= bottom:
            if is_low_point(i, candles_low):
                rlp_1 = i
                break

    if rlp_1 is None:
        return False

    now_point = len(candles)

    if candles['Low'][now_point-2] < candles['Low'][rlp_1] and candles_info['RSI'][now_point-2] > candles_info['RSI'][rlp_1]:
        return True
    else:
        return False

# 상승 다이버전스 발견 (과거부터 탐색 -> 처움 만나는 다이버전스 Return) = 과거 Test 용
def detect_bullish_divergence(candles, candles_info, top, k):
    frhp = None # First RSI Low Point
    srhp = None # Second RSI Low Point

    candles_high = candles['High'].apply(pd.to_numeric).to_list()

    for i, e in enumerate(candles_info['RSI'][k:], start=k):
        if e >= top:
            if is_high_point(i, candles_high):
                frhp = i
                break

    if frhp is None:
        return 0

    while 1:
        for i, e in enumerate(candles_info['RSI'][frhp+1:], start=frhp+1):
            if e >= top:
                if is_high_point(i, candles_high):
                    srhp = i
                    break

        if srhp is None:
            return 0

        if candles['High'][frhp] > candles['High'][srhp] or candles_info['RSI'][frhp] < candles_info['RSI'][srhp]:
            frhp = srhp
        else:
            return candles_info['Time'][frhp], candles_info['Time'][srhp], srhp, srhp-frhp


def detect_bullish_divergences(candles, candles_info, top):
    bull_div_list = []
    k = 0
    while 1:
        result = detect_bullish_divergence(candles, candles_info, top, k)
        if result==0:
            return bull_div_list
        else:
            time1, time2, next, term = result
            if 60 > term > 5: # 봉 간 60개 이상 차이가 안 나야 한다.
                bull_div_list.append([time1, time2])
            k = next



# 상승 다이버전스 감시 (현재 데이터에서 다이버전스가 일어났나?), 일반 다이버전스만 구현.
def spectate_bullish_divergence(candles, candles_info, top):

    rhp_1 = None # RSI High Point // 최근에 가까운 기준
    rhp_2 = None # Second RSI High Point # 삼중, 사중 다이버전스 후보군
    rhp_3 = None # Third RSI High Point

    candles_high = candles['High'].apply(pd.to_numeric).to_list()

    # 가장 최근 RSI 저점 구하기
    for i, v in reversed(list(enumerate(candles_info['RSI']))):
        if v >= top:
            if is_low_point(i, candles_high):
                rlp_1 = i
                break

    if rhp_1 is None:
        return False

    now_point = len(candles)

    if candles['Low'][now_point-2] > candles['Low'][rhp_1] and candles_info['RSI'][now_point-2] < candles_info['RSI'][rlp_1]:
        return True
    else:
        return False

# 현재 포지션 설정
def get_position(positions, symbol):

    global isOrdered
    global is_Long
    global is_Short

    for position in positions:
        if position['symbol'] == symbol:
            if float(position['positionAmt']) > 0:
                print("현재 포지션 : Long")
                isOrdered = True
                is_Long = True
                is_Short = False
                return
            elif float(position['positionAmt']) < 0:
                print("현재 포지션 : Short")
                isOrdered = True
                is_Long = False
                is_Short = True
                return
            else:
                print("현재 포지션 : 없음")
                isOrdered = False
                is_Long = True
                is_Short = False
                return


# 기울기 구하는 함수 # close는 Array, Numeric
def calculate_incline(close, i, j):
    return (close[j]-close[i])/(j-i)/(close[i])*1000

# 1시간 봉 라이브 기울기 구하는 함수1
def cal_inc_1h1d(ytdc, cp):
    ytdc = float(ytdc)  # ytdc 변수를 부동소수점으로 변환
    cp = float(cp)  # cp 변수를 부동소수점으로 변환
    return (cp-ytdc)/cp*1000

# 1시간 봉 라이브 기울기 구하는 함수2
def calculate_incline_1h_1d(yesterday_close, cur_price, yesterday_incline):
    return (yesterday_incline*2+cal_inc_1h1d(yesterday_close, cur_price))/3


### 장 추세 구별 함수
# 종가 기준 기울기를 통해 현재 장이 상승장 or 하락장을 구분할 것임
# 최소 1시간 이상 봉을 이용하는 게 좋아 보인다.
# cal 일봉:9.0 4시간봉:1 1시간봉:0.2 사용하자
def calculate_trends(candles_info, start):

    inclination_mean_list = []
    inclination_mean_list.append(0)

    timestamps = []
    candles_close = candles_info["Close"].apply(pd.to_numeric).to_list()

    # i = 인덱스, e = 종가
    for i, e in enumerate(candles_info['Close'][start:], start=start):
        sum_inclination = 0
        count = 0
        for j in range(-10 ,0):
            if i+j<0 or i+j>=len(candles_info) or j==0:
                continue

            sum_inclination += calculate_incline(candles_close, i, i+j)

            count+=1
        if count == 0:
            continue
        inclination_mean = sum_inclination / count
        inclination_mean_list.append(inclination_mean)
        timestamps.append(candles_info['Time'][i])

    return pd.DataFrame({'Inclination': inclination_mean_list})

def read_csv_data(time):
    candles_history = pd.read_csv(f"candle_data/candle_data_{time}.csv")
    return candles_history

def add_tag(row):
    if row['Inclination'] >= 8:
        return '상승장'
    elif row['Inclination'] >= -8:
        return '횡보장'
    else:
        return '하락장'

# sjh = 손절 상수, lev = 배율상수(전체시드 기준)
def backTesting_hwengbo(candles_history_info_1h, candles_history_info_1d, inc, sjh, lev):

    is_bought = False
    is_hwengbo = False # 횡보 중인가?

    trade_count = 0 # 매매 횟수
    win_count = 0 # 승리 횟수
    lose_count = 0 # 패배 횟수

    start_cash = 10000 # 초기 시드
    current_cash = start_cash # 현재 시드
    highest_cash = start_cash # 최고 시드
    lowest_cash = start_cash # 최저 시드

    day_inclination = None # 현재 기울기
    dayrow = None # 당일 데이터
    yest_inc = None # 전일 기울기

    in_price = 0 # 진입가
    sonjul = 0 # 손절가
    is_Long = False
    is_Short = False
    inc_count = 0 # 기울기 카운트. 1시간봉 종가 기준 횡보 기울기가 연속으로 몇개가 나오면 들어갈 건지에 대해

    for idx, row in candles_history_info_1h.iloc[:].iterrows():


        if idx < 100:
            continue

        upperband = row['UpperBand']
        lowerband = row['LowerBand']
        close = row['Close']
        volume = row['Volume']
        vol_sma = row['Vol_SMA']
        low = row['Low']
        high = row['High']
        dt = row['Time']

        # 사지 않은 상태라면,
        if is_bought == False:

            # 당일 일봉 기울기 가져오기
            if dt.hour == 9 and dt.minute == 0 and dt.second == 0:
                dayrow = candles_history_info_1d[candles_history_info_1d['Time'] == dt]
                if yest_inc is None:
                    yest_inc = dayrow['Inclination'].values[0]
                    continue
            # 예외 처리
            if dayrow is None:
                continue
            # day_inclination = dayrow['Inclination'].values[0]

            # 기울기 계산(1시간 봉 종가 기준마다)
            dayrow_open = dayrow['Open'].values[0]
            day_inclination = calculate_incline_1h_1d(dayrow_open, close, yest_inc) # 내적으로 보완. 당일 기울기 : 전일까지의 기울기평균 = 1:2
            #print(dt, day_inclination)
            # 전일 기울기 업데이트

            # 예외 처리
            if day_inclination is None:
                print("None")
                continue
            yest_inc = dayrow['Inclination'].values[0]
            # 기울기 = 횡보일 때
            if -inc < day_inclination < inc:
                inc_count+=1
            else:

                inc_count=0
                is_hwengbo = False
                continue
            # 기울기 연속으로 얼마나 횡보였나 판단.

            if inc_count>=1:
                is_hwengbo = True
                inc_count=0
            else:
                continue

            # 횡보라고 판정됨 -> 매수 준비
            if is_hwengbo:
                # 1시간 봉 기준 볼린저밴드 아래에서 마감
                if lowerband > close:
                    # 거래량이 평균보다 20% 높으면 롱 진입.
                    if (volume > vol_sma * 1.2).all():
                        in_price = close
                        is_bought = True
                        sonjul = close - close * 0.01 * sjh
                        is_Long = True
                        is_Short = False
                        trade_count += 1
                        #print("오늘 기울기: ",day_inclination)
                        #print(row['Time'], "롱 진입", in_price, f"손절가 :{sonjul:<2.0f}", "총 거래 횟수 : ", trade_count)
                        continue
                # 1시간 봉 기준 볼린저밴드 위에서 마감
                if close > upperband:
                    # 거래량이 평균보다 20% 높으면 숏 진입.
                    if (volume > vol_sma * 1.2).all():
                        in_price = close
                        is_bought = True
                        sonjul = close + close * 0.01 * sjh
                        is_Long = False
                        is_Short = True
                        trade_count += 1
                        #print("오늘 기울기: ", day_inclination)
                        #print(row['Time'], "숏 진입", in_price,f"손절가 :{sonjul:<2.0f}", "총 거래 횟수 : ", trade_count)
                        continue
        else:
            if is_Long:
                # 익절
                if low <= upperband <= high:
                    is_bought = False
                    suik = current_cash * (upperband / in_price - 1-0.0008) * lev * 0.5 # 수익 계산
                    current_cash += suik
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    win_count += 1
                    #print(row['Time'], f"롱 승리! 익절가 :{upperband:<2.0f}, 수익 :{suik:<2.0f},                현재 시드 :{current_cash:<2.0f}\n")
                    continue

                # 손절
                elif low <= sonjul <= high:
                    is_bought = False
                    current_cash += current_cash * (sonjul / in_price - 1-0.0008) * lev* 0.5
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    lose_count += 1
                    #print(row['Time'], f"롱 패배...                                 현재 시드 :{current_cash:<2.0f}\n")
                    continue

            elif is_Short:
                # 익절
                if low <= lowerband <= high:
                    is_bought = False
                    suik = current_cash * (in_price / lowerband - 1-0.0008) * lev * 0.5 # 수익 계산
                    current_cash += suik
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    win_count += 1
                    #print(row['Time'], f"숏 승리! 익절가 :{lowerband:<2.0f}, 수익 :{suik:<2.0f},              현재 시드 :{current_cash:<2.0f}\n")
                    continue
                # 손절
                elif low <= sonjul <= high:
                    is_bought = False
                    current_cash += current_cash * (in_price / sonjul - 1-0.0008) * lev * 0.5
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    lose_count += 1
                    #print(row['Time'], f"숏 패배...                                            현재 시드 :{current_cash:<2.0f}\n")
                    continue
    print(
        f"Trade count: {str(trade_count):<2}  Win count: {str(win_count):<2}  Lose count: {str(lose_count):<2}"
        f"  Current cash: {current_cash:<8.0f}  Highest cash: {highest_cash:<8.0f}  Lowest cash: {lowest_cash:<8.0f}"
    )

def do_trading_hwengbo(inc, sjh, lev, client, symbol):


    global isOrdered
    global is_Long
    global is_Short

    is_hwengbo = False # 횡보 중인가?
    is_Long = False
    is_Short = False
    is_zeongak = False

    trade_count = 0 # 매매 횟수
    win_count = 0 # 승리 횟수
    lose_count = 0 # 패배 횟수

    day_inclination = None # 현재 기울기
    dayrow = None # 당일 데이터
    yest_inc = None # 전일 기울기

    in_price = 0 # 진입가
    sonjul = 0 # 손절가


    while True:

        ### 캔들 정보 가져오기 (현재)
        candles_1m, candles_5m, candles_15m, candles_1h, candles_4h, candles_1d, candles_1w = get_candles(client,
                                                                                                          symbol,
                                                                                                          limit)

        ### 캔들 정보 가져오기 (특정 시각)
        # start_time = datetime(2023, 5, 20)
        # end_time = datetime(2023, 6, 26)
        # candles_15m = get_klines_by_date(client, symbol, limit, Client.KLINE_INTERVAL_15MINUTE, start_time, end_time)

        ### 보조지표 추출
        candles_info_15m = get_candle_subdatas(candles_15m)
        candles_info_1h = get_candle_subdatas(candles_1h)
        candles_info_4h = get_candle_subdatas(candles_4h)
        candles_info_1d = get_candle_subdatas(candles_1d)

        row = candles_info_1h.iloc[-1]
        upperband = float(row['UpperBand'])
        lowerband = float(row['LowerBand'])
        close = float(row['Close'])
        volume = row['Volume']
        vol_sma = row['Vol_SMA']
        low = row['Low']
        high = row['High']
        dt = row['Time']

        now = datetime.now()

        if now.minute == 0 and now.second<3:
            is_zeongak = True
        else:
            is_zeongak = False

        dt_day = None

        if dt.hour >=9:
            dt_day = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        else:
            dt = dt - timedelta(days=1, hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
            dt_day = dt.replace(hour=9, minute=0, second=0, microsecond=0)

        dt_yest = dt - timedelta(days=2, hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
        dt_yest = dt.replace(hour=9, minute=0, second=0, microsecond=0)

        # 사지 않은 상태라면,
        if not isOrdered:

            leverage, satoshi = get_future_client_info(client, symbol, lev)

            amount = satoshi * leverage * 0.5

            # 전날 일봉 기울기 가져오기
            yestrow = candles_info_1d[candles_info_1d['Time'] == dt_yest]
            yest_inc = yestrow['Inclination'].values[0]
            # 당일 일봉 OPEN 값 가져오기
            dayrow = candles_info_1d[candles_info_1d['Time'] == dt_day]
            dayrow_open = dayrow['Open'].values[0]
            # 기울기 계산(1시간 봉 종가 기준마다)
            day_inclination = calculate_incline_1h_1d(dayrow_open, close, yest_inc) # 내적으로 보완. 당일 기울기 : 전일까지의 기울기평균 = 1:2
            # 전일 기울기 업데이트

            # 기울기 = 횡보일 때
            if -inc < day_inclination < inc:
                is_hwengbo = True
            else:
                return

            # 횡보라고 판정됨 -> 매수 준비
            if is_hwengbo:
                # 1시간 봉 기준 볼린저밴드 아래에서 마감
                if lowerband > close:
                    # 거래량이 평균보다 20% 높으면 롱 진입.
                    if (volume > vol_sma * 1.2).all():
                        in_price = close
                        is_bought = True
                        sonjul = close - close * 0.01 * sjh
                        is_Long = True
                        is_Short = False
                        future_order('BUY', amount, sonjul)
                        print(f"롱. 양:{amount}, 손절:{sonjul}")

                # 1시간 봉 기준 볼린저밴드 위에서 마감
                if close > upperband:
                    # 거래량이 평균보다 20% 높으면 숏 진입.
                    if (volume > vol_sma * 1.2).all():
                        in_price = close
                        is_bought = True
                        sonjul = close + close * 0.01 * sjh
                        is_Long = False
                        is_Short = True
                        trade_count += 1
                        #print("오늘 기울기: ", day_inclination)
                        future_order('SELL', amount, sonjul)
                        print(f"숏. 양:{amount}, 손절:{sonjul}")

        else:
            if is_Long:
                # 익절
                if low <= upperband <= high:
                    is_bought = False
                    suik = current_cash * (upperband / in_price - 1-0.0008) * lev * 0.5 # 수익 계산
                    current_cash += suik
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    win_count += 1
                    future_order_close('SELL', amount)
                    print("익절")

                # 손절
                elif low <= sonjul <= high:
                    is_bought = False
                    current_cash += current_cash * (sonjul / in_price - 1-0.0008) * lev* 0.5
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    lose_count += 1
                    future_order_close('SELL', amount)
                    print("손절")

            elif is_Short:
                # 익절
                if low <= lowerband <= high:
                    is_bought = False
                    suik = current_cash * (in_price / lowerband - 1-0.0008) * lev * 0.5 # 수익 계산
                    current_cash += suik
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    win_count += 1
                    future_order_close('BUY', amount)
                    print("익절")

                # 손절
                elif low <= sonjul <= high:
                    is_bought = False
                    current_cash += current_cash * (in_price / sonjul - 1-0.0008) * lev * 0.5
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    lose_count += 1
                    future_order_close('BUY', amount)
                    print("손절")

        print(f"보류. 기울기 : {day_inclination}")
        time.sleep(3)

def backTesting_bull():


    return print("Hi")



# 메인 함수
if __name__ == '__main__':

    ### Initiation
    # row 생략 없이 출력
    pd.set_option('display.max_rows', None)
    # col 생략 없이 출력
    pd.set_option('display.max_columns', None)
    # 캔들 데이터 가져오기
    symbol = "BTCUSDT"
    limit = 500  # 가져올 분봉 데이터의 개수 (최대 500개까지 가능)
    # 최대 매수 BTC / 레버리지
    satoshi = None
    leverage = None

    time1 = time.time()
    ### 계좌 연결
    binance_access_key, binance_secret_key = read_api_keys(api_key_file_path)
    try:
        client = Client(binance_access_key, binance_secret_key)
        server_time = client.get_server_time()
        set_system_time(server_time)

    except BinanceAPIException as e:
        print(e)
        exit()

    ### 현재 포지션 정보
    positions = client.futures_position_information()
    get_position(positions, symbol)
    ### Client 정보 설정 및 잔고 출력
    get_usdt_balance(client)
    leverage, satoshi = get_future_client_info(client, symbol, 10) # 현재 거래 중일 시 레버리지 움직이면 오류.
    print(leverage, satoshi)


    # ### 캔들 정보 가져오기 (현재)
    # candles_1m, candles_5m, candles_15m, candles_1h, candles_4h, candles_1d, candles_1w = get_candles(client, symbol, limit)
    #
    # ### 캔들 정보 가져오기 (특정 시각)
    # # start_time = datetime(2023, 5, 20)
    # # end_time = datetime(2023, 6, 26)
    # # candles_15m = get_klines_by_date(client, symbol, limit, Client.KLINE_INTERVAL_15MINUTE, start_time, end_time)
    #
    # ### 보조지표 추출
    # candles_info_15m = get_candle_subdatas(candles_15m)
    # candles_info_1h = get_candle_subdatas(candles_1h)
    # candles_info_4h = get_candle_subdatas(candles_4h)
    # candles_info_1d = get_candle_subdatas(candles_1d)
    # time2 = time.time()

    ### 과거 데이터 & 보조지표 (Timestamp 안 이상함)
    # candles_history_15m = read_csv_data("15m")
    # candles_history_info_15m = get_candle_subdatas(candles_history_15m)
    #
    # candles_history_1h = read_csv_data("1h")
    # candles_history_info_1h = get_candle_subdatas(candles_history_1h)
    #
    # candles_history_4h = read_csv_data("4h")
    # candles_history_info_4h = get_candle_subdatas(candles_history_4h)
    #
    # candles_history_1d = read_csv_data("1d")
    # candles_history_info_1d = get_candle_subdatas(candles_history_1d)
    #
    # candles_history_1h_21 = pd.read_csv(f"candle_data/candle_data_1h_before_21.csv")
    # candles_history_info_1h_21 = get_candle_subdatas(candles_history_1h_21)
    #
    # candles_history_4h_21 = pd.read_csv(f"candle_data/candle_data_4h_before_21.csv")
    # candles_history_info_4h_21 = get_candle_subdatas(candles_history_4h_21)
    #
    # candles_history_1d_21 = pd.read_csv(f"candle_data/candle_data_1d_before_21.csv")
    # candles_history_info_1d_21 = get_candle_subdatas(candles_history_1d_21)

    ### 하락 다이버전스 발견(과거 데이터)(리스트 형식) 출력 = [(time1, time2)]
    # print(detect_bullish_divergences(candles_15m, candles_info_15m, 70))
    # print(detect_bearish_divergences(candles_history_15m, candles_history_info_15m, 30))

    ### 하락 다이버전스 감지(현재 데이터)
    # 문제 : 이걸 분마다 계산하는 게 이득일까? 다른 데 저장해놨다가 새로 들어오는 분에 대해서만 새로운 연산을 수행하면 되지 않나? -> 최적화 문제
    # print(spectate_bearish_divergence(candles_15m, candles_info_15m, 30))
    # print(spectate_bullish_divergence(candles_15m, candles_info_15m, 70))

    ### 장 추세 계산함수 (일봉 9.0, 4시간봉 1.5, 1시간봉 0.3) 오늘 계산 = len(candles)-1
    # incinc = pd.concat([candles_history_info_4h['Time'], candles_history_info_4h['Inclination']],axis=1)
    # incinc['Tag'] = incinc.apply(add_tag, axis=1)
    # print(incinc)

    # filtered_df = incinc[incinc.iloc[:, 1] > 10]
    # print(filtered_df)


    ### 함수 걸리는 시간 계산
    # time1 = time.time()
    # time2 = time.time()
    # print(time2-time1)

    ### 횡보장 전략 확정 기울기 -5~5, 손실 2%, 레버리지 10배, 기울기 보수지표 = 1
    # backTesting_hwengbo(candles_history_info_1h_21, candles_history_info_1d_21, 5, 2, 10)

    #do_trading_hwengbo(5, 2, leverage, client, symbol)
    future_order("BUY", 0.001, '30100')

