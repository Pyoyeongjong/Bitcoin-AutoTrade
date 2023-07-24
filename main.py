# 바이낸스 API
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *

# 텔레그램
import telegram
import asyncio

### Telegram
bot = telegram.Bot(token="6332731064:AAEOlgnRBgM8RZxW9CnkUPJHvEo54SZoEH8")
chat_id = 1735838793

# 시간 동기화
# import win32api
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
# api_key_file_path = "/home/ubuntu/Bitcoin/Binance/api.txt"


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
# def set_system_time(serv_time):
#     gmtime = time.gmtime(int((serv_time["serverTime"])/1000))
#     win32api.SetSystemTime(gmtime[0],
#                            gmtime[1],
#                            0,
#                            gmtime[2],
#                            gmtime[3],
#                            gmtime[4],
#                            gmtime[5],
#                            0)

def create_client():

    global client

    ### 계좌 연결
    binance_access_key, binance_secret_key = read_api_keys(api_key_file_path)
    try:
        client = Client(binance_access_key, binance_secret_key)
        server_time = client.get_server_time()
        # set_system_time(server_time)
    except BinanceAPIException as e:
        print(e)
        exit()

    return


# isOrdered는 밖에서 관리하자.
def future_create_order_market(symbol, side, quantity):
    f_order = client.futures_create_order(
        symbol=symbol,
        side=side,
        type=FUTURE_ORDER_TYPE_MARKET,
        quantity=quantity
    )
    return

def future_create_stop_loss_order(symbol, side, quantity, sonjul):

    stop_loss_order = client.futures_create_order(
        symbol=symbol,
        side=side,
        type=FUTURE_ORDER_TYPE_STOP_MARKET,
        quantity=quantity,
        stopprice=sonjul
    )
    return

def future_cancel_all_open_order(symbol):

    order = client.futures_cancel_all_open_orders(
        symbol=symbol,
        timestamp=time.time()
    )
    return

# 선물 거래 ( Not Margin!! )
def future_order(symbol, side, amount, sonjul):

    otherside = None

    if side is SIDE_BUY:
        otherside = SIDE_SELL
    else:
        otherside = SIDE_BUY

    # 진입.
    f_order = future_create_order_market(symbol, side, amount)
    stop_loss_order = future_create_stop_loss_order(symbol, otherside, amount, sonjul)

    return f_order, stop_loss_order


def future_order_take_profit(symbol, side, amount):

    f_order = future_create_order_market(symbol, side, amount)
    future_cancel_all_open_order(symbol)

    return f_order

def get_current_price(symbol):
    # 비트코인 현재 가격
    ticker = client.get_ticker(symbol=symbol)
    current_price = ticker['lastPrice']
    return current_price

# USDT 잔고 출력
def get_usdt_balance(client, isprint):
    usdt_balance = None
    futures_account = client.futures_account_balance()
    for asset in futures_account:
        if asset['asset'] == "USDT":
            usdt_balance = float(asset['balance'])
            break
    if usdt_balance is not None:
        if isprint:
            print(f"USDT 잔고: {usdt_balance}")
    else:
        print("USDT 잔고를 찾을 수 없습니다.")
    return usdt_balance

def get_position_amt(symbol):
    position_info = client.futures_position_information(symbol=symbol)
    return math.floor(float(position_info[0]['positionAmt'])*1000)/1000


# 선물 계좌 잔고/레버리지 출력
def change_leverage(symbol, lev):

    global isOrdered
    leverage = None

    # 레버리지 변경
    if isOrdered is False:
        # print(isOrdered)
        try:
            leverage_info = client.futures_change_leverage(symbol=symbol, leverage=lev)
            leverage = leverage_info['leverage']
            # print(f"레버리지: {leverage}")
        except BinanceAPIException as e:
            print(e)
    else:
        print("현재 포지션을 가지고 있어 레버리지가 변경되지 않습니다.")
        return get_current_leverage(symbol)

    # print(f"최대 매수(매도) 가능 BTC: {satoshi}")
    return leverage

def get_current_leverage(symbol):

    timestamp = int(time.time() * 1000)
    try:
        account_info = client.futures_account(timestamp=timestamp)
        for position in account_info['positions']:
            if position['symbol'] == symbol:
                return position['leverage']
        return None
    except Exception as e:
        print("Error occurred:", e)
        return None

def get_lev_amt(symbol, is_print):
    # 현재 레버리지
    leverage = get_current_leverage(symbol)
    # 레버리지 변경 함수
    # leverage = change_leverage(symbol, 10)
    # 비트코인 현재 가격
    current_price = get_current_price(symbol)
    # 최대 몇사토시까지 살 수 있는가?
    # 형 변환 / 최대 가용 사토시 계산
    if leverage is not None:
        satoshi = math.floor(float(usdt_balance) * float(leverage) / float(current_price) * 1000) / 1000
        amount = math.floor(satoshi * 0.5 * 1000) / 1000
    if is_print:
        print(f"현재 레버리지 :{leverage}, 시드 절반:{amount}")
    return leverage, amount

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
    data = pd.concat([candles, sma7, sma20, sma60, sma120, rsi, volume, volume_sma, upperband, lowerband, inclination],
                     axis=1)
    return data


# 저점 구하는 함수
# Low값을 활용 + Array 숫자 변환 후 사용
def is_low_point(point, candles):
    count = 0
    temp_low = candles[point]
    for i in range(-8, 9):  # 좌우 8개의 값을 비교
        if point + i >= len(candles):
            break
        if candles[point + i] < temp_low:
            count += 1  #
    if count > 2:  # 꼭 저점 아니어도 저점 부근이면 OK
        return False
    else:
        return True


# 고점 구하는 함수
# High값을 활용 + Array 숫자 변환 후 사용
def is_high_point(point, candles):
    count = 0
    temp_low = candles[point]
    for i in range(-8, 9):  # 좌우 8개의 값을 비교
        if point + i >= len(candles):
            break
        if candles[point + i] > temp_low:
            count += 1
    if count > 2:  # 꼭 고점 아니어도 고점 부근이면 OK
        return False
    else:
        return True


# 하락 다이버전스 발견 (과거부터 탐색 -> 처움 만나는 다이버전스 Return) = 과거 Test 용
def detect_bearish_divergence(candles, candles_info, bottom, k):
    frlp = None  # First RSI Low Point
    srlp = None  # Second RSI Low Point

    candles_low = candles['Low'].apply(pd.to_numeric).to_list()

    for i, e in enumerate(candles_info['RSI'][k:], start=k):
        if e <= bottom:
            if is_low_point(i, candles_low):
                frlp = i
                break

    if frlp is None:
        return 0

    while 1:
        for i, e in enumerate(candles_info['RSI'][frlp + 1:], start=frlp + 1):
            if e <= bottom:
                if is_low_point(i, candles_low):
                    srlp = i
                    break

        if srlp is None:
            return 0

        if candles['Low'][frlp] < candles['Low'][srlp] or candles_info['RSI'][frlp] > candles_info['RSI'][srlp]:
            frlp = srlp
        else:
            return candles_info['Time'][frlp], candles_info['Time'][srlp], srlp, srlp - frlp


def detect_bearish_divergences(candles, candles_info, bottom):
    bear_div_list = []
    k = 0

    while 1:
        result = detect_bearish_divergence(candles, candles_info, bottom, k)
        if result == 0:
            return bear_div_list
        else:
            time1, time2, next, term = result
            if 5 < term < 60:  # 봉 간 60개 이상 차이가 안 나야 한다.
                bear_div_list.append([time1, time2])
            k = next

    # bear_div_list.append((time1, time2))


# 하락 다이버전스 감시 (현재 데이터에서 다이버전스가 일어났나?), 일반 다이버전스만 구현.
def spectate_bearish_divergence(candles, candles_info, bottom):
    rlp_1 = None  # First RSI Low Point // 최근에 가까운 기준
    rlp_2 = None  # Second RSI Low Point # 삼중, 사중 다이버전스 후보군
    rlp_3 = None  # Third RSI Low Point

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

    if candles['Low'][now_point - 2] < candles['Low'][rlp_1] and candles_info['RSI'][now_point - 2] > \
            candles_info['RSI'][rlp_1]:
        return True
    else:
        return False


# 상승 다이버전스 발견 (과거부터 탐색 -> 처움 만나는 다이버전스 Return) = 과거 Test 용
def detect_bullish_divergence(candles, candles_info, top, k):
    frhp = None  # First RSI Low Point
    srhp = None  # Second RSI Low Point

    candles_high = candles['High'].apply(pd.to_numeric).to_list()

    for i, e in enumerate(candles_info['RSI'][k:], start=k):
        if e >= top:
            if is_high_point(i, candles_high):
                frhp = i
                break

    if frhp is None:
        return 0

    while 1:
        for i, e in enumerate(candles_info['RSI'][frhp + 1:], start=frhp + 1):
            if e >= top:
                if is_high_point(i, candles_high):
                    srhp = i
                    break

        if srhp is None:
            return 0

        if candles['High'][frhp] > candles['High'][srhp] or candles_info['RSI'][frhp] < candles_info['RSI'][srhp]:
            frhp = srhp
        else:
            return candles_info['Time'][frhp], candles_info['Time'][srhp], srhp, srhp - frhp


def detect_bullish_divergences(candles, candles_info, top):
    bull_div_list = []
    k = 0
    while 1:
        result = detect_bullish_divergence(candles, candles_info, top, k)
        if result == 0:
            return bull_div_list
        else:
            time1, time2, next, term = result
            if 60 > term > 5:  # 봉 간 60개 이상 차이가 안 나야 한다.
                bull_div_list.append([time1, time2])
            k = next


# 상승 다이버전스 감시 (현재 데이터에서 다이버전스가 일어났나?), 일반 다이버전스만 구현.
def spectate_bullish_divergence(candles, candles_info, top):
    rhp_1 = None  # RSI High Point // 최근에 가까운 기준
    rhp_2 = None  # Second RSI High Point # 삼중, 사중 다이버전스 후보군
    rhp_3 = None  # Third RSI High Point

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

    if candles['Low'][now_point - 2] > candles['Low'][rhp_1] and candles_info['RSI'][now_point - 2] < \
            candles_info['RSI'][rlp_1]:
        return True
    else:
        return False


# 현재 포지션 설정
def get_position(symbol, is_print):

    global isOrdered
    global is_Long
    global is_Short

    positions = client.futures_position_information()

    for position in positions:
        if position['symbol'] == symbol:
            if float(position['positionAmt']) > 0:
                if is_print:
                    print("현재 포지션 : Long")
                isOrdered = True
                is_Long = True
                is_Short = False
                return
            elif float(position['positionAmt']) < 0:
                if is_print:
                    print("현재 포지션 : Short")
                isOrdered = True
                is_Long = False
                is_Short = True
                return
            else:
                if is_print:
                    print("현재 포지션 : 없음")
                isOrdered = False
                is_Long = False
                is_Short = False
                return


# 기울기 구하는 함수 # close는 Array, Numeric
def calculate_incline(close, i, j):
    return (close[j] - close[i]) / (j - i) / (close[i]) * 1000


# 1시간 봉 라이브 기울기 구하는 함수1
def cal_inc_1h1d(ytdc, cp):
    ytdc = float(ytdc)  # ytdc 변수를 부동소수점으로 변환
    cp = float(cp)  # cp 변수를 부동소수점으로 변환
    return (cp - ytdc) / cp * 1000


# 1시간 봉 라이브 기울기 구하는 함수2
def calculate_incline_1h_1d(yesterday_close, cur_price, yesterday_incline):
    return (yesterday_incline * 2 + cal_inc_1h1d(yesterday_close, cur_price)) / 3

def calculate_incline_1h_1d_history(cur_price, candles_info_1d):

    day_close = candles_info_1d['Close']

    sum_inclination = 0
    count = 0

    for j in range(-12, -1):
        sum_inclination += (float(cur_price) - float(day_close.iloc[j])) / (-j-1) / float(cur_price) * 1000
        count += 1

    inclination_mean = sum_inclination / count

    return inclination_mean




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
        for j in range(-10, 0):
            if i + j < 0 or i + j >= len(candles_info) or j == 0:
                continue

            sum_inclination += calculate_incline(candles_close, i, i + j)

            count += 1
        if count == 0:
            continue
        inclination_mean = sum_inclination / count
        inclination_mean_list.append(inclination_mean)
        timestamps.append(candles_info['Time'][i])

    return pd.DataFrame({'Inclination': inclination_mean_list})


def read_csv_data(time):
    candles_history = pd.read_csv(f"candle_data/candle_data_{time}.csv")
    return candles_history
def read_csv_data_eth(time):
    candles_history = pd.read_csv(f"candle_data/eth_candle_data_{time}.csv")
    return candles_history


def add_tag(row):
    if row['Inclination'] >= 8:
        return '상승장'
    elif row['Inclination'] >= -8:
        return '횡보장'
    else:
        return '하락장'


# sjh = 손절 상수, lev = 배율상수(전체시드 기준)
# 일봉 기울기 실시간 업데이트하는 백테스팅 : 결과가 매우 안좋다. 왜지??
def backTesting_hwengbo_2(candles_history_info_1h, candles_history_info_1d, inc, sjh, lev):

    is_bought = False
    is_hwengbo = False  # 횡보 중인가?

    trade_count = 0  # 매매 횟수
    win_count = 0  # 승리 횟수
    lose_count = 0  # 패배 횟수

    start_cash = 10000  # 초기 시드
    current_cash = start_cash  # 현재 시드
    highest_cash = start_cash  # 최고 시드
    lowest_cash = start_cash  # 최저 시드

    day_inclination = None  # 현재 기울기
    dayrows = None  # 당일 데이터
    yest_inc = None  # 전일 기울기

    in_price = 0  # 진입가
    sonjul = 0  # 손절가
    is_Long = False
    is_Short = False
    inc_count = 0  # 기울기 카운트. 1시간봉 종가 기준 횡보 기울기가 연속으로 몇개가 나오면 들어갈 건지에 대해

    for idx, row in candles_history_info_1h.iloc[:].iterrows():

        if idx < 300:
            continue

        upperband = float(row['UpperBand'])
        lowerband = float(row['LowerBand'])
        close = float(row['Close'])
        volume = float(row['Volume'][0])
        vol_sma = float(row['Vol_SMA'])
        low = float(row['Low'])
        high = float(row['High'])
        dt = row['Time']

        # 사지 않은 상태라면,
        if is_bought == False:

            dayrows = candles_history_info_1d[candles_history_info_1d['Time'] <= dt]

            # 예외 처리
            if dayrows is None:
                continue
            # day_inclination = dayrow['Inclination'].values[0]

            # 기울기 계산(1시간 봉 종가 기준마다)
            day_inclination = calculate_incline_1h_1d_history(close, dayrows)

            # 예외 처리
            if day_inclination is None:
                print("None")
                continue

            # 기울기 = 횡보일 때
            if -inc < day_inclination < inc:
                is_hwengbo = True
            else:
                is_hwengbo = False
                continue

            # 횡보라고 판정됨 -> 매수 준비
            if is_hwengbo:
                # 1시간 봉 기준 볼린저밴드 아래에서 마감
                if lowerband > close:
                    # 거래량이 평균보다 20% 높으면 롱 진입.
                    if volume > vol_sma * 1.2:
                        in_price = close
                        is_bought = True
                        sonjul = close - close * 0.01 * sjh
                        is_Long = True
                        is_Short = False
                        trade_count += 1
                        print("오늘 기울기: ",day_inclination)
                        print(row['Time'], "롱 진입", in_price, f"손절가 :{sonjul:<2.0f}", "총 거래 횟수 : ", trade_count)
                        continue
                # 1시간 봉 기준 볼린저밴드 위에서 마감
                if close > upperband:
                    # 거래량이 평균보다 20% 높으면 숏 진입.
                    if volume > vol_sma * 1.2:
                        in_price = close
                        is_bought = True
                        sonjul = close + close * 0.01 * sjh
                        is_Long = False
                        is_Short = True
                        trade_count += 1
                        print("오늘 기울기: ", day_inclination)
                        print(row['Time'], "숏 진입", in_price,f"손절가 :{sonjul:<2.0f}", "총 거래 횟수 : ", trade_count)
                        continue
        else:
            if is_Long:
                # 익절
                if low <= upperband <= high:
                    is_bought = False
                    suik = current_cash * (upperband / in_price - 1 - 0.0008) * lev * 0.5  # 수익 계산
                    current_cash += suik
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    win_count += 1
                    print(row['Time'], f"롱 승리! 익절가 :{upperband:<2.0f}, 수익 :{suik:<2.0f},                현재 시드 :{current_cash:<2.0f}\n")
                    continue

                # 손절
                elif low <= sonjul <= high:
                    is_bought = False
                    current_cash += current_cash * (sonjul / in_price - 1 - 0.0008) * lev * 0.5
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    lose_count += 1
                    print(row['Time'], f"롱 패배...                                 현재 시드 :{current_cash:<2.0f}\n")
                    continue

            elif is_Short:
                # 익절
                if float(low) <= float(lowerband) <= float(high):
                    is_bought = False
                    suik = current_cash * (in_price / lowerband - 1 - 0.0008) * lev * 0.5  # 수익 계산
                    current_cash += suik
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    win_count += 1
                    print(row['Time'], f"숏 승리! 익절가 :{lowerband:<2.0f}, 수익 :{suik:<2.0f},              현재 시드 :{current_cash:<2.0f}\n")
                    continue
                # 손절
                elif low <= sonjul <= high:
                    is_bought = False
                    current_cash += current_cash * (in_price / sonjul - 1 - 0.0008) * lev * 0.5
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    lose_count += 1
                    print(row['Time'], f"숏 패배...                                            현재 시드 :{current_cash:<2.0f}\n")
                    continue
    print(
        f"Trade count: {str(trade_count):<2}  Win count: {str(win_count):<2}  Lose count: {str(lose_count):<2}"
        f"  Current cash: {current_cash:<8.0f}  Highest cash: {highest_cash:<8.0f}  Lowest cash: {lowest_cash:<8.0f}"
    )

# sjh = 손절 상수, lev = 배율상수(전체시드 기준)
# greed = 볼린저밴드만 찍고 반대로 가지 않고 보통 볼린저밴드를 타고 흐른다. 따라서 볼린저밴드를 초과하는 곳을 익절라인으로 잡으면 수익이 극대화되지 않을까?
#   볼린저밴드 상단을 터치하자마자 생성되게 만든 greed_value는 백테스팅 결과가 좋지 못했다.
# 각 봉에 따른 upperband를 업데이트 하는것이 더 수익면에서 좋았다.
def backTesting_hwengbo(candles_history_info_1h, candles_history_info_1d, inc, sjh, lev, greed, isprint, k):
    is_bought = False
    is_hwengbo = False  # 횡보 중인가?

    trade_count = 0  # 매매 횟수
    win_count = 0  # 승리 횟수
    lose_count = 0  # 패배 횟수

    start_cash = 10000  # 초기 시드
    current_cash = start_cash  # 현재 시드
    highest_cash = start_cash  # 최고 시드
    lowest_cash = start_cash  # 최저 시드

    day_inclination = None  # 현재 기울기
    dayrow = None  # 당일 데이터
    yest_inc = None  # 전일 기울기

    in_price = 0  # 진입가
    sonjul = 0  # 손절가
    is_Long = False
    is_Short = False
    inc_count = 0  # 기울기 카운트. 1시간봉 종가 기준 횡보 기울기가 연속으로 몇개가 나오면 들어갈 건지에 대해

    for idx, row in candles_history_info_1h.iloc[24*30*k:24*30*(k+1)].iterrows():#24*30*k:24*30*(k+1)

        if idx < 100:
            continue

        upperband = float(row['UpperBand'])
        lowerband = float(row['LowerBand'])
        close = float(row['Close'])
        volume = float(row['Volume'][0])
        vol_sma = float(row['Vol_SMA'])
        low = float(row['Low'])
        high = float(row['High'])
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
            day_inclination = calculate_incline_1h_1d(dayrow_open, close,
                                                      yest_inc)  # 내적으로 보완. 당일 기울기 : 전일까지의 기울기평균 = 1:2
            # print(dt, day_inclination)
            # 전일 기울기 업데이트

            # 예외 처리
            if day_inclination is None:
                print("None")
                continue
            yest_inc = dayrow['Inclination'].values[0]
            # 기울기 = 횡보일 때
            if -inc < day_inclination < inc:
                inc_count += 1
            else:

                inc_count = 0
                is_hwengbo = False
                continue
            # 기울기 연속으로 얼마나 횡보였나 판단.

            if inc_count >= 1:
                is_hwengbo = True
                inc_count = 0
            else:
                continue

            # 횡보라고 판정됨 -> 매수 준비
            if is_hwengbo:
                # 1시간 봉 기준 볼린저밴드 아래에서 마감
                if lowerband > close:
                    # 거래량이 평균보다 20% 높으면 롱 진입.
                    if volume > vol_sma * 0:
                        in_price = close
                        is_bought = True
                        sonjul = close - close * 0.01 * sjh
                        is_Long = True
                        is_Short = False
                        trade_count += 1
                        if isprint:
                            print("오늘 기울기: ",day_inclination)
                            print(row['Time'], "롱 진입", in_price, f"손절가 :{sonjul:<2.0f}", "총 거래 횟수 : ", trade_count)
                        continue
                # 1시간 봉 기준 볼린저밴드 위에서 마감
                if close > upperband:
                    # 거래량이 평균보다 20% 높으면 숏 진입.
                    if volume > vol_sma * 0:
                        in_price = close
                        is_bought = True
                        sonjul = close + close * 0.01 * sjh
                        is_Long = False
                        is_Short = True
                        trade_count += 1
                        if isprint:
                            print("오늘 기울기: ", day_inclination)
                            print(row['Time'], "숏 진입", in_price,f"손절가 :{sonjul:<2.0f}", "총 거래 횟수 : ", trade_count)
                        continue
        else:
            if is_Long:

                upperband = upperband*(1+greed)

                # 익절
                if low <= upperband <= high:
                    is_bought = False
                    suik = current_cash * (upperband / in_price - 1 - 0.0008) * lev * 0.5  # 수익 계산
                    current_cash += suik
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    win_count += 1
                    if isprint:
                        print(row['Time'], f"롱 승리! 익절가 :{upperband:<2.0f}, 수익 :{suik:<2.0f},                현재 시드 :{current_cash:<2.0f}\n")
                    continue

                # 손절
                elif low <= sonjul <= high:
                    is_bought = False
                    current_cash += current_cash * (sonjul / in_price - 1 - 0.0008) * lev * 0.5
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    lose_count += 1
                    if isprint:
                        print(row['Time'], f"롱 패배...                                 현재 시드 :{current_cash:<2.0f}\n")
                    continue

            elif is_Short:
                # 익절
                lowerband = lowerband*(1-greed)

                if low <= lowerband <= high:
                    is_bought = False
                    suik = current_cash * (in_price / lowerband - 1 - 0.0008) * lev * 0.5  # 수익 계산
                    current_cash += suik
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    win_count += 1
                    if isprint:
                        print(row['Time'], f"숏 승리! 익절가 :{lowerband:<2.0f}, 수익 :{suik:<2.0f},              현재 시드 :{current_cash:<2.0f}\n")
                    continue
                # 손절
                elif low <= sonjul <= high:
                    is_bought = False
                    current_cash += current_cash * (in_price / sonjul - 1 - 0.0008) * lev * 0.5
                    if current_cash > highest_cash:
                        highest_cash = current_cash
                    elif current_cash < lowest_cash:
                        lowest_cash = current_cash
                    lose_count += 1
                    if isprint:
                        print(row['Time'], f"숏 패배...                                            현재 시드 :{current_cash:<2.0f}\n")
                    continue
    print(
        f"Trade count: {str(trade_count):<2}  Win count: {str(win_count):<2}  Lose count: {str(lose_count):<2}"
        f"  Current cash: {current_cash:<8.0f}  Highest cash: {highest_cash:<8.0f}  Lowest cash: {lowest_cash:<8.0f}"
    )


def do_trading_hwengbo(inc, sjh, client, symbol, is_print):

    global isOrdered
    global is_Long
    global is_Short

    is_hwengbo = False  # 횡보 중인가?
    is_zeongak = False # 1시간봉 종가 끝난지 얼마 안되었는가?

    day_inclination = None  # 현재 기울기
    dayrow = None  # 당일 데이터
    yest_inc = None  # 전일 기울기

    dt_day = None # 오늘 datetime

    leverage = None
    satoshi = None
    amount = None

    clockprint = False
    position_text = None

    bot.send_message(chat_id=chat_id, text="#[자동매매 봇]# - COPYRIGHT by 표영종")

    while True:

        # 현재 시각
        now = datetime.now()
        # 현재 포지션 업데이트
        get_position(symbol, is_print=False)

        if now.minute > 0:
            clockprint = False
            is_zeongak = False
        # 현재 시각이 정각 지난지 3초 이내일 때
        if now.minute == 0 and 1 < now.second < 10 and clockprint==False: # now.minute == 0 and
            print("### 정각 !", now)
            if is_Long:
                position_text = "LONG"
            elif is_Short:
                position_text = "SHORT"
            else:
                position_text = "No Position"
            bot.send_message(chat_id = chat_id, text = f"[자동매매 봇] - 현재 시각은 {now}입니다.\n포지션: {position_text}")
            is_zeongak = True
            clockprint = True
        else:
            is_zeongak = False
            time.sleep(1)

        ### 캔들 정보 가져오기 (현재)
        candles_1m, candles_5m, candles_15m, candles_1h, candles_4h, candles_1d, candles_1w = get_candles(client,
                                                                                                          symbol,
                                                                                                          limit)

        ### 보조지표 추출
        candles_info_15m = get_candle_subdatas(candles_15m)
        candles_info_1h = get_candle_subdatas(candles_1h)
        candles_info_4h = get_candle_subdatas(candles_4h)
        candles_info_1d = get_candle_subdatas(candles_1d)

        # Dataframe 계산 삑나서.. (전 봉)
        row = candles_info_1h.iloc[-2]
        upperband = row['UpperBand']
        lowerband = row['LowerBand']
        close = float(row['Close'])
        volume = float(row['Volume'][0])
        vol_sma = float(row['Vol_SMA'])
        low = row['Low']
        high = row['High']
        dt = row['Time']
        print(row)
        # 현 봉
        row2 = candles_info_1h.iloc[-1]
        upperband2 = row2['UpperBand']
        lowerband2 = row2['LowerBand']
        close2 = float(row2['Close'])
        volume2 = float(row2['Volume'][0])
        vol_sma2 = float(row2['Vol_SMA'])
        low2 = row2['Low']
        high2 = row2['High']
        dt2 = row2['Time']
        print(row2)

        # 사지 않은 상태라면
        if not isOrdered:
            # 종가가 끝나지 않았다면
            if not is_zeongak:
                continue

            if dt2.hour >= 9:  # 9시가 지나면 당일이 오늘
                dt_day = dt2.replace(hour=9, minute=0, second=0, microsecond=0)
            else:  # 9시가 안지나면 어제가 오늘
                dt2 = dt2 - timedelta(days=1, hours=dt2.hour, minutes=dt2.minute, seconds=dt2.second,
                                    microseconds=dt2.microsecond)
                dt_day = dt2.replace(hour=9, minute=0, second=0, microsecond=0)

            # 오늘 기준 어제 9시 datetime
            dt_yest = dt2 - timedelta(days=2, hours=dt2.hour, minutes=dt2.minute, seconds=dt2.second,
                                     microseconds=dt2.microsecond)
            dt_yest = dt2.replace(hour=9, minute=0, second=0, microsecond=0)

            # 전날 일봉 기울기 가져오기
            yestrow = candles_info_1d[candles_info_1d['Time'] == dt_yest]
            yest_inc = yestrow['Inclination'].values[0]
            # 당일 일봉 OPEN 값 가져오기
            dayrow = candles_info_1d[candles_info_1d['Time'] == dt_day]
            dayrow_open = dayrow['Open'].values[0]
            # 기울기 계산(1시간 봉 종가 기준마다)
            day_inclination = calculate_incline_1h_1d(dayrow_open, close,
                                                      yest_inc)  # 내적으로 보완. 당일 기울기 : 전일까지의 기울기평균 = 1:2
            # 기울기 = 횡보일 때
            if -inc < day_inclination < inc:
                is_hwengbo = True
            else:
                continue

            # 횡보라고 판정됨 -> 매수 준비
            if is_hwengbo:
                print("횡보!")

                # 1시간 봉 기준 볼린저밴드 아래에서 마감
                if lowerband > close:
                    print("롱 각")
                    # 거래량이 평균보다 20% 높으면 롱 진입.
                    if volume > vol_sma * 0: # 이게 오히려 낫다.

                        leverage, amount = get_lev_amt(symbol, True)
                        in_price = close
                        is_bought = True
                        sonjul = math.floor(close - close * 0.01 * sjh)

                        future_create_order_market(symbol, SIDE_BUY, amount)
                        future_create_stop_loss_order(symbol, SIDE_SELL, amount, sonjul)


                        if is_print:
                            bot.send_message(chat_id = chat_id, text = f"[자동매매 봇] - 롱 진입 \n수량: {amount}\n손절: {sonjul}")
                            print(f"롱. 양:{amount}, 손절:{sonjul}")
                    else:
                        bot.send_message(chat_id=chat_id,
                                         text=f"[자동매매 봇] - 거래량이 적습니다.\nVOL: {volume}, VOL_SMA: {vol_sma}")
                        print(f"거래량이 적습니다. VOL: {volume}, VOL_SMA: {vol_sma}")
                else:
                    bot.send_message(chat_id=chat_id, text=f"[자동매매 봇] - 볼린저 밴드 Lowerband 위입니다.\nLowerband: {lowerband}\n가격: {close}")
                    print("숏치려 했는데 볼린저밴드 아래")

                # 1시간 봉 기준 볼린저밴드 위에서 마감
                if close > upperband:
                    print("숏 각")
                    # 거래량이 평균보다 20% 높으면 숏 진입.
                    if volume > vol_sma * 0: # 이게 오히려 낫다.

                        leverage, amount = get_lev_amt(symbol, True)
                        in_price = close
                        is_bought = True
                        sonjul = math.floor(close + close * 0.01 * sjh)

                        future_create_order_market(symbol, SIDE_SELL, amount)
                        future_create_stop_loss_order(symbol, SIDE_BUY, amount, sonjul)
                        if is_print:
                            bot.send_message(chat_id = chat_id, text = f"[자동매매 봇] - 숏 진입 \n수량: {amount}\n손절: {sonjul}")
                            print(f"숏. 양:{amount}, 손절:{sonjul}")
                    else:
                        bot.send_message(chat_id=chat_id, text=f"[자동매매 봇] - 거래량이 적습니다.\nVOL: {volume}, VOL_SMA: {vol_sma}")
                        print(f"거래량이 적습니다. VOL: {volume}, VOL_SMA: {vol_sma}")
                else:
                    bot.send_message(chat_id=chat_id, text=f"[자동매매 봇] - 볼린저 밴드 Upperband 아래입니다.\nUpperband: {upperband}\n가격: {close}")
                    print("롱치려 했는데 볼린저밴드 위")
            else:
                print(f"보류. 기울기 : {day_inclination}")
                bot.send_message(chat_id=chat_id,
                                 text=f"[자동매매 봇] - 횡보가 아닙니다. : \n기울기: {day_inclination}\n가격: {close}")


        else:

            amount = get_position_amt(symbol)

            if is_Long:
                # 익절
                print(f"Low: {low2}, Upperband: {upperband2}, High: {high2}", amount)
                if float(low2) <= float(upperband2) <= float(high2):
                    future_create_order_market(symbol, SIDE_SELL, amount)
                    future_cancel_all_open_order(symbol)
                    bot.send_message(chat_id = chat_id, text = f"[자동매매 봇] - 익절 완료")
                    print("익절")
                else:
                    print("롱 진입중... ")
                    time.sleep(1)
                    continue

            elif is_Short:
                # 익절
                print(f"Low: {low2}, Lowerband: {lowerband2}, High: {high2}", abs(amount))
                if float(low2) <= float(lowerband2) <= float(high2):
                    future_create_order_market(symbol, SIDE_BUY, abs(amount))
                    future_cancel_all_open_order(symbol)
                    bot.send_message(chat_id = chat_id, text = "[자동매매 봇] - 익절 완료")
                    print("익절")
                else:

                    print("숏 진입중... ")
                    time.sleep(1)
                    continue

        print(f"보류. 기울기 : {day_inclination}")

def backTesting_bull():
    return print("Hi")

# 메인 함수
if __name__ == '__main__':

    time0 = time.time()
    ### Initiation
    # row 생략 없이 출력
    pd.set_option('display.max_rows', 20)
    # col 생략 없이 출력
    pd.set_option('display.max_columns', None)
    # 가져올 분봉 데이터의 개수 (최대 500개까지 가능)
    limit = 500
    # 캔들 데이터 가져오기
    symbol = "BTCUSDT"
    # 최대 매수 BTC / 레버리지

    # 계좌 연결
    create_client()

    ### 현재 포지션 정보
    get_position(symbol, is_print=True)
    ### Client 정보 설정 및 잔고 출력
    usdt_balance = get_usdt_balance(client, isprint=True)
    # 레버리지 변경 함수
    # leverage = change_leverage(symbol, 10)
    leverage, amount = get_lev_amt(symbol, True)

    leverage = change_leverage(symbol, 10)

    do_trading_hwengbo(5, 2, client, symbol, is_print=True)













