### 이 파일은 저장용입니다. 주로 일회성 함수를 저장하는 데 쓰입니다.

# ### csv 데이터 추출
# # 디렉토리 생성
# data_dir = 'candle_data'
# os.makedirs(data_dir, exist_ok=True)
# # csv 파일 생성
# filename = "candle_data_15m.csv"
# filepath = os.path.join(data_dir, filename)
#
# with open(filepath, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
#
#     print("Open Ok")
#
#     klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, "1 Jan, 2021", "15 Jun, 2023")
#     print("Get Candles OK")
#
#     for k in klines:
#         timestamp = k[0]
#         open_price = k[1]
#         high_price = k[2]
#         low_price = k[3]
#         close_price = k[4]
#         volume = k[5]
#         writer.writerow([timestamp, open_price, high_price, low_price, close_price, volume])
#
# print("Data fetching and saving completed.")

# ### 캔들 정보 가져오기 (현재)
# candles_1m, candles_5m, candles_15m, candles_1h, candles_4h, candles_1d, candles_1w = get_candles(client, symbol,
#                                                                                                   limit)

### 캔들 정보 가져오기 (특정 시각)
# start_time = datetime(2023, 5, 20)
# end_time = datetime(2023, 6, 26)
# candles_15m = get_klines_by_date(client, symbol, limit, Client.KLINE_INTERVAL_15MINUTE, start_time, end_time)

### 보조지표 추출
# candles_info_15m = get_candle_subdatas(candles_15m)
# candles_info_1h = get_candle_subdatas(candles_1h)
# candles_info_4h = get_candle_subdatas(candles_4h)
# candles_info_1d = get_candle_subdatas(candles_1d)

# print(candles_info_15m)

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
# future_order(symbol, SIDE_BUY, 0.001, "29900")

# do_trading_hwengbo(5, 2, leverage, client, symbol)