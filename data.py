### 이 파일은 저장용입니다. 주로 일회성 함수를 저장하는 데 쓰입니다.

### csv 데이터 추출
# 디렉토리 생성
data_dir = 'candle_data'
os.makedirs(data_dir, exist_ok=True)
# csv 파일 생성
filename = "candle_data_15m.csv"
filepath = os.path.join(data_dir, filename)

with open(filepath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

    print("Open Ok")

    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, "1 Jan, 2021", "15 Jun, 2023")
    print("Get Candles OK")

    for k in klines:
        timestamp = k[0]
        open_price = k[1]
        high_price = k[2]
        low_price = k[3]
        close_price = k[4]
        volume = k[5]
        writer.writerow([timestamp, open_price, high_price, low_price, close_price, volume])

print("Data fetching and saving completed.")