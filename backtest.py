from datetime import datetime, timedelta, time as dtime
import pandas as pd
import pytz
from kiteconnect import KiteConnect
import time
import matplotlib.pyplot as plt
import numpy as np

import logging
from kiteconnect import KiteConnect

logging.basicConfig(level=logging.DEBUG)

kite = KiteConnect(api_key=('API_KEY'))

print(kite.login_url())

request_token = input("Enter the request token: ")

data = kite.generate_session(request_token, api_secret='avcts2er4x6d9jlk6ls2nblf9b5c8l5o')

kite.set_access_token(data["access_token"])


instrument_token = 115876871 
interval = "day"
days = 500

india = pytz.timezone("Asia/Kolkata")
to_date = datetime.now(india)
from_date = to_date - timedelta(days=days)

records = kite.historical_data(instrument_token, from_date, to_date, interval)

data = pd.DataFrame(records)
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)


# Strategy starts here.

# --- Bollinger Band Squeeze with ADX Strategy for Kite Data ---
# No TA-Lib or backtesting.py used. All indicators and backtest logic are implemented with pandas/numpy.

def sma(series, window):
    return series.rolling(window=window, min_periods=window).mean()

def std(series, window):
    return series.rolling(window=window, min_periods=window).std()

def atr(high, low, close, window):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

def adx(high, low, close, window):
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_ = tr.rolling(window=window, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=window, min_periods=window).sum() / atr_
    minus_di = 100 * pd.Series(minus_dm).rolling(window=window, min_periods=window).sum() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_ = dx.rolling(window=window, min_periods=window).mean()
    return adx_

# --- Parameters ---
bb_window = 20
bb_std = 2.0
keltner_window = 20
keltner_atr_mult = 1.5
adx_period = 14
adx_threshold = 25
take_profit = 0.05  # 5%
stop_loss = 0.03   # 3%

# --- Indicator Calculations ---
data['bb_middle'] = sma(data['close'], bb_window)
data['bb_std'] = std(data['close'], bb_window)
data['bb_upper'] = data['bb_middle'] + bb_std * data['bb_std']
data['bb_lower'] = data['bb_middle'] - bb_std * data['bb_std']

data['kc_middle'] = sma(data['close'], keltner_window)
data['kc_atr'] = atr(data['high'], data['low'], data['close'], keltner_window)
data['kc_upper'] = data['kc_middle'] + keltner_atr_mult * data['kc_atr']
data['kc_lower'] = data['kc_middle'] - keltner_atr_mult * data['kc_atr']

data['squeeze_on'] = (data['bb_upper'] < data['kc_upper']) & (data['bb_lower'] > data['kc_lower'])
data['adx'] = adx(data['high'], data['low'], data['close'], adx_period)

# --- Backtest Logic ---
initial_cash = 100000
cash = initial_cash
position = 0  # 0: no position, 1: long, -1: short
entry_price = 0
stop_price = 0
take_price = 0
trade_log = []

for i in range(max(bb_window, keltner_window, adx_period) + 2, len(data)):
    row = data.iloc[i]
    prev_row = data.iloc[i-1]
    prev2_row = data.iloc[i-2]
    price = row['close']

    # Squeeze release detection
    squeeze_prev = prev_row['squeeze_on']
    squeeze_now = row['squeeze_on']
    squeeze_released = (squeeze_prev == True) and (squeeze_now == False)

    # Entry logic
    if position == 0 and squeeze_released and row['adx'] > adx_threshold:
        # Long breakout
        if price > row['bb_upper']:
            position = 1
            entry_price = price
            stop_price = price * (1 - stop_loss)
            take_price = price * (1 + take_profit)
            trade_log.append({'type': 'BUY', 'price': price, 'index': data.index[i]})
        # Short breakout
        elif price < row['bb_lower']:
            position = -1
            entry_price = price
            stop_price = price * (1 + stop_loss)
            take_price = price * (1 - take_profit)
            trade_log.append({'type': 'SELL', 'price': price, 'index': data.index[i]})

    # Exit logic
    if position == 1:
        # Check stop loss or take profit
        if row['low'] <= stop_price:
            cash *= (1 - stop_loss)
            trade_log.append({'type': 'SELL (SL)', 'price': stop_price, 'index': data.index[i]})
            position = 0
        elif row['high'] >= take_price:
            cash *= (1 + take_profit)
            trade_log.append({'type': 'SELL (TP)', 'price': take_price, 'index': data.index[i]})
            position = 0
    elif position == -1:
        if row['high'] >= stop_price:
            cash *= (1 - stop_loss)
            trade_log.append({'type': 'BUY (SL)', 'price': stop_price, 'index': data.index[i]})
            position = 0
        elif row['low'] <= take_price:
            cash *= (1 + take_profit)
            trade_log.append({'type': 'BUY (TP)', 'price': take_price, 'index': data.index[i]})
            position = 0

# --- Results ---
print("🌟 BACKTEST COMPLETE - Default Parameters 🌟")
print(f"Initial Cash: {initial_cash}")
print(f"Final Cash: {cash:.2f}")
print(f"Total Return: {(cash/initial_cash - 1)*100:.2f}%")
print(f"Number of Trades: {len([t for t in trade_log if t['type'] in ['BUY', 'SELL']])}")

# Print trade log summary
for t in trade_log:
    print(f"{t['index']} - {t['type']} at {t['price']:.2f}")
