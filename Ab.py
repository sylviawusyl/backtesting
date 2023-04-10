#install yfinance
#pip install yfinance --upgrade --no-cache-dir
import yfinance as yf
import datetime as dt
import pytz


import numpy as np
import pandas as pd

try:
    import cupy as cp
    import cudf as cd
    print('GPU acceleration is available')
except:
    print('GPU acceleration is NOT available')
    pass

import sqlite3 as sql
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt


class StockData(object):
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = None
        self.start_date = None
        self.end_date = None

    def get_data_from_yfinance(self, ticker: str, start_date: dt.datetime, end_date: dt.datetime, interval: str = '1d'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=interval)
        self.start_date = self.data.index[0]
        self.end_date = self.data.index[-1]
        self.data['Weekday'] = self.data.index.weekday
        self.index = pd.to_datetime(self.data.index)

    def get_data_history_from_yfinance(self, ticker: str, period: str, interval: str, start_date, end_date):
        """
        :Parameters:
            period : str
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                Either Use period parameter or use start and end
            interval : str
                Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                Intraday data cannot extend last 60 days
            start: str or datetime
                Download start date string (YYYY-MM-DD) or _datetime.
                Default is 1900-01-01
                start_date = dt.datetime(2003,3,10)
            end: str str or datetime
                Download end date string (YYYY-MM-DD) or _datetime.
                Default is now
            prepost : bool
                Include Pre and Post market data in results?
                Default is False
            auto_adjust: bool
                Adjust all OHLC automatically? Default is True
            back_adjust: bool
                Back-adjusted data to mimic true historical prices
            proxy: str
                Optional. Proxy server URL scheme. Default is None
            rounding: bool
                Round values to 2 decimal places?
                Optional. Default is False = precision suggested by Yahoo!
            tz: str
                Optional timezone locale for dates.
                (default data is returned as non-localized dates)
            timeout: None or float
                If not None stops waiting for a response after given number of
                seconds. (Can also be a fraction of a second e.g. 0.01)
                Default is None.
            **kwargs: dict
                debug: bool
                    Optional. If passed as False, will suppress
                    error message printing to console.
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.Ticker(self.ticker).history(period=self.period, interval=self.interval, start=self.start_date, end=self.end_date,
                                                   prepost=False, actions=True, auto_adjust=True, back_adjust=False, proxy=None, rounding=False, timeout=None)
        if interval == '1wk':
            self.data.index = self.data.index - dt.timedelta(days=3)
        self.data['Weekday'] = self.data.index.weekday
        self.data.index = pd.to_datetime(self.data.index)


    def get_data_from_csv(self, path: str):
        self.data = pd.read_csv(path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index(['Date'], inplace=True)
        self.data.sort_index(inplace=True)

    def get_data_from_db(self, db_path: str = 'data/stock_data.db', limit: int = 100000):
        conn = sql.connect(db_path)
        print("SELECT * FROM stock_history Where Ticker='{}' limit {}".format(self.ticker, limit))
        self.data = pd.read_sql_query(
            "SELECT * FROM stock_history Where Ticker='{}' limit {}".format(self.ticker, limit), conn)
        self.data.set_index(['Date'], inplace=True)
        conn.close()

    def get_indicators(self, column='Close', ma_windows=[5, 10, 20, 50, 200], below_thresholds=[30], above_thresholds=[15]):
        for ma_window in ma_windows:
            self.data['MA{}'.format(ma_window)] = self.data[column].rolling(
                window=ma_window, min_periods=1).mean()
            self.data['price_to_MA{}'.format(
                ma_window)] = self.data[column] / self.data['MA{}'.format(ma_window)]

    def get_thresholds(self, column='Close', ma_windows=[5, 10, 20, 50, 200], below_thresholds=[30], above_thresholds=[15]):
        for below_threshold in below_thresholds:
            self.data['below{}'.format(below_threshold)] = np.where(
                self.data[column] < below_threshold, 1, 0)
        for above_threshold in above_thresholds:
            self.data['above{}'.format(above_threshold)] = np.where(
                self.data[column] > above_threshold, 1, 0)
    def get_sma(self, column='Close', sma_window=21):
        self.data['{}-SMA{}'.format(column, sma_window)] = self.data[column].rolling(
            window=sma_window, min_periods=1).mean()
    def get_ema(self, column='Close', ema_window=21):
        self.data['{}-EMA{}'.format(column, ema_window)] = self.data[column].ewm(
            span=ema_window, adjust=False).mean()
    def get_k(self, column='Close', k_window=14):
        #K(fast line) = Close - 14Days low / 14Days high - 14Days low * 100
        self.data['{}-K{}'.format(column, k_window)] = (self.data[column] - self.data[column].rolling(window=k_window, min_periods=1).min()) / (self.data[column].rolling(window=k_window, min_periods=1).max() - self.data[column].rolling(window=k_window, min_periods=1).min()) * 100

class Strategy(metaclass=ABCMeta):
    def __init__(self, name: str, stop_loss: float, take_profit: float):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = pd.DataFrame(columns=['Date', 'Signal'])
        self.trades.set_index(['Date'], inplace=True)
        self.name = name
        self.joined_data = None
        # Action: Buy, Sell, StopLoss, TakeProfit, BuyAll, SellAll

    @abstractmethod
    def run_strategy(self, indicators, start_date: dt.datetime, end_date: dt.datetime , verbose: bool = False):
        pass

# Simple implementation of Buy and Hold strategy of above Strategy class, run in pandas


class BuyAndHold(Strategy):
    def __init__(self, name: str = 'Buy and Hold', stop_loss: float = 0, take_profit: float = 0):
        super().__init__(name, stop_loss, take_profit)

    def run_strategy(self, indicator, start_date: dt.datetime, end_date: dt.datetime):
        # get the start and end date, if the start date is before the stock data start date, use the stock data start date
        # the min and max trade date between the entered start date and end date
        sd = max(start_date, indicator.data.index[0])
        ed = min(end_date, indicator.data.index[-1])

        # check if sd is in indicator index
        if sd not in indicator.data.index:
            sd = indicator.data.index[indicator.data.index.get_loc(
                sd, method='backfill')]

        self.trades = pd.DataFrame({'Date': [sd, ed], 'Signal': [1, -1]})
        self.trades.set_index(['Date'], inplace=True)


class MACross(Strategy):
    def __init__(self, name: str = 'MA Cross', short_window: int = 50, long_window: int = 200, stop_loss: float = 0, take_profit: float = 0):
        stg_name = '{} {}/{}'.format(name, short_window, long_window)
        super().__init__(stg_name, stop_loss, take_profit)
        self.short_window = short_window
        self.long_window = long_window

    def run_strategy(self, indicator, start_date: dt.datetime, end_date: dt.datetime):
        # clear the trades
        self.trades = pd.DataFrame(columns=['Date', 'Signal'])
        self.trades.set_index('Date', inplace=True)
        # check if the columns are present in the indicator
        if 'MA{}'.format(self.short_window) not in indicator.data.columns:
            indicator.get_indicators('Close', ma_windows=[self.short_window])
        if 'MA{}'.format(self.long_window) not in indicator.data.columns:
            indicator.get_indicators('Close', ma_windows=[self.long_window])
        # get the start and end date
        # if there is input for sd and ed, then filter the data for the date range only
        if start_date:
            self.joined_data = indicator.data.loc[(indicator.data.index >= start_date)
                                                  & (indicator.data.index <= end_date)].copy()
        else:
            self.joined_data = indicator.data.copy()

        # Calculate the short and long moving average
        self.joined_data['ShortMA'] = indicator.data['MA{}'.format(
            self.short_window)]
        self.joined_data['LongMA'] = indicator.data['MA{}'.format(
            self.long_window)]

        # Calculate the signal
        self.joined_data['Signal'] = 0.0
        # Calculate the Buy signal, if the short moving average is greater than the long moving average first time, then buy
        self.joined_data['Signal'] = np.where(
            self.joined_data['ShortMA'] > self.joined_data['LongMA'], 1.0, 0.0)
        # Calculate the Sell signal
        self.joined_data['Signal'] = np.where(
            self.joined_data['ShortMA'] < self.joined_data['LongMA'], -1.0, self.joined_data['Signal'])

        self.trades = self.joined_data[['Signal']]


class MAThreshold(Strategy):
    def __init__(self, name='MA TH', ma_window: int = 20, buy_threshold: float = 1, sell_threshold: float = 1, stop_loss: float = 0, take_profit: float = 0):
        stg_name = '{} {}/{} MA {}'.format(name,
                                           buy_threshold, sell_threshold, ma_window)
        super().__init__(stg_name, stop_loss, take_profit)
        self.ma_window = ma_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def run_strategy(self, indicator, start_date: dt.datetime, end_date: dt.datetime):
        self.trades = pd.DataFrame(columns=['Date', 'Ticker', 'Signal'])
        self.trades.set_index(['Date'], inplace=True)
        self.stock_ticker = indicator.ticker
        ma_str = 'MA{}'.format(self.ma_window)
        if ma_str not in indicator.data.columns:
            indicator.get_indicators('Close', ma_windows=[self.ma_window])

        if start_date:
            self.joined_data = indicator.data.loc[(indicator.data.index >= start_date)
                                                  & (indicator.data.index <= end_date)].copy()
        else:
            self.joined_data = indicator.data.copy()

        self.joined_data['price_to_MA'] = self.joined_data['Close'] / \
            self.joined_data[ma_str]

        # Calculate the signal
        self.joined_data['Signal'] = np.where((self.joined_data['price_to_MA'] < self.sell_threshold), -1.0,
                                              np.where(
                                                  (self.joined_data['price_to_MA'] > self.buy_threshold), 1.0, 0.0)
                                              )
        self.trades = self.joined_data[['Signal']]


class Threshold(Strategy):
    def __init__(self, name: str = 'TH', buy_threshold: float = 15, sell_threshold: float = 30, signal_ma_window=20, stop_loss: float = 0, take_profit: float = 0):
        stg_name = '{} {}/{} MA {}'.format(name, buy_threshold,
                                           sell_threshold, signal_ma_window)
        super().__init__(stg_name, stop_loss, take_profit)

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.signal_ma_window = signal_ma_window

    def run_strategy(self, indicator, start_date: dt.datetime, end_date: dt.datetime):
        # clear the trades
        self.trades = pd.DataFrame(columns=['Date', 'Signal'])
        self.trades.set_index('Date', inplace=True)
        self.joined_data = indicator.data.copy()
        # it should only include dates that both signal data and stock data are available
        self.joined_data = self.joined_data.loc[start_date:end_date]
        ma_str = 'MA{}'.format(self.signal_ma_window)
        self.joined_data[ma_str] = self.joined_data['Close'].rolling(
            window=self.signal_ma_window).mean()
        # fix the NA value in the MA column, cupy is not happy with NA value
        self.joined_data[ma_str].fillna(method='bfill', inplace=True)

        # Calculate the signal
        # sell signal: price < 30 (sell threshold), and in a downward trend. use sma20>price as a proxy of downward trend
        # buy signal: price > 15 (buy threshold), and in an upward trend. use sma20<price as a proxy of upward trend
        # otherwise, when buy threshold < sell threshold, it won't work (sell rule will always dominates, eg, when price < 30, all sell)

        self.joined_data['Signal'] = 0
        self.joined_data['Signal'] = np.where((self.joined_data['Close'] > self.buy_threshold) & (
            self.joined_data[ma_str] < self.joined_data['Close']), 1.0, 0.0)
        self.joined_data['Signal'] = np.where((self.joined_data['Close'] < self.sell_threshold) & (
            self.joined_data[ma_str] > self.joined_data['Close']), -1.0, self.joined_data['Signal'])
        #rename the 'Close' to indicator.ticker
        self.joined_data.rename(columns={'Close': indicator.ticker}, inplace=True)

        # only keep the Date and Signal columns
        self.trades = self.joined_data[['Signal']]

# Stochastic Oscillator Strategy
# https://www.investopedia.com/terms/s/stochasticoscillator.asp
#https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

#Buy Rule:
#Need to meet all the following conditions:
#Daily %K above %D
#Weekly %K Going Up or Flat
#Daily %K above 20

#Sell Rule:
#Need to meet all the following conditions:
#Weekly %K under %D
#Weekly %K under 80

#Stop Loss Rule:
#TQQQ drop 10% or more in a day
class StochasticCross(Strategy):
    def __init__(self, name: str = 'Stochastic', k_window=14, full_k_window=5, full_d_window=5, overbought=80, oversold=10, stop_loss: float = 0, take_profit: float = 0, var=0, ma_notrade=0):
        super().__init__(name, stop_loss, take_profit)
        self.k_window = k_window
        self.full_k_window = full_k_window
        self.full_d_window = full_d_window
        self.oversold = oversold
        self.overbought = overbought
        self.var = var
        self.ma_notrade = ma_notrade

    def run_strategy(self, indicators, sd: dt.datetime, ed: dt.datetime):
        self.trades = pd.DataFrame(columns=['Date', 'Ticker', 'Signal'])
        self.trades.set_index('Date', inplace=True)
        daily_data = indicators[0]
        weekly_data = indicators[1]

        Fast_K_str = 'Close-K{}'.format(self.k_window)
        Fast_D_str = 'Close-K{}-EMA{}'.format(self.k_window, self.full_k_window)
        Full_K_str = Fast_D_str
        Full_D_str = '{}-EMA{}'.format(Full_K_str, self.full_d_window)
        #Daily Stochastic Oscillator:

        #Fast Stochastic Oscillator:
        #Fast %K = %K basic calculation
        #Fast %D = 5-period SMA of Fast %K
        if 'D%D' not in daily_data.data.columns:
            daily_data.get_k('Close', self.k_window)
            daily_data.get_ema(Fast_K_str, self.full_k_window)

        #Full Stochastic Oscillator:
        #Full %K = Fast %D
        #Full %D = 5-period EMA of Full %K
        if 'D%K' not in daily_data.data.columns:
            daily_data.get_ema(Full_K_str, self.full_d_window)

        daily_data.data['DClose'] = daily_data.data['Close']
        daily_data.data.rename(columns={Full_K_str: 'D%K', Full_D_str: 'D%D'}, inplace=True)

        #Weekly Stochastic Oscillator:
        if 'W%K' not in weekly_data.data.columns:
            weekly_data.get_k('Close', self.k_window)
            weekly_data.get_ema(Fast_K_str, self.full_k_window)
        if 'W%D' not in weekly_data.data.columns:
            weekly_data.get_ema(Full_K_str, self.full_d_window)

        weekly_data.data['WClose'] = weekly_data.data['Close']
        weekly_data.data.rename(columns={Full_K_str: 'W%K', Full_D_str: 'W%D'}, inplace=True)

        #join daily and weekly data together into one dataframe joined_data, keep DClose, D%K, D%D, WClose, W%K, W%D
        #keep all daily data, patch weekly data that has a corresponding daily data
        self.joined_data = daily_data.data[['DClose','D%K','D%D']].merge(weekly_data.data[['WClose','W%K','W%D']], how='left', left_index=True, right_index=True)
        #fill in the NA value in W%K, W%D with the next available value
        self.joined_data['W%K'] = self.joined_data['W%K'].fillna(method='ffill')
        self.joined_data['W%D'] = self.joined_data['W%D'].fillna(method='ffill')
        self.joined_data['WClose'] = self.joined_data['WClose'].fillna(method='ffill')

        #determin if Weekly %K Going Up or Flat, into a new column W%K-UP
        self.joined_data['W%K-UP'] = np.where(self.joined_data['W%K'] >= self.joined_data['W%K'].shift(1), 1.0, 0.0)

        if self.var == 0:
            #buy rule: Daily %K above %D, Weekly %K Going Up, Daily %K above oversold
            self.joined_data['Signal'] = np.where((self.joined_data['D%K'] > self.joined_data['D%D']) & (self.joined_data['W%K-UP'] == 1.0) & (self.joined_data['D%K'] > self.oversold), 1.0, 0.0)
            #sell rule: Weekly %K under %D, Weekly %K under overbought
            self.joined_data['Signal'] = np.where((self.joined_data['W%K'] < self.joined_data['W%D']) & (self.joined_data['W%K'] < self.overbought), -1.0, self.joined_data['Signal'])
        elif self.var == 1:
            #buy rule: Daily %K above %D, Daily %K above oversold
            self.joined_data['Signal'] = np.where((self.joined_data['D%K'] > self.joined_data['D%D']) & (self.joined_data['D%K'] > self.oversold), 1.0, 0.0)
            #sell rule: Daily %K under %D, Daily %K under overbought
            self.joined_data['Signal'] = np.where((self.joined_data['D%K'] < self.joined_data['D%D']) & (self.joined_data['D%K'] < self.overbought), -1.0, self.joined_data['Signal'])

        if self.ma_notrade != 0:
            #sell rule: when under moving average, sell
            daily_data.get_sma('Close', self.ma_notrade)
            self.joined_data = self.joined_data.merge(daily_data.data[['Close-SMA{}'.format(self.ma_notrade)]], how='left', left_index=True, right_index=True)
            self.joined_data['Signal'] = np.where((self.joined_data['DClose'] < self.joined_data['Close-SMA{}'.format(self.ma_notrade)]), -1.0, self.joined_data['Signal'])

        #stop loss rule: TQQQ drop 10% or more in a day
        self.joined_data['Signal'] = np.where((self.joined_data['DClose'] < self.joined_data['DClose'].shift(1) * 0.9), -1.0, self.joined_data['Signal'])
        #for resarch purpose, keep the stop loss rule in a separate column
        self.joined_data['Stoploss'] = np.where((self.joined_data['DClose'] < self.joined_data['DClose'].shift(1) * 0.9), -1.0, 0)
        #only keep sd to ed data
        self.joined_data = self.joined_data.loc[sd:ed]
        self.trades = self.joined_data[['Signal']]


class CustomizedStrategy(Strategy):
    def __init__(self, signals_df, name: str = 'Customized', stop_loss: float = 0, take_profit: float = 0):
        super().__init__(name, stop_loss, take_profit)
        self.signals_df = signals_df

    def run_strategy(self, stock_data: StockData, start_date: dt.datetime, end_date: dt.datetime):
        # make sure dates and rows are aligned in signal data and stock data
        # it should only include dates that both signal data and stock data are available
        self.joined_data = stock_data.data[['Close']].rename(columns={'Close': 'Price'}).merge(
            self.signals_df[['Signal']],
            how='inner', left_index=True, right_index=True).sort_index()

        # if there is input for sd and ed, then filter the data for the date range only
        if start_date:
            self.joined_data = self.joined_data.loc[(self.joined_data.index >= start_date) & (
                self.joined_data.index <= end_date)].copy()
        else:
            pass

        self.trades = self.joined_data[['Signal']]


class Portfolio(metaclass=ABCMeta):
    def __init__(self, principal, trade_size, pyramiding, margin):
        self.principal = principal
        self.pyramiding = pyramiding
        self.trade_size = trade_size
        self.margin = margin
        # Real account balance should be Cash+Stock-Margin = Total
        self.balance = pd.DataFrame(
            columns=['Date', 'Cash', 'Stock', 'Total', 'Margin'])
        self.balance.set_index('Date', inplace=True)
        self.joined_data = None
        self.name = None
        self.trade_records = pd.DataFrame(columns=[
                                          'Buy Date', 'Sell Date', 'Ticker', 'Quant', 'Buy Price', 'Sell Price', 'Profit', 'Profit %'])

    @abstractmethod
    def run_backtest(self, strategy: Strategy, stock_data: StockData):
        pass

    @abstractmethod
    def performance_summary(self, verbose=True):
        portValue = self.balance[['Total']]

        # cumulative return
        self.cumulative_return = portValue.iloc[-1] / portValue.iloc[0] - 1

        # max_drawdown
        drawdown_window = 252
        rolling_max = portValue.rolling(drawdown_window, min_periods=1).max()
        daily_drawdown = portValue/rolling_max - 1.0
        max_daily_drawdown = daily_drawdown.rolling(
            drawdown_window, min_periods=1).min()
        self.max_drawdown = max_daily_drawdown.min()

        # daily return and sharpe ratio
        daily_return = (portValue / portValue.shift(1) - 1)[1:]
        self.avg_return = daily_return.mean()
        self.std_return = daily_return.std()
        self.sharp_ratio = self.avg_return/self.std_return
        # get the first and last trading date from balance ['Date']
        start_date = self.balance.index[0]
        end_date = self.balance.index[-1]
        trading_dates = end_date - start_date
        # annual return
        if round((trading_dates.days/252)) == 0:
            self.annual_return = self.cumulative_return.values[0] + 1
        else:
            self.annual_return = np.power(
                self.cumulative_return.values[0], 1/round((trading_dates.days/252)))
        # number of trades
        self.num_trades = self.balance.Stock.nunique()

        # trade_record analysis
        # betting average
        gain = len(self.trade_records.loc[self.trade_records['Profit %'] > 0])
        loss = len(self.trade_records.loc[self.trade_records['Profit %'] <= 0])
        bet_avg = gain/(gain+loss)
        gain_avg = self.trade_records.loc[self.trade_records['Profit %'] > 0, 'Profit %'].mean()
        loss_avg = self.trade_records.loc[self.trade_records['Profit %'] <= 0, 'Profit %'].mean()
        gain_std = self.trade_records.loc[self.trade_records['Profit %'] > 0, 'Profit %'].std()
        loss_std = self.trade_records.loc[self.trade_records['Profit %'] <= 0, 'Profit %'].std()

        if verbose:
            print("""
{}:
cumulative return      : {:.2%}
compound anual return  : {:.4%}
max_drawdown           : {:.2%}
sharp_ratio            : {:.2%}
average of daily return: {:.4%}
std of daily return    : {:.4%}
number of trades       : {},
trading days           : {},
Betting Average        : {:.2%}
Gain Average           : {:.2%}
Loss Average           : {:.2%}
Risk Reward Ratio      : {:.2f}
Gain STD               : {:.2%}
Loss STD               : {:.2%}
        """.format(
                self.name,
                self.cumulative_return.values[0],
                self.annual_return - 1,
                self.max_drawdown.values[0],
                self.sharp_ratio.values[0],
                self.avg_return.values[0],
                self.std_return.values[0],
                self.num_trades,
                trading_dates.days,
                bet_avg,
                gain_avg,
                loss_avg,
                gain_avg/-loss_avg,
                gain_std,
                loss_std
            ))

        stats_names = ['name', 'num_trades',
                       'cumulative_return', 'annual_return', 'max_drawdown',
                       'sharp_ratio',  'avg_daily_return',
                       'std_daily_return', 'num_trading_days',
                       'Betting Average',
                       'Gain Average',
                       'Loss Average',
                       'Risk Reward Ratio',
                       'Gain STD',
                       'Loss STD',
                       ]

        stats = [self.name, self.num_trades,
                 self.cumulative_return.values[0], self.annual_return -
                 1, self.max_drawdown.values[0],
                 self.sharp_ratio.values[0], self.avg_return.values[0], self.std_return.values[0],
                 trading_dates.days,
                 bet_avg,
                 gain_avg,
                 loss_avg,
                 gain_avg/-loss_avg,
                 gain_std,
                 loss_std
                 ]

        self.summary_result = pd.DataFrame([stats], columns=stats_names)


class BackTest(Portfolio):
    def __init__(self, principal=1, trade_size=1, pyramiding=1):
        super().__init__(principal, trade_size, pyramiding, 0)
        self.pyramiding_count = 0

    def __record_buy(self, ticker: str, date, price: float, quantity: float):
        # put the buy order into the trade list
        self.trade_records.loc[len(self.trade_records)] = [date, np.nan, ticker, quantity, price, np.nan, np.nan, np.nan]
        self.balance.loc[date, 'Trade'] = 1

    def __record_sell(self, ticker: str, date, price: float, quantity: float):
        self.balance.loc[date, 'Trade'] = -1
        # put the sell order into the trade list,find the last buy order with empty sell date
        i = self.trade_records[self.trade_records['Ticker'] ==ticker][self.trade_records['Sell Date'].isnull()].index
        if i.empty:
            print('No buy order for ticker {} on date {}'.format(ticker, date))
            return
        self.trade_records.loc[i, 'Sell Date'] = date
        self.trade_records.loc[i, 'Sell Price'] = price
        self.trade_records.loc[i, 'Profit'] = (price - self.trade_records.loc[i, 'Buy Price'].values[0]) * quantity
        self.trade_records.loc[i, 'Profit %'] = (price - self.trade_records.loc[i, 'Buy Price'].values[0]) / self.trade_records.loc[i, 'Buy Price'].values[0]

    def __copy_balance(self, i, cash, stock, total):
        self.balance.loc[i[0], 'Cash'] = cash
        self.balance.loc[i[0], 'Stock'] = stock
        self.balance.loc[i[0], 'Total'] = total

    def run_backtest(self, strategy: Strategy, stock_data: StockData, start_date, end_date, weekly_buy=False, weekly_sell=False, verbose=False):
        self.name = strategy.name
        sd = max(start_date, stock_data.data.index.min())
        ed = min(end_date, stock_data.data.index.max())



        #copy the stock data index to the self.balance['Date'] within the given date range
        self.balance = stock_data.data[['Close','Weekday']].loc[sd:ed].copy()
        #rename Close to stocker_data.ticker
        self.balance.rename(columns={'Close': stock_data.ticker}, inplace=True)

        #merge the strategy signal to the balance
        self.balance = self.balance.merge(strategy.trades['Signal'], how='left', left_index=True, right_index=True)

        #fill Signal the NaN with 0
        self.balance['Signal'].fillna(0, inplace=True)

        self.balance['Cash'] = 0
        self.balance['Stock'] = 0
        self.balance['Total'] = 0
        self.balance['Margin'] = 0
        self.balance['Trade'] = 0
        self.balance['Buy Price'] = 0
        self.balance['Profit'] = 0

        self.balance.loc[self.balance.index[0], 'Cash'] = self.principal
        self.balance.loc[self.balance.index[0], 'Total'] = self.principal

        p_cash = self.principal
        p_stock = 0
        #iterate through the balance
        for i in self.balance.iterrows():
            c_price = i[1][stock_data.ticker]
            if(i[1]['Signal'] == 1 and self.pyramiding_count < self.pyramiding):
                self.pyramiding_count = self.pyramiding_count + 1
                self.balance.loc[i[0],'Stock'] = p_stock + p_cash / c_price
                self.balance.loc[i[0], 'Cash'] = 0
                self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0],'Cash'] + self.balance.loc[i[0], 'Stock'] * c_price
                self.__record_buy(stock_data.ticker, i[0], c_price,  self.balance.loc[i[0], 'Stock'])
                self.balance.loc[i[0], 'Buy Price'] = self.balance.loc[i[0], 'Total']
                if (verbose):
                    print('{} Buy {}'.format(i[0], self.balance.loc[i[0], 'Stock']))

            elif(i[1]['Signal'] == -1 and self.pyramiding_count > 0):
                self.balance.loc[i[0], 'Stock'] = 0
                self.balance.loc[i[0],'Cash'] = p_cash + p_stock * c_price
                self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0],'Cash'] + self.balance.loc[i[0], 'Stock'] * c_price
                self.pyramiding_count = self.pyramiding_count - 1
                self.__record_sell(stock_data.ticker, i[0], c_price, p_stock)
                if (verbose):
                    print('{} Sell {}'.format(i[0], p_stock))
            else:
                if verbose:
                    print('{} No action'.format(i[0]))
                self.__copy_balance(i, p_cash, p_stock,p_cash + p_stock * c_price)

            p_stock = self.balance.loc[i[0], 'Stock']
            p_cash = self.balance.loc[i[0], 'Cash']

        if self.balance.iloc[-1]['Stock'] > 0:
            # if there is still stock left, force to sell it
            self.balance.loc[self.balance.index[-1], 'Total'] = self.balance.iloc[-1]['Stock'] * self.balance.iloc[-1][stock_data.ticker]
            self.balance.loc[self.balance.index[-1], 'Stock'] = 0
            self.balance.loc[self.balance.index[-1], 'Cash'] = 0
            self.__record_sell(stock_data.ticker, self.balance.index[-1], self.balance.iloc[-1][stock_data.ticker], self.balance.iloc[-1]['Stock'])

        self.balance['Buy Price'] = self.balance.loc[(self.balance['Stock'] != 0) | (self.balance['Trade'] != 0), 'Buy Price'].replace(0, np.nan).ffill()
        self.balance['Profit'] = (self.balance['Total'] - self.balance['Buy Price'])/self.balance['Buy Price']
        #replace the Profit with NaN for better plot

        if(strategy.joined_data is not None):
            self.joined_data = self.balance.merge(strategy.joined_data, how='left', left_index=True, right_index=True)


    def plot_records(self):
        plt.figure(figsize=(16, 4))
        plt.bar(self.trade_records.index,
                self.trade_records['Profit %'], label=self.name)
        plt.legend()

    def plot_balance(self):
        plt.figure(figsize=(16, 4))
        plt.plot(self.balance.index, self.balance['Total'], label=self.name)
        plt.legend()
    def plot_joined_data(self, indicator_column:[str], start_date, end_date, ydash_low = None, ydash_high = None):
        plt.figure(figsize=(16, 3))
        #plot stock with indicator_columns bewteen start_date and end_date
        for i in indicator_column:
            plt.plot(self.joined_data.loc[start_date:end_date][i], label=i)

        for idx, row in self.joined_data.iterrows():
            if idx < start_date or idx > end_date:
                continue
            if row['Trade']<0:
                plt.axvline(x=idx, color = 'red', linestyle='dashed')
            if row['Trade']>0:
                plt.axvline(x=idx, color = 'green', linestyle='dashed')

        if ydash_low is not None:
            plt.axhline(y=ydash_low, color = 'black', linestyle='solid')
        if ydash_high is not None:
            plt.axhline(y=ydash_high, color = 'black', linestyle='solid')
        plt.legend()
    def performance_summary(self, v=True):
        return super().performance_summary(verbose=v)
