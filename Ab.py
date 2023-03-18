import pandas as pd
import yfinance as yf
import datetime as dt   
import numpy as np
import sqlite3 as sql
from abc import abstractmethod, ABCMeta

class StockData(object):
    def __init__(self, ticker:str):
        self.ticker = ticker
        self.data = None
        self.start_date = None
        self.end_date = None

    def get_data_from_yfinance(self, ticker:str, start_date:dt.datetime, end_date:dt.datetime):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.start_date = self.data.index[0]
        self.end_date = self.data.index[-1]
        #self.data.reset_index(inplace=True)
    
    def get_data_history_from_yfinance(self, ticker:str, period:str, interval:str,start_date, end_date):
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
        self.data = yf.Ticker(self.ticker).history(period= self.period,interval =self.interval,start =self.start_date,end = self.end_date,prepost=False, actions=True,auto_adjust=True, back_adjust=False,proxy=None, rounding=False, timeout=None)#.reset_index()
        self.data.index = self.data.index.strftime('%Y-%m-%d')
        self.data.index = pd.to_datetime(self.data.index)


    def get_data_from_csv(self, path:str):
        self.data = pd.read_csv(path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        #self.data.sort_values(by='Date', inplace=True)
    def get_data_from_db(self, db_path:str='data/stock_data.db', limit:int=100000):
        conn = sql.connect(db_path)
        print("SELECT * FROM stock_history Where Ticker='{}' limit {}".format(self.ticker,limit))
        self.data = pd.read_sql_query("SELECT * FROM stock_history Where Ticker='{}' limit {}".format(self.ticker,limit), conn)
        self.data.set_index(['Date'],inplace=True)
        conn.close()
        
    def get_indicators(self, column = 'Close', ma_windows = [5,10,20,50,200], below_thresholds = [30], above_thresholds = [15]):
        for ma_window in ma_windows:
            self.data['MA{}'.format(ma_window)] = self.data[column].rolling(window=ma_window).mean()
            self.data['price_to_MA{}'.format(ma_window)] = self.data[column] / self.data['MA{}'.format(ma_window)]
        for below_threshold in below_thresholds:
            self.data['below{}'.format(below_threshold)] = np.where(self.data[column]<below_threshold, 1, 0)
        for above_threshold in above_thresholds:
            self.data['above{}'.format(above_threshold)] = np.where(self.data[column]>above_threshold, 1, 0)        
        
        
class Strategy(metaclass=ABCMeta):
    def __init__(self, name:str, stop_loss:float, take_profit:float):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = pd.DataFrame(columns=['Date','Ticker','Action', 'Price'])
        self.name = name
        #Action: Buy, Sell, StopLoss, TakeProfit, BuyAll, SellAll
    @abstractmethod
    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        pass


    
#Simple implementation of Buy and Hold strategy of above Strategy class
class BuyAndHold(Strategy):
    def __init__(self, name:str='Buy and Hold', stop_loss:float=0, take_profit:float=0):
        super().__init__(name, stop_loss, take_profit)

    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        #get the start and end date, if the start date is before the stock data start date, use the stock data start date
        # the min and max trade date between the entered start date and end date
        self.trades = pd.DataFrame(columns=['Date','Ticker','Action', 'Price'])
        sd = min( [ i  for i in stock_data.data.index if i >= start_date and i <= end_date])
        ed = max( [ i  for i in stock_data.data.index if i >= start_date and i <= end_date])

        self.trades.loc[len(self.trades.index)] = [sd, stock_data.ticker, 'BuyAll', stock_data.data.loc[sd]['Close']]
        self.trades.loc[len(self.trades.index)] = [ed, stock_data.ticker, 'SellAll', stock_data.data.loc[ed]['Close']]

class MACross(Strategy):
    def __init__(self,name:str='MA Cross', short_window:int=50, long_window:int=200, stop_loss:float=0, take_profit:float=0):
        stg_name = '{} {}/{}'.format(name,short_window,long_window)
        super().__init__(stg_name, stop_loss, take_profit)
        self.short_window = short_window
        self.long_window = long_window
        
    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        #clear the trades
        self.trades = pd.DataFrame(columns=['Date','Ticker','Action', 'Price'])
        self.trades.set_index('Date', inplace=True)
        self.stock_ticker = stock_data.ticker
        #check if the columns are present in the stock data
        if 'MA{}'.format(self.short_window) not in stock_data.data.columns:
            stock_data.get_indicators('Close',ma_windows=[self.short_window])
        if 'MA{}'.format(self.long_window) not in stock_data.data.columns:
            stock_data.get_indicators('Close',ma_windows=[self.long_window])
        # get the start and end date
        # if there is input for sd and ed, then filter the data for the date range only 
        if start_date:
            self.joined_data = stock_data.data.loc[(stock_data.data.index >= start_date) 
                                                   & (stock_data.data.index <= end_date)].copy()
        else:
            self.joined_data = stock_data.data.copy()
        
        #Calculate the short and long moving average
        self.joined_data['ShortMA'] = stock_data.data['MA{}'.format(self.short_window)]
        self.joined_data['LongMA'] = stock_data.data['MA{}'.format(self.long_window)]

        #Calculate the signal
        self.joined_data['Signal'] = 0.0
        #Calculate the Buy signal, if the short moving average is greater than the long moving average first time, then buy
        self.joined_data['Signal'] = np.where(self.joined_data['ShortMA'] > self.joined_data['LongMA'], 1.0, 0.0)
        #Calculate the Sell signal
        self.joined_data['Signal'] = np.where(self.joined_data['ShortMA'] < self.joined_data['LongMA'], -1.0, self.joined_data['Signal'])

        #copy signal and close columns with index to trades
        self.trades = self.joined_data[['Signal', 'Close']].copy()
        #drop signal = 0 rows
        self.trades = self.trades[self.trades['Signal'] != 0]
        self.trades['Action'] = np.where(self.trades['Signal'] == 1.0, 'Buy', 
                                         np.where(self.trades['Signal'] == -1.0, 'Sell', 'Hold'))
        #drop signal column Signal
        self.trades = self.trades.drop(columns=['Signal'])
        self.trades['Ticker'] = self.stock_ticker
        self.trades = self.trades.rename(columns={'Close':'Price'})

class MAThreshold(Strategy):
    def __init__(self, name='MA TH', ma_window:int = 20, buy_threshold:float=15, sell_threshold:float=30, stop_loss:float=0, take_profit:float=0):
        stg_name = '{} {}/{} MA {}'.format(name,buy_threshold,sell_threshold,ma_window)
        super().__init__(stg_name, stop_loss, take_profit)
        self.ma_window = ma_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        self.trades = pd.DataFrame(columns=['Date','Ticker','Action', 'Price'])
        self.stock_ticker = stock_data.ticker
        ma_str = 'MA{}'.format(self.ma_window)
        if ma_str not in stock_data.data.columns:
            stock_data.get_indicators('Close',ma_windows=[self.ma_window])

        if start_date:
            self.joined_data = stock_data.data.loc[(stock_data.data.index >= start_date) 
                                                   & (stock_data.data.index <= end_date)].copy()
        else:
            self.joined_data = stock_data.data.copy()
            
        self.joined_data['price_to_MA'] = self.joined_data['Close'] / self.joined_data[ma_str]

        #Calculate the signal
        self.joined_data['Signal'] = np.where((self.joined_data['price_to_MA'] < self.sell_threshold) , -1.0, 
                                            np.where((self.joined_data['price_to_MA'] > self.buy_threshold), 1.0, 0.0)
                                            )
                                            
        #Generate the trade list
        for row in self.joined_data.iterrows():
            if row[1]['Signal'] == 1:
                self.trades.loc[len(self.trades.index)] = [row[0], self.stock_ticker, 'Buy', row[1]['Close']]
            elif row[1]['Signal'] == -1:
                self.trades.loc[len(self.trades.index)] = [row[0], self.stock_ticker, 'Sell', row[1]['Close']]      

class WeeklyMAThreshold(Strategy):
    def __init__(self, name:str='W MA TH', ma_window:int=20,  buy_threshold:float=15, sell_threshold:float=30,stop_loss:float=0, take_profit:float=0):
        stg_name = '{} {}/{} MA {}'.format(name,buy_threshold,sell_threshold,ma_window)
        super().__init__(stg_name, stop_loss, take_profit)
        self.ma_window = ma_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        self.trades = pd.DataFrame(columns=['Date','Ticker','Action', 'Price'])
        self.stock_ticker = stock_data.ticker
        ma_str = 'MA{}'.format(self.ma_window)
        if ma_str not in stock_data.data.columns:
            stock_data.get_indicators('Close',ma_windows=[self.ma_window])
        if start_date:
            self.joined_data = stock_data.data.loc[(stock_data.data.index >= start_date) 
                                                   & (stock_data.data.index <= end_date)].copy()
        else:
            self.joined_data = stock_data.data.copy()
        
        self.joined_data['dayofweek'] = self.joined_data.index.dayofweek
        self.joined_data =  self.joined_data[self.joined_data['dayofweek']==4].copy()
        
        self.joined_data['price_to_MA'] = self.joined_data['Close'] / self.joined_data[ma_str]
        self.joined_data['price_to_MA_long'] = self.joined_data.apply(lambda row: 1 if row['price_to_MA']>self.buy_threshold else 0, axis =1)
        self.joined_data['price_to_MA_short'] = self.joined_data.apply(lambda row: 1 if row['price_to_MA']<self.sell_threshold else 0, axis =1)
        
        self.joined_data['MA_price_to_MA_long'] = self.joined_data['price_to_MA_long'].rolling(window=2).mean()
        self.joined_data['MA_price_to_MA_short'] = self.joined_data['price_to_MA_short'].rolling(window=2).mean()
        

        #Calculate the signal
        self.joined_data['Signal'] = np.where((self.joined_data['MA_price_to_MA_short'] ==1) , -1.0, 
                                            np.where((self.joined_data['MA_price_to_MA_long'] ==1), 1.0, 0.0) )
                                            
        #Generate the trade list
        for row in self.joined_data.iterrows():
            if row[1]['Signal'] == 1:
                self.trades.loc[len(self.trades.index)] = [row[0], self.stock_ticker, 'Buy', row[1]['Close']]
            elif row[1]['Signal'] == -1:
                self.trades.loc[len(self.trades.index)] = [row[0], self.stock_ticker, 'Sell', row[1]['Close']] 
                
                
class Threshold(Strategy):
    def __init__(self, signal_data:StockData,indicator, name:str='TH', buy_threshold:float=15, sell_threshold:float=30, signal_ma_window = 20, stop_loss:float=0, take_profit:float=0):
        stg_name = '{} {}/{} MA {}'.format(name,buy_threshold,sell_threshold,signal_ma_window)
        super().__init__(stg_name, stop_loss, take_profit)
        self.signal_data = signal_data
        self.indicator = indicator
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.signal_ma_window = signal_ma_window

    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        #clear the trades
        self.trades = pd.DataFrame(columns=['Date','Ticker','Action', 'Price'])

        self.stock_ticker = stock_data.ticker
        self.signal_ticker = self.signal_data.ticker
       
        # make sure dates and rows are aligned in signal data and stock data
        # it should only include dates that both signal data and stock data are available 
        self.joined_data = stock_data.data[['Close']].rename(columns={'Close':'StockPrice'}).merge(
                                             self.signal_data.data[['Close']].rename(columns = {'Close':'SignalPrice'}),
                                             how = 'inner', left_index = True, right_index = True).sort_index()

        # if there is input for sd and ed, then filter the data for the date range only 
        if start_date:
            self.joined_data = self.joined_data.loc[(self.joined_data.index >= start_date) & (self.joined_data.index <= end_date)].copy()
        else:
            pass

        #Calculate the indicator (TODO: here the indicator comes from the signal data price, in the future can use other indicators passed to the function)
       
        # self.joined_data['Indicator'] = self.joined_data[self.indicator]
        self.joined_data['SignalMA'] = self.joined_data['SignalPrice'].rolling(window=self.signal_ma_window).mean()

        #Calculate the signal
        # sell signal: price < 30 (sell threshold), and in a downward trend. use sma20>price as a proxy of downward trend
        # buy signal: price > 15 (buy threshold), and in an upward trend. use sma20<price as a proxy of upward trend
        # otherwise, when buy threshold < sell threshold, it won't work (sell rule will always dominates, eg, when price < 30, all sell)
        # (TODO) is there a better way to define the signal or a better proxy of downward/upward trend ?
        self.joined_data['Signal'] = np.where((self.joined_data['SignalPrice'] < self.sell_threshold) & (self.joined_data['SignalMA']>self.joined_data['SignalPrice']), -1.0, 
                                            np.where((self.joined_data['SignalPrice'] > self.buy_threshold)  & (self.joined_data['SignalMA']<self.joined_data['SignalPrice']), 1.0, 0.0)
                                            )
        #Generate the trade list
        #first copy the joined_data to trades , only keep the index, Signal, StockPrice
        self.trades = self.joined_data[['Signal','StockPrice']].copy()
        #drop signal = 0 rows
        self.trades = self.trades[self.trades['Signal'] != 0]
        self.trades['Action'] = np.where(self.trades['Signal'] == 1.0, 'Buy', 
                                         np.where(self.trades['Signal'] == -1.0, 'Sell', 'Hold'))
        #drop signal column Signal
        self.trades = self.trades.drop(columns=['Signal'])
        self.trades['Ticker'] = self.stock_ticker
        self.trades = self.trades.rename(columns={'StockPrice':'Price'})

class CustomizedStrategy(Strategy):
    def __init__(self, signals_df, name:str='Customized',stop_loss:float=0, take_profit:float=0):
        super().__init__(name, stop_loss, take_profit)
        self.signals_df = signals_df

    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        self.stock_ticker = stock_data.ticker
       
        # make sure dates and rows are aligned in signal data and stock data
        # it should only include dates that both signal data and stock data are available 
        self.joined_data = stock_data.data[['Close']].rename(columns={'Close':'Price'}).merge(
                                             self.signals_df[['Signal']],
                                             how = 'inner', left_index = True, right_index = True).sort_index()

        # if there is input for sd and ed, then filter the data for the date range only 
        if start_date:
            self.joined_data = self.joined_data.loc[(self.joined_data.index >= start_date) & (self.joined_data.index <= end_date)].copy()
        else:
            pass

        # The signals are already from signals_df: self.joined_data['Signal']

        #Generate the trade list
        #first copy the joined_data to trades , only keep the index, Signal, Price & drop signal = 0 rows
        self.trades = self.joined_data.loc[self.joined_data.Signal!=0, ['Signal','Price']].copy()
        self.trades['Ticker'] = self.stock_ticker
        self.trades['Action'] = np.where(self.trades['Signal'] == 1.0, 'Buy', 
                                         np.where(self.trades['Signal'] == -1.0, 'Sell', 'Hold'))
        #drop signal column Signal
        self.trades = self.trades.drop(columns=['Signal'])
        

class Portfolio(metaclass=ABCMeta):
    def __init__(self, principal, trade_size, prymiding, margin):
        self.principal = principal
        self.prymiding = prymiding
        self.trade_size = trade_size
        self.margin = margin
        #Real account balance should be Cash+Stock-Margin = Total
        self.balance = pd.DataFrame(columns=['Date','Cash','Stock','Total','Margin'])
        self.name = None
        self.trade_records = pd.DataFrame(columns=['Buy Date','Sell Date','Ticker','Quant','Buy Price','Sell Price','Profit', 'Profit %'])
        
    @abstractmethod
    def run_backtest(self, strategy:Strategy, stock_data:StockData):
        pass
    
    @abstractmethod
    def performance_summary(self, start_date, end_date, verbose = True):
        portValue = self.balance[['Total']]
        
        # cumulative return
        self.cumulative_return = portValue.iloc[-1] / portValue.iloc[0] - 1
        
        # max_drawdown
        drawdown_window = 252
        rolling_max = portValue.rolling(drawdown_window, min_periods=1).max()
        daily_drawdown = portValue/rolling_max - 1.0
        max_daily_drawdown = daily_drawdown.rolling(drawdown_window, min_periods=1).min()
        self.max_drawdown = max_daily_drawdown.min()

        # daily return and sharpe ratio
        daily_return = (portValue / portValue.shift(1) - 1) [1:]
        self.avg_return = daily_return.mean()
        self.std_return = daily_return.std()
        self.sharp_ratio = self.avg_return/self.std_return
        trading_dates = end_date - start_date
        # annual return
        if round((trading_dates.days/252)) == 0:
            self.annual_return = self.cumulative_return.values[0] + 1
        else:
            self.annual_return = np.power(self.cumulative_return.values[0], 1/round((trading_dates.days/252)))
        # number of trades
        self.num_trades = self.balance.Stock.nunique()
        
        #trade_record analysis
        #betting average
        gain = len(self.trade_records.loc[self.trade_records['Profit %']> 0])
        loss = len(self.trade_records.loc[self.trade_records['Profit %']<= 0])
        bet_avg = gain/(gain+loss)
        gain_avg = self.trade_records.loc[self.trade_records['Profit %'] > 0, 'Profit %'].mean()
        loss_avg = self.trade_records.loc[self.trade_records['Profit %'] <= 0,'Profit %'].mean()
        gain_std = self.trade_records.loc[self.trade_records['Profit %'] > 0, 'Profit %'].std()
        loss_std = self.trade_records.loc[self.trade_records['Profit %'] <= 0,'Profit %'].std()                                                                                      

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
        
        stats_names = [ 'name','num_trades',
        'cumulative_return','annual_return','max_drawdown',
        'sharp_ratio',  'avg_daily_return', 
        'std_daily_return','num_trading_days',
        'Betting Average',
        'Gain Average',
        'Loss Average',
        'Risk Reward Ratio',
        'Gain STD',
        'Loss STD',
        ]

        stats = [self.name, self.num_trades,
                self.cumulative_return.values[0], self.annual_return - 1, self.max_drawdown.values[0],
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
    def __init__(self, sd:dt.datetime, ed:dt.datetime, principal=1, trade_size=1 , prymiding=1):
        super().__init__(principal, trade_size, prymiding, 0)
        self.prymiding_count = 0
        self.start_date = sd
        self.end_date = ed
        

    def __record_buy(self, ticker:str, date:dt.datetime, price:float, quantity:float):
        #put the buy order into the trade list
        self.trade_records.loc[len(self.trade_records)] = [date, np.nan, ticker, quantity, price, np.nan, np.nan, np.nan]

    def __record_sell(self, ticker:str, date:dt.datetime, price:float, quantity:float):
        #put the sell order into the trade list,find the last buy order with empty sell date
        i = self.trade_records[self.trade_records['Ticker'] == ticker][self.trade_records['Sell Date'].isnull()].index
        if i.empty:
            return
        self.trade_records.loc[i, 'Sell Date'] = date
        self.trade_records.loc[i, 'Sell Price'] = price
        self.trade_records.loc[i, 'Profit'] = (price - self.trade_records.loc[i, 'Buy Price'].values[0]) * quantity
        self.trade_records.loc[i, 'Profit %'] = (price - self.trade_records.loc[i, 'Buy Price'].values[0]) / self.trade_records.loc[i, 'Buy Price'].values[0]

    def run_backtest(self, strategy:Strategy, stock_data:StockData , weekly_buy=False, weekly_sell=False):
        sd = min( [ i  for i in stock_data.data.index if i >= self.start_date and i <= self.end_date])
        ed = max( [ i  for i in stock_data.data.index if i >= self.start_date and i <= self.end_date])
        self.name = strategy.name
        self.balance['Date'] = stock_data.data.loc[sd:ed].index
        
        self.balance['Cash'] = 0
        self.balance['Stock'] = 0
        self.balance['Total'] = 0
        self.balance['Margin'] = 0
        
        self.balance.loc[0, 'Cash'] = self.principal
        self.balance.loc[0, 'Stock'] = 0                
        self.balance.loc[0, 'Total'] = self.principal
        
        #reset strategy index
        strategy.trades = strategy.trades.reset_index(drop=False)
        x = self.balance.iterrows()
        y = strategy.trades.iterrows()
        i = next(x, None)
        j = next(y, None)
        if j is None:
            print("No action on this strategy")
            return
        
        while i is not None:
            if j is None:
                #No action on this day, copy the previous day's row value except for Date
                self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0]-1, 'Cash']
                self.balance.loc[i[0], 'Stock'] =  self.balance.loc[i[0]-1, 'Stock']
                self.balance.loc[i[0], 'Total'] = stock_data.data.loc[i[1]['Date'], 'Close'] * self.balance.loc[i[0], 'Stock'] + self.balance.loc[i[0], 'Cash']
            elif i[1]['Date'] == j[1]['Date']:
                if (weekly_buy and i[1]['Date'].weekday() != 4 and j[1]['Action'] == 'Buy') or (weekly_sell and i[1]['Date'].weekday() != 4 and j[1]['Action'] == 'Sell'):
                    #No action on this day, copy the previous day's row value except for Date
                    self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0]-1, 'Cash']
                    self.balance.loc[i[0], 'Stock'] =  self.balance.loc[i[0]-1, 'Stock']
                    self.balance.loc[i[0], 'Total'] = stock_data.data.loc[i[1]['Date'], 'Close'] * self.balance.loc[i[0], 'Stock'] + self.balance.loc[i[0], 'Cash']

                elif j[1]['Action'] == 'BuyAll':
                    self.balance.loc[i[0], 'Stock'] = self.trade_size * self.balance.loc[i[0], 'Cash'] / j[1]['Price']
                    self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0], 'Cash'] - self.trade_size * self.balance.loc[i[0], 'Cash']
                    self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0], 'Cash'] + self.balance.loc[i[0], 'Stock'] * j[1]['Price']
                    self.__record_buy(stock_data.ticker, i[1]['Date'], j[1]['Price'], self.balance.loc[i[0], 'Stock'])

                elif j[1]['Action'] == 'SellAll':
                    self.__record_sell(stock_data.ticker, i[1]['Date'], j[1]['Price'],self.balance.loc[i[0], 'Stock'])
                    self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0]-1, 'Stock'] * j[1]['Price']
                    self.balance.loc[i[0], 'Stock'] = 0
                    self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0], 'Cash'] + self.balance.loc[i[0], 'Stock'] * j[1]['Price']

                elif j[1]['Action'] == 'Buy':
                    if self.prymiding_count < self.prymiding:
                        self.prymiding_count = self.prymiding_count + 1
                        # today's stock = previous day stock + trade % * previous day cash / today price
                        if i[0] == 0:
                            self.balance.loc[i[0], 'Stock'] = self.trade_size * self.balance.loc[i[0], 'Cash'] / j[1]['Price']
                            self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0], 'Cash'] - self.balance.loc[i[0], 'Stock']*j[1]['Price']
                            self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0], 'Cash'] + self.balance.loc[i[0], 'Stock'] * j[1]['Price']
                        else:
                            self.balance.loc[i[0], 'Stock'] = self.balance.loc[i[0] - 1, 'Stock'] + self.trade_size * self.balance.loc[i[0] - 1, 'Cash'] / j[1]['Price']
                            # today's cash = previous day cash - trade % * previous day cash 
                            self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0] - 1, 'Cash'] - self.trade_size * self.balance.loc[i[0] - 1, 'Cash']
                            # today's balance = cash + stock value
                            self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0], 'Cash'] + self.balance.loc[i[0], 'Stock'] * j[1]['Price']
                            self.__record_buy(stock_data.ticker, i[1]['Date'], j[1]['Price'],  self.balance.loc[i[0], 'Stock'])
                    else:                        
                        #No action on this day, copy the previous day's row value except for Date
                        self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0]-1, 'Cash']
                        self.balance.loc[i[0], 'Stock'] =  self.balance.loc[i[0]-1, 'Stock']
                        self.balance.loc[i[0], 'Total'] = stock_data.data.loc[i[1]['Date'], 'Close'] * self.balance.loc[i[0], 'Stock'] + self.balance.loc[i[0], 'Cash']
                elif j[1]['Action'] == 'Sell':
                    if self.prymiding_count > 0:
                        # today's cash = previous day cash + trade % * previous day stock * today price
                        self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0] - 1, 'Cash'] + self.trade_size * self.balance.loc[i[0]-1, 'Stock'] * j[1]['Price']
                        # today's stock = previous day stock - trade % * previous day stock
                        self.balance.loc[i[0], 'Stock'] = self.balance.loc[i[0] - 1, 'Stock'] - self.trade_size * self.balance.loc[i[0] - 1, 'Stock']
                        self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0], 'Cash'] + self.balance.loc[i[0], 'Stock'] * j[1]['Price']
                        self.prymiding_count = self.prymiding_count - 1
                        self.__record_sell(stock_data.ticker, i[1]['Date'], j[1]['Price'],self.balance.loc[i[0] - 1, 'Stock'])
                        
                    else:
                        #No action on this day, copy the previous day's row value except for Date
                        self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0]-1, 'Cash']
                        self.balance.loc[i[0], 'Stock'] =  self.balance.loc[i[0]-1, 'Stock']
                        self.balance.loc[i[0], 'Total'] = stock_data.data.loc[i[1]['Date'], 'Close'] * self.balance.loc[i[0], 'Stock'] + self.balance.loc[i[0], 'Cash']
                else:
                    pass
                j = next(y, None)
            else:
                if i[0] == 0:
                    self.balance.loc[i[0], 'Cash'] = self.principal
                    self.balance.loc[i[0], 'Stock'] = 0
                    self.balance.loc[i[0], 'Total'] = self.principal
                else:
                    #No action on this day, copy the previous day's row value except for Date
                    self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0]-1, 'Cash']
                    self.balance.loc[i[0], 'Stock'] =  self.balance.loc[i[0]-1, 'Stock']
                    self.balance.loc[i[0], 'Total'] = stock_data.data.loc[i[1]['Date'], 'Close'] * self.balance.loc[i[0], 'Stock'] + self.balance.loc[i[0], 'Cash']
            i = next(x, None)

            if self.balance.iloc[-1]['Stock'] > 0:
                self.__record_sell(stock_data.ticker, self.end_date, stock_data.data.loc[self.end_date, 'Close'], self.balance.iloc[-1]['Stock'])

    def performance_summary(self, v = True):
        super().performance_summary(start_date=self.start_date,end_date=self.end_date,verbose=v)



        
