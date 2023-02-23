import pandas as pd
import yfinance as yf
import datetime as dt   
import numpy as np

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

        
        
        
class Strategy(metaclass=ABCMeta):
    def __init__(self, stop_loss:float, take_profit:float):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = pd.DataFrame(columns=['Date','Ticker','Action', 'Price'])
        #Action: Buy, Sell, StopLoss, TakeProfit, BuyAll, SellAll
    @abstractmethod
    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        pass


    
#Simple implementation of Buy and Hold strategy of above Strategy class
class BuyAndHold(Strategy):
    def __init__(self,stop_loss:float=0, take_profit:float=0):
        super().__init__(stop_loss, take_profit)

    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        #get the start and end date, if the start date is before the stock data start date, use the stock data start date
        #sd = stock_data.data.index[0] if stock_data.data.index[0] > start_date else start_date
        #ed = stock_data.data.index[-1] if stock_data.data.index[-1] < end_date else end_date
        # the min and max trade date between the entered start date and end date
        sd = min( [ i  for i in stock_data.data.index if i >= start_date and i <= end_date])
        ed = max( [ i  for i in stock_data.data.index if i >= start_date and i <= end_date])

        for row in stock_data.data.iterrows():
            if row[0] == sd:
                self.trades.loc[len(self.trades.index)] = [row[0], stock_data.ticker, 'BuyAll', row[1]['Close']]
            elif row[0] == ed:
                self.trades.loc[len(self.trades.index)] = [row[0], stock_data.ticker, 'SellAll', row[1]['Close']]

class MACross(Strategy):
    def __init__(self, short_window:int, long_window:int, stop_loss:float=0, take_profit:float=0):
        super().__init__(stop_loss, take_profit)
        self.short_window = short_window
        self.long_window = long_window
    def run_strategy(self, stock_data:StockData, start_date:dt.datetime, end_date:dt.datetime):
        #get the start and end date, if the start date is before the stock data start date, use the stock data start date
        #sd = stock_data.data.index[0] if stock_data.data.index[0] > start_date else start_date
        #ed = stock_data.data.index[-1] if stock_data.data.index[-1] < end_date else end_date
        # the min and max trade date between the entered start date and end date
        sd = min( [ i  for i in stock_data.data.index if i >= start_date and i <= end_date])
        ed = max( [ i  for i in stock_data.data.index if i >= start_date and i <= end_date])
        
        #Calculate the short and long moving average
        stock_data.data['ShortMA'] = stock_data.data['Close'].rolling(window=self.short_window).mean()
        stock_data.data['LongMA'] = stock_data.data['Close'].rolling(window=self.long_window).mean()
        #Calculate the signal
        stock_data.data['Signal'] = 0.0
        stock_data.data['Signal'] = np.where(stock_data.data['ShortMA'] > stock_data.data['LongMA'], 1.0, 0.0)
        #Calculate the Sell signal
        stock_data.data['Signal'] = np.where(stock_data.data['ShortMA'] < stock_data.data['LongMA'], -1.0, stock_data.data['Signal'])

        #Generate the trade list
        for row in stock_data.data.iterrows():
 #           if row[1]['Date'] < sd or row[1]['Date'] > ed:
 #               continue
            if row[1]['Signal'] == 1:
                self.trades.loc[len(self.trades.index)] = [row[0], stock_data.ticker, 'Buy', row[1]['Close']]
            elif row[1]['Signal'] == -1:
                self.trades.loc[len(self.trades.index)] = [row[0], stock_data.ticker, 'Sell', row[1]['Close']]       

class Portfolio(metaclass=ABCMeta):
    def __init__(self, principal, trade_size, prymiding, margin):
        self.principal = principal
        self.prymiding = prymiding
        self.trade_size = trade_size
        self.margin = margin
        #Real account balance should be Cash+Stock-Margin = Total
        self.balance = pd.DataFrame(columns=['Date','Cash','Stock','Total','Margin'])
        
    @abstractmethod
    def run_backtest(self, strategy:Strategy, stock_data:StockData):
        pass


class BackTest(Portfolio):
    def __init__(self, sd:dt.datetime, ed:dt.datetime, principal=1000000, trade_size=1 , prymiding=1):
        super().__init__(principal, trade_size, prymiding, 0)
        self.prymiding_count = 0
        self.start_date = sd
        self.end_date = ed
        
    def run_backtest(self, strategy:Strategy, stock_data:StockData):
        sd = min( [ i  for i in stock_data.data.index if i >= self.start_date and i <= self.end_date])
        ed = max( [ i  for i in stock_data.data.index if i >= self.start_date and i <= self.end_date])
        self.balance['Date'] = stock_data.data.loc[sd:ed].index
        
        #self.balance['Date'] = np.where((stock_data.data.index >= self.start_date) & (stock_data.data.index <= self.end_date), stock_data.data.index, None)
        
        self.balance['Cash'] = 0
        self.balance['Stock'] = 0
        self.balance['Total'] = 0
        self.balance['Margin'] = 0
        
        self.balance.loc[0, 'Cash'] = self.principal
        self.balance.loc[0, 'Stock'] = 0                
        self.balance.loc[0, 'Total'] = self.principal
        

        x = self.balance.iterrows()
        y = strategy.trades.iterrows()
        i = next(x, None)
        j = next(y, None)
        while i is not None and j is not None:
 
            if i[1]['Date'] == j[1]['Date']:
                if j[1]['Action'] == 'BuyAll':
                    self.balance.loc[i[0], 'Stock'] = self.trade_size * self.balance.loc[i[0], 'Cash'] / j[1]['Price']
                    self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0], 'Cash'] - self.trade_size * self.balance.loc[i[0], 'Cash']
                    self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0], 'Cash'] + self.balance.loc[i[0], 'Stock'] * j[1]['Price']
                    

                elif j[1]['Action'] == 'SellAll':
                    self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0]-1, 'Stock'] * j[1]['Price']
                    self.balance.loc[i[0], 'Stock'] = 0
                    self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0], 'Cash'] + self.balance.loc[i[0], 'Stock'] * j[1]['Price']
                elif j[1]['Action'] == 'Buy':
                    if self.prymiding_count < self.prymiding:
                        self.prymiding_count = self.prymiding_count + 1
                        # today's stock = previous day stock + trade % * previous day cash / today price
                        self.balance.loc[i[0], 'Stock'] = self.balance.loc[i[0] - 1, 'Stock'] + self.trade_size * self.balance.loc[i[0] - 1, 'Cash'] / j[1]['Price']
                        # today's cash = previous day cash - trade % * previous day cash 
                        self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0] - 1, 'Cash'] - self.trade_size * self.balance.loc[i[0] - 1, 'Cash']
                        # today's balance = cash + stock value
                        self.balance.loc[i[0], 'Total'] = self.balance.loc[i[0], 'Cash'] + self.balance.loc[i[0], 'Stock'] * j[1]['Price']
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
                    #self.balance.loc[i[0], 'Total'] = stock_data.data.loc[i[1]['Date'], 'Close'] * self.balance.loc[i[0], 'Stock'] + self.balance.loc[i[0], 'Cash']
                    self.balance.loc[i[0], 'Total'] =  self.balance.loc[i[0]-1, 'Total']
            i = next(x, None)

    
    
        
#Test Code, need to remove
qqq = StockData('QQQ')
qqq.get_data_from_yfinance('QQQ', dt.datetime(1998,12,4), dt.datetime(2022,3,2))
#need to preprocess the csv to remove gargage data, need to reverse the order of the data.
print(qqq.data.index[0], qqq.data['Close'][0])

buy_and_hold = BuyAndHold()
print(buy_and_hold.trades)
