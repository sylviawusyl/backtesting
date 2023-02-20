import pandas as pd
import yfinance as yf
import datetime as dt   
import numpy as np
import matplotlib.pyplot as plt
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

    def get_data_from_csv(self, path:str):
        self.data = pd.read_csv(path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)


#Test Code, need to remove
qqq = StockData('QQQ')
qqq.get_data_from_csv('data/QQQ.csv')
qqq2 = StockData('QQQ')
qqq2.get_data_from_yfinance('QQQ', dt.datetime(1998,12,4), dt.datetime(2022,3,2))
#need to preprocess the csv to remove gargage data, need to reverse the order of the data.
print(qqq.data.index[0], qqq.data['Close'][0])
print(qqq2.data.index[-1], qqq2.data['Close'][-1])


class Strategy(metaclass=ABCMeta):
    def __init__(self, stop_loss:float, take_profit:float):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = pd.DataFrame(columns=['Date','Ticker','Action', 'Price'])
        #Action: Buy, Sell, StopLoss, TakeProfit, BuyAll, SellAll
    @abstractmethod
    def buyS(self, stock_data:StockData):
        pass
    @abstractmethod
    def sellS(self, stock_data:StockData):
        pass
    @abstractmethod
    def run_strategy(self, stock_data:StockData):
        pass

#Simple implementation of Buy and Hold strategy of above Strategy class
class BuyAndHold(Strategy):
    def __init__(self,stop_loss:float=0, take_profit:float=0):
        super().__init__(stop_loss, take_profit)
    def buyS(self, stock_data:StockData):
        self.trades.loc[len(self.trades.index)] = [stock_data.data.index[0], stock_data.ticker, 'BuyAll', stock_data.data['Close'][0]]

    def sellS(self, stock_data:StockData):
        self.trades.loc[len(self.trades.index)] = [stock_data.data.index[-1], stock_data.ticker, 'SellAll', stock_data.data['Close'][-1]]

    def run_strategy(self, stock_data:StockData):
        self.buyS(stock_data)
        self.sellS(stock_data)

#Test Code, need to remove:
buy_and_hold = BuyAndHold()
buy_and_hold.run_strategy(qqq2)
print(buy_and_hold.trades)

class MACross(Strategy):
    def __init__(self, short_window:int, long_window:int, stop_loss:float=0, take_profit:float=0):
        super().__init__(stop_loss, take_profit)
        self.short_window = short_window
        self.long_window = long_window

    def buyS(self, stock_data:StockData):
        pass
    def sellS(self, stock_data:StockData):
        pass
    def run_strategy(self, stock_data:StockData):
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
            if row[1]['Signal'] == 1:
                self.trades.loc[len(self.trades.index)] = [row[0], stock_data.ticker, 'Buy', row[1]['Close']]
            elif row[1]['Signal'] == -1:
                self.trades.loc[len(self.trades.index)] = [row[0], stock_data.ticker, 'Sell', row[1]['Close']]       


macross_strategy = MACross(50, 200)
macross_strategy.run_strategy(qqq2)
print(qqq2.data)
print(macross_strategy.trades)



class Portfolio(metaclass=ABCMeta):
    def __init__(self, principal, trade_size, prymiding, margin):
        self.principal = principal
        self.prymiding = prymiding
        self.trade_size = trade_size
        self.margin = margin
        #Real account balance should be Cash+Stock-Margin = Total
        self.balance = pd.DataFrame(columns=['Date','Cash','Stock','Total','Margin'])
        
    @abstractmethod
    def run_strategy(self, strategy:Strategy, stock_data:StockData):
        pass


class BackTest(Portfolio):
    def __init__(self, sd:dt.datetime, ed:dt.datetime, principal=1000000, trade_size=1 , prymiding=1):
        super().__init__(principal, trade_size, prymiding, 0)
        self.prymiding_count = 0
        
    def run_strategy(self, strategy:Strategy, stock_data:StockData):
        self.balance['Date'] = stock_data.data.index
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
                    self.balance.loc[i[0], 'Total'] = stock_data.data.loc[i[1]['Date'], 'Close'] * self.balance.loc[i[0], 'Stock'] + self.balance.loc[i[0], 'Cash']
            i = next(x, None)
        
#Test Code, need to remove:
bt = BackTest(dt.datetime(1998,12,4), dt.datetime(2022,3,2))
bt.run_strategy(buy_and_hold, qqq2)
ma_cross_bt = BackTest(dt.datetime(1998,12,4), dt.datetime(2022,3,2))
ma_cross_bt.run_strategy(macross_strategy, qqq2)

#plot the result
plt.plot(bt.balance['Date'], bt.balance['Total'])
plt.plot(ma_cross_bt.balance['Date'], ma_cross_bt.balance['Total'])
plt.savefig('BackTest.png')
