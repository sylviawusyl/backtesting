import pandas as pd
import yfinance as yf
import datetime as dt   
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
        
    def run_strategy(self, strategy:Strategy, stock_data:StockData):
        self.balance['Date'] = stock_data.data.index
        self.balance['Cash'] = self.principal
        self.balance['Stock'] = 0
        self.balance['Total'] = self.principal
        self.balance['Margin'] = 0

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
                else:
                    print('Error: Action not recognized')
                j = next(y, None)
            else:
                #No action on this day, copy the previous day's row value except for Date
                self.balance.loc[i[0], 'Cash'] = self.balance.loc[i[0]-1, 'Cash']
                self.balance.loc[i[0], 'Stock'] =  self.balance.loc[i[0]-1, 'Stock']
                self.balance.loc[i[0], 'Total'] = stock_data.data.loc[i[1]['Date'], 'Close'] * self.balance.loc[i[0], 'Stock'] + self.balance.loc[i[0], 'Cash']
            i = next(x, None)
        
#Test Code, need to remove:
bt = BackTest(dt.datetime(1998,12,4), dt.datetime(2022,3,2))
bt.run_strategy(buy_and_hold, qqq2)
#plot the result
plt.plot(bt.balance['Date'], bt.balance['Total'])