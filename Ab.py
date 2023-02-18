import pandas as pd
import yfinance as yf
import datetime as dt   
import csv
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
    def __init__(self, principal=1, tradeSize=0.1):
        self.principal = principal
        self.balance = []
        self.trades = []
        self.primade = 1
        self.tradeSize = tradeSize
    @abstractmethod
    def buy(strategyEvent):
        pass
    def sel(strategyEvent):
        pass