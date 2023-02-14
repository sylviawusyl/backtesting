
from abc import abstractmethod, ABCMeta
class stockData(object):
    def __init__(self):
        self.ticker = None
        self.price = None
        self.dates = None

class strategy(metaclass=ABCMeta):
    def __init__(self, stockData, stoploss, takeprofit):
        self.stoploss = stoploss
        self.takeprofit = takeprofit
        self.stockData = stockData
    @abstractmethod
    def buyS(self):
        pass
    @abstractmethod
    def sellS(self):
        pass

class strategyEvent(object):
    def __init__(self, event, ticker, eventDate):
        self.event = event
        self.ticker = ticker
        self.eventDate = eventDate

class portfolio(metaclass=ABCMeta):
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