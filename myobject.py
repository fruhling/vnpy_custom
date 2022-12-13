"""
Basic data structure used 
"""


from dataclasses import dataclass
from datetime import datetime, date
from logging import INFO

from vnpy.trader.constant import Direction, Exchange, Offset, Status, Interval

ACTIVE_STATUSES = set([Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED])


@dataclass
class BaseData:
    """
    Any data object needs a gateway_name as source
    and should inherit base data.
    """

    gateway_name: str


@dataclass
class MainData(BaseData):
    """
    MainlData load from Dolphindb, could used by strategy.
    main  like Rqdata    RM2205
    symbol RM
    """

    symbol: str
    main: str
    datetime: datetime = None

@dataclass
class MyTradeData(BaseData):
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    strategy_class: str
    strategy_name: str
    strategy_num: str
    strategy_period: int

    symbol: str
    exchange: Exchange
    orderid: str
    tradeid: str
    direction: Direction = None

    offset: Offset = Offset.NONE
    price: float = 0.0
    volume: float = 0.0
    datetime: datetime = None

    display: bool = False
    calculate: bool = False


@dataclass
class SignData(BaseData):
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    tradingday: str
    order_time: datetime
    strategy_group: str
    strategy_id: str
    instrument: str
    period: int
    sign: str
    remark: str
    insert_time: datetime


@dataclass
class SignalData(BaseData):
    """
    SignalData load from Dolphindb, could used by strategy.
    symbol  like Rqdata    RM2205
    """

    strategy_num: str
    symbol: str
    datetime: datetime = None
    interval: Interval = None
    pos: float = 0.0


@dataclass
class MyPositionData(BaseData):
    """
    Positon data is used for tracking each individual position holding.
    """

    symbol: str
    exchange: Exchange

    direction: Direction = Direction.LONG
    volume: float = 0.0
    price: float = 0.0

    def __post_init__(self):
        """"""
        self.vt_symbol = f"{self.symbol}.{self.exchange.value}"
        self.vt_positionid = f"{self.vt_symbol}.{self.direction.value}"

@dataclass
class DailyBarData(BaseData):
    """
    Candlestick bar data of a certain trading period.
    """

    symbol: str
    exchange: Exchange
    datetime: datetime

    interval: Interval = None
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0
    settlement: float = 0
    prev_settlement: float = 0
    limit_up: float = 0
    limit_down: float = 0

    def __post_init__(self):
        """"""
        self.vt_symbol = f"{self.symbol}.{self.exchange.value}"

@dataclass
class MemberRankData(BaseData):
    """
    MemberRank data of a certain trading period.
    """

    symbol: str
    exchange: Exchange
    datetime: datetime
    rank_by: str
    
    member_name: str
    interval: Interval = None
    rank: int = 0
    volume: float = 0
    volume_change: float = 0 
    

    def __post_init__(self):
        """"""
        self.vt_symbol = f"{self.symbol}.{self.exchange.value}"

@dataclass
class TickOverview:
    """
    Overview of tick data stored in database.
    """

    symbol: str = ""
    exchange: Exchange = None
    count: int = 0
    start: datetime = None
    end: datetime = None

#kaki add
@dataclass
class DailyBarOverview:
    """
    Overview of bar data stored in database.
    """

    symbol: str = ""
    exchange: Exchange = None
    interval: Interval = None
    count: int = 0
    start: datetime = None
    end: datetime = None

@dataclass
class TrendFeaturesData(BaseData):
    """
    TrendFeatures data of a certain trading period.
    """

    index_name: str
    symbol: str
    exchange: Exchange
    datetime: datetime
    trend_point_date: datetime

    interval: Interval = None
    close_price: float = 0
    index_trend_var: str = "0"
    index_trend_now: float = 0
    trend_point_price: float = 0
    trend_temp_point_price: float = 0
    trend_cum_rate: float = 0
    trend_up_down_range: float = 0
    trend_cum_revers: float = 0
    trend_period_days: int = 0
    trend_up_nums: int = 0
    trend_down_nums: int = 0
    trend_linear_coef: float = 0
    trend_linear_r2: float = 0
    trend_linear_score: float = 0
