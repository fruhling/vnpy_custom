""""""
from datetime import datetime
from vnpy.trader.constant import Direction, Offset
from typing import List, Dict
import copy
from pathlib import Path

from peewee import (
    AutoField,
    CharField,
    DateTimeField,
    FloatField,
    IntegerField,
    BooleanField,
    Model,
    PostgresqlDatabase as PeeweePostgresqlDatabase,
    ModelSelect,
    ModelDelete,
    fn,
    chunked,
)

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.utility import load_json
from vnpy_custom.myobject import (
    MyTradeData, 
    SignData, 
    DailyBarData, 
    SignalData, 
    MemberRankData,
    MainData, 
    DailyBarOverview,
    TrendFeaturesData
    )
#kaki add
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import (
    BaseDatabase,
    BarOverview,
    TickOverview,
    DB_TZ,
    convert_tz
)

from vnpy.trader.setting import SETTINGS

vnpy_home_path = Path.home().joinpath(".vntrader")

db = PeeweePostgresqlDatabase(
        database=SETTINGS["database.database"],
        user=SETTINGS["database.user"],
        password=SETTINGS["database.password"],
        host=SETTINGS["database.host"],
        port=SETTINGS["database.port"],
        autorollback=True
)

#write by kaki, mytradedata model for db

class DbBarData(Model):
    """K线数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    datetime: datetime = DateTimeField()
    interval: str = CharField()

    volume: float = FloatField()
    turnover: float = FloatField()
    open_interest: float = FloatField()
    open_price: float = FloatField()
    high_price: float = FloatField()
    low_price: float = FloatField()
    close_price: float = FloatField()

    class Meta:
        database: PeeweePostgresqlDatabase = db
        indexes: tuple = ((("symbol", "exchange", "interval", "datetime"), True),)

class DbTickData(Model):
    """TICK数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    datetime: datetime = DateTimeField()

    name: str = CharField()
    volume: float = FloatField()
    turnover: float = FloatField()
    open_interest: float = FloatField()
    last_price: float = FloatField()
    last_volume: float = FloatField()
    limit_up: float = FloatField()
    limit_down: float = FloatField()

    open_price: float = FloatField()
    high_price: float = FloatField()
    low_price: float = FloatField()
    pre_close: float = FloatField()

    bid_price_1: float = FloatField()
    bid_price_2: float = FloatField(null=True)
    bid_price_3: float = FloatField(null=True)
    bid_price_4: float = FloatField(null=True)
    bid_price_5: float = FloatField(null=True)

    ask_price_1: float = FloatField()
    ask_price_2: float = FloatField(null=True)
    ask_price_3: float = FloatField(null=True)
    ask_price_4: float = FloatField(null=True)
    ask_price_5: float = FloatField(null=True)

    bid_volume_1: float = FloatField()
    bid_volume_2: float = FloatField(null=True)
    bid_volume_3: float = FloatField(null=True)
    bid_volume_4: float = FloatField(null=True)
    bid_volume_5: float = FloatField(null=True)

    ask_volume_1: float = FloatField()
    ask_volume_2: float = FloatField(null=True)
    ask_volume_3: float = FloatField(null=True)
    ask_volume_4: float = FloatField(null=True)
    ask_volume_5: float = FloatField(null=True)

    localtime: datetime = DateTimeField(null=True)

    class Meta:
        database: PeeweePostgresqlDatabase = db
        indexes: tuple = ((("symbol", "exchange", "datetime"), True),)

class DbBarOverview(Model):
    """K线汇总数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    interval: str = CharField()
    count: int = IntegerField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    class Meta:
        database: PeeweePostgresqlDatabase = db
        indexes: tuple = ((("symbol", "exchange", "interval"), True),)

class DbTickOverview(Model):
    """Tick汇总数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    count: int = IntegerField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    class Meta:
        database: PeeweePostgresqlDatabase = db
        indexes: tuple = ((("symbol", "exchange"), True),)

class DbMyTradeData(Model):
    """mytradedata model for db"""
    id = AutoField()

    strategy_class: str = CharField()
    strategy_name: str = CharField()
    strategy_num: str = CharField()
    strategy_period: int = IntegerField()

    datetime: datetime = DateTimeField()
    symbol: str = CharField()
    exchange: str = CharField()
    orderid: str = CharField()
    tradeid: str = CharField()
    direction: str = CharField()
    offset: str = CharField()
    price: float = FloatField()
    volume: float = FloatField()

    display: bool = BooleanField()
    calculate: bool = BooleanField()

    class Meta:
        database = db
        indexes = ((("strategy_class","strategy_name","strategy_num","symbol", "exchange", "tradeid"), True),)

class strategy_sign(Model):
    """mytradedata model for db"""
    id = AutoField()

    tradingday: str = CharField()
    order_time: str = DateTimeField()
    strategy_group: str = CharField()
    strategy_id: str = CharField()
    instrument: str = CharField()
    period: int = IntegerField()
    sign: str = CharField()
    remark: str = CharField()
    insert_time = DateTimeField()  


    class Meta:
        database = db
        indexes = ((("order_time","instrument","period","strategy_id"), True),)

class DbSignal(Model):
    """mytradedata model for db"""
    id = AutoField()

    symbol: str = CharField()
    datetime: datetime = DateTimeField()
    interval: str = CharField()
    strategy_num: str = CharField()
    pos: float = FloatField()


    class Meta:
        database = db
        indexes = ((("symbol","interval","strategy_num","datetime"), True),)

class DbDailyBar(Model):
    """K线数据表映射对象"""

    id = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    datetime: datetime = DateTimeField()
    interval: str = CharField()

    volume: float = FloatField()
    turnover: float = FloatField()
    open_interest: float = FloatField()
    open_price: float = FloatField()
    high_price: float = FloatField()
    low_price: float = FloatField()
    close_price: float = FloatField()
    settlement: float = FloatField()
    prev_settlement: float = FloatField()
    limit_up: float = FloatField()
    limit_down: float = FloatField()

    class Meta:
        database = db
        indexes = ((("symbol", "exchange", "interval", "datetime"), True),)

class DbDailyBarOverview(Model):
    """K线汇总数据表映射对象"""

    id = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    interval: str = CharField()
    count: int = IntegerField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    class Meta:
        database = db
        indexes = ((("symbol", "exchange", "interval"), True),)

class DbMemberRank(Model):
    """会员持仓报告"""

    id = AutoField()

    symbol: str = CharField()
    datetime: datetime = DateTimeField()
    member_name: str = CharField()
    rank: int = IntegerField()
    volume: float = FloatField()
    volume_change: float = FloatField()
    rank_by: str = CharField()

    class Meta:
        database = db
        indexes = ((("datetime","symbol","rank_by","rank"), True),)

class DbTrendFeatures(Model):
    """会员持仓报告"""

    id = AutoField()

    index_name: str = CharField()
    symbol: str = CharField()
    exchange: str = CharField()
    datetime: datetime = DateTimeField()
    trend_point_date: datetime = DateTimeField()

    interval: str = CharField()
    close_price: float = FloatField()
    index_trend_var: str = CharField()
    index_trend_now: float = FloatField()
    trend_point_price: float = FloatField()
    trend_temp_point_price: float = FloatField()
    trend_cum_rate: float = FloatField()
    trend_up_down_range: float = FloatField()
    trend_cum_revers: float = FloatField()
    trend_period_days: int = IntegerField()
    trend_up_nums: int = IntegerField()
    trend_down_nums: int = IntegerField()
    trend_linear_coef: float = FloatField()
    trend_linear_r2: float = FloatField()
    trend_linear_score: float = FloatField()

    class Meta:
        database = db
        indexes = ((("datetime","symbol","index_name","index_trend_var"), True),)

class NewPostgresqlDatabase(BaseDatabase):
    """PostgreSQL数据库接口"""

    def __init__(self) -> None:
        """"""
        self.db: PeeweePostgresqlDatabase = db
        self.db.connect()
        # add dbtradedata
        # self.db.create_tables([DbMyTradeData, DbBarData, DbTickData, DbBarOverview, strategy_sign,DbDailyBar, DbDailyBarOverview, DbSignal,DbMemberRank，DbTrendFeatures])

    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """保存K线数据"""
        # 读取主键参数
        bar: BarData = bars[0]
        symbol: str = bar.symbol
        exchange: Exchange = bar.exchange
        interval: Interval = bar.interval

        # 将BarData数据转换为字典，并调整时区
        data: list = []

        for bar in bars:
            bar.datetime = convert_tz(bar.datetime)

            d: dict = bar.__dict__
            d["exchange"] = d["exchange"].value
            d["interval"] = d["interval"].value
            d.pop("gateway_name")
            d.pop("vt_symbol")
            data.append(d)

        # 使用upsert操作将数据更新到数据库中 chunked批量操作加快速度
        with self.db.atomic():
            for c in chunked(data, 100):
                DbBarData.insert_many(c).on_conflict(
                    update={
                        DbBarData.volume: DbBarData.volume,
                        DbBarData.turnover: DbBarData.turnover,
                        DbBarData.open_interest: DbBarData.open_interest,
                        DbBarData.open_price: DbBarData.open_price,
                        DbBarData.high_price: DbBarData.high_price,
                        DbBarData.low_price: DbBarData.low_price,
                        DbBarData.close_price: DbBarData.close_price
                    },
                    conflict_target=(
                        DbBarData.symbol,
                        DbBarData.exchange,
                        DbBarData.interval,
                        DbBarData.datetime,
                    ),
                ).execute()

        # 更新K线汇总数据
        overview: DbBarOverview = DbBarOverview.get_or_none(
            DbBarOverview.symbol == symbol,
            DbBarOverview.exchange == exchange.value,
            DbBarOverview.interval == interval.value,
        )

        if not overview:
            overview = DbBarOverview()
            overview.symbol = symbol
            overview.exchange = exchange.value
            overview.interval = interval.value
            overview.start = convert_tz(bars[0].datetime)
            overview.end = convert_tz(bars[-1].datetime)
            overview.count = len(bars)
        elif stream:
            overview.end = convert_tz(bars[-1].datetime)
            overview.count += len(bars)
        else:
            overview.start = min(convert_tz(bars[0].datetime), overview.start)
            overview.end = max(convert_tz(bars[-1].datetime), overview.end)

            s: ModelSelect = DbBarData.select().where(
                (DbBarData.symbol == symbol)
                & (DbBarData.exchange == exchange.value)
                & (DbBarData.interval == interval.value)
            )
            overview.count = s.count()

        overview.save()

        return True

    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """保存TICK数据"""
        # 读取主键参数
        tick: TickData = ticks[0]
        symbol: str = tick.symbol
        exchange: Exchange = tick.exchange

        # 将TickData数据转换为字典，并调整时区
        data: list = []

        for tick in ticks:
            tick.datetime = convert_tz(tick.datetime)

            d: dict = tick.__dict__
            d["exchange"] = d["exchange"].value
            d.pop("gateway_name")
            d.pop("vt_symbol")
            data.append(d)

        # 使用upsert操作将数据更新到数据库中
        with self.db.atomic():
            for d in data:
                DbTickData.insert(d).on_conflict(
                    update=d,
                    conflict_target=(
                        DbTickData.symbol,
                        DbTickData.exchange,
                        DbTickData.datetime,


                    ),
                ).execute()

            for c in chunked(data, 100):
                DbTickData.insert_many(c).on_conflict(
                    update={
                        DbTickData.name: DbTickData.name,
                        DbTickData.volume: DbTickData.volume,
                        DbTickData.turnover: DbTickData.turnover,
                        DbTickData.open_interest: DbTickData.open_interest,
                        DbTickData.last_price: DbTickData.last_price,
                        DbTickData.last_volume: DbTickData.last_volume,
                        DbTickData.limit_up: DbTickData.limit_up,
                        DbTickData.limit_down: DbTickData.limit_down,
                        DbTickData.open_price: DbTickData.open_price,
                        DbTickData.high_price: DbTickData.high_price,
                        DbTickData.low_price: DbTickData.low_price,
                        DbTickData.pre_close: DbTickData.pre_close,
                        DbTickData.bid_price_1: DbTickData.bid_price_1,
                        DbTickData.bid_price_2: DbTickData.bid_price_2,
                        DbTickData.bid_price_3: DbTickData.bid_price_3,
                        DbTickData.bid_price_4: DbTickData.bid_price_4,
                        DbTickData.bid_price_5: DbTickData.bid_price_5,
                        DbTickData.ask_price_1: DbTickData.ask_price_1,
                        DbTickData.ask_price_2: DbTickData.ask_price_2,
                        DbTickData.ask_price_3: DbTickData.ask_price_3,
                        DbTickData.ask_price_4: DbTickData.ask_price_4,
                        DbTickData.ask_price_5: DbTickData.ask_price_5,
                        DbTickData.bid_volume_1: DbTickData.bid_volume_1,
                        DbTickData.bid_volume_2: DbTickData.bid_volume_2,
                        DbTickData.bid_volume_3: DbTickData.bid_volume_3,
                        DbTickData.bid_volume_4: DbTickData.bid_volume_4,
                        DbTickData.bid_volume_5: DbTickData.bid_volume_5,
                        DbTickData.ask_volume_1: DbTickData.ask_volume_1,
                        DbTickData.ask_volume_2: DbTickData.ask_volume_2,
                        DbTickData.ask_volume_3: DbTickData.ask_volume_3,
                        DbTickData.ask_volume_4: DbTickData.ask_volume_4,
                        DbTickData.ask_volume_5: DbTickData.ask_volume_5,
                        DbTickData.localtime: DbTickData.localtime,
                    },
                    conflict_target=(
                        DbTickData.symbol,
                        DbTickData.exchange,
                        DbTickData.datetime,
                    ),
                ).execute()

        # 更新Tick汇总数据
        overview: DbTickOverview = DbTickOverview.get_or_none(
            DbTickOverview.symbol == symbol,
            DbTickOverview.exchange == exchange.value,
        )

        if not overview:
            overview: DbTickOverview = DbTickOverview()
            overview.symbol = symbol
            overview.exchange = exchange.value
            overview.start = ticks[0].datetime
            overview.end = ticks[-1].datetime
            overview.count = len(ticks)
        elif stream:
            overview.end = ticks[-1].datetime
            overview.count += len(ticks)
        else:
            overview.start = min(ticks[0].datetime, overview.start)
            overview.end = max(ticks[-1].datetime, overview.end)

            s: ModelSelect = DbTickData.select().where(
                (DbTickData.symbol == symbol)
                & (DbTickData.exchange == exchange.value)
            )
            overview.count = s.count()

        overview.save()

        return True

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """读取K线数据"""
        s: ModelSelect = (
            DbBarData.select().where(
                (DbBarData.symbol == symbol)
                & (DbBarData.exchange == exchange.value)
                & (DbBarData.interval == interval.value)
                & (DbBarData.datetime >= start)
                & (DbBarData.datetime <= end)
            ).order_by(DbBarData.datetime)
        )

        bars: List[BarData] = []
        for db_bar in s:
            bar: BarData = BarData(
                symbol=db_bar.symbol,
                exchange=Exchange(db_bar.exchange),
                datetime=datetime.fromtimestamp(db_bar.datetime.timestamp(), DB_TZ),
                interval=Interval(db_bar.interval),
                volume=db_bar.volume,
                turnover=db_bar.turnover,
                open_interest=db_bar.open_interest,
                open_price=db_bar.open_price,
                high_price=db_bar.high_price,
                low_price=db_bar.low_price,
                close_price=db_bar.close_price,
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """读取TICK数据"""
        s: ModelSelect = (
            DbTickData.select().where(
                (DbTickData.symbol == symbol)
                & (DbTickData.exchange == exchange.value)
                & (DbTickData.datetime >= start)
                & (DbTickData.datetime <= end)
            ).order_by(DbTickData.datetime)
        )

        ticks: List[TickData] = []
        for db_tick in s:
            tick: TickData = TickData(
                symbol=db_tick.symbol,
                exchange=Exchange(db_tick.exchange),
                datetime=datetime.fromtimestamp(db_tick.datetime.timestamp(), DB_TZ),
                name=db_tick.name,
                volume=db_tick.volume,
                turnover=db_tick.turnover,
                open_interest=db_tick.open_interest,
                last_price=db_tick.last_price,
                last_volume=db_tick.last_volume,
                limit_up=db_tick.limit_up,
                limit_down=db_tick.limit_down,
                open_price=db_tick.open_price,
                high_price=db_tick.high_price,
                low_price=db_tick.low_price,
                pre_close=db_tick.pre_close,
                bid_price_1=db_tick.bid_price_1,
                bid_price_2=db_tick.bid_price_2,
                bid_price_3=db_tick.bid_price_3,
                bid_price_4=db_tick.bid_price_4,
                bid_price_5=db_tick.bid_price_5,
                ask_price_1=db_tick.ask_price_1,
                ask_price_2=db_tick.ask_price_2,
                ask_price_3=db_tick.ask_price_3,
                ask_price_4=db_tick.ask_price_4,
                ask_price_5=db_tick.ask_price_5,
                bid_volume_1=db_tick.bid_volume_1,
                bid_volume_2=db_tick.bid_volume_2,
                bid_volume_3=db_tick.bid_volume_3,
                bid_volume_4=db_tick.bid_volume_4,
                bid_volume_5=db_tick.bid_volume_5,
                ask_volume_1=db_tick.ask_volume_1,
                ask_volume_2=db_tick.ask_volume_2,
                ask_volume_3=db_tick.ask_volume_3,
                ask_volume_4=db_tick.ask_volume_4,
                ask_volume_5=db_tick.ask_volume_5,
                localtime=db_tick.localtime,
                gateway_name="DB"
            )
            ticks.append(tick)

        return ticks

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除K线数据"""
        d: ModelDelete = DbBarData.delete().where(
            (DbBarData.symbol == symbol)
            & (DbBarData.exchange == exchange.value)
            & (DbBarData.interval == interval.value)
        )
        count: int = d.execute()

        # 删除K线汇总数据
        d2: ModelDelete = DbBarOverview.delete().where(
            (DbBarOverview.symbol == symbol)
            & (DbBarOverview.exchange == exchange.value)
            & (DbBarOverview.interval == interval.value)
        )
        d2.execute()
        return count

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除TICK数据"""
        d: ModelDelete = DbTickData.delete().where(
            (DbTickData.symbol == symbol)
            & (DbTickData.exchange == exchange.value)
        )
        count: int = d.execute()

        # 删除Tick汇总数据
        d2: ModelDelete = DbTickOverview.delete().where(
            (DbTickOverview.symbol == symbol)
            & (DbTickOverview.exchange == exchange.value)
        )
        d2.execute()

        return count

    def get_bar_overview(self) -> List[BarOverview]:
        """查询数据库中的K线汇总信息"""
        # 如果已有K线，但缺失汇总信息，则执行初始化
        data_count: int = DbBarData.select().count()
        overview_count: int = DbBarOverview.select().count()
        if data_count and not overview_count:
            self.init_bar_overview()

        s: ModelSelect = DbBarOverview.select()
        overviews: List[BarOverview] = []
        for overview in s:
            overview.exchange = Exchange(overview.exchange)
            overview.interval = Interval(overview.interval)
            overviews.append(overview)
        return overviews

    def get_tick_overview(self) -> List[TickOverview]:
        """查询数据库中的Tick汇总信息"""
        s: ModelSelect = DbTickOverview.select()
        overviews: list = []
        for overview in s:
            overview.exchange = Exchange(overview.exchange)
            overviews.append(overview)
        return overviews

    def init_bar_overview(self) -> None:
        """初始化数据库中的K线汇总信息"""
        s: ModelSelect = (
            DbBarData.select(
                DbBarData.symbol,
                DbBarData.exchange,
                DbBarData.interval,
                fn.COUNT(DbBarData.id).alias("count")
            ).group_by(
                DbBarData.symbol,
                DbBarData.exchange,
                DbBarData.interval
            )
        )

        for data in s:
            overview: DbBarOverview = DbBarOverview()
            overview.symbol = data.symbol
            overview.exchange = data.exchange
            overview.interval = data.interval
            overview.count = data.count

            start_bar: DbBarData = (
                DbBarData.select()
                .where(
                    (DbBarData.symbol == data.symbol)
                    & (DbBarData.exchange == data.exchange)
                    & (DbBarData.interval == data.interval)
                )
                .order_by(DbBarData.datetime.asc())
                .first()
            )
            overview.start = start_bar.datetime

            end_bar: DbBarData = (
                DbBarData.select()
                .where(
                    (DbBarData.symbol == data.symbol)
                    & (DbBarData.exchange == data.exchange)
                    & (DbBarData.interval == data.interval)
                )
                .order_by(DbBarData.datetime.desc())
                .first()
            )
            overview.end = end_bar.datetime

            overview.save()

    # save mytradedata
    def save_trade_data(self, trades: List[MyTradeData]) -> bool:
        """"""
        # Store key parameters
        #"strategy_class","strategy_property","strategy_name","strategy_num","symbol", "exchange", "tradeid", "datetime"


        # Convert bar object to dict and adjust timezone
        data = []

        for trade in trades:
            d = copy.deepcopy(trade.__dict__)
            d['datetime'] = convert_tz(d['datetime'])
            d_temp : Dict={
                "strategy_class":d["strategy_class"],
                "strategy_name": d["strategy_name"],
                "strategy_num": d["strategy_num"],
                "strategy_period": int(d["strategy_period"]),
                "datetime": d['datetime'],
                "symbol": d["symbol"],
                "exchange": d["exchange"].value,
                "orderid": d["orderid"],
                "tradeid": d["tradeid"],
                "direction": d["direction"].value,
                "offset": d["offset"].value,
                "price": float(d["price"]),
                "volume": float(d["volume"]),
                "display": bool(d["display"]),
                "calculate": bool(d["calculate"])
                }

            data.append(d_temp)

        # Upsert data into database
        with self.db.atomic():
            for d in data:
                DbMyTradeData.insert(d).on_conflict(
                    update=d,
                    conflict_target=(
                        DbMyTradeData.strategy_class,
                        DbMyTradeData.strategy_name,
                        DbMyTradeData.strategy_num,
                        DbMyTradeData.symbol,
                        DbMyTradeData.exchange,                                                                                                
                        DbMyTradeData.tradeid,
                    ),
                ).execute()
    # save mytradedata
    def save_sign_data(self, signs: List[SignData]) -> bool:
        """"""
        # Store key parameters
        #"strategy_class","strategy_property","strategy_name","strategy_num","symbol", "exchange", "tradeid", "datetime"

        # Convert bar object to dict and adjust timezone
        data = []

        for sign in signs:
            d = copy.deepcopy(sign.__dict__)
            d['order_time'] = convert_tz(d['order_time'])
            d['insert_time'] = convert_tz(d['insert_time'])
            d_temp:Dict={
                "tradingday":d["tradingday"],
                "order_time": d['order_time'],
                "strategy_group": d["strategy_group"],
                "strategy_id": d["strategy_id"],
                "instrument": d["instrument"],
                "period": int(d["period"]),
                "sign": d["sign"],
                "remark": d["remark"],
                "insert_time": d["insert_time"]
            }
            data.append(d_temp)

        # Upsert data into database
        with self.db.atomic():
            for d in data:
                strategy_sign.insert(d).on_conflict(
                    update=d,
                    conflict_target=(
                        strategy_sign.order_time,
                        strategy_sign.instrument,
                        strategy_sign.period,
                        strategy_sign.strategy_id,
                    ),
                ).execute()
    # save signal data
    def save_signal_data(self, signals: List[SignalData]) -> bool:
        """"""
        # Store key parameters
        # "symbol", "interval", "strategy_num", "datetime"
        # Convert signal object to dict and adjust timezone
        data = []

        for signal in signals:
            d = copy.deepcopy(signal.__dict__)
            d['datetime'] = convert_tz(d['datetime'])
            d_temp: Dict = {
            "symbol": d['symbol'],
            "datetime": d['datetime'],
            "interval": d["interval"].value,
            "strategy_num": d['strategy_num'],
            "pos": float(d['pos'])
            }
            # print(d_temp)
            data.append(d_temp)

        # Upsert data into database
        with self.db.atomic():
            for d in data:
                DbSignal.insert(d).on_conflict(
                    update=d,
                    conflict_target=(
                        DbSignal.interval,
                        DbSignal.strategy_num,
                        DbSignal.symbol,
                        DbSignal.datetime,                                                                                                
                    ),
                ).execute()
        return True

    def save_daily_bar_data(self, bars: List[DailyBarData]) -> bool:
        """保存K线数据"""
        # 读取主键参数
        bar = bars[0]
        symbol = bar.symbol
        exchange = bar.exchange
        interval = bar.interval

        # 将BarData数据转换为字典，并调整时区
        data = []

        for bar in bars:
            dt = convert_tz(bar.datetime)
            d = {
                "symbol": symbol,
                "exchange": exchange.value,
                "datetime": dt,
                "interval": interval.value,
                "volume": float(bar.volume),
                "turnover": float(bar.turnover),
                "open_interest": float(bar.open_interest),
                "open_price": float(bar.open_price),
                "high_price": float(bar.high_price),
                "low_price": float(bar.low_price),
                "close_price": float(bar.close_price),
                "settlement": float(bar.settlement),
                "prev_settlement": float(bar.prev_settlement),
                "limit_up": float(bar.limit_up),
                "limit_down": float(bar.limit_down)
                }
            data.append(d)

        # 使用upsert操作将数据更新到数据库中
        with self.db.atomic():
            for d in data:
                DbDailyBar.insert(d).on_conflict(
                    update=d,
                    conflict_target=(
                        DbDailyBar.symbol,
                        DbDailyBar.exchange,
                        DbDailyBar.interval,
                        DbDailyBar.datetime,
                    ),
                ).execute()

        # 更新K线汇总数据
        overview: DbDailyBarOverview = DbDailyBarOverview.get_or_none(
            DbDailyBarOverview.symbol == symbol,
            DbDailyBarOverview.exchange == exchange.value,
            DbDailyBarOverview.interval == interval.value,
        )

        if not overview:
            overview = DbDailyBarOverview()
            overview.symbol = symbol
            overview.exchange = exchange.value
            overview.interval = interval.value
            overview.start = convert_tz(bars[0].datetime)
            overview.end = convert_tz(bars[-1].datetime)
            overview.count = len(bars)
        else:
            overview.start = min(convert_tz(bars[0].datetime), overview.start)
            overview.end = max(convert_tz(bars[-1].datetime), overview.end)

            s: ModelSelect = DbDailyBar.select().where(
                (DbDailyBar.symbol == symbol)
                & (DbDailyBar.exchange == exchange.value)
                & (DbDailyBar.interval == interval.value)
            )
            overview.count = s.count()

        overview.save()

        return True
     # save member data
    def save_member_rank_data(self, members: List[MemberRankData]) -> bool:
        """"""
        # Store key parameters
        # "datetime","symbol","rank_by","rank"

        data = []

        for member in members:
            d = copy.deepcopy(member.__dict__)
            d['datetime'] = convert_tz(d['datetime'])
            d_temp: Dict = {
            "symbol": d['symbol'],
            "datetime": d['datetime'],
            "member_name": d["member_name"],
            "rank": int(d['rank']),
            "volume": float(d['volume']),
            "volume_change": float(d['volume_change']),
            "rank_by": d['rank_by']
            }
            # print(d_temp)
            data.append(d_temp)

        # Upsert data into database
        with self.db.atomic():
            for d in data:
                DbMemberRank.insert(d).on_conflict(
                    update=d,
                    conflict_target=(
                        DbMemberRank.datetime,
                        DbMemberRank.symbol,
                        DbMemberRank.rank_by,
                        DbMemberRank.rank,                                                                                                
                    ),
                ).execute()
        return True

    def save_trend_features_data(self, bars: List[TrendFeaturesData]) -> bool:
        """保存k线数据"""
        # 读取主键参数
        bar0: TrendFeaturesData = bars[0]
        symbol: str = bar0.symbol
        exchange: Exchange = bar0.exchange
        interval: Interval = bar0.interval
        index_name: str = bar0.index_name
        index_trend_var:str = str(bar0.index_trend_var)

        # 转换为DataFrame写入数据库
        data: List[dict] = []

        for bar in bars:
            # bar.datetime = convert_tz(bar.datetime)
            # bar.trend_point_date = convert_tz(bar.trend_point_date)
            dt = convert_tz(bar.datetime)
            trend_point_dt = convert_tz(bar.trend_point_date)

            d = {
                "symbol": symbol,
                "exchange": exchange.value,
                "interval": interval.value,
                "datetime": dt,
                "close_price":float(bar.close_price),
                "index_name" : index_name,
                "index_trend_var" : str(index_trend_var),
                "index_trend_now":int(bar.index_trend_now),
                "trend_point_date":trend_point_dt,
                "trend_point_price":float(bar.trend_point_price),
                "trend_temp_point_price":float(bar.trend_temp_point_price),
                "trend_cum_rate":float(bar.trend_cum_rate),
                "trend_up_down_range":float(bar.trend_up_down_range),
                "trend_cum_revers":float(bar.trend_cum_rate),
                "trend_period_days":int(bar.trend_period_days),
                "trend_up_nums":int(bar.trend_up_nums),
                "trend_down_nums":int(bar.trend_down_nums),
                "trend_linear_coef":float(bar.trend_linear_coef),
                "trend_linear_r2":float(bar.trend_linear_r2),
                "trend_linear_score":float(bar.trend_linear_score),
            }

            data.append(d)

         # 使用upsert操作将数据更新到数据库中
        with self.db.atomic():
            for d in data:
                DbTrendFeatures.insert(d).on_conflict(
                    update=d,
                    conflict_target=(
                        DbTrendFeatures.symbol,
                        DbTrendFeatures.index_name,
                        DbTrendFeatures.index_trend_var,
                        DbTrendFeatures.datetime,
                    ),
                ).execute()

        return True

    def load_daily_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[DailyBarData]:
        """读取K线数据"""
        s: ModelSelect = (
            DbDailyBar.select().where(
                (DbDailyBar.symbol == symbol)
                & (DbDailyBar.exchange == exchange.value)
                & (DbDailyBar.interval == interval.value)
                & (DbDailyBar.datetime >= start)
                & (DbDailyBar.datetime <= end)
            ).order_by(DbDailyBar.datetime)
        )

        bars: List[DailyBarData] = []
        for db_bar in s:
            bar = DailyBarData(
                symbol=db_bar.symbol,
                exchange=Exchange(db_bar.exchange),
                datetime=datetime.fromtimestamp(db_bar.datetime.timestamp(), DB_TZ),
                interval=Interval(db_bar.interval),
                volume=db_bar.volume,
                turnover=db_bar.turnover,
                open_interest=db_bar.open_interest,
                open_price=db_bar.open_price,
                high_price=db_bar.high_price,
                low_price=db_bar.low_price,
                close_price=db_bar.close_price,
                settlement=db_bar.settlement,
                prev_settlement=db_bar.prev_settlement,
                limit_up=db_bar.limit_up,
                limit_down=db_bar.limit_down,
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def load_trade_data(
        self,
        strategy_num: str,
        start: datetime,
        end: datetime
    ) -> List[MyTradeData]:
        """"""
        s: ModelSelect = (
            DbMyTradeData.select().where(
                (DbMyTradeData.strategy_num == strategy_num)
                & (DbMyTradeData.datetime >= start)
                & (DbMyTradeData.datetime <= end)
            ).order_by(DbMyTradeData.datetime)
        )

        # vt_symbol = f"{symbol}.{exchange.value}"

        
        trades: List[MyTradeData] = []
        for db_trade in s:
            dt = datetime.fromtimestamp(db_trade.datetime.timestamp(), DB_TZ)
            # dt = DB_TZ.localize(db_trade.datetime)
            temp_trade = MyTradeData(
                strategy_class=db_trade.strategy_class,
                strategy_name= db_trade.strategy_name,
                strategy_num= strategy_num,
                strategy_period= int(db_trade.strategy_period),
                datetime= dt,
                symbol= db_trade.symbol,
                exchange= Exchange(db_trade.exchange),
                orderid= db_trade.orderid,
                tradeid= db_trade.tradeid,
                direction= Direction(db_trade.direction),
                offset= Offset(db_trade.offset),
                price= float(db_trade.price),
                volume= float(db_trade.volume),
                display= bool(db_trade.display),
                calculate= bool(db_trade.calculate),
                gateway_name="DB"
                )
            trades.append(temp_trade)

        return trades

    def load_trade_all(
        self,
        start: datetime,
        end: datetime
        ) -> List[MyTradeData]:
        """"""
        s: ModelSelect = (
            DbMyTradeData.select().where(
                 (DbMyTradeData.datetime >= start)
                & (DbMyTradeData.datetime <= end)
            ).order_by(DbMyTradeData.datetime)
        )

        # vt_symbol = f"{symbol}.{exchange.value}"

        
        trades: List[MyTradeData] = []
        for db_trade in s:
            # dt = DB_TZ.localize(db_trade.datetime)
            dt = datetime.fromtimestamp(db_trade.datetime.timestamp(), DB_TZ)
            temp_trade = MyTradeData(
                strategy_class=db_trade.strategy_class,
                strategy_name= db_trade.strategy_name,
                strategy_num= db_trade.strategy_num,
                strategy_period= int(db_trade.strategy_period),
                datetime= dt,
                symbol= db_trade.symbol,
                exchange= Exchange(db_trade.exchange),
                orderid= db_trade.orderid,
                tradeid= db_trade.tradeid,
                direction= Direction(db_trade.direction),
                offset= Offset(db_trade.offset),
                price= float(db_trade.price),
                volume= float(db_trade.volume),
                display= bool(db_trade.display),
                calculate= bool(db_trade.calculate),
                gateway_name="DB"
                )
            trades.append(temp_trade)

        return trades

    def load_sign_data(
        self,
        strategy_num: str,
        start: datetime,
        end: datetime
    ) -> List[SignData]:
        """"""
        s: ModelSelect = (
            strategy_sign.select().where(
                (strategy_sign.strategy_id == strategy_num)
                & (strategy_sign.order_time >= start)
                & (strategy_sign.order_time <= end)
            ).order_by(strategy_sign.order_time)
        )

        # vt_symbol = f"{symbol}.{exchange.value}"

        
        signs: List[SignData] = []
        for db_sign in s:
            dt_order_time = datetime.fromtimestamp(db_sign.order_time.timestamp(), DB_TZ)
            dt_insert_time = datetime.fromtimestamp(db_sign.insert_time.timestamp(), DB_TZ)
            # dt_order_time = DB_TZ.localize(db_sign.order_time)
            # dt_insert_time = DB_TZ.localize(db_sign.insert_time)
            temp_sign = SignData(
                tradingday=db_sign.tradingday,
                order_time= dt_order_time,
                strategy_group= db_sign.strategy_group,
                strategy_id= db_sign.strategy_id,
                instrument= db_sign.instrument,
                period= int(db_sign.period),
                sign= db_sign.sign,
                remark= db_sign.remark,
                insert_time= dt_insert_time,
                gateway_name="DB"
                )
            signs.append(temp_sign)

        return signs

    def load_signal_data(
        self,
        strategy_num: str,
        interval:Interval,
        start: datetime,
        end: datetime
        ) -> List[SignalData]:
        """"""
        s: ModelSelect = (
            DbSignal.select().where(
                (DbSignal.strategy_num == strategy_num)
                & (DbSignal.interval == interval.value)
                & (DbSignal.datetime >= start)
                & (DbSignal.datetime <= end)
            ).order_by(DbSignal.datetime)
        )

        signals: List[SignalData] = []
        for signal in s:
            signal = SignalData(
                symbol=signal.symbol,
                datetime=datetime.fromtimestamp(signal.datetime.timestamp(), DB_TZ),
                interval=Interval(signal.interval),
                strategy_num=signal.strategy_num,
                pos=signal.pos,
                gateway_name="DB"
            )
            signals.append(signal)

        return signals

    def load_sign_all(
        self,
        start: datetime,
        end: datetime
        ) -> list:
        """"""
        s: ModelSelect = (
            strategy_sign.select().where(
                 (strategy_sign.order_time >= start)
                & (strategy_sign.order_time <= end)
            ).order_by(strategy_sign.order_time)
        )

        # vt_symbol = f"{symbol}.{exchange.value}"

        
        signs: List[SignData] = []
        for db_sign in s:
            dt_order_time = datetime.fromtimestamp(db_sign.order_time.timestamp(), DB_TZ)
            dt_insert_time = datetime.fromtimestamp(db_sign.insert_time.timestamp(), DB_TZ)
            temp_sign = SignData(
                tradingday=db_sign.tradingday,
                order_time= dt_order_time,
                strategy_group= db_sign.strategy_group,
                strategy_id= db_sign.strategy_id,
                instrument= db_sign.instrument,
                period= int(db_sign.period),
                sign= db_sign.sign,
                remark= db_sign.remark,
                insert_time= dt_insert_time,
                gateway_name="DB"
                )
            signs.append(temp_sign)

        return signs

    def load_symbol_same_trades(
        self,
        symbol_same_strategy_num: str,
        start: datetime,
        end: datetime
        ) -> List[MyTradeData]:
        """"""
        s: ModelSelect = (
            DbMyTradeData.select().where(
                 (DbMyTradeData.datetime >= start)
                & (DbMyTradeData.datetime <= end)
                & (DbMyTradeData.strategy_num.contains(symbol_same_strategy_num))
            ).order_by(DbMyTradeData.datetime)
        )

        # vt_symbol = f"{symbol}.{exchange.value}"

        
        trades: List[MyTradeData] = []
        for db_trade in s:
            dt = datetime.fromtimestamp(db_trade.datetime.timestamp(), DB_TZ)
            temp_trade = MyTradeData(
                strategy_class=db_trade.strategy_class,
                strategy_name= db_trade.strategy_name,
                strategy_num= db_trade.strategy_num,
                strategy_period= int(db_trade.strategy_period),
                datetime= dt,
                symbol= db_trade.symbol,
                exchange= Exchange(db_trade.exchange),
                orderid= db_trade.orderid,
                tradeid= db_trade.tradeid,
                direction= Direction(db_trade.direction),
                offset= Offset(db_trade.offset),
                price= float(db_trade.price),
                volume= float(db_trade.volume),
                display= bool(db_trade.display),
                calculate= bool(db_trade.calculate),
                gateway_name="DB"
                )
            trades.append(temp_trade)

        return trades

    def load_trend_features_data(
        self,
        symbol: str='',
        interval: Interval=Interval.DAILY,
        index_name: str='com_nanhua',
        index_trend_var: str='4.0',
        start: datetime=datetime(2010,1,1),
        end: datetime=datetime(2029,12,31)
    ) -> List[TrendFeaturesData]:
        """读取K线数据"""
        # 转换时间格式
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        
        if not symbol:
            s: ModelSelect = (
                DbTrendFeatures.select().where(
                    (DbTrendFeatures.interval == interval.value)
                    & (DbTrendFeatures.index_name == index_name)
                    & (DbTrendFeatures.index_trend_var == index_trend_var)
                    & (DbTrendFeatures.datetime >= start)
                    & (DbTrendFeatures.datetime <= end)
                ).order_by(DbTrendFeatures.datetime)
            )
        else:
            s: ModelSelect = (
                DbTrendFeatures.select().where(
                    (DbTrendFeatures.symbol == symbol)
                    & (DbTrendFeatures.interval == interval.value)
                    & (DbTrendFeatures.index_name == index_name)
                    & (DbTrendFeatures.index_trend_var == index_trend_var)
                    & (DbTrendFeatures.datetime >= start)
                    & (DbTrendFeatures.datetime <= end)
                ).order_by(DbTrendFeatures.datetime)
            )

        bars: List[TrendFeaturesData] = []
        # 转换为BarData格式

        for db_bar in s:
            dt = datetime.fromtimestamp(db_bar.datetime.timestamp(), DB_TZ)
            trend_point_dt = datetime.fromtimestamp(db_bar.trend_point_date.timestamp(), DB_TZ)

            bar = TrendFeaturesData(
                symbol=db_bar.symbol,
                exchange=Exchange(db_bar.exchange),
                interval=Interval(db_bar.interval),
                datetime=dt,
                close_price=db_bar.close_price,
                index_name = db_bar.index_name,
                index_trend_var = index_trend_var,
                index_trend_now=db_bar.index_trend_now,
                trend_point_date=trend_point_dt,
                trend_point_price=db_bar.trend_point_price,
                trend_temp_point_price=db_bar.trend_temp_point_price,
                trend_cum_rate=db_bar.trend_cum_rate,
                trend_up_down_range=db_bar.trend_up_down_range,
                trend_cum_revers=db_bar.trend_cum_revers,
                trend_period_days=db_bar.trend_period_days,
                trend_up_nums=db_bar.trend_up_nums,
                trend_down_nums=db_bar.trend_down_nums,
                trend_linear_coef=db_bar.trend_linear_coef,
                trend_linear_r2=db_bar.trend_linear_r2,
                trend_linear_score=db_bar.trend_linear_score,
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除K线数据"""
        d: ModelDelete = DbBarData.delete().where(
            (DbBarData.symbol == symbol)
            & (DbBarData.exchange == exchange.value)
            & (DbBarData.interval == interval.value)
        )
        count = d.execute()

        # 删除K线汇总数据
        d2: ModelDelete = DbBarOverview.delete().where(
            (DbBarOverview.symbol == symbol)
            & (DbBarOverview.exchange == exchange.value)
            & (DbBarOverview.interval == interval.value)
        )
        d2.execute()
        return count

    def delete_daily_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除K线数据"""
        d: ModelDelete = DbDailyBar.delete().where(
            (DbDailyBar.symbol == symbol)
            & (DbDailyBar.exchange == exchange.value)
            & (DbDailyBar.interval == interval.value)
        )
        count = d.execute()

        # 删除K线汇总数据
        d2: ModelDelete = DbDailyBarOverview.delete().where(
            (DbDailyBarOverview.symbol == symbol)
            & (DbDailyBarOverview.exchange == exchange.value)
            & (DbDailyBarOverview.interval == interval.value)
        )
        d2.execute()
        return count
    #kaki_add
    def delete_trade_data(
        self,
        strategy_num: str,
        start: datetime,
        end: datetime
    ) -> int:
        """删除trade数据"""
        d: ModelSelect = (
            DbMyTradeData.delete().where(
                (DbMyTradeData.strategy_num == strategy_num)
                & (DbMyTradeData.datetime >= start)
                & (DbMyTradeData.datetime <= end)
            )
        )
        count = d.execute()
        return count
    #kaki_add
    def delete_sign_data(
        self,
        strategy_id: str,
        start: datetime,
        end: datetime
    ) -> int:
        """删除sign数据"""
        d: ModelSelect = (
            strategy_sign.delete().where(
                (strategy_sign.strategy_id == strategy_id)
                & (strategy_sign.order_time >= start)
                & (strategy_sign.order_time <= end)
            )
        )
        count = d.execute()
        return count

    def delete_signal_data(
        self,
        strategy_num: str,
        interval:Interval,
        start: datetime,
        end: datetime
    ) -> int:
        """删除signal数据，需要strategy_num(str), interval(Interval), start(datetime),end"""
        # 加载数据表
        d: ModelDelete = DbSignal.delete().where(
            (DbSignal.strategy_num == strategy_num)
            & (DbSignal.interval == interval.value)
            & (DbSignal.datetime >= start)
            & (DbSignal.datetime <= end)
        )
        count = d.execute()

        return count

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除TICK数据"""
        d: ModelDelete = DbTickData.delete().where(
            (DbTickData.symbol == symbol)
            & (DbTickData.exchange == exchange.value)
        )
        count = d.execute()
        return count

    def delete_trend_features_data(
        self,
        symbol: str,
        interval: Interval=Interval.DAILY,
        index_name: str='com_nanhua',
        index_trend_var: str='4.0',
        start: datetime=datetime(2010,1,1),
        end: datetime=datetime(2029,12,31)
    ) -> int:
        """删除TrendFeatures数据"""
        d: ModelDelete = DbTrendFeatures.delete().where(
            (DbTrendFeatures.symbol == symbol)
                & (DbTrendFeatures.interval == interval.value)
                & (DbTrendFeatures.index_name == index_name)
                & (DbTrendFeatures.index_trend_var == index_trend_var)
                & (DbTrendFeatures.datetime >= start)
                & (DbTrendFeatures.datetime <= end)
        )
        count = d.execute()
        return count

    def get_daily_bar_overview(self) -> List[DailyBarOverview]:
        """查询数据库中的K线汇总信息"""
        # 如果已有K线，但缺失汇总信息，则执行初始化
        data_count = DbDailyBar.select().count()
        overview_count = DbDailyBarOverview.select().count()
        if data_count and not overview_count:
            self.init_daily_bar_overview()

        s: ModelSelect = DbDailyBarOverview.select()
        overviews = []
        for overview in s:
            overview.exchange = Exchange(overview.exchange)
            overview.interval = Interval(overview.interval)
            overviews.append(overview)
        return overviews

    def init_daily_bar_overview(self) -> None:
        """初始化数据库中的K线汇总信息"""
        s: ModelSelect = (
            DbDailyBar.select(
                DbDailyBar.symbol,
                DbDailyBar.exchange,
                DbDailyBar.interval,
                fn.COUNT(DbDailyBar.id).alias("count")
            ).group_by(
                DbDailyBar.symbol,
                DbDailyBar.exchange,
                DbDailyBar.interval
            )
        )

        for data in s:
            dailybaroverview = DbDailyBarOverview()
            dailybaroverview.symbol = data.symbol
            dailybaroverview.exchange = data.exchange
            dailybaroverview.interval = data.interval
            dailybaroverview.count = data.count

            start_bar: DbDailyBar = (
                DbDailyBar.select()
                .where(
                    (DbDailyBar.symbol == data.symbol)
                    & (DbDailyBar.exchange == data.exchange)
                    & (DbDailyBar.interval == data.interval)
                )
                .order_by(DbDailyBar.datetime.asc())
                .first()
            )
            dailybaroverview.start = start_bar.datetime

            end_bar: DbDailyBar = (
                DbDailyBar.select()
                .where(
                    (DbDailyBar.symbol == data.symbol)
                    & (DbDailyBar.exchange == data.exchange)
                    & (DbDailyBar.interval == data.interval)
                )
                .order_by(DbDailyBar.datetime.desc())
                .first()
            )
            dailybaroverview.end = end_bar.datetime

            dailybaroverview.save()
