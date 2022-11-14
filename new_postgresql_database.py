""""""
from datetime import datetime
from vnpy.trader.constant import Direction, Offset
from typing import List, Dict
import copy
from pathlib import Path

from peewee import (
    AutoField,
    BooleanField,
    CharField,
    DateTimeField,
    FloatField, IntegerField,
    Model,
    PostgresqlDatabase as PeeweePostgresqlDatabase,
    ModelSelect,
    ModelDelete,
    fn
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
    DailyBarOverview
    )
#kaki add

from vnpy.trader.database import (
    BaseDatabase,
    DB_TZ,
    convert_tz
)

from vnpy_postgresql.postgresql_database import (
    PostgresqlDatabase,
    DbBarData,
    DbTickData,
    DbBarOverview,
    DbTickOverview
    )
vnpy_home_path = Path.home().joinpath(".vntrader")
custom_setting_filename = vnpy_home_path.joinpath("vnpy_custom_setting.json")
custom_setting = load_json(custom_setting_filename)

db = PeeweePostgresqlDatabase(
    database=custom_setting["custom_database_name"],
    user=custom_setting["custom_database_user"],
    password=custom_setting["custom_database_password"],
    host=custom_setting["custom_database_host"],
    port=custom_setting["custom_database_port"],
    autorollback=True
)

#write by kaki, mytradedata model for db

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

class NewPostgresqlDatabase(PostgresqlDatabase):
    """PostgreSQL数据库接口"""

    def __init__(self) -> None:
        """"""
        self.db = db
        self.db.connect()
        # add dbtradedata
        # self.db.create_tables([DbMyTradeData, DbBarData, DbTickData, DbBarOverview, strategy_sign,DbDailyBar, DbDailyBarOverview, DbSignal,DbMemberRank])

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
            overview.start = bars[0].datetime
            overview.end = bars[-1].datetime
            overview.count = len(bars)
        else:
            overview.start = min(bars[0].datetime, overview.start)
            overview.end = max(bars[-1].datetime, overview.end)

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
            dt = DB_TZ.localize(db_trade.datetime)
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
            dt = DB_TZ.localize(db_trade.datetime)
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
            dt_order_time = DB_TZ.localize(db_sign.order_time)
            dt_insert_time = DB_TZ.localize(db_sign.insert_time)
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
            dt_order_time = DB_TZ.localize(db_sign.order_time)
            dt_insert_time = DB_TZ.localize(db_sign.insert_time)
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
            dt = DB_TZ.localize(db_trade.datetime)
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
