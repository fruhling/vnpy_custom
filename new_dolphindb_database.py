from typing import Dict, List
from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
import re
import dolphindb as ddb

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import (
    BaseDatabase,
    BarOverview,
    TickOverview,
    DB_TZ,
    convert_tz
)
from vnpy.trader.setting import SETTINGS

from vnpy_dolphindb.dolphindb_script import (
    CREATE_DATABASE_SCRIPT,
    CREATE_BAR_TABLE_SCRIPT,
    CREATE_TICK_TABLE_SCRIPT,
    CREATE_OVERVIEW_TABLE_SCRIPT
)

from .new_dolphindb_script import (
    CREATE_TICK_OVERVIEW_TABLE_SCRIPT,
    CREATE_DAILY_BAR_TABLE_SCRIPT,
    CREATE_DAILY_OVERVIEW_TABLE_SCRIPT,
    CREATE_TRADE_DATA_TABLE_SCRIPT,
    CREATE_SIGN_DATA_TABLE_SCRIPT,
    CREATE_SIGNAL_TABLE_SCRIPT,
    CREATE_TRENDFEATURES_TABLE_SCRIPT
)

from vnpy.trader.constant import Direction, Offset
from .myobject import (
    MyTradeData, 
    SignData, 
    DailyBarData, 
    SignalData, 
    MemberRankData,
    MainData, 
    DailyBarOverview,
    TrendFeaturesData
)

import copy

from vnpy_dolphindb.dolphindb_database import DolphindbDatabase

class NewDolphindbDatabase(DolphindbDatabase):
    """DolphinDB数据库接口"""

    def __init__(self) -> None:
        """"""
        self.user: str = SETTINGS["database.user"]
        self.password: str = SETTINGS["database.password"]
        self.host: str = SETTINGS["database.host"]
        self.port: int = int(SETTINGS["database.port"])
        self.db_path: str = "dfs://vnpy"

        # 连接数据库
        self.session = ddb.session()
        self.session.connect(self.host, self.port, self.user, self.password)

        # 创建连接池（用于数据写入）
        self.pool = ddb.DBConnectionPool(self.host, self.port, 1, self.user, self.password)

        # 初始化数据库和数据表
        if not self.session.existsDatabase(self.db_path):
            self.session.run(CREATE_DATABASE_SCRIPT)
            self.session.run(CREATE_BAR_TABLE_SCRIPT)
            self.session.run(CREATE_TICK_TABLE_SCRIPT)
            self.session.run(CREATE_OVERVIEW_TABLE_SCRIPT)
            self.session.run(CREATE_TICK_OVERVIEW_TABLE_SCRIPT)
            self.session.run(CREATE_DAILY_BAR_TABLE_SCRIPT)
            self.session.run(CREATE_DAILY_OVERVIEW_TABLE_SCRIPT)
            self.session.run(CREATE_TRADE_DATA_TABLE_SCRIPT)
            self.session.run(CREATE_SIGN_DATA_TABLE_SCRIPT)
            self.session.run(CREATE_SIGNAL_TABLE_SCRIPT)
            self.session.run(CREATE_TRENDFEATURES_TABLE_SCRIPT)

    def save_bar_data(self, bars: List[BarData]) -> bool:
        """保存k线数据"""
        # 读取主键参数
        bar: BarData = bars[0]
        symbol: str = bar.symbol
        exchange: Exchange = bar.exchange
        interval: Interval = bar.interval

        # 转换为DatFrame写入数据库
        data: List[dict] = []

        for bar in bars:
            dt = np.datetime64(convert_tz(bar.datetime))

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
                "close_price": float(bar.close_price)
            }

            data.append(d)

        df: pd.DataFrame = pd.DataFrame.from_records(data)

        appender = ddb.PartitionedTableAppender(self.db_path, "bar", "datetime", self.pool)
        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)

        # 计算已有K线数据的汇总
        table = self.session.loadTable(tableName="bar", dbPath=self.db_path)

        df_start: pd.DataFrame = (
            table.select('*')
            .where(f'symbol="{symbol}"')
            .where(f'exchange="{exchange.value}"')
            .where(f'interval="{interval.value}"')
            .sort(bys=["datetime"]).top(1)
            .toDF()
        )

        df_end: pd.DataFrame = (
            table.select('*')
            .where(f'symbol="{symbol}"')
            .where(f'exchange="{exchange.value}"')
            .where(f'interval="{interval.value}"')
            .sort(bys=["datetime desc"]).top(1)
            .toDF()
        )

        df_count: pd.DataFrame = (
            table.select('count(*)')
            .where(f'symbol="{symbol}"')
            .where(f'exchange="{exchange.value}"')
            .where(f'interval="{interval.value}"')
            .toDF()
        )

        count: int = df_count["count"][0]
        start: datetime = df_start["datetime"][0]
        end: datetime = df_end["datetime"][0]

        # 更新K线汇总数据
        data: List[dict] = []

        dt = np.datetime64(datetime(2022, 1, 1))    # 该时间戳仅用于分区

        d: Dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value,
            "count": count,
            "start": start,
            "end": end,
            "datetime": dt,
        }
        data.append(d)

        df: pd.DataFrame = pd.DataFrame.from_records(data)

        appender = ddb.PartitionedTableAppender(self.db_path, "overview", "datetime", self.pool)

        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)

        return True

    def save_tick_data(self, ticks: List[TickData]) -> bool:
        """保存TICK数据"""
        data: List[dict] = []

        for tick in ticks:
            dt = np.datetime64(convert_tz(tick.datetime))

            d: Dict = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "datetime": dt,

                "name": tick.name,
                "volume": float(tick.volume),
                "turnover": float(tick.turnover),
                "open_interest": float(tick.open_interest),
                "last_price": float(tick.last_price),
                "last_volume": float(tick.last_volume),
                "limit_up": float(tick.limit_up),
                "limit_down": float(tick.limit_down),

                "open_price": float(tick.open_price),
                "high_price": float(tick.high_price),
                "low_price": float(tick.low_price),
                "pre_close": float(tick.pre_close),

                "bid_price_1": float(tick.bid_price_1),
                "bid_price_2": float(tick.bid_price_2),
                "bid_price_3": float(tick.bid_price_3),
                "bid_price_4": float(tick.bid_price_4),
                "bid_price_5": float(tick.bid_price_5),

                "ask_price_1": float(tick.ask_price_1),
                "ask_price_2": float(tick.ask_price_2),
                "ask_price_3": float(tick.ask_price_3),
                "ask_price_4": float(tick.ask_price_4),
                "ask_price_5": float(tick.ask_price_5),

                "bid_volume_1": float(tick.bid_volume_1),
                "bid_volume_2": float(tick.bid_volume_2),
                "bid_volume_3": float(tick.bid_volume_3),
                "bid_volume_4": float(tick.bid_volume_4),
                "bid_volume_5": float(tick.bid_volume_5),

                "ask_volume_1": float(tick.ask_volume_1),
                "ask_volume_2": float(tick.ask_volume_2),
                "ask_volume_3": float(tick.ask_volume_3),
                "ask_volume_4": float(tick.ask_volume_4),
                "ask_volume_5": float(tick.ask_volume_5),

                "localtime": np.datetime64(tick.localtime),
            }

            data.append(d)

        df: pd.DataFrame = pd.DataFrame.from_records(data)

        appender = ddb.PartitionedTableAppender(self.db_path, "tick", "datetime", self.pool)
        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)

        return True

    def save_trade_data(self, trades: List[MyTradeData]) -> bool:
        """"""
        # Store key parameters
        #"strategy_class","strategy_property","strategy_name","strategy_num","symbol", "exchange", "tradeid", "datetime"


        # Convert bar object to dict and adjust timezone
        data: List[dict] =[]

        for trade in trades:
            d = copy.deepcopy(trade.__dict__)
            d['datetime'] = np.datetime64(convert_tz(d['datetime']))
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

        df: pd.DataFrame = pd.DataFrame.from_records(data)
        appender = ddb.PartitionedTableAppender(self.db_path, "tradedata", "datetime", self.pool)
        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)

    def save_sign_data(self, signs: List[SignData]) -> bool:
        """"""
        # Store key parameters
        #"strategy_class","strategy_property","strategy_name","strategy_num","symbol", "exchange", "tradeid", "datetime"


        # Convert bar object to dict and adjust timezone
        data: List[dict] = []

        for sign in signs:
            d = copy.deepcopy(sign.__dict__)
            d['order_time'] = np.datetime64(convert_tz(d['order_time']))
            d['insert_time'] = np.datetime64(convert_tz(d['insert_time']))
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

        df: pd.DataFrame = pd.DataFrame.from_records(data)
        appender = ddb.PartitionedTableAppender(self.db_path, "signdata", "order_time", self.pool)
        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)

    def save_signal_data(self, signals: List[SignalData]) -> bool:
        """存入signal_data，要求输入[SignalData]"""
        # Store key parameters
        #"strategy_num","symbol", "datetime"
        # Convert bar object to dict and adjust timezone
        data: List[dict] = []

        for signal in signals:
            d = copy.deepcopy(signal.__dict__)
            d['datetime'] = np.datetime64(convert_tz(d['datetime']))
            d_temp: Dict = {
            "symbol": d['symbol'],
            "datetime": d['datetime'],
            "interval": d["interval"].value,
            "strategy_num": d['strategy_num'],
            "pos": d['pos']
            }
            # print(d_temp)
            data.append(d_temp)

        df: pd.DataFrame = pd.DataFrame.from_records(data)
        appender = ddb.PartitionedTableAppender(self.db_path, "Signal", "datetime", self.pool)
        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)
        return True
    
    def save_daily_bar_data(self, bars: List[DailyBarData]) -> bool:
        """保存k线数据"""
        # 读取主键参数
        bar0: BarData = bars[0]
        symbol: str = bar0.symbol
        exchange: Exchange = bar0.exchange
        interval: Interval = bar0.interval

        # 转换为DatFrame写入数据库
        data: List[dict] = []

        for bar in bars:
            bar.datetime = convert_tz(bar.datetime)
            dt = np.datetime64(convert_tz(bar.datetime))

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

        df: pd.DataFrame = pd.DataFrame.from_records(data)

        appender = ddb.PartitionedTableAppender(self.db_path, "dailybar", "datetime", self.pool)
        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)

        # 计算已有K线数据的汇总
        table = self.session.loadTable(tableName="dailybar", dbPath=self.db_path)

        df_start: pd.DataFrame = (
            table.select("*")
            .where(f"symbol='{symbol}'")
            .where(f"exchange='{exchange.value}'")
            .where(f'interval="{interval.value}"')
            .sort(bys=["datetime"]).top(1)
            .toDF()
        )

        df_end: pd.DataFrame = (
            table.select("*")
            .where(f"symbol='{symbol}'")
            .where(f"exchange='{exchange.value}'")
            .where(f'interval="{interval.value}"')
            .sort(bys=["datetime desc"]).top(1)
            .toDF()
        )

        df_count: pd.DataFrame = (
            table.select("count(*)")
            .where(f"symbol='{symbol}'")
            .where(f"exchange='{exchange.value}'")
            .where(f'interval="{interval.value}"')
            .toDF()
        )

        count: int = df_count["count"][0]
        start: datetime = df_start["datetime"][0]
        end: datetime = df_end["datetime"][0]

        # 更新K线汇总数据
        data: List[dict] = []

        dt = np.datetime64(datetime(2022, 1, 1))    # 该时间戳仅用于分区

        d: Dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value,
            "count": count,
            "start": start,
            "end": end,
            "datetime": dt,
        }
        data.append(d)

        df: pd.DataFrame = pd.DataFrame.from_records(data)

        appender = ddb.PartitionedTableAppender(self.db_path, "dailybar_overview", "datetime", self.pool)
        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)

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
            dt = np.datetime64(convert_tz(bar.datetime))
            trend_point_dt = np.datetime64(convert_tz(bar.trend_point_date))

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

        df: pd.DataFrame = pd.DataFrame.from_records(data)

        appender = ddb.PartitionedTableAppender(self.db_path, "trendfeatures", "datetime", self.pool)
        while True:
            try:
                appender.append(df)
                break
            except RuntimeError:
                sleep(5)

        # # 计算已有K线数据的汇总
        # table = self.session.loadTable(tableName="trendfeatures", dbPath=self.db_path)

        # df_start: pd.DataFrame = (
        #     table.select("*")
        #     .where(f"symbol='{symbol}'")
        #     .where(f"exchange='{exchange.value}'")
        #     .where(f'interval="{interval.value}"')
        #     .sort(bys=["datetime"]).top(1)
        #     .toDF()
        # )

        # df_end: pd.DataFrame = (
        #     table.select("*")
        #     .where(f"symbol='{symbol}'")
        #     .where(f"exchange='{exchange.value}'")
        #     .where(f'interval="{interval.value}"')
        #     .sort(bys=["datetime desc"]).top(1)
        #     .toDF()
        # )

        # df_count: pd.DataFrame = (
        #     table.select("count(*)")
        #     .where(f"symbol='{symbol}'")
        #     .where(f"exchange='{exchange.value}'")
        #     .where(f'interval="{interval.value}"')
        #     .toDF()
        # )

        # count: int = df_count["count"][0]
        # start: datetime = df_start["datetime"][0]
        # end: datetime = df_end["datetime"][0]

        # # 更新K线汇总数据
        # data: List[dict] = []

        # dt = np.datetime64(datetime(2022, 1, 1))    # 该时间戳仅用于分区

        # d: Dict = {
        #     "symbol": symbol,
        #     "exchange": exchange.value,
        #     "interval": interval.value,
        #     "count": count,
        #     "start": start,
        #     "end": end,
        #     "datetime": dt,
        # }
        # data.append(d)

        # df: pd.DataFrame = pd.DataFrame.from_records(data)

        # appender = ddb.PartitionedTableAppender(self.db_path, "dailybar_overview", "datetime", self.pool)
        # while True:
        #     try:
        #         appender.append(df)
        #         break
        #     except RuntimeError:
        #         sleep(5)

        return True

    def save_member_rank_data(self, members: List[MemberRankData]) -> bool:
        member: MemberRankData = members[0]
        symbol: str = member.symbol

        # 转换为DatFrame写入数据库
        data: List[dict] = []

        for member in members:
            dt = np.datetime64(convert_tz(member.datetime), 'ns')
            #dt = np.array(convert_tz(bar.datetime), dtype='datetime64[ns]')

            d = {
                "symbol": symbol,
                "datetime": dt,
                "member_name":member.member_name,
                "rank": int(member.rank),
                "volume": float(member.volume),
                "volume_change": float(member.volume_change),
                "rank_by": member.rank_by
            }

            data.append(d)

        df: pd.DataFrame = pd.DataFrame.from_records(data)
        while True:
            try:
                self.session.run("append!{{loadTable('{db}', `{tb})}}".format(db=self.db_path, tb="memberrank"), df)
                break
            except RuntimeError:
                sleep(5)
        

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
        # 转换时间格式
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")

        table = self.session.loadTable(tableName="dailybar", dbPath=self.db_path)

        df: pd.DataFrame = (
            table.select("*")
            .where(f"symbol='{symbol}'")
            .where(f"exchange='{exchange.value}'")
            .where(f'interval="{interval.value}"')
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )

        bars: List[DailyBarData] = []
        # 转换为BarData格式

        for tp in df.itertuples():
            dt = datetime.fromtimestamp(tp.datetime.to_pydatetime().timestamp(), DB_TZ)

            bar = DailyBarData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                interval=interval,
                volume=tp.volume,
                turnover=tp.turnover,
                open_interest=tp.open_interest,
                open_price=tp.open_price,
                high_price=tp.high_price,
                low_price=tp.low_price,
                close_price=tp.close_price,
                settlement=tp.settlement,
                prev_settlement=tp.prev_settlement,
                limit_up=tp.limit_up,
                limit_down=tp.limit_down,
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
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")

        table = self.session.loadTable(tableName="tradedata", dbPath=self.db_path)

        df: pd.DataFrame = (
            table.select("*")
            .where(f"strategy_num='{strategy_num}'")
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )
        # vt_symbol = f"{symbol}.{exchange.value}"

        
        trades: List[MyTradeData] = []
        for tp in df.itertuples():
            dt = datetime.fromtimestamp(tp.datetime.to_pydatetime().timestamp(), DB_TZ)
            temp_trade = MyTradeData(
                strategy_class=tp.strategy_class,
                strategy_name= tp.strategy_name,
                strategy_num= strategy_num,
                strategy_period= int(tp.strategy_period),
                datetime= dt,
                symbol= tp.symbol,
                exchange= Exchange(tp.exchange),
                orderid= tp.orderid,
                tradeid= tp.tradeid,
                direction= Direction(tp.direction),
                offset= Offset(tp.offset),
                price= float(tp.price),
                volume= float(tp.volume),
                display= bool(tp.display),
                calculate= bool(tp.calculate),
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
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")

        table = self.session.loadTable(tableName="tradedata", dbPath=self.db_path)

        df: pd.DataFrame = (
            table.select("*")
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )

        # vt_symbol = f"{symbol}.{exchange.value}"

        
        trades: List[MyTradeData] = []
        for tp in df.itertuples():
            dt = datetime.fromtimestamp(tp.datetime.to_pydatetime().timestamp(), DB_TZ)
            temp_trade = MyTradeData(
                strategy_class=tp.strategy_class,
                strategy_name= tp.strategy_name,
                strategy_num= tp.strategy_num,
                strategy_period= int(tp.strategy_period),
                datetime= dt,
                symbol= tp.symbol,
                exchange= Exchange(tp.exchange),
                orderid= tp.orderid,
                tradeid= tp.tradeid,
                direction= Direction(tp.direction),
                offset= Offset(tp.offset),
                price= float(tp.price),
                volume= float(tp.volume),
                display= bool(tp.display),
                calculate= bool(tp.calculate),
                gateway_name="DB"
                )
            trades.append(temp_trade)

        return trades
    
    def load_sign_data(
        self,
        strategy_id: str,
        start: datetime,
        end: datetime
    ) -> List[SignData]:
        """"""
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")

        table = self.session.loadTable(tableName="signdata", dbPath=self.db_path)

        df: pd.DataFrame = (
            table.select("*")
            .where(f"strategy_id='{strategy_id}'")
            .where(f"order_time>={start}")
            .where(f"order_time<={end}")
            .toDF()
        )
        # vt_symbol = f"{symbol}.{exchange.value}"

        signs: List[SignData] = []
        for tp in df.itertuples():
            dt_order_time = datetime.fromtimestamp(tp.order_time.to_pydatetime().timestamp(), DB_TZ)
            dt_insert_time = datetime.fromtimestamp(tp.insert_time.to_pydatetime().timestamp(), DB_TZ)
            temp_sign = SignData(
                tradingday=tp.tradingday,
                order_time= dt_order_time,
                strategy_group= tp.strategy_group,
                strategy_id= tp.strategy_id,
                instrument= tp.instrument,
                period= int(tp.period),
                sign= tp.sign,
                remark= tp.remark,
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
        """输出signal数据，需要strategy_num(str), interval(Interval), start(datetime),end"""
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")

        table = self.session.loadTable(tableName="Signal", dbPath=self.db_path)

        df: pd.DataFrame = (
            table.select("*")
            .where(f"strategy_num='{strategy_num}'")
            .where(f'interval="{interval.value}"')
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )
        # vt_symbol = f"{symbol}.{exchange.value}"

        
        signals: List[SignalData] = []
        for tp in df.itertuples():
            dt = datetime.fromtimestamp(tp.datetime.to_pydatetime().timestamp(), DB_TZ)

            signal = SignalData(
                symbol=tp.symbol,
                datetime=dt,
                interval=Interval(tp.interval),
                strategy_num=tp.strategy_num,
                pos=tp.pos,
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
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")

        table = self.session.loadTable(tableName="signdata", dbPath=self.db_path)

        df: pd.DataFrame = (
            table.select("*")
            .where(f"order_time>={start}")
            .where(f"order_time<={end}")
            .toDF()
        )
        
        # vt_symbol = f"{symbol}.{exchange.value}"
        
        signs: List[SignData] = []
        for tp in df.itertuples():
            dt_order_time = datetime.fromtimestamp(tp.order_time.to_pydatetime().timestamp(), DB_TZ)
            dt_insert_time = datetime.fromtimestamp(tp.insert_time.to_pydatetime().timestamp(), DB_TZ)
            temp_sign = SignData(
                tradingday=tp.tradingday,
                order_time= dt_order_time,
                strategy_group= tp.strategy_group,
                strategy_id= tp.strategy_id,
                instrument= tp.instrument,
                period= int(tp.period),
                sign= tp.sign,
                remark= tp.remark,
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
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")

        table = self.session.loadTable(tableName="tradedata", dbPath=self.db_path)

        df: pd.DataFrame = (
            table.select("*")
            .where(f"strategy_num like '%{symbol_same_strategy_num}%'")
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )

        trades: List[MyTradeData] = []
        for tp in df.itertuples():
            dt = datetime.fromtimestamp(tp.datetime.to_pydatetime().timestamp(), DB_TZ)
            temp_trade = MyTradeData(
                strategy_class=tp.strategy_class,
                strategy_name= tp.strategy_name,
                strategy_num= tp.strategy_num,
                strategy_period= int(tp.strategy_period),
                datetime= dt,
                symbol= tp.symbol,
                exchange= Exchange(tp.exchange),
                orderid= tp.orderid,
                tradeid= tp.tradeid,
                direction= Direction(tp.direction),
                offset= Offset(tp.offset),
                price= float(tp.price),
                volume= float(tp.volume),
                display= bool(tp.display),
                calculate= bool(tp.calculate),
                gateway_name="DB"
                )
            trades.append(temp_trade)

        return trades

    def load_domain_info(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> List[MainData]:
        """输出main数据，需要symbol(str), start(datetime),end"""
        symbol=''.join(re.findall(r'[A-Za-z]',symbol)).upper()#只取品种英文大写代码

        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")
        table = self.session.loadTable(tableName="main", dbPath=self.db_path)

        df: pd.DataFrame = (
            table.select("*")
            .where(f"symbol=\"{symbol}\"")
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )
        # vt_symbol = f"{symbol}.{exchange.value}"

        
        mains: List[MainData] = []
        for tp in df.itertuples():
            dt = datetime.fromtimestamp(tp.datetime.to_pydatetime().timestamp(), DB_TZ)

            domain = MainData(
                symbol=tp.symbol,
                datetime=dt,
                main=tp.main,
                gateway_name="DB"
            )
            mains.append(domain)

        return mains

    def load_trend_features_data(
        self,
        interval: Interval,
        index_name: str,
        index_trend_var: str,
        symbol: str='',
        start: datetime='2010-01-01',
        end: datetime='2029-12-31'
    ) -> List[TrendFeaturesData]:
        """读取K线数据"""
        # 转换时间格式
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")

        table = self.session.loadTable(tableName="trendfeatures", dbPath=self.db_path)

        if not symbol:
            df: pd.DataFrame = (
                table.select("*")
                .where(f'interval="{interval.value}"')
                .where(f"index_name='{index_name}'")
                .where(f"index_trend_var='{index_trend_var}'")
                .where(f"datetime>={start}")
                .where(f"datetime<={end}")
                .toDF()
            )
        else:
            df: pd.DataFrame = (
                table.select("*")
                .where(f"symbol='{symbol}'")
                .where(f'interval="{interval.value}"')
                .where(f"index_name='{index_name}'")
                .where(f"index_trend_var='{index_trend_var}'")
                .where(f"datetime>={start}")
                .where(f"datetime<={end}")
                .toDF()
            )

        bars: List[TrendFeaturesData] = []
        # 转换为BarData格式

        for tp in df.itertuples():
            dt = datetime.fromtimestamp(tp.datetime.to_pydatetime().timestamp(), DB_TZ)
            trend_point_dt = datetime.fromtimestamp(tp.trend_point_date.to_pydatetime().timestamp(), DB_TZ)

            bar = TrendFeaturesData(
                symbol=tp.symbol,
                exchange=Exchange(tp.exchange),
                interval=Interval(tp.interval),
                datetime=dt,
                close_price=tp.close_price,
                index_name = tp.index_name,
                index_trend_var = index_trend_var,
                index_trend_now=tp.index_trend_now,
                trend_point_date=trend_point_dt,
                trend_point_price=tp.trend_point_price,
                trend_temp_point_price=tp.trend_temp_point_price,
                trend_cum_rate=tp.trend_cum_rate,
                trend_up_down_range=tp.trend_up_down_range,
                trend_cum_revers=tp.trend_cum_revers,
                trend_period_days=tp.trend_period_days,
                trend_up_nums=tp.trend_up_nums,
                trend_down_nums=tp.trend_down_nums,
                trend_linear_coef=tp.trend_linear_coef,
                trend_linear_r2=tp.trend_linear_r2,
                trend_linear_score=tp.trend_linear_score,
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def delete_daily_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除日线K线数据"""
        # 加载数据表
        table = self.session.loadTable(tableName="dailybar", dbPath=self.db_path)

        # 统计数据量
        df: pd.DataFrame = (
            table.select("count(*)")
            .where(f"symbol='{symbol}'")
            .where(f"exchange='{exchange.value}'")
            .where(f'interval="{interval.value}"')
            .toDF()
        )
        count = df["count"][0]

        # 删除K线数据
        (
            table.delete()
            .where(f"symbol='{symbol}'")
            .where(f"exchange='{exchange.value}'")
            .where(f'interval="{interval.value}"')
            .execute()
        )

        # 删除K线汇总
        table = self.session.loadTable(tableName="dailybar_overview", dbPath=self.db_path)
        (
            table.delete()
            .where(f"symbol='{symbol}'")
            .where(f"exchange='{exchange.value}'")
            .where(f'interval="{interval.value}"')
            .execute()
        )

        return count

    def delete_trend_features_data(
        self,
        symbol: str,
        interval: Interval,
        index_name: str,
        index_trend_var: str,
        start: datetime='2010-01-01',
        end: datetime='2029-12-31'
    ) -> int:
        """删除日线K线数据"""
        # 加载数据表
        table = self.session.loadTable(tableName="trendfeatures", dbPath=self.db_path)

        # 统计数据量
        df: pd.DataFrame = (
            table.select("count(*)")
            .where(f"symbol='{symbol}'")
            .where(f'interval="{interval.value}"')
            .where(f"index_name='{index_name}'")
            .where(f"index_trend_var='{index_trend_var}'")
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )
        count = df["count"][0]

        # 删除K线数据
        (
            table.delete()
            .where(f"symbol='{symbol}'")
            .where(f'interval="{interval.value}"')
            .where(f"index_name='{index_name}'")
            .where(f"index_trend_var='{index_trend_var}'")
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .execute()
        )

        # # 删除K线汇总
        # table = self.session.loadTable(tableName="dailybar_overview", dbPath=self.db_path)
        # (
        #     table.delete()
        #      .where(f"symbol='{symbol}'")
        #     .where(f'interval="{interval.value}"')
        #     .where(f"index_name='{index_name}'")
        #     .where(f"index_trend_var='{index_trend_var}'")
        #     .where(f"datetime>={start}")
        #     .where(f"datetime<={end}")
        #     .execute()
        # )

        return count

    def delete_trade_data(
        self,
        strategy_num: str,
        start: datetime,
        end: datetime
    ) -> int:
        """删除tradedata数据"""
        # 加载数据表
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")
        table = self.session.loadTable(tableName="tradedata", dbPath=self.db_path)

        # 统计数据量
        df: pd.DataFrame = (
            table.select("count(*)")
            .where(f"strategy_num='{strategy_num}'")
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )
        count = df["count"][0]

        # 删除K线数据
        while True:
            try:
                (
                    table.delete()
                    .where(f"strategy_num='{strategy_num}'")
                    .where(f"datetime>={start}")
                    .where(f"datetime<={end}")
                    .execute()
                )
                break
            except RuntimeError:
                sleep(5)

        return count

    def delete_sign_data(
        self,
        strategy_id: str,
        start: datetime,
        end: datetime
    ) -> int:
        """删除sign数据"""
        # 加载数据表
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")
        table = self.session.loadTable(tableName="signdata", dbPath=self.db_path)

        # 统计数据量
        df: pd.DataFrame = (
            table.select("count(*)")
            .where(f"strategy_id='{strategy_id}'")
            .where(f"order_time>={start}")
            .where(f"order_time<={end}")
            .toDF()
        )
        count = df["count"][0]

        # 删除K线数据
        while True:
            try:
                (
                    table.delete()
                    .where(f"strategy_id='{strategy_id}'")
                    .where(f"order_time>={start}")
                    .where(f"order_time<={end}")
                    .execute()
                )
                break
            except RuntimeError:
                # print("Runtime Error")
                sleep(5)
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
        start = np.datetime64(start)
        start: str = str(start).replace("-", ".")

        end = np.datetime64(end)        
        end: str = str(end).replace("-", ".")
        table = self.session.loadTable(tableName="Signal", dbPath=self.db_path)

        # 统计数据量
        df: pd.DataFrame = (
            table.select("count(*)")
            .where(f"strategy_num='{strategy_num}'")
            .where(f'interval="{interval.value}"')
            .where(f"datetime>={start}")
            .where(f"datetime<={end}")
            .toDF()
        )
        count = df["count"][0]

        # 删除信号数据
        while True:
            try:
                (
                    table.delete()
                    .where(f"strategy_num='{strategy_num}'")
                    .where(f'interval="{interval.value}"')
                    .where(f"datetime>={start}")
                    .where(f"datetime<={end}")
                    .execute()
                )
                break
            except RuntimeError:
                # print("Runtime Error")
                sleep(5)

        return count

    def get_tick_overview(self) -> List[TickOverview]:
        """"查询数据库中的K线汇总信息"""
        table = self.session.loadTable(tableName="tick_overview", dbPath=self.db_path)
        df: pd.DataFrame = table.select("*").toDF()

        overviews: List[TickOverview] = []

        for tp in df.itertuples():
            overview = TickOverview(
                symbol=tp.symbol,
                exchange=Exchange(tp.exchange),
                count=tp.count,
                start=datetime.fromtimestamp(tp.start.to_pydatetime().timestamp(), DB_TZ),
                end=datetime.fromtimestamp(tp.end.to_pydatetime().timestamp(), DB_TZ),
            )
            overviews.append(overview)

        return overviews

    def get_daily_bar_overview(self) -> List[DailyBarOverview]:
        """"查询数据库中的K线汇总信息"""
        table = self.session.loadTable(tableName="dailybar_overview", dbPath=self.db_path)
        df: pd.DataFrame = table.select("*").toDF()

        overviews: List[DailyBarOverview] = []

        for tp in df.itertuples():
            overview = DailyBarOverview(
                symbol=tp.symbol,
                exchange=Exchange(tp.exchange),
                interval=Interval(tp.interval),
                count=tp.count,
                start=datetime.fromtimestamp(tp.start.to_pydatetime().timestamp(), DB_TZ),
                end=datetime.fromtimestamp(tp.end.to_pydatetime().timestamp(), DB_TZ),
            )
            overviews.append(overview)

        return overviews
