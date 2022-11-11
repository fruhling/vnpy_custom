from typing import List, Optional
from datetime import timedelta


from rqdatac.services.future import get_member_rank
from rqdatac.services.get_price import get_price

from vnpy.trader.constant import Exchange
from .myobject import DailyBarData, MemberRankData
from vnpy.trader.object import HistoryRequest
from vnpy.trader.utility import round_to



from vnpy_rqdata.rqdata_datafeed import RqdataDatafeed, INTERVAL_VT2RQ, INTERVAL_ADJUSTMENT_MAP, CHINA_TZ

def to_rq_symbol(symbol: str, exchange: Exchange) -> str:
    """将交易所代码转换为米筐代码"""
    # 股票
    if exchange in [Exchange.SSE, Exchange.SZSE]:
        if exchange == Exchange.SSE:
            rq_symbol: str = f"{symbol}.XSHG"
        else:
            rq_symbol: str = f"{symbol}.XSHE"
    # 金交所现货
    elif exchange in [Exchange.SGE]:
        for char in ["(", ")", "+"]:
            symbol: str = symbol.replace(char, "")
        symbol = symbol.upper()
        rq_symbol: str = f"{symbol}.SGEX"
    # 期货和期权
    elif exchange in [Exchange.SHFE, Exchange.CFFEX, Exchange.DCE, Exchange.CZCE, Exchange.INE]:
        for count, word in enumerate(symbol):
            if word.isdigit():
                break

        product: str = symbol[:count]
        time_str: str = symbol[count:]

        # 期货
        if time_str.isdigit():
            if exchange is not Exchange.CZCE:
                return symbol.upper()

            if exchange is Exchange.CZCE and len(time_str)==4:
                return symbol.upper()

            # 检查是否为连续合约或者指数合约
            if time_str in ["88", "888", "99", "889"]:
                return symbol

            year: str = symbol[count]
            month: str = symbol[count + 1:]

            if year in ["9","8","7","6","5"]:
                year = "1" + year
            else:
                year = "2" + year

            rq_symbol: str = f"{product}{year}{month}".upper()
        # 期权
        else:
            if exchange in [Exchange.CFFEX, Exchange.DCE, Exchange.SHFE]:
                rq_symbol: str = symbol.replace("-", "").upper()
            elif exchange == Exchange.CZCE:
                year: str = symbol[count]
                suffix: str = symbol[count + 1:]

                if year in ["9","8","7","6","5"]:
                    year = "1" + year
                else:
                    year = "2" + year

                rq_symbol: str = f"{product}{year}{suffix}".upper()
    else:
        rq_symbol: str = f"{symbol}.{exchange.value}"

    return rq_symbol


class NewRqdataDatafeed(RqdataDatafeed):
    """米筐RQData数据服务接口"""

    def __init__(self):
        """"""
        super().__init__()
        

    def query_daily_bar_history(self, req: HistoryRequest) -> Optional[List[DailyBarData]]:
        """查询K线数据"""
        if not self.inited:
            n = self.init()
            if not n:
                return []

        symbol = req.symbol
        exchange = req.exchange
        interval = req.interval
        start = req.start
        end = req.end

        rq_symbol = to_rq_symbol(symbol, exchange)

        rq_interval = INTERVAL_VT2RQ.get(interval)
        if not rq_interval:
            return None

        # 为了将米筐时间戳（K线结束时点）转换为vn.py时间戳（K线开始时点）
        adjustment = INTERVAL_ADJUSTMENT_MAP[interval]

        # 为了查询夜盘数据
        end += timedelta(1)

        # 只对衍生品合约才查询持仓量数据
        fields = ["open", "high", "low", "close", "volume", "total_turnover","settlement","prev_settlement","limit_up","limit_down"]
        if not symbol.isdigit():
            fields.append("open_interest")

        df = get_price(
            rq_symbol,
            frequency=rq_interval,
            fields=fields,
            start_date=start,
            end_date=end,
            adjust_type="none"
        )

        data: List[DailyBarData] = []

        if df is not None:
            for ix, row in df.iterrows():
                dt = row.name[1].to_pydatetime() - adjustment
                # dt = CHINA_TZ.localize(dt)
                dt = dt.replace(tzinfo=CHINA_TZ)
                # kaki add，过滤错误数据
                if(str(row["open"]) =="nan" or str(row["high"]) =="nan" or \
                    str(row["low"]) =="nan" or str(row["close"]) =="nan" or \
                        str(row["settlement"]) =="nan" or str(row["prev_settlement"]) =="nan" or \
                            str(row["limit_up"]) =="nan" or str(row["limit_down"]) =="nan"):
                    continue

                bar = DailyBarData(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    datetime=dt,
                    open_price=round_to(row["open"], 0.000001),
                    high_price=round_to(row["high"], 0.000001),
                    low_price=round_to(row["low"], 0.000001),
                    close_price=round_to(row["close"], 0.000001),
                    settlement=round_to(row["settlement"], 0.000001),
                    prev_settlement=round_to(row["prev_settlement"], 0.000001),
                    limit_up=round_to(row["limit_up"], 0.000001),
                    limit_down=round_to(row["limit_down"], 0.000001),
                    volume=row["volume"],
                    turnover=row["total_turnover"],
                    open_interest=row.get("open_interest", 0),
                    gateway_name="RQ"
                )

                data.append(bar)

        return data

    def query_member_rank_history(self, req: HistoryRequest) -> Optional[List[MemberRankData]]:
        """查询K线数据"""
        if not self.inited:
            self.init()

        symbol = req.symbol
        exchange = req.exchange
        interval = req.interval
        start = req.start
        end = req.end

        rq_symbol = to_rq_symbol(symbol, exchange)

        rq_interval = INTERVAL_VT2RQ.get(interval)
        if not rq_interval:
            return None

        # 为了将米筐时间戳（K线结束时点）转换为vn.py时间戳（K线开始时点）
        adjustment = INTERVAL_ADJUSTMENT_MAP[interval]

        # 为了查询夜盘数据
        end += timedelta(1)

        # 只对short合约才查询持仓量数据
        #df0 = get_member_rank('A1901',trading_date=20180910,rank_by='short')
        #df0 = get_member_rank('A1901',start_date=20180910,end_date=end,rank_by='short')
        df = get_member_rank(
            rq_symbol,
            start_date=start,
            end_date=end,
            rank_by="short"
        )
        dflong = get_member_rank(
            rq_symbol,
            start_date=start,
            end_date=end,
            rank_by="long"
        )

        data: List[MemberRankData] = []

        if df is not None:
            for ix, row in df.iterrows():
                dt = row.name.to_pydatetime() 
                # dt = CHINA_TZ.localize(dt)
                dt = dt.replace(tzinfo=CHINA_TZ)

                MemberRank = MemberRankData(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    datetime=dt,
                    member_name=row["member_name"],
                    rank=int(row["rank"]),
                    volume=round_to(row["volume"], 0.000001),
                    volume_change=round_to(row["volume_change"], 0.000001),
                    rank_by="short",
                    gateway_name="RQ"
                )

                data.append(MemberRank)

        if dflong is not None:
            for ix, row in dflong.iterrows():
                dt = row.name.to_pydatetime()
                # dt = CHINA_TZ.localize(dt)
                dt = dt.replace(tzinfo=CHINA_TZ)

                MemberRank = MemberRankData(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    datetime=dt,
                    member_name=row["member_name"],
                    rank=int(row["rank"]),
                    volume=round_to(row["volume"], 0.000001),
                    volume_change=round_to(row["volume_change"], 0.000001),
                    rank_by="long",
                    gateway_name="RQ"
                )

                data.append(MemberRank)

        return data
    