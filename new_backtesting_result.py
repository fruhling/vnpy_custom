import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import date, datetime, timedelta, time
import re
from typing import List, Dict
from pandas import DataFrame
from vnpy.trader.constant import Exchange, Interval, Offset, Direction, Status
from vnpy_custom.cta_utility import get_contract_rule,get_symbol_head,to_CTP_symbol,generate_trading_day,wavg
from dataclasses import dataclass

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np


exchange_dict = {"中国金融交易所":"CFFEX","上海期货交易所":"SHFE","郑州商品交易所":"CZCE","大连商品交易所":"DCE","上海国际能源交易中心股份有限公司":"INE"}
direction_dict = {"买":"多","卖":"空"}
offset_dict = {'平今':'平今', '平仓':'平', '平昨':'平昨', '开仓':'开'}
net_pos_direction_dict = {Direction.LONG:1.0, Direction.SHORT:-1.0}
yf_sns_axes_style={'xtick.bottom': True,'ytick.left': True}
win_loss_rate_bins=[-np.inf,-10.0,-5.0,-2.0,-1.0,-0.5,0,0.5,1.0,2.0,5.0,10.0,np.inf] 
win_loss_rate_labels = ["小于-10%","-10%~-5%","-5%~-2%","-2%~-1%","-1%~-0.5%","-0.5%~0","0~0.5%","0.5%~1%","1%~2%","2%~5%","5%~10%","大于10%"] 
holding_time_bins = [0,1,2,3,5,10,np.inf]
holding_time_labels = ["日内","1天","2天","3-5天","1-2周","2周以上"]

@dataclass
class CalTradeData:
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    symbol: str
    exchange: Exchange
    tradingday: date = None
    direction: Direction = None

    offset: Offset = Offset.NONE
    price: float = 0.0
    volume: float = 0.0
    fee: float = 0.0
    turnover: float = 0.0
    datetime: datetime = None

@dataclass
class CalCloseTradeData:
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    symbol: str
    symbol_class: str
    exchange: Exchange
    closingday: date = None
    direction: Direction = None

    open_price: float = 0.0
    close_price: float = 0.0
    volume: float = 0.0
    fee: float = 0.0
    turnover: float = 0.0
    profit_or_loss: float = 0.0
    profit_or_loss_in_price: float = 0.0
    holding_time: float = 0.0
    open_datetime: datetime = None
    close_datetime: datetime = None

@dataclass
class CalPositionData:
    """
    Positon data is used for tracking each individual position holding.
    """

    symbol: str
    exchange: Exchange
    datetime: date = None

    direction: Direction = Direction.LONG

    volume: float = 0.0
    price: float = 0.0
    settlement: float = 0.0
    marginCompany: float = 0.0
    marginCompanyRate: float = 0.0
    marginExchange: float = 0.0
    marginExchangeRate: float = 0.0

@dataclass
class CalCapitalData:
    """
    Positon data is used for tracking each individual position holding.
    """
    datetime: date = None
    initialCapital: float = 0.0
    finalCapital: float = 0.0
    deposit: float = 0.0
    withdrawal: float = 0.0
    commission: float = 0.0
    riskCompany: float = 0.0
    riskExchange: float = 0.0
    marginCompany: float = 0.0
    marginCompanyRate: float = 0.0
    marginExchange: float = 0.0
    marginExchangeRate: float = 0.0
    freeCapCompany: float = 0.0
    freeCapExchange: float = 0.0
    profitOrLoss: float = 0.0

@dataclass
class CalSettlementData:
    """
    Positon data is used for tracking each individual position holding.
    """
    symbol: str
    datetime: date = None
    settlement: float = 0.0


 
def transform_trade_data(
        backtesting_trades: list
    ) -> List[CalTradeData]:
        """从回测交易列表，构建交易分析数据list"""
        #处理由backtesting返回的trades列表trade in engine.trades
       
        symbol_rq = backtesting_trades[0].symbol
        symbol_contract_rule = get_contract_rule(symbol_rq)
        symbol_size = symbol_contract_rule['ContractSize']
        ifFeeJ = symbol_contract_rule['IfFeeJ']
        # symbol_fee_rate = symbol_contract_rule['feeOpenJ']
        trades: List[CalTradeData] = []
        for trade in backtesting_trades:
            #提前处理
          
#             print(row[['成交时间','品种名称','成交价格','成交手数']])
            order_time = trade.datetime
            trading_day = generate_trading_day(order_time)
            if ifFeeJ:
                if trade.offset in [Offset.OPEN, Offset.CLOSE, Offset.CLOSEYESTERDAY]:
                    trade_fee = trade.price * symbol_size * trade.volume * symbol_contract_rule["feeOpenJ"]
                else:
                    trade_fee = trade.price * symbol_size * trade.volume * symbol_contract_rule["feeCloseTodayJ"]
            else:
                if trade.offset in [Offset.OPEN, Offset.CLOSE, Offset.CLOSEYESTERDAY]:
                    trade_fee = trade.volume * symbol_contract_rule["feeOpenS"]
                else:
                    trade_fee = trade.volume * symbol_contract_rule["feeCloseTodayS"]


            temp_trade = CalTradeData(
                symbol= trade.symbol,
                exchange= trade.exchange,
                direction= trade.direction,
                offset= trade.offset,
                price= trade.price,
                volume= trade.volume,
                fee= trade_fee,
                turnover = trade.price * symbol_size * trade.volume,
                datetime= order_time,
                tradingday = trading_day
                )
            trades.append(temp_trade)

        return trades
    

# 计算两个日期之间的工作日数,非天数.
class workDays():
    def __init__(self, start_date, end_date, days_off=None):
        """days_off:休息日,默认周六日, 以0(星期一)开始,到6(星期天)结束, 传入tupple
        没有包含法定节假日,
        """
        self.start_date = start_date
        self.end_date = end_date
        self.days_off = days_off
        if self.start_date > self.end_date:
            self.start_date,self.end_date = self.end_date, self.start_date
        if days_off is None:
            self.days_off = 5,6
        # 每周工作日列表
        self.days_work = [x for x in range(7) if x not in self.days_off]

    def workDays(self):
        """实现工作日的 iter, 从start_date 到 end_date , 如果在工作日内,yield 日期
        """
        # 还没排除法定节假日
        tag_date = self.start_date
        while True:
            if tag_date > self.end_date:
                break
            if tag_date.weekday() in self.days_work:
                yield tag_date
            tag_date += timedelta(days=1)

    def daysCount(self):
        """工作日统计,返回数字"""
        return len(list(self.workDays()))

    def weeksCount(self, day_start=0):
        """统计所有跨越的周数,返回数字
        默认周从星期一开始计算
        """
        day_nextweek = self.start_date
        while True:
            if day_nextweek.weekday() == day_start:
                break
            day_nextweek += timedelta(days=1)
        # 区间在一周内
        if day_nextweek > self.end_date:
            return 1
        weeks = ((self.end_date - day_nextweek).days + 1)/7
        weeks = int(weeks)
        if ((self.end_date - day_nextweek).days + 1)%7:
            weeks += 1
        if self.start_date < day_nextweek:
            weeks += 1
        return weeks

class DailyResult:
    """"""

    def __init__(self, date: date, close_price: float):
        """"""
        self.date = date
        self.close_price = close_price
        self.pre_close = 0.0

        self.trades = []
        self.trade_count = 0.0

        self.start_pos = 0.0
        self.end_pos = 0.0

        self.turnover = 0.0
        self.commission = 0.0

        self.trading_pnl = 0.0
        self.holding_pnl = 0.0
        self.total_pnl = 0.0
        self.net_pnl = 0.0

    def add_trade(self, trade: CalTradeData):
        """"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: int   
    ):
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 0.01

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * \
                (self.close_price - self.pre_close) * size
        

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            # For normal contract
            self.trading_pnl += pos_change * \
                    (self.close_price - trade.price) * size
            
            self.commission += trade.fee
            self.turnover += trade.turnover

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission


class BacktestingCloseTrade(object):
    """生成交易分析报告
    
    """

    def __init__(self, tradesdata:List[CalTradeData]):
        self.strategy_trades = tradesdata
        self.strategy_contracts = {contract: get_symbol_head(contract) for contract in list(set([trade.symbol for trade in tradesdata if trade.offset.value != '开']))}
        self.strategy_close_trades = self.get_close_trades()
        self.close_trade_statistics_dict = self.calculate_account_close_trade_statistics()
        self.close_trade_statistics_dataframe = pd.DataFrame([value for value in self.close_trade_statistics_dict.values()],index=self.close_trade_statistics_dict.keys())

    def ready_close_trade(self, trades:CalTradeData, positions:CalPositionData, contract_rule:dict):
        """"""
        contract_size = contract_rule['ContractSize']
        ifFeeJ = contract_rule['IfFeeJ']
        open_trade = [trade for trade in trades if trade.offset.value[0] == '开']
        close_trade = [trade for trade in trades if trade.offset.value[0] == '平']
        temp_trades = trades.copy()

        if len(positions)>0:
            if len(open_trade)>0:
                if ifFeeJ:
                    trade_fee = (open_trade[0].fee / open_trade[0].turnover) * \
                    (positions[0].price * contract_size * positions[0].volume)
                else:
                    trade_fee = (open_trade[0].fee / open_trade[0].volume) * positions[0].volume
            else:
                if ifFeeJ:
                    trade_fee = (close_trade[0].fee / close_trade[0].turnover) * \
                    (positions[0].price * contract_size * positions[0].volume)
                else:
                    trade_fee = (close_trade[0].fee / close_trade[0].volume) * positions[0].volume

            temp_trade = CalTradeData(
                symbol= positions[0].symbol,
                exchange= positions[0].exchange,
                direction= positions[0].direction,
                offset= Offset('开'),
                price= float(positions[0].price),
                volume= float(positions[0].volume),
                fee= float(trade_fee),
                turnover = float(positions[0].price*contract_size*positions[0].volume),
                datetime= datetime.combine(self.start,time(15,0,0,0)),
                tradingday = self.start
                )
            temp_trades = [temp_trade] + temp_trades
        return temp_trades

    def generate_close_trade(self, trades:CalTradeData, contract_rule:dict):
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error

        # Holding pnl is the pnl from holding position at day start
        
        CloseTradeList = []
        temp_closetrade = CalCloseTradeData
        pos = 0.0
        contract_size = contract_rule['ContractSize']
        open_datetime = None
        holding_price = 0.0
        fee_sum = 0.0
        turnover_sum = 0.0

        for trade in trades:
    #         print(trade)
            if trade.offset.value=='开':
                holding_price = (holding_price * pos + trade.price * trade.volume) / (pos + trade.volume)
                fee_sum += trade.fee
                turnover_sum += trade.turnover
                open_datetime = trade.datetime
                pos += trade.volume
            else:
                if pos == 0.0:
                    continue
                else:
                    data_fee = trade.fee + (fee_sum / pos) * trade.volume
                    data_turnover = trade.turnover + (holding_price*contract_size) * trade.volume
                    holding_days = 0 if trade.tradingday == generate_trading_day(open_datetime) else \
                        workDays(generate_trading_day(open_datetime),trade.tradingday).daysCount()
                    if trade.direction.value == '多':
                        data_direction = Direction('空')
                        data_profit_or_loss = (holding_price-trade.price) * contract_size * trade.volume
                        data_profit_or_loss_in_price = holding_price-trade.price
                    else:
                        data_direction = Direction('多')
                        data_profit_or_loss = (trade.price-holding_price) * contract_size * trade.volume
                        data_profit_or_loss_in_price = trade.price-holding_price

                    temp_closetrade = CalCloseTradeData(
                        symbol = trade.symbol,
                        symbol_class= get_symbol_head(trade.symbol),
                        exchange= trade.exchange,
                        closingday = trade.tradingday,
                        direction = data_direction,
                        open_price = holding_price,
                        close_price = trade.price,
                        volume = trade.volume,
                        fee = data_fee,
                        turnover = data_turnover,
                        profit_or_loss = data_profit_or_loss,
                        profit_or_loss_in_price=data_profit_or_loss_in_price,
                        holding_time = holding_days if holding_days else 0.5,
                        open_datetime = open_datetime,
                        close_datetime = trade.datetime
                    )

                    if pos >= trade.volume:
        #                 print(pos)
                        fee_sum = max(0,(fee_sum / pos) * (pos-trade.volume))
                        turnover_sum = max(0,(turnover_sum / pos) * (pos-trade.volume))
                        pos -= trade.volume
                    else:
                        pos = 0.0
                        open_datetime = None
                        holding_price = 0.0
                        fee_sum = 0.0
                        turnover_sum = 0.0
                    CloseTradeList.append(temp_closetrade)
        return(CloseTradeList)

    def get_close_trades(self):
        close_trades_data = []
        contract_trades = self.strategy_trades
        contract = contract_trades[0].symbol
        # for contract in self.strategy_contracts.keys():
        #     temp_contract = 'm2209'
        contract_rule = get_contract_rule(contract)
        contract_positions=[]
        temp_up = []
        temp_down = []
        for trade in contract_trades:
            if (trade.direction.value == '多' and trade.offset.value[0]=='开') or \
            (trade.direction.value == '空' and trade.offset.value[0]=='平'):
                temp_up.append(trade)
            else:
                temp_down.append(trade)
        temp_up_position = []
        temp_down_position = []
    #     print(temp_up, temp_up_position)
        contract_trades_up_ready = self.ready_close_trade(temp_up, temp_up_position, contract_rule)
        temp_close_trades_up = self.generate_close_trade(contract_trades_up_ready, contract_rule)
    #     print(temp_down, temp_down_position)
        contract_trades_down_ready = self.ready_close_trade(temp_down, temp_down_position, contract_rule)
        temp_close_trades_down = self.generate_close_trade(contract_trades_down_ready, contract_rule)
        close_trades_data = close_trades_data + temp_close_trades_up + temp_close_trades_down
        return close_trades_data

    def calculate_symbol_close_trade_statistics(self,symbol:str):
        """"""
        symbol_statistic_dict = {}
        temp_close_trades = [close_trade for close_trade in self.strategy_close_trades \
            if close_trade.symbol_class == symbol]
        temp_close_trades_sheet = pd.DataFrame([trade.__dict__ for trade in temp_close_trades])
        temp_close_trades_win = temp_close_trades_sheet[temp_close_trades_sheet['profit_or_loss_in_price']>0]
        temp_close_trades_loss = temp_close_trades_sheet[temp_close_trades_sheet['profit_or_loss_in_price']<0]
        symbol_statistic_dict['symbol_cn' ]= get_contract_rule(symbol)['CNname']
        symbol_statistic_dict['sum_profit' ]= temp_close_trades_sheet['profit_or_loss'].sum() if len(temp_close_trades_sheet) > 0 else 0 
        symbol_statistic_dict["net_profit"] = (temp_close_trades_sheet['profit_or_loss'].sum() - temp_close_trades_sheet['fee'].sum()) if len(temp_close_trades_sheet) > 0 else 0
        symbol_statistic_dict["trade_times"] = len(temp_close_trades_sheet)
        symbol_statistic_dict["win_times"] = len(temp_close_trades_win)
        symbol_statistic_dict["win_times_rate"] = len(temp_close_trades_win) / len(temp_close_trades_sheet) if len(temp_close_trades_sheet) >0 else 0
        symbol_statistic_dict["max_profit"] = max(temp_close_trades_win['profit_or_loss']) if len(temp_close_trades_win) > 0 else 0
        symbol_statistic_dict["max_loss"] = min(temp_close_trades_loss['profit_or_loss']) if len(temp_close_trades_loss) > 0 else 0
        symbol_statistic_dict["fee_sum"] = temp_close_trades_sheet['fee'].sum() if len(temp_close_trades_sheet) > 0 else 0
        symbol_statistic_dict["holding_mean"] = temp_close_trades_sheet['holding_time'].mean() if len(temp_close_trades_sheet) > 0 else 0
        symbol_statistic_dict["profit_mean"] = temp_close_trades_win['profit_or_loss'].mean() if len(temp_close_trades_win) > 0 else 0
        symbol_statistic_dict["loss_mean"] = temp_close_trades_loss['profit_or_loss'].mean() if len(temp_close_trades_loss) > 0 else 0
        symbol_statistic_dict["profit_price_mean"] = temp_close_trades_win['profit_or_loss_in_price'].mean() if len(temp_close_trades_win) > 0 else 0
        symbol_statistic_dict["profit_price_max"] = temp_close_trades_win['profit_or_loss_in_price'].max() if len(temp_close_trades_win) > 0 else 0
        symbol_statistic_dict["loss_price_min"] = temp_close_trades_loss['profit_or_loss_in_price'].min() if len(temp_close_trades_loss) > 0 else 0
        symbol_statistic_dict["loss_price_mean"] = temp_close_trades_loss['profit_or_loss_in_price'].mean() if len(temp_close_trades_loss) > 0 else 0
        symbol_statistic_dict["profit_divide_loss"] = (temp_close_trades_win['profit_or_loss_in_price'].mean()+0.01) \
            / (abs(temp_close_trades_loss['profit_or_loss_in_price'].mean())+0.01)
        # symbol_statistic_dict["profit_divide_loss_top_20_rate"] = temp_close_trades_win[temp_close_trades_win['profit_or_loss_in_price']\
        #     .rank(pct=True, ascending=False)<=0.2]['profit_or_loss_in_price'].mean() / abs(temp_close_trades_loss[temp_close_trades_loss['profit_or_loss_in_price']\
        #         .rank(pct=True, ascending=False)>=0.8]['profit_or_loss_in_price'].mean())
        # symbol_statistic_dict["profit_divide_loss_holdingday_top_20_rate"] = temp_close_trades_win[temp_close_trades_win['profit_or_loss_in_price']\
        #     .rank(pct=True, ascending=False)<=0.2]['holding_time'].mean() / temp_close_trades_loss[temp_close_trades_loss['profit_or_loss_in_price']\
        #         .rank(pct=True, ascending=False)>=0.8]['holding_time'].mean()
        symbol_statistic_dict_str = symbol_statistic_dict.copy()
        for key, value in symbol_statistic_dict_str.items():
            if key in ["sum_profit","net_profit","max_profit","max_loss","fee_sum","profit_mean","loss_mean","profit_price_max"]:
                value = str(round(value,2))
            elif key in ["win_times"]:
                value = str(int(value))
            elif key in ["win_times_rate"]:
                value = str(round(value*100,2))+"%"
            elif key in ["holding_mean","loss_price_mean","profit_price_mean","profit_divide_loss","loss_price_min"]:
                value = str(round(value,1))
            else:
                pass
            symbol_statistic_dict_str[key]=value

        statistic_item_cn = {"symbol_cn":"品种","sum_profit":"总收益","net_profit":"净收益","trade_times":"交易笔数",\
                        "win_times":"盈利次数","win_times_rate":"胜率","max_profit":"单笔最大获利","max_loss":"单笔最大损失",\
                            "fee_sum":"手续费","holding_mean":"平均持仓天数","profit_mean":"平均单笔获利","loss_mean":"平均单笔损失",\
                            "profit_price_mean":"平均单笔获利点数","profit_price_max":"最大单笔获利点数","loss_price_min":"最大单笔损失点数",\
                            "loss_price_mean":"平均单笔损失点数","profit_divide_loss":"盈亏比"}
        statistic_dict_cn = { statistic_item_cn[key]:value for key,value in symbol_statistic_dict_str.items()}
        return symbol_statistic_dict_str,statistic_dict_cn

    def calculate_account_close_trade_statistics(self):
        account_close_trades_stastic_dict = {}
        for symbol in list(set(self.strategy_contracts.values())):
            # print(symbol)
            account_close_trades_stastic_dict[symbol]=self.calculate_symbol_close_trade_statistics(symbol)
        return account_close_trades_stastic_dict


    def get_symbol_stastic_sheet(self):
        symbol_stastic_sheet = DataFrame(columns=['品种','总盈亏','交易笔数','盈利笔数','平均赢利点数','平均亏损点数','盈亏比','胜率','持仓均期'])
        for symbol in list(set(self.strategy_contracts.values())):
            symbol_stastic_dict, _ = self.calculate_symbol_statistics(symbol)

        return symbol_stastic_sheet

    def show_results(self):      
        close_trade_statistics_dict = self.calculate_account_close_trade_statistics()
        close_trade_statistics_dataframe = pd.DataFrame([value for value in close_trade_statistics_dict.values()],index=close_trade_statistics_dict.keys())
        close_trade_statistics_dataframe.index.name = "symbol"
        print(close_trade_statistics_dataframe)


