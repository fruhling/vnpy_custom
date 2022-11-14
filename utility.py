"""
General utility functions.
"""

from cmath import log
from datetime import date,datetime, timedelta
from time import sleep
from dateutil.relativedelta import relativedelta
import copy
from vnpy.trader.object import TradeData, HistoryRequest, BarData, TickData, PositionData
from vnpy_custom.myobject import MyTradeData, SignData, MemberRankData
from vnpy.trader.constant import Exchange, Interval, Offset, Direction
from pandas import DataFrame, concat, read_csv
import pandas as pd
from vnpy.trader.database import BaseDatabase, get_database
from vnpy.trader.datafeed import BaseDatafeed, get_datafeed
from tzlocal import get_localzone

from vnpy.trader.utility import generate_vt_symbol, extract_vt_symbol, load_json, save_json, BarGenerator, ArrayManager, get_file_path
from vnpy_ctastrategy.backtesting import DailyResult

import re
import json
import interval
from collections import defaultdict
from pathlib import Path

from typing import Callable, Dict, Tuple, Union, Optional
from math import floor, ceil

import numpy as np
import talib

from vnpy.trader.object import BarData, TickData

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
from email.mime.application import MIMEApplication
from seatable_api import Base, context
import rqdatac

LOCAL_TZ = get_localzone()

database_manager: BaseDatabase = get_database()
vnpy_home_path = Path.home().joinpath(".vntrader")

domain_contracts_dict_filename = vnpy_home_path.joinpath("data/json/domain_contracts.json")
domain_contracts_dict = load_json(domain_contracts_dict_filename)

def add_model_str(model):
    """返回用于通用回测脚本中，导入模型的命令文本"""
    modelClass = get_model_class(model)
    return  f'engine.add_strategy({modelClass}, settingdict)'

def add_model_str_in_class(model):
    """返回在策略中，导入模型的命令文本"""
    modelClass = get_model_class(model)
    return  f'self.engine.add_strategy({modelClass}, settingdict)'

def cal_daily_var(open, high, low):
    """cal daily max up or down"""
    return max((high - open), (open-low))/open
    
def cal_strategy_symbol_pos(capital:float, proportion:float,risk_level:float,price:float,size:float)->int:
    """返回策略持仓，用于策略按自定义资金和持仓比重计算收益"""
    return floor((capital * proportion * risk_level)/(price * size))

def cal_daily_sheet(daily_sheet_temp, nlag=5):
    """返回日度市场情况报告，策略星报告统计使用"""
    daily_sheet_temp['今日涨跌'] = round((daily_sheet_temp['close_price'] - daily_sheet_temp['close_price'].shift(1))/daily_sheet_temp['close_price'].shift(1),4)*100
    daily_sheet_temp['close_price'] = daily_sheet_temp['close_price']
    daily_sheet_temp['pre_close_price'] =  daily_sheet_temp['close_price'].shift(1)
    daily_sheet_temp['近5日平均涨跌'] = (pow(daily_sheet_temp['close_price'].shift(1)/daily_sheet_temp['close_price'].shift(nlag),1/(nlag-1))-1)*100
    daily_sheet_temp['ROC'] = round(talib.ROC(daily_sheet_temp['close_price'],1).rolling(nlag*5).std()*np.sqrt(250),2)
    daily_sheet_temp['趋势度']=round((daily_sheet_temp['close_price']-daily_sheet_temp['open_price'])/(daily_sheet_temp['high_price']-daily_sheet_temp['low_price']),2)*100
    daily_sheet_temp['乖离率']=round(daily_sheet_temp['close_price']/daily_sheet_temp['close_price'].rolling(6).mean()-1,4)*100
    daily_sheet_temp['历史波动']=daily_sheet_temp['close_price']/daily_sheet_temp['close_price'].shift(1)
    daily_sheet_temp['历史波动率']=round(daily_sheet_temp['历史波动'].rolling(5).std(),4)*100
    ATR=talib.ATR(daily_sheet_temp['high_price'],daily_sheet_temp['low_price'],daily_sheet_temp['close_price'],1)

    max_atr=ATR.rolling(nlag).max()
    min_atr=ATR.rolling(nlag).min()
    NATR=(ATR-min_atr)/(max_atr-min_atr)
    daily_sheet_temp['se_ATR']=round(NATR.rolling(nlag).std(),2)
    #daily_sheet_temp['ATR']=round(talib.ATR(daily_sheet_temp['high_price'],daily_sheet_temp['low_price'],daily_sheet_temp['close_price'],1),2)
    #daily_sheet_temp['NATR']=round(talib.NATR(daily_sheet_temp['high_price'],daily_sheet_temp['low_price'],daily_sheet_temp['close_price'],1),2)
    #daily_sheet_temp.loc[daily_sheet_temp['effect']>100,'effect'] = 100
    return daily_sheet_temp

def custom_resampler(df):
    """自定义K线周期转换，用于pandas中，通过1分钟线合成其他周期K线"""
    if len(df) == 0:
        return np.nan
    else:
        if df.name == "open":
            return np.asarray(df)[0]
        if df.name == "close":
            return np.asarray(df)[-1]
        if df.name == "low":
            return np.min(df)
        if df.name == "high":
            return np.max(df)
        if df.name == "volume":
            return np.sum(df)

def download_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime
    ) -> int:
    """
    Query bar data from RQData.
    """
    datafeed: BaseDatafeed = get_datafeed()
    datafeed.init()
   
    req = HistoryRequest(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start=start,
        end=datetime.now(LOCAL_TZ)
    )

    data = datafeed.query_bar_history(req)
    rqdatac.reset()
    if data:
        while True:
            try:
                database_manager.save_bar_data(data)
                break
            except RuntimeError:
                # print("Runtime Error")
                sleep(5)

    return 0

def download_daily_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime
    ) -> int:
    """
    Query bar data from RQData.
    """
    datafeed: BaseDatafeed = get_datafeed()
    datafeed.init()
   
    req = HistoryRequest(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start=start,
        end=datetime.now(LOCAL_TZ)
    )

    data = datafeed.query_daily_bar_history(req)
    rqdatac.reset()
    if data:
        while True:
            try:
                database_manager.save_daily_bar_data(data)
                break
            except RuntimeError:
                # print("Runtime Error")
                sleep(5)
        return(len(data))

    return 0

def download_memberrank_data(
        symbol: str,
        exchange: Exchange,
        interval: str,
        start: datetime
    ) -> int:
        """
        Query memberrank data from datafeed.
        """
        datafeed: BaseDatafeed = get_datafeed()
        datafeed.init()
        req = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=Interval(interval),
            start=start,
            end=datetime.now(LOCAL_TZ)
        )
        """
        vt_symbol = f"{symbol}.{exchange.value}"
        contract = self.main_engine.get_contract(vt_symbol)

        # If history data provided in gateway, then query
        if contract and contract.history_data:
            data = self.main_engine.query_history(
                req, contract.gateway_name
            )
        # Otherwise use datafeed to query data
        else:
        """
        data = datafeed.query_member_rank_history(req)
        rqdatac.reset()
        
        if data:
            while True:
                try:
                    database_manager.save_member_rank_data(data)
                    break
                except RuntimeError:
                    # print("Runtime Error")
                    sleep(5)
            return(len(data))

        return 0

def delete_contract_paper_account_data(paper_account_file,contract):
    positions_data = load_json(paper_account_file)
    GATEWAY_NAME = 'PAPER'

    #make positions
    paper_account_positions = dict()

    for d in positions_data:
        d_vt_symbol = d["vt_symbol"]
        d_direction = Direction(d["direction"])
        d_symbol, d_exchange = extract_vt_symbol(d_vt_symbol)
        d_key = (d_vt_symbol, d_direction)
        position = PositionData(
                    symbol=d_symbol,
                    exchange=d_exchange,
                    direction=d_direction,
                    volume = d['volume'],
                    price = d['price'],
                    gateway_name=GATEWAY_NAME
                )
        paper_account_positions[d_key] = position
        
    contractRule = get_contract_rule(contract)
    exchange = Exchange(contractRule['exchange'])
    contract_ctp = to_CTP_symbol(contract)
    contract_vt_symbol = generate_vt_symbol(contract_ctp,exchange)
    long_position_key = (contract_vt_symbol, Direction.LONG)
    short_position_key = (contract_vt_symbol, Direction.SHORT)
    if long_position_key in paper_account_positions:
        paper_account_positions.pop(long_position_key)
        print(f"delete:{long_position_key}")
    
    if short_position_key in paper_account_positions:
        paper_account_positions.pop(short_position_key)
        print(f"delete:{short_position_key}")
    
    paper_account_position_data = []
    for position in paper_account_positions.values():
        if not position.volume:
            continue

        d = {
            "vt_symbol": position.vt_symbol,
            "volume": position.volume,
            "price": position.price,
            "direction": position.direction.value
        }
        paper_account_position_data.append(d)

    save_json(paper_account_file, paper_account_position_data)

def get_30minsign_time(trade_time,min_period_str):
    """用于转换交易信号中时间为对应的30min为单位的交易时间"""
    result = []
    time_period_list=[x.split('-') for x in min_period_str.split(',')]
    time_period_len = len(time_period_list)
    for i in range(time_period_len):
        time_period = interval.Interval(time_period_list[i][0],time_period_list[i][1],lower_closed=False, upper_closed=True)
        if time_period_list[i][0] in ['09:00:00','21:00:00','13:00:00','00:00:00']:
            time_period = interval.Interval(time_period_list[i][0],time_period_list[i][1],lower_closed=True, upper_closed=True)
        result.append(time_period)
        
    trade_time_str = trade_time.strftime('%H:%M:%S')
    
    for period in result:
        if trade_time_str in period:
            if period.upper_bound == '23:59:59':
                sign_time = datetime.strptime('00:00:00','%H:%M:%S')
            else:
                sign_time = datetime.strptime(period.upper_bound,'%H:%M:%S')
            return trade_time.replace(hour=sign_time.hour, minute = sign_time.minute, second=0,microsecond=0)
    
    return trade_time.replace(minute = int(trade_time.minute - trade_time.minute % 10), second=0,microsecond=0)

def generate_strategy_json():
    """通过strategy_sheet，生成strategy_setting.json"""
    strategy_sheet1 = pd.read_excel(Path.home().joinpath(".vntrader").joinpath("strategy_sheet.xlsx"),sheet_name='strategy1')
    strategy_sheet2 = pd.read_excel(Path.home().joinpath(".vntrader").joinpath("strategy_sheet.xlsx"),sheet_name='strategy2')
    dict_all = {}
    dict_1 = {}
    dict_2 = {}


    for index, row in strategy_sheet1.iterrows():
        strategy_wholename = row['strategy_class']+'_'+ row['strategy_name'] +'_'+str(row['strategy_num'])+'_'+str(row['strategy_period'])
        strategy_dict = {}
        strategy_dict['class_name'] = row['strategy_name']
        symbol = row['symbol']
        contractRule = get_contract_rule(symbol)
        exchange = Exchange(contractRule['exchange'])
        vt_symbol = generate_vt_symbol(symbol,exchange)
        strategy_dict['vt_symbol'] = vt_symbol
        strategy_dict['setting'] = setting_dict(row)
        dict_1[strategy_wholename] = strategy_dict
    
    for index, row in strategy_sheet2.iterrows():
        strategy_wholename = row['strategy_class']+'_'+ row['strategy_name'] +'_'+str(row['strategy_num'])+'_'+str(row['strategy_period'])
        strategy_dict = {}
        strategy_dict['class_name'] = row['strategy_name']
        symbol = row['symbol']
        contractRule = get_contract_rule(symbol)
        exchange = Exchange(contractRule['exchange'])
        vt_symbol = generate_vt_symbol(symbol,exchange)
        strategy_dict['vt_symbol'] = vt_symbol
        strategy_dict['setting'] = setting_dict(row)
        dict_2[strategy_wholename] = strategy_dict
    dict_all = {**dict_2, **dict_1}
    
    save_json('./cta_strategy_setting'+datetime.now().strftime('%Y%m%d%H%M')+'.json',dict_all)
    return 'file saved'

def get_strategy_group(n):
    """用于策略汇总，从策略表中读取策略分组"""
    strategy_sheet = pd.read_excel(Path.home().joinpath(".vntrader").joinpath("strategy_sheet.xlsx"),sheet_name='group'+str(n))
    strategy_sheet['symbol_num'] = strategy_sheet['symbol_num'].astype(str)
    # strategy_sheet2 = pd.read_excel(Path.home().joinpath(".vntrader").joinpath("strategy_sheet.xlsx"),sheet_name='strategy2')
    # dict_all = {}
    # dict_1 = {}
    # dict_2 = {}
    return strategy_sheet

def get_strategynum_list(start:str, end:str):
    """用于策略汇总计算，从数据库中读取时间段内的所有交易信号"""
    startday = datetime.fromisoformat(start)
    endday = datetime.fromisoformat(end)
    endday_time = endday.replace(hour = 16)
    symbol_trades = database_manager.load_trade_all(start = startday,end = endday_time)
    # strategy_sheet2 = pd.read_excel(Path.home().joinpath(".vntrader").joinpath("strategy_sheet.xlsx"),sheet_name='strategy2')
    # dict_all = {}
    # dict_1 = {}
    # dict_2 = {}
    return list(set([trade.strategy_num for trade in symbol_trades]))

def get_strategy_sheet():
    """读取strategy_setting的内部，生成策略sheet"""
    strategy_setting = load_json(Path.home().joinpath(".vntrader").joinpath("cta_strategy_setting.json"))
    strategy_dict = {'strategy_num':[],'strategy_class':[], 'strategy_name': [],'vt_symbol':[]}
    for key, value in strategy_setting.items():
        strategy_class, strategy_name, strategy_num,_ = key.split('_')
        vt_symbol = value['vt_symbol']
        strategy_dict['strategy_num'].append(strategy_num)
        strategy_dict['strategy_class'].append(strategy_class)
        strategy_dict['strategy_name'].append(strategy_name)
        strategy_dict['vt_symbol'].append(vt_symbol)
        
    strategy_group = DataFrame.from_dict(strategy_dict)
    return strategy_group

def get_mystrategy_sheet():
    """读取策略strategy_setting，返回策略设置sheet"""
    strategy_setting = load_json(Path.home().joinpath(".vntrader").joinpath("cta_strategy_setting.json"))
    strategy_dict = {'mystrategy_name':[],'vt_symbol':[],'strategy_setting':[]}
    for key, value in strategy_setting.items():
        strategy_dict['mystrategy_name'].append(key)
        strategy_dict['vt_symbol'].append(value['vt_symbol'])
        strategy_dict['strategy_setting'].append(value['setting'])
        
    strategy_group = DataFrame.from_dict(strategy_dict)
    return strategy_group

def get_daily_sheet2(symbol,exchange,start)->DataFrame:
    """get symbol n days dailybar"""
   
    daily_bars = database_manager.load_bar_data(
        symbol = symbol, 
        exchange = exchange, 
        interval = Interval.DAILY, 
        start = start, 
        end=datetime.now().date()
    )
    daily_data = defaultdict(list)
    for bar in daily_bars:
        for key,value in bar.__dict__.items():
            daily_data[key].append(value)
    daily_df = pd.DataFrame.from_dict(daily_data).set_index("datetime")
    return daily_df

def get_daily_sheet(symbol,exchange,start)->DataFrame:
    """get symbol n days dailybar"""
   
    download_bar_data(symbol = symbol,
        exchange = exchange,
        interval = Interval.DAILY,
        start=start
        )
    daily_bars = database_manager.load_bar_data(
        symbol = symbol, 
        exchange = exchange, 
        interval = Interval.DAILY, 
        start = start, 
        end=datetime.now().date()
    )
    daily_data = defaultdict(list)
    for bar in daily_bars:
        for key,value in bar.__dict__.items():
            daily_data[key].append(value)
    daily_df = pd.DataFrame.from_dict(daily_data).set_index("datetime")
    return daily_df

def get_x_minutes_sheet(symbol,exchange,bar_interval,start)->DataFrame:
    """get symbol x minutes bar"""
   
    data_bars = database_manager.load_bar_data(
        symbol = symbol, 
        exchange = exchange, 
        interval = Interval.MINUTE, 
        start = start, 
        end=datetime.now().date()
    )
    bar_datas = defaultdict(list)
    for bar in data_bars:
        for key,value in bar.__dict__.items():
            bar_datas[key].append(value)
    minute_bar_df = pd.DataFrame.from_dict(bar_datas).set_index("datetime")
    minute_bar_df.rename(columns={"open_price":"open","high_price":"high","low_price":"low","close_price":"close"},inplace=True)
    data = pd.DataFrame()
    data["open_price"] = minute_bar_df["open"].resample(bar_interval).apply(custom_resampler).dropna()
    data["high_price"] = minute_bar_df["high"].resample(bar_interval).apply(custom_resampler).dropna()
    data["low_price"] = minute_bar_df["low"].resample(bar_interval).apply(custom_resampler).dropna()
    data["close_price"] = minute_bar_df["close"].resample(bar_interval).apply(custom_resampler).dropna()
    data["volume"] = minute_bar_df["volume"].resample(bar_interval).apply(custom_resampler).dropna()
    return data

def get_model_class(model:str):
    """按合约规则由策略文件名小写用_分隔，转至类名，驼峰命名规则"""
    model_name_list = model.split('_')
    model_class = ''
    i = 0
    for i in range(len(model_name_list)-1):
        model_class += model_name_list[i].capitalize()
    return model_class+'Strategy'

def import_model_str(model):
    """返回文本格式的，引入策略的import命令，用于通用exec方式导入"""
    modelClass = get_model_class(model)
    # return  f'from vnpy_ctastrategy.strategies.{model} import {modelClass}'
    return  f'from strategies.{model} import {modelClass}'

def cal_trading_statistic(trading_list):
    """计算最近几次交易的重要数据
    sum_profit, avg_profit, win_rate,holding_klines_mean,open_price_deviate_mean_value
    """
    if trading_list:
        profit_list = [(trade['close_price'] - trade['open_price'])/(trade['open_price']+0.01) if trade['close_direction']=='空' else (trade['open_price'] - trade['close_price'])/(trade['open_price']+0.01) for trade in trading_list]
        sum_profit = str(round(np.sum(profit_list),3) * 100) + '%'
        avg_profit = str(round(np.mean(np.abs(profit_list)),3) * 100)+ '%'
        win_rate = str(round(np.sum([1 if profit > 0 else 0 for profit in profit_list]) / len(profit_list),2) *100)+ '%'
        holding_klines_mean = np.mean([trade['holding_klines'] for trade in trading_list])
        open_price_deviate_mean_value = str(round(abs((trading_list[-1]['open_price'] / np.mean([trade['open_price'] for trade in trading_list if trade['close_direction']== trading_list[-1]['close_direction']]))- 1),3) * 100)+ '%'
        return {"trading_counts":len(trading_list),"sum_profit":sum_profit, "avg_profit":avg_profit, "win_rate":win_rate,"holding_klines_mean":holding_klines_mean,"open_price_bias_avg":open_price_deviate_mean_value}
    else:
        return None

def return_domain_lastday(contract,today_domain=domain_contracts_dict):
    startday='2010-01-01'
    endday='2025-01-01'
    contracts=database_manager.load_domain_info(contract,startday,endday)
    d=[c.datetime for c in contracts if c.main==contract.upper()]
    if d==[]:
        return None
    if contract==today_domain[''.join(re.findall(r'[A-Za-z]',contract)).upper()]:
        return None
    else:
        return str(d[-1])[0:10]

def return_domain_firstday(contract):
    startday='2010-01-01'
    endday='2025-01-01'
    contracts=database_manager.load_domain_info(contract,startday,endday)
    d=[c.datetime for c in contracts if c.main==contract.upper()]
    if d==[]:
        return None
    # today_domain=load_json('/root/.vntrader/data/json/domain_contracts.json')
    # if contract==today_domain[''.join(re.findall(r'[A-Za-z]',contract)).upper()]:
    #     return None
    return str(d[0])[0:10]

#汇总工具
      
class StrategyDaily(object):
    """根据数据库中的交易数据，得出策略报告所需的信息
    实例：strategy_daily = StrategyDaily(capital = 500000, strategy_name = 'TrendStar', start='2017-03-01', end='2021-04-01')
        指定初始资金capital，strategy_name,策略名，起止日期
        cal_strategy_daily()
    """

    def __init__(self, capital, sheet_num,strategy_group, risk_level, start, end=None):
        self.capital = capital
        self.risk_level = risk_level
        self.strategy_group = strategy_group
        self.start = start
        self.end = end
        if not end:
            self.end = datetime.strftime(datetime.today(),'%Y-%m-%d')
        self.strategy_trades = []
        self.strategy_daily = DataFrame()
        self.symbol_same_strategy_daily = DataFrame()
        self.strategy_statistics = DataFrame()
        self.strategy_holdings = DataFrame()
        strategys_group_sheet = get_strategy_group(sheet_num)
        self.strategys_group_sheet = strategys_group_sheet
        if not strategy_group:
            self.group_sheet=strategys_group_sheet.copy()
        else:
            self.group_sheet = strategys_group_sheet[strategys_group_sheet['group']==strategy_group].copy()
        if len(self.group_sheet) ==0:
            raise ValueError(f"不存在{strategy_group}策略组")
        self.group_sheet['vt_symbol'] = self.group_sheet['symbol'].apply(get_vt_symbol)
        

    def cal_symbol_daily(self,
                        vt_symbol:str, 
                        strategy_num:str,
                        start:str,
                        end:str,
                        proportion:float
        ):

        symbol, exchange = extract_vt_symbol(vt_symbol)
        interval = Interval.DAILY
        startday = datetime.fromisoformat(start)
        endday = datetime.fromisoformat(end)
        self.contractRule = get_contract_rule(vt_symbol)
        contractRule = get_contract_rule(vt_symbol)
        risk_level = self.risk_level
        download_bar_data(symbol = symbol,exchange = exchange,interval = Interval.DAILY,start = self.start)
        daily_data = database_manager.load_bar_data(symbol,exchange,interval, startday, endday)
        symbol_trades = database_manager.load_trade_data(strategy_num = strategy_num,start = str(startday)[0:10],end = str(endday)[0:10] + " 16:00:00")
        self.symbol_trades = symbol_trades

        daily_results = {}
        daily_close = []
        for bar in daily_data:
            daily_results[bar.datetime.date()] = DailyResult(bar.datetime.date(), bar.close_price)#字典，每个日期对应一个dailyresult；
            daily_close.append(bar.close_price)

        close_average = np.mean(daily_close)
        symbol_pos = max(1,cal_strategy_symbol_pos(self.capital, proportion,risk_level,close_average,contractRule['ContractSize']))
        print(symbol_pos)
        for trade in symbol_trades:
            trade.volume = symbol_pos
            self.strategy_trades.append(trade)
            d = generate_trading_day(trade.datetime).date()
            daily_result = daily_results[d]#daily_results的键值为bardata的日期，这里索引d为tradedata的日期，在当天交易日恰好有交易且在交易日未结束时运行，如果endday取当天或者未来日期，那么bardata比tradedata少一天，报错datetime(d)；
            daily_result.add_trade(trade)
            
        pre_close = 0
        start_pos = 0
        for daily_result in daily_results.values():
            daily_result.calculate_pnl(pre_close,start_pos,contractRule['ContractSize'],contractRule['feeOpenJ'],contractRule['PriceMinMove']+contractRule['feeOpenS']/contractRule['ContractSize'],False)
            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos
            
        results = defaultdict(list)
        for daily_result in daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)
                
        daily_df = DataFrame.from_dict(results).set_index("date")
        return daily_df

    def cal_symbol_same_strategy_daily(self):
        symbol_same_strategy_daily = DataFrame()
        # symbol_same_strategy_trades = database_manager.load_trade_data(strategy_num = strategy_num,start = self.start,end = self.end)
        # for index, row in self.strategy_sheet.iterrows():
        for index, row in self.group_sheet.drop_duplicates(subset=['startd','endd']).iterrows():
            # vt_symbol = row['vt_symbol']
            s=row['startd']
            e=row['endd']
            for index,row in self.group_sheet.loc[(self.group_sheet['startd']==s)&(self.group_sheet['endd']==e),['symbol','symbol_num']].iterrows():
                symbol_same_strategy_num = str(row['symbol_num'])#策略代码加品种排序，如100106；
                symbol_contracts = [row['symbol']]#非完整合约代码，如y2201；
                symbol_same_strategy_trades = database_manager.load_symbol_same_trades(symbol_same_strategy_num,str(s)[0:10],str(e)[0:10] + " 16:00:00")
                symbol_contracts_all = list(set(symbol_contracts+[trade.symbol for trade in symbol_same_strategy_trades]))#所有非完整合约代码；
                for contract in symbol_contracts_all:
            # symbolHead = ''.join(re.findall(r'[A-Za-z]', vt_symbol.split(".")[0].upper()))
            # contractRule = contractRuleDict[symbolHead]
                    vt_symbol = get_vt_symbol(contract)#完整合约代码，如y2201.DCE;
                    contract_num = ''.join(re.findall(r'[0-9]', contract))#合约年月，如2201；
                    strategy_num = symbol_same_strategy_num+contract_num#可定位代码，如1001062201;
                    p=self.group_sheet.loc[self.group_sheet['startd']==s,['symbol_num','proportion']][self.group_sheet['symbol_num']==strategy_num[:6]]['proportion']
                    print(vt_symbol)
                    symbol_daily = self.cal_symbol_daily(vt_symbol,strategy_num,str(s)[0:10],str(e)[0:10],p)
                    symbol_same_strategy_daily=concat([symbol_same_strategy_daily,symbol_daily])
        self.symbol_same_strategy_daily = symbol_same_strategy_daily.groupby(symbol_same_strategy_daily.index).sum()
        return symbol_same_strategy_daily
    
    
    # def cal_strategy_daily(self):
    #     strategy_daily = DataFrame()
    #     # for index, row in self.strategy_sheet.iterrows():
    #     for index, row in self.group_sheet.iterrows():
    #         vt_symbol = row['vt_symbol']
    #         strategy_num = row['strategy_num']
    #         # symbolHead = ''.join(re.findall(r'[A-Za-z]', vt_symbol.split(".")[0].upper()))
    #         # contractRule = contractRuleDict[symbolHead]
    #         symbol_daily = self.cal_symbol_daily(vt_symbol,strategy_num,self.start,self.end)
    #         strategy_daily=concat([strategy_daily,symbol_daily])
    #     self.strategy_daily = strategy_daily.groupby(strategy_daily.index).sum()
    #     return strategy_daily

    def calculate_statistics(self):
        """"""
        if not self.strategy_trades:
            # self.cal_strategy_daily()
            self.cal_symbol_same_strategy_daily()

        # df = self.strategy_daily
        df = self.symbol_same_strategy_daily
        annual_days = 240
        risk_free = 0.02

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date = ""
            end_date = ""
            total_days = 0
            profit_days = 0
            loss_days = 0
            end_balance = 0
            max_drawdown = 0
            max_ddpercent = 0
            max_drawdown_duration = 0
            total_net_pnl = 0
            daily_net_pnl = 0
            total_commission = 0
            daily_commission = 0
            total_slippage = 0
            daily_slippage = 0
            total_turnover = 0
            daily_turnover = 0
            total_trade_count = 0
            daily_trade_count = 0
            total_return = 0
            annual_return = 0
            daily_return = 0
            return_std = 0
            sharpe_ratio = 0
            return_drawdown_ratio = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital

            # When balance falls below 0, set daily return to 0
            x = df["balance"] / df["balance"].shift(1)
            y=df['balance']/df['balance'].shift(4)
            x[x <= 0] = np.nan
            y[x <= 0] = np.nan
            df["return"] = np.log(x).fillna(0)
            df['weekly_return']=np.log(y).fillna(0)

            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = len(df[df["net_pnl"] > 0])
            loss_days = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            # weekly_max_ddpercent= df['ddpercent'].rolling(5).min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            weekly_trade_count= df['trade_count'].rolling(5).sum().iloc[-1]
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital- 1) * 100
            annual_return = total_return / total_days * annual_days
            daily_return = df["return"].mean() * 100
            weekly_return=df['weekly_return'].iloc[-1]
            
            return_std = df["return"].std() * 100
            weekly_return_std=df['return'].rolling(5).std().iloc[-1]

            if return_std:
                daily_risk_free = risk_free / np.sqrt(annual_days)
                sharpe_ratio = (daily_return - daily_risk_free) / return_std * np.sqrt(annual_days)
            else:
                sharpe_ratio = 0
            if weekly_return_std:
                weekly_risk_free = risk_free*5 / np.sqrt(annual_days)
                weekly_sharpe_ratio = (weekly_return - weekly_risk_free)/ weekly_return_std
            else:
                weekly_sharpe_ratio = 0
            return_drawdown_ratio = -total_return /max_ddpercent

        statistics = {
            "开始时间": start_date,
            "截至时间": end_date,
            "总交易天数": total_days,
            "盈利天数": profit_days,
            "亏损天数": loss_days,
            "初始权益": self.capital,
            "当前权益": end_balance,
            "最大回撤": max_drawdown,
            "百分比最大回撤": max_ddpercent,
            "最大回撤持续时间": max_drawdown_duration,
            "总盈亏": total_net_pnl,
            "日均盈亏": daily_net_pnl,
            "总交易成本": total_commission + total_slippage,
            "日均交易成本": daily_commission + daily_slippage,
            "总成交额": total_turnover,
            "日均成交额": daily_turnover,
            "总成交笔数": total_trade_count,
            "日均成交笔数": daily_trade_count,
            "总收益率": total_return,
            "年化收益": annual_return,
            "日均收益率": daily_return,
            "收益标准差": return_std,
            "夏普比率": sharpe_ratio,
            "收益回撤比": return_drawdown_ratio,
            "近一周成交笔数":weekly_trade_count,
            "近一周收益率":weekly_return,
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)
        # self.strategy_statistics = statistics

        return statistics

    def show_chart(self):
        """"""
        if not self.strategy_trades:
            # self.cal_strategy_daily()
            self.cal_symbol_same_strategy_daily()

        # df = self.strategy_daily
        df = self.symbol_same_strategy_daily
        # print("balance" in df.columns)
        # Check for init DataFrame
        # if df is None:
        if "balance" not in df.columns or "drawdown" not in df.columns:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["资金余额", "每日回撤", "每日净盈亏", "盈亏分布"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="资金余额"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="每日回撤"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="每日净盈亏")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="天数")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        self.strategy_fig = fig
        # fig.write_image(datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')+'.png')
        
        fig.show()
    
    # def show_strategy_symbol(self):
    #     strategys = self.strategy_sheet.copy()
    #     strategys.rename(columns={'strategy_num':'策略编号','vt_symbol':'合约','strategy_class':'策略大类','strategy_name':'策略名称'},inplace=True)
    #     return strategys
    
    def show_trades(self,start=None) ->DataFrame:
        if not self.strategy_trades:
            # self.cal_strategy_daily()
            self.cal_symbol_same_strategy_daily()

        if not start:
            start = self.start
        trades = [trade.__dict__ for trade in self.strategy_trades]
        trades_sheet = DataFrame(trades)
        if not trades_sheet.empty:
            trades_sheet['datetime'] = trades_sheet['datetime'].apply(lambda x: x.replace(tzinfo=None))
            # trades_sheet_show
            trades_sheet_show = trades_sheet[(trades_sheet['calculate'])&(trades_sheet['datetime']>=datetime.strptime(start,'%Y-%m-%d'))]
            trades_sheet_show = trades_sheet_show[['datetime','symbol','direction','offset','price']]
            trades_sheet_show.rename(columns={'datetime':'时间','symbol':'合约','direction':'买卖','offset':'开平','price':'参考价格'},inplace=True)
            trades_sheet_show['时间'] = trades_sheet_show['时间'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M'))
            trades_sheet_show['买卖'] = trades_sheet_show['买卖'].apply(lambda x: x.value)
            trades_sheet_show['开平'] = trades_sheet_show['开平'].apply(lambda x: x.value)
            direction_dict = {'空':'卖出','多':'买入'}
            offset_dict = {'开':'开仓','平':'平仓','平今':'平仓'}
            trades_sheet_show['买卖']= trades_sheet_show['买卖'].map(direction_dict)
            trades_sheet_show['开平']=trades_sheet_show['开平'].map(offset_dict)
            pd.options.display.float_format = '{:.2f}'.format
        else:
            trades_sheet_show = pd.DataFrame(columns=['时间','合约','买卖','开平','参考价格'])
        return trades_sheet_show

    def show_stastics_sheet(self):
        strategy_statistics_dict =  self.calculate_statistics()
        stastic_heads1 = {'开始时间', '截至时间', '总交易天数', '盈利天数','总盈亏','总收益率'}
        stastic_heads2 = {'初始权益', '当前权益', '日均收益率', '收益标准差','夏普比率','收益回撤比'}
        stastic_heads3 = {'最大回撤', '百分比最大回撤', '最大回撤持续时间', '总交易成本','总成交额','总成交笔数','近一周成交笔数','近一周收益率'}
        stastic_dict1 = {key: value for key, value in strategy_statistics_dict.items() if key in stastic_heads1}
        stastic_dict2 = {key: value for key, value in strategy_statistics_dict.items() if key in stastic_heads2}
        stastic_dict3 = {key: value for key, value in strategy_statistics_dict.items() if key in stastic_heads3}
        stastic_sheet1 = pd.DataFrame.from_dict(stastic_dict1,orient='index')
        stastic_sheet1['title'] = stastic_sheet1.index
        stastic_sheet1 =  stastic_sheet1[['title',0]].reset_index(drop=True)
        stastic_sheet2 = pd.DataFrame.from_dict(stastic_dict2,orient='index')
        stastic_sheet2['title'] = stastic_sheet2.index
        stastic_sheet2 =  stastic_sheet2[['title',0]].reset_index(drop=True)
        stastic_sheet3 = pd.DataFrame.from_dict(stastic_dict3,orient='index')
        stastic_sheet3['title'] = stastic_sheet3.index
        stastic_sheet3 =  stastic_sheet3[['title',0]].reset_index(drop=True)
        stastic_sheet = pd.concat([stastic_sheet1,stastic_sheet2,stastic_sheet3],axis=1)
        stastic_sheet.columns = ['A','B','C','D','E','F']
        pd.options.display.float_format = '{:.2f}'.format
        self.strategy_statistics = stastic_sheet
        return stastic_sheet

    def show_results(self):
        # self.cal_strategy_daily()
        self.cal_symbol_same_strategy_daily()
        print(self.show_stastics_sheet())
        print('---------------------------')
        print(self.cal_strategy_holdings())
        self.show_chart()

    def cal_symbol_holding(self,trades_sheet:DataFrame)->dict:
        trades_sheet_temp = trades_sheet.copy()
        #datetime to str
        trades_sheet_temp['datetime'] = trades_sheet_temp['datetime'].apply(lambda x: x.replace(tzinfo=None))
    #     trades_sheet_temp['datetime'] = trades_sheet_temp['datetime'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M'))
        #get symbol and exchange
        trades_sheet_dict = trades_sheet_temp.iloc[-1,:].to_dict()
        symbol = trades_sheet_dict['symbol']
        exchange = trades_sheet_dict['exchange'].value
        #holding_dict items
        holding_names = ['symbol','exchange','holding_direction','holding_num','trading_time','trading_price']
        
        #init long and short dict
        symbol_holding_long = {}
        symbol_holding_short = {}
        long_init = [symbol,exchange,'多单',0,'',0]
        long_dict=dict(zip(holding_names,long_init))

        short_init = [symbol,exchange,'空单',0,'',0]
        short_dict=dict(zip(holding_names,short_init))

        for key, value in long_dict.items():
            symbol_holding_long[key]=value

        for key, value in short_dict.items():
            symbol_holding_short[key]=(value)
        
        #symbol holding, 1 mean long   -1 mean short
        symbol_holding= 0
        
        #calculate long short dict
        for index, row in trades_sheet_temp.iterrows():
            # symbol_holding_old = symbol_holding
            if row['direction'].value == '多' and row['offset'].value == '开':
                symbol_holding += row['volume']
                symbol_holding_long['trading_price'] = (symbol_holding_long['trading_price']*symbol_holding_long['holding_num']\
                                                    +row['price']*row['volume'])/(symbol_holding_long['holding_num']+row['volume'])
                symbol_holding_long['holding_num'] += row['volume']
                symbol_holding_long['trading_time']=row['datetime']
            elif row['direction'].value == '空' and (row['offset'].value == '平' or row['offset'].value == '平今'):
                symbol_holding -= row['volume']
                if (symbol_holding_long['holding_num']-row['volume']) ==0:
                    symbol_holding_long['trading_price'] = 0
                    symbol_holding_long['trading_time'] = ''
                symbol_holding_long['holding_num'] =max(0, symbol_holding_long['holding_num']- row['volume'])
            elif row['direction'].value == '空' and row['offset'].value == '开':
                symbol_holding -= row['volume']
                symbol_holding_short['trading_price'] = (symbol_holding_short['trading_price']*symbol_holding_short['holding_num']\
                                                    +row['price']*row['volume'])/(symbol_holding_short['holding_num']+row['volume'])
                symbol_holding_short['holding_num'] += row['volume']
                symbol_holding_short['trading_time']=row['datetime']
            elif row['direction'].value == '多' and (row['offset'].value == '平' or row['offset'].value == '平今'):
                symbol_holding += row['volume']
                if (symbol_holding_short['holding_num']-row['volume']) ==0:
                    symbol_holding_short['trading_price'] = 0
                    symbol_holding_short['trading_time'] = ''
                symbol_holding_short['holding_num'] =max(0, symbol_holding_short['holding_num']- row['volume'])
        symbol_holding = {}
        symbol_holding['long'] = symbol_holding_long
        symbol_holding['short'] = symbol_holding_short
        
        return symbol_holding

    def cal_strategy_holdings(self, endday:str=None):
    #get trading sheet
        trades_sheet = DataFrame([trade.__dict__ for trade in self.strategy_trades])
        if not trades_sheet.empty:
            trades_sheet['datetime'] = trades_sheet['datetime'].apply(lambda x: x.replace(tzinfo=None))
            
            if endday:
                trades_sheet = trades_sheet[trades_sheet['datetime']<=datetime.fromisoformat(endday)].copy()
            
            strategylist = list(set(trades_sheet['strategy_num'].tolist()))
            strategy_holdings ={}
            for strategy_num in strategylist:
                strategy_sheet = trades_sheet[trades_sheet['strategy_num']==strategy_num].copy()
                symbollist = list(set(strategy_sheet['symbol'].tolist()))
                symbols_holding ={}
                for symbol in symbollist:
                    symbol_sheet = strategy_sheet[strategy_sheet['symbol']==symbol].copy()
                    symbol_holding = self.cal_symbol_holding(symbol_sheet)
                    symbols_holding[symbol] = symbol_holding
                strategy_holdings[strategy_num] = symbols_holding

            strategy_holding_list = []
            for k, v in strategy_holdings.items():
            #     strategy_num = k
                for ik, iv in v.items():
                    for iik, iiv in iv.items():
                        row = pd.DataFrame([iiv])
                        row['strategy_num']=k
                        strategy_holding_list.append(row.T.to_dict(orient='dict')[0])
            strategy_holding_sheet = pd.DataFrame(strategy_holding_list).dropna().reset_index(drop=True)
            strategy_holding_sheet = strategy_holding_sheet[['strategy_num','symbol','holding_direction','holding_num','trading_time','trading_price']].copy()
            strategy_holding_sheet.rename(columns={'strategy_num':'策略编号','symbol':'合约','holding_direction':'持仓方向','holding_num':'持仓数量','trading_time':'入场时间','trading_price':'持仓成本'},inplace=True)
            pd.options.display.float_format = '{:.2f}'.format
            self.strategy_holdings = strategy_holding_sheet
        else:
            strategy_holding_sheet = pd.DataFrame(columns=['策略编号','合约','持仓方向','持仓数量','入场时间','持仓成本'])
        return strategy_holding_sheet
        
class NewStrategyDaily(StrategyDaily):
    """根据数据库中的交易数据，得出策略报告所需的信息
    实例：strategy_daily = StrategyDaily(capital = 500000, strategy_name = 'TrendStar', start='2017-03-01', end='2021-04-01')
        指定初始资金capital，strategy_name,策略名，起止日期
        cal_strategy_daily()
    """

    def __init__(self, capital, sheet_num,strategy_group, risk_level, start, end=None):
        super().__init__(capital, sheet_num,strategy_group, risk_level, start, end)
    
    def cal_symbol_daily(self,
                    vt_symbol:str, 
                    strategy_num:str,
                    start:str,
                    end:str,
                    proportion:float,
                    end_close:bool
        ):

        symbol, exchange = extract_vt_symbol(vt_symbol)
        interval = Interval.DAILY
        startday = datetime.fromisoformat(start)
        endday = datetime.fromisoformat(end)
        self.contractRule = get_contract_rule(vt_symbol)
        contractRule = get_contract_rule(vt_symbol)
        risk_level = self.risk_level
        download_bar_data(symbol = symbol,exchange = exchange,interval = Interval.DAILY,start = self.start)
        daily_data = database_manager.load_bar_data(symbol,exchange,interval, startday, endday)
        symbol_trades = database_manager.load_trade_data(strategy_num = strategy_num,start = str(startday)[0:10],end = str(endday)[0:10] + " 16:00:00")
        #---开始增加配对交易
        #判断是否有开仓
        if len(symbol_trades) > 0:
            if symbol_trades[0].offset.value != '开':
                #补充开仓交易
                temp_trades1 = copy.deepcopy(symbol_trades[0])
                #id减1
                temp_trades1.id = temp_trades1.id -1
                #将交易时间变成统计时段首日9点
                temp_trades1.datetime = daily_data[0].datetime.replace(hour=9)
                #direction变换方向
                temp_trades1.direction = (Direction.SHORT if temp_trades1.direction.value == '多' else Direction.LONG)
                #变开仓
                temp_trades1.offset = Offset.OPEN
                #变price为开仓
                temp_trades1.price = daily_data[0].open_price
                symbol_trades = [temp_trades1] + symbol_trades

            if (symbol_trades[-1].offset.value == '开') and (end_close):
                #补充平仓交易
                temp_trades2 = copy.deepcopy(symbol_trades[-1])
                #id加1
                temp_trades2.id = temp_trades2.id +1
                #将交易时间变成统计时段末日15点
                temp_trades2.datetime = daily_data[-1].datetime.replace(hour=15)
                #direction变换方向
                temp_trades2.direction = (Direction.SHORT if temp_trades2.direction.value == '多' else Direction.LONG)
                #变开仓
                temp_trades2.offset = Offset.CLOSE
                #变price为开仓
                temp_trades2.price = daily_data[-1].close_price
                symbol_trades = symbol_trades + [temp_trades2]

        self.symbol_trades = symbol_trades

        daily_results = {}
        daily_close = []
        for bar in daily_data:
            daily_results[bar.datetime.date()] = DailyResult(bar.datetime.date(), bar.close_price)#字典，每个日期对应一个dailyresult；
            daily_close.append(bar.close_price)

        close_average = np.mean(daily_close)
        symbol_pos = max(1,cal_strategy_symbol_pos(self.capital, proportion,risk_level,close_average,contractRule['ContractSize']))
        print(symbol_pos)
        for trade in symbol_trades:
            trade.volume = symbol_pos
            self.strategy_trades.append(trade)
            d = generate_trading_day(trade.datetime).date()
            daily_result = daily_results[d]#daily_results的键值为bardata的日期，这里索引d为tradedata的日期，在当天交易日恰好有交易且在交易日未结束时运行，如果endday取当天或者未来日期，那么bardata比tradedata少一天，报错datetime(d)；
            daily_result.add_trade(trade)
            
        pre_close = 0
        start_pos = 0
        for daily_result in daily_results.values():
            daily_result.calculate_pnl(pre_close,start_pos,contractRule['ContractSize'],contractRule['feeOpenJ'],contractRule['PriceMinMove']+contractRule['feeOpenS']/contractRule['ContractSize'],False)
            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos
            
        results = defaultdict(list)
        for daily_result in daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)
                
        daily_df = DataFrame.from_dict(results).set_index("date")
        return daily_df

    def cal_symbol_same_strategy_daily(self):
        symbol_same_strategy_daily = DataFrame()
        # symbol_same_strategy_trades = database_manager.load_trade_data(strategy_num = strategy_num,start = self.start,end = self.end)
        # for index, row in self.strategy_sheet.iterrows():
        for index, row in self.group_sheet.iterrows():
            # vt_symbol = row['vt_symbol']
            s=row['startd']
            e=row['endd']
            if 'endd_close_trade' in self.group_sheet.columns:
                end_close = row['endd_close_trade']
            else:
                if datetime.fromisoformat(str(e)[0:10])> datetime.today():
                    end_close = False
                else:
                    end_close = True

            
            vt_symbol = get_vt_symbol(row['symbol'])#完整合约代码，如y2201.DCE;
            strategy_num = row['strategy_num']#可定位代码，如1001062201;
            p=row['proportion']
            print(vt_symbol)
            symbol_daily = self.cal_symbol_daily(vt_symbol,strategy_num,str(s)[0:10],str(e)[0:10],p,end_close)
            symbol_same_strategy_daily=concat([symbol_same_strategy_daily,symbol_daily])
        self.symbol_same_strategy_daily = symbol_same_strategy_daily.groupby(symbol_same_strategy_daily.index).sum()
        return symbol_same_strategy_daily


    """通过bar，计算拐点队列，并计算队列指标"""
    def __init__(self,size: int = 50, var: float = 0.68) -> None:
        self.var = var
        self.size = size
        self.count:int = 0
        self.inited: bool = False
        self.trend_array = ArrayManager(size)
        self.trend_up_array = ArrayManager(20)
        self.trend_down_array = ArrayManager(20)
        self.trend_now = 1
        self.temp_max_verse = 0.0
        self.temp_init_bar = None
        self.temp_trend_bar = None
        
        
    def update_bar(self, bar: BarData) -> None:
        self.trend_point(bar)
    
    def update_trend_bar(self, bar:BarData, trend_direction:int=0) -> None:
        if not trend_direction:
            self.trend_array.update_bar(bar)
            if self.trend_array.inited:
                self.inited = True
        elif trend_direction == -1:
            self.trend_down_array.update_bar(bar)
        else:
            self.trend_up_array.update_bar(bar)
    
    
    def trend_point(self,bar: BarData):
        if not self.temp_init_bar:
            self.temp_init_bar = bar
            self.temp_trend_bar = bar
            
        #新bar比较，如果形成新的拐点，update_trend_point_array
        #累加每日涨跌幅
        temp_trend = ((bar.close_price - self.temp_init_bar.close_price) / self.temp_init_bar.close_price) * 100
        temp_max_trend = ((self.temp_trend_bar.close_price - self.temp_init_bar.close_price) / self.temp_init_bar.close_price) * 100
        if self.trend_now == 1:
            if temp_trend > temp_max_trend:
                self.temp_trend_bar = bar
            self.temp_max_verse = (min(self.temp_max_verse,(bar.close_price - self.temp_trend_bar.close_price) / self.temp_trend_bar.close_price*100))
            if self.temp_max_verse < - self.var:
                self.update_trend_bar(self.temp_trend_bar)
                self.update_trend_bar(self.temp_trend_bar,1)
                self.trend_now = -1
                self.temp_init_bar = self.temp_trend_bar
                self.temp_trend_bar = bar
                self.temp_max_verse = 0.0

        else:
            if temp_trend < temp_max_trend:
                self.temp_trend_bar = bar
            self.temp_max_verse = (max(self.temp_max_verse, (bar.close_price - self.temp_trend_bar.close_price) / self.temp_trend_bar.close_price* 100)) 
            if self.temp_max_verse > self.var:
                self.update_trend_bar(self.temp_trend_bar)
                self.update_trend_bar(self.temp_trend_bar,-1)
                self.trend_now = 1
                self.temp_init_bar = self.temp_trend_bar
                self.temp_trend_bar = bar
                self.temp_max_verse = 0