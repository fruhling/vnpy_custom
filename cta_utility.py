"""
General utility functions.
"""

from cmath import log
from datetime import datetime, timedelta
from time import sleep
from dateutil.relativedelta import relativedelta
from tzlocal import get_localzone
import re
import json
import interval
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import pandas as pd
from pandas import concat, read_csv
from math import floor, ceil
import numpy as np
import talib
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from seatable_api import Base, context
from vnpy.trader.object import TradeData, BarData
from vnpy_custom.myobject import MyTradeData, SignData
from vnpy.trader.constant import Exchange, Interval, Offset, Direction
from vnpy.trader.database import BaseDatabase, get_database
from vnpy.trader.utility import generate_vt_symbol, extract_vt_symbol, load_json, save_json, BarGenerator, ArrayManager, get_file_path

LOCAL_TZ = get_localzone()

#导入数据库
database_manager: BaseDatabase = get_database()
#导入邮箱密码等设置文件
vnpy_home_path = Path.home().joinpath(".vntrader")
custom_setting_filename = vnpy_home_path.joinpath("vnpy_custom_setting.json")
custom_setting = load_json(custom_setting_filename)
contracts_rule_json_filename= vnpy_home_path.joinpath("data/json/contracts_rule.json")
domain_contracts_rule = load_json(contracts_rule_json_filename)

trading_days_filename = vnpy_home_path.joinpath("data/json/trading_days.json")
trading_day_dict = load_json(trading_days_filename)

domain_contracts_dict_filename = vnpy_home_path.joinpath("data/json/domain_contracts.json")
domain_contracts_dict = load_json(domain_contracts_dict_filename)

def bartime_to_signtime(bartime:datetime, timecover_dict:dict,interval:int = 5)->datetime:
    """cover bartime to signtime"""
    if interval == 5:
        sign_datetime = bartime.replace(tzinfo=None,minute=int(bartime.minute - bartime.minute%interval), second=0,microsecond=0) + timedelta(minutes=5)
    else:
        try:
            signtime_temp = timecover_dict[bartime.replace(tzinfo=None).time()]
        except KeyError:
            signtime_temp = bartime.replace(tzinfo=None,minute=int(bartime.minute - bartime.minute%interval), second=0,microsecond=0)
            # print(signtime_temp)
            signtime_temp = timecover_dict[signtime_temp.time()]
        sign_datetime = datetime(bartime.year, bartime.month, bartime.day,signtime_temp.hour,signtime_temp.minute)
    return sign_datetime

#计算品种日内波幅
def cal_daily_var(open, high, low):
    """cal daily max up or down"""
    return max((high - open), (open-low))/open

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

def check_trade_time(now_localtime,time_period_str):
    """检查是否是交易时间，通过交易时段文本与当前时间比较"""
    result = []
    time_period_list=[x.split('-') for x in time_period_str.split(',')]
    time_period_len = len(time_period_list)
    for i in range(time_period_len):
        time_period = interval.Interval(time_period_list[i][0],time_period_list[i][1],lower_closed=True, upper_closed=False)
        result.append(time_period)
    
    for period in result:
        if now_localtime in period:
            return True
    
    return False

def contract_rule_rq(contract:str, domain_contracts_rule = domain_contracts_rule):
    """通过策略中symbol(CTP样式或rqdata样式)，从contracts_rule.json中找到相关合约规则"""
    symbolHead = get_symbol_head(contract)
    symbolNumber = get_contract_number(contract)
    symbol_key = symbolHead + symbolNumber
    return domain_contracts_rule.get(symbol_key,None)

def generate_my_tradedata(strategy_str:str ,trade: TradeData, display:bool ,calculate:bool):
    """生成用于入postgresql中的交易数据，增加了策略名，策略周期等信息"""
    trades = []
    strategy_str = str(strategy_str)

    mytrade = MyTradeData(
        strategy_class = strategy_str.split("_")[0],
        strategy_name = strategy_str.split("_")[1],
        strategy_num = strategy_str.split("_")[2],
        strategy_period = int(strategy_str.split("_")[3]),
        symbol = to_CTP_symbol(trade.symbol),
        exchange = trade.exchange,
        # orderid = trade.datetime.strftime('%Y%m%d%H%M%S')+str(trade.orderid),
        orderid = str(trade.orderid),
        tradeid = str(trade.tradeid),
        direction = trade.direction,
        offset = trade.offset,
        price = trade.price,
        volume = trade.volume,
        datetime = trade.datetime.replace(second=0,microsecond=0),

        display = display,
        calculate = calculate,
        gateway_name = trade.gateway_name
    )
    trades.append(mytrade)
    return trades

def get_timecover_dict(symbol):
    """获取合约交易时间转换映射字典，通过bartimetosigntimeindex与bartimetosigntime转换。
    5分钟以上周期，不同软件K线时间定义不同，sign信号入库需要对应时间"""
    symbolHead = get_symbol_head(symbol)
    index = symbolHead in ['IF','IH','IC']
    if index:
        timedictlist = read_csv(Path.home().joinpath(".vntrader").joinpath("bartimeToSigntimeIndex.csv"))
    else:
        timedictlist = read_csv(Path.home().joinpath(".vntrader").joinpath("bartimeToSigntime.csv"))
    timecover_dict = {}
    for index, row in timedictlist.iterrows():
        bartime = datetime.strptime(row['bartime'],'%H:%M').time()
        signtime = datetime.strptime(row['signtime'],'%H:%M').time()
    #     bartime = row['bartime']
    #     signtime = row['signtime']
        timecover_dict[bartime] = signtime
    return timecover_dict

def generate_trading_day(order_time: datetime) -> datetime:
    """返回交易日，用于处理夜盘跨日与跨周末"""
    if order_time.hour > 8 and order_time.hour < 16:
        # return order_time.strftime('%Y%m%d')
        return order_time.replace(hour=0,minute=0, second=0, microsecond=0)
    else:
        if order_time.hour <= 8 and order_time.weekday() <= 4:
            return order_time.replace(hour=0,minute=0, second=0, microsecond=0)
        elif order_time.hour <= 8 and order_time.weekday() > 4:
            return (order_time + timedelta(days=2)).replace(hour=0,minute=0, second=0, microsecond=0)
        elif order_time.hour >= 16 and order_time.weekday() < 4:
            return (order_time + timedelta(days=1)).replace(hour=0,minute=0, second=0, microsecond=0)
        elif order_time.hour >= 16 and order_time.weekday() >= 4:
            return (order_time + timedelta(days=3)).replace(hour=0,minute=0, second=0, microsecond=0)
        else:
            pass

def get_vt_symbol(symbol:str):
    """通过symbol，直接得到vt_symbol，交易所数据来自contractRule.csv"""
    contractRule = get_contract_rule(symbol)
    exchange = Exchange(contractRule['exchange'])
    return generate_vt_symbol(symbol,exchange)

def generate_rqdata_symbol(symbol:str):
    """symbol转换成rqdata格式的symbol"""
    contract_head = get_symbol_head(symbol)
    contract_number = get_contract_number(symbol)
    return contract_head+contract_number

def generate_trading_sign(trade_dict: dict) -> dict:
    """交易信号入sign_data为前转换使用"""
    sign = {}
    if trade_dict['direction'].value == '多':
        sign['color'] = 'red'
    else:
        sign['color'] = 'green'
    sign['price']=trade_dict['price']
    return sign

def generate_trading_sign2(trade_dict: dict,open_switch: bool) -> dict:
    """交易信号入sign_data为前转换使用"""
    sign = {}
    if trade_dict['direction'].value == '多':
        if open_switch:
            sign['color'] = 'red'
        else:
            sign['color'] = 'magenta'
    else:
        if open_switch:
            sign['color'] = 'green'
        else:
            sign['color'] = 'cyan'
    sign['price']=trade_dict['price']
    return sign

def generate_sign_data(strategy_str:str,timecover_dict:dict,trade: TradeData, bartime: datetime):
    """生成可入库的交易信号数据"""
    signs = []
    trade_dict = trade.__dict__
    interval = int(strategy_str.split("_")[3])
    signtime = bartime_to_signtime(bartime,timecover_dict,interval)
    sign = SignData(
        strategy_group = strategy_str.split("_")[0],
        strategy_id = strategy_str.split("_")[2],
        period = strategy_str.split("_")[3],
        instrument = to_CTP_symbol(trade_dict['symbol']),
        order_time = signtime,
        tradingday = generate_trading_day(trade_dict['datetime']).strftime('%Y%m%d'),
        sign = json.dumps(generate_trading_sign(trade.__dict__)),
        remark = '',
        insert_time=datetime.now(),
        gateway_name = trade.gateway_name
    )
    signs.append(sign)
    return signs

def generate_newsign_data(strategy_str:str,min_period_str:str,trade: TradeData):
    """生成可入库的交易信号数据，处理了交易时间"""
    signs = []
    trade_dict = trade.__dict__
    interval = int(strategy_str.split("_")[3])
    if interval == 5:
        signtime = trade.datetime.replace(tzinfo=None,minute=int(trade.datetime.minute - trade.datetime.minute%interval), second=0,microsecond=0) + timedelta(minutes=5)
    else:
        # adjust_tradetime = trade.datetime.replace(tzinfo=None,minute=int(trade.datetime.minute - trade.datetime.minute%interval), second=0,microsecond=0)
        adjust_tradetime = trade.datetime.replace(tzinfo=None, second=0,microsecond=0)
        signtime = get_30minsign_time(adjust_tradetime,min_period_str)
    sign = SignData(
        strategy_group = strategy_str.split("_")[0],
        strategy_id = strategy_str.split("_")[2],
        period = strategy_str.split("_")[3],
        instrument = to_CTP_symbol(trade_dict['symbol']),
        order_time = signtime,
        tradingday = generate_trading_day(trade_dict['datetime']).strftime('%Y%m%d'),
        sign = json.dumps(generate_trading_sign(trade.__dict__)),
        remark = '',
        insert_time=datetime.now(),
        gateway_name = trade.gateway_name
    )
    signs.append(sign)
    return signs

def generate_newsign_data2(strategy_str:str,min_period_str:str,trade: TradeData,open_switch: bool):
    """生成可入库的交易信号数据，处理了交易时间"""
    signs = []
    trade_dict = trade.__dict__
    interval = int(strategy_str.split("_")[3])
    if interval == 5:
        signtime = trade.datetime.replace(tzinfo=None,minute=int(trade.datetime.minute - trade.datetime.minute%interval), second=0,microsecond=0) + timedelta(minutes=5)
    else:
        # adjust_tradetime = trade.datetime.replace(tzinfo=None,minute=int(trade.datetime.minute - trade.datetime.minute%interval), second=0,microsecond=0)
        adjust_tradetime = trade.datetime.replace(tzinfo=None, second=0,microsecond=0)
        signtime = get_30minsign_time(adjust_tradetime,min_period_str)
    sign = SignData(
        strategy_group = strategy_str.split("_")[0],
        strategy_id = strategy_str.split("_")[2],
        period = strategy_str.split("_")[3],
        instrument = to_CTP_symbol(trade_dict['symbol']),
        order_time = signtime,
        tradingday = generate_trading_day(trade_dict['datetime']).strftime('%Y%m%d'),
        sign = json.dumps(generate_trading_sign2(trade.__dict__,open_switch)),
        remark = '',
        insert_time=datetime.now(),
        gateway_name = trade.gateway_name
    )
    signs.append(sign)
    return signs

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

def get_symbol_head(contract:str):
    """获取合约代码英文部分，大写"""
    if len(contract.split('.'))>1:
        symbolHead = ''.join(re.findall(r'[A-Za-z]', contract.split(".")[0].upper()))
    else:
        symbolHead = ''.join(re.findall(r'[A-Za-z]', contract.upper()))
    return symbolHead

def get_contract_rule(symbol:str):
    """通过contractRule.csv获取品种基础规则"""
    symbolHead = get_symbol_head(symbol)
    contractRuleDict = read_csv(Path.home().joinpath(".vntrader").joinpath("contractRule.csv"), index_col='symbolHead',encoding='gbk').to_dict(orient='index')
    return contractRuleDict[symbolHead]

def get_contract_number(symbol:str)-> str:
    """获取合约数字部分，郑州补足4位数字，用于生成rqdata规划的合约代码"""
    exchange =  get_contract_rule(symbol)['exchange']
    time_str = ''.join(re.findall(r'[0-9]', symbol))
    
    if time_str in ["88", "888", "99", "889"]:
        #原为return "9999"，改为return相应数字
        return time_str
    
    if exchange == 'CZCE' and len(time_str)<4:
        year = time_str[0]
        if year in ["9","8","7","6","5"]:
            year = "1" + year
        else:
            year = "2" + year

        month = time_str[1:]
        return f"{year}{month}"
    else:
        return f"{symbol[-4:]}"

def get_trading_nday_before(date:str, n:int, trading_day_dict=trading_day_dict)->str:
    """返回某日前第n个交易日日期，包含当天"""
    if not isinstance(date,str):
        date = datetime.strftime(date,'%Y-%m-%d')
    query_day = datetime.strptime(date,'%Y-%m-%d')
    trading_day_list = [key for (key, value) in trading_day_dict.items() if datetime.strptime(key,'%Y-%m-%d') < query_day if value]
    if len(trading_day_list)>=n:
        return trading_day_list[-n]
    else:
        print('日期超出可查询范围')
        return None

def get_last_trading_day_in_the_same_month(date:str,trading_day_dict=trading_day_dict)->str:
    """返回trading_days.json中某月的最后一个交易日"""
    if not isinstance(date,str):
        date = datetime.strftime(date,'%Y-%m-%d')
    #返回查询日期当月的最后自然日
    query_day = (datetime.strptime(date,'%Y-%m-%d') + relativedelta(months=+1)).replace(day=1) - timedelta(days=1)
    
    #获取到查询日期列表中，交易日的日期列表
    trading_day_list = [key for (key, value) in trading_day_dict.items() if datetime.strptime(key,'%Y-%m-%d') <= query_day if value]
    if trading_day_list:
        return trading_day_list[-1]
    else:
        return None

def get_last_trading_day_in_the_before_month(date:str,trading_day_dict=trading_day_dict)->str:
    """返回trading_days.json中某月的最后一个交易日"""
    if not isinstance(date,str):
        date = datetime.strftime(date,'%Y-%m-%d')
    #返回查询日期当月的最后自然日
    query_day = (datetime.strptime(date,'%Y-%m-%d')).replace(day=1) - timedelta(days=1)
    
    #获取到查询日期列表中，交易日的日期列表
    trading_day_list = [key for (key, value) in trading_day_dict.items() if datetime.strptime(key,'%Y-%m-%d') <= query_day if value]
    if trading_day_list:
        return trading_day_list[-1]
    else:
        return None

def get_nearest_trading_day(date:str,trading_day_dict=trading_day_dict)->str:
    """返回距离日期最近的交易日"""
    if not isinstance(date,str):
        date = datetime.strftime(date,'%Y-%m-%d')
    if trading_day_dict[date]:
        return date
    else:
        query_day = datetime.strptime(date,'%Y-%m-%d')
        trading_day_list = [key for (key, value) in trading_day_dict.items() if datetime.strptime(key,'%Y-%m-%d') <= query_day if value]
        if trading_day_list:
            return trading_day_list[-1]
        else:
            return None

def hurst(ts):
    """Hurst指数计算"""
    ts = list(ts)
    N = len(ts)
    if N < 20:
        raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

    max_k = int(np.floor(N/2))
    R_S_dict = []
    for k in range(10,max_k+1):
        R,S = 0,0
        # split ts into subsets
        subset_list = [ts[i:i+k] for i in range(0,N,k)]
        if np.mod(N,k)>0:
            subset_list.pop()
            #tail = subset_list.pop()
            #subset_list[-1].extend(tail)
        # calc mean of every subset
        mean_list=[np.mean(x) for x in subset_list]
        for i in range(len(subset_list)):
            cumsum_list = pd.Series(subset_list[i]-mean_list[i]).cumsum()
            R += max(cumsum_list)-min(cumsum_list)
            S += np.std(subset_list[i])
        R_S_dict.append({"R":R/len(subset_list),"S":S/len(subset_list),"n":k})
    
    log_R_S = []
    log_n = []
    # print(R_S_dict)
    for i in range(len(R_S_dict)):
        R_S = (R_S_dict[i]["R"]+np.spacing(1)) / (R_S_dict[i]["S"]+np.spacing(1))
        log_R_S.append(np.log(R_S))
        log_n.append(np.log(R_S_dict[i]["n"]))

    Hurst_exponent = np.polyfit(log_n,log_R_S,1)[0]
    return Hurst_exponent

def is_number(s):
    """判断是否为字符是否为数字"""
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def klines_in_a_day(tradingtime:str, interval:int)->int:
    """"根据品种交易时间，返回对应周期的K线数量"""
    kline_day = 0
    for period in tradingtime.split(','):
        start, end = period.split('-')
        timedelta = datetime.strptime(end, '%H:%M:%S') - datetime.strptime(start, '%H:%M:%S')
        kline_num = round(timedelta.seconds/(60 * int(interval)),0)
#         print(f"start:{start},end:{end},timedelta:{timedelta},klin_num:{kline_num},")
        kline_day +=kline_num
    return kline_day

def last_trade_day(symbol,contract_rule):
    """返回合约可交易的最后交易日，通过contractRule.csv中，到期月份减自然日天数计算得出"""
    symbolNum = ''.join(re.findall(r'[0-9]', symbol))
    if symbolNum in ['88','888','889','99']:
        return datetime(2099,1,1)
    symbol_year = '202'+symbolNum[-3:-2]
    symbol_month = symbolNum[-2:]
    temp_day = datetime(int(symbol_year),int(symbol_month),1) 
    lastday = temp_day + timedelta(days=contract_rule['LastTradingDay'])
    return lastday

def make_trade_info(strategy_name: str,trade: TradeData):
    """生成策略邮件提示信息，通过策略名与交易信号"""
    trade_dict = trade.__dict__
    interval = int(strategy_name.split("_")[3])
    strategy_name = strategy_name.split("_")[1]
    strategy_conversion_dict = {'XtrendStarStrategy':'趋势星','RBreakerStrategy':'5M星','TrendStarStrategy':'趋势星'}
    strategy_name_cn = strategy_conversion_dict[strategy_name]
    order_time = datetime.strftime(trade.datetime.replace(tzinfo=None),'%Y-%m-%d %H:%M')
    symbol = trade_dict['symbol']
    contractRule = get_contract_rule(symbol)
    symbol_cn = f"{contractRule['CNname']}{''.join(re.findall(r'[0-9]', symbol))}"
    
    direction= trade_dict['direction']
    offset= trade_dict['offset']
    price = round(trade_dict['price'],1)
    
    if offset == Offset.OPEN:
        s = (f'{order_time}，{strategy_name_cn}策略提示,{symbol_cn}在{interval}分钟线,出现{direction.value}头交易机会，关注{price}附近的机会')
    else:
        if direction == Direction.LONG:
            holding_direction = '空'
        else:
            holding_direction = '多'
        s = (f'{order_time}，{strategy_name_cn}策略提示,{symbol_cn}在{interval}分钟线的{holding_direction}头机会，可能在{price}价格附近暂停或中止了')
    return s

def make_mail_content(strategy_name: str,trade: TradeData):
    """制作提示邮件的邮件正文，包含风险提示的html格式"""
    
    trade_dict = trade.__dict__
    interval = int(strategy_name.split("_")[3])
    strategy_name = strategy_name.split("_")[1]
    strategy_conversion_dict = {'XtrendStarStrategy':'趋势星','RBreakerStrategy':'5M星','TrendStarStrategy':'趋势星'}
    strategy_name_cn = strategy_conversion_dict[strategy_name]
    order_time = datetime.strftime(trade.datetime.replace(tzinfo=None),'%Y-%m-%d %H:%M')
    symbol = trade_dict['symbol']
    contractRule = get_contract_rule(symbol)
    symbol_cn = f"{contractRule['CNname']}{''.join(re.findall(r'[0-9]', symbol))}"
    
    direction= trade_dict['direction']
    offset= trade_dict['offset']
    price = round(trade_dict['price'],1)
    
    if offset == Offset.OPEN:
        s = (f'{order_time}，{strategy_name_cn}策略提示,{symbol_cn}在{interval}分钟线,出现{direction.value}头交易机会，关注{price}附近的机会')
    else:
        if direction == Direction.LONG:
            holding_direction = '空'
        else:
            holding_direction = '多'
        s = (f'{order_time}，{strategy_name_cn}策略提示,{symbol_cn}在{interval}分钟线的{holding_direction}头机会，可能在{price}价格附近暂停或中止了')

    risk_information = f"""
        <html>
        <head></head>
        <body>
            <section class="_editor" data-tools="编辑器" data-id="86521" >
                <section style="box-sizing:border-box;margin:10px 0;padding-top: 10px; padding-bottom: 10px; padding-left: 10px;display: inline-block; width: 100%; border: 1px solid #1f497d; box-shadow: #cccccc 2px 2px 4px;" data-width="100%">
                    <section class="brush" data-brushtype="text" style="box-sizing:border-box;margin-top: -3px; margin-bottom: -3px;font-size:16px;display: inline-block; width: 88.0295%;" data-width="88.0295%" hm_fix="319:220">
                        {s}
                    </section>
                </section>
            </section>
            <section style="margin: 10px auto;text-align: center;">
                <section style="font-size: 24px;letter-spacing: 1.5px;color: #0b79ff;text-align: left;margin: 0 0 -25px 40px;transform: rotate(0deg);" class="">
                    <strong>风险提示</strong>
                </section>
            <section style="display: flex;justify-content: flex-start;align-items: center;margin: 0 0 -4px 25px;">
                <section style="box-sizing:border-box;width: 7px;height: 7px;background-color: #0b79ff;border-radius: 50%;"></section>
                <section style="box-sizing:border-box;width: 120px;height: 1px;background-color: #fff;" data-width="120px"></section>
            </section>
            <section style="border: 1px solid #5ca6ff;padding: 30px 25px;border-radius: 12px;" class="">
                <section data-autoskip="1" class="brush" style="text-align: justify;line-height:1.75em;letter-spacing: 1.5px;font-size:14px;color:#333333;background: transparent;">
                    <section class="box-edit _editor" style="display: flex;justify-content: flex-start;align-items: flex-start;margin-top: 15px;">
                        <section class="assistant" style="box-sizing:border-box;width: 7px;height: 7px;background-color: #0b79ff;margin: 7px 10px 0 0;flex-shrink: 0;"></section>
                        <section data-autoskip="1" class="brush" style="text-align: justify;line-height:1.75em;letter-spacing: 1.5px;font-size:14px;color:#333333;background: transparent;">
                            <p>
                                趋势星量化策略是光大期货客户服务产品线中的一员，仅对具有自主交易能力以及自主风险识别与控制能力的客户提供策略展示。
                            </p>
                        </section>
                    </section>
                    <section class="box-edit _editor" style="display: flex;justify-content: flex-start;align-items: flex-start;margin-top: 15px;">
                        <section class="assistant" style="box-sizing:border-box;width: 7px;height: 7px;background-color: #0b79ff;margin: 7px 10px 0 0;flex-shrink: 0;"></section>
                        <section data-autoskip="1" class="brush" style="text-align: justify;line-height:1.75em;letter-spacing: 1.5px;font-size:14px;color:#333333;background: transparent;">
                            <p>
                                趋势星指标信号提供的信息不作为本公司及本公司员工向您提供的任何投资依据，不论是否使用该功能，您的所有交易均应由您自主做出投资决策并独立承担投资风险。
                            </p>
                        </section>
                    </section>
                    <section class="box-edit _editor" style="display: flex;justify-content: flex-start;align-items: flex-start;margin-top: 15px;">
                        <section class="assistant" style="box-sizing:border-box;width: 7px;height: 7px;background-color: #0b79ff;margin: 7px 10px 0 0;flex-shrink: 0;"></section>
                        <section data-autoskip="1" class="brush" style="text-align: justify;line-height:1.75em;letter-spacing: 1.5px;font-size:14px;color:#333333;background: transparent;">
                            <p>
                                趋势星指标30分钟K线上的信号是用于追踪活跃品种的主力合约与次主力合约的趋势机会而计算的提示信号，但该信号无法涵盖持仓过夜时可能遭受价格跳空、涨跌停等极端风险。
                            </p>
                        </section>
                    </section>
                    <section class="box-edit _editor" style="display: flex;justify-content: flex-start;align-items: flex-start;margin-top: 15px;">
                        <section class="assistant" style="box-sizing:border-box;width: 7px;height: 7px;background-color: #0b79ff;margin: 7px 10px 0 0;flex-shrink: 0;"></section>
                        <section data-autoskip="1" class="brush" style="text-align: justify;line-height:1.75em;letter-spacing: 1.5px;font-size:14px;color:#333333;background: transparent;">
                            <p>
                                势星指标信号基于历史数据和系统默认的交易规则计算所得，本功能具有一定的局限性，本公司仅是提供在默认系统的交易框架下计算得出的客观数据供投资者参考，不代表本公司对该回测信息的完备性、科学性、合理性、准确性等的认同。
                            </p>
                        </section>
                    </section>
                </section>
            </section>
        </section>
        <section class="_135editor" data-role="paragraph">
            <p>
                <br/>
        &nbsp; &nbsp;
            </p>
            <section class="_135editor" data-tools="编辑器" data-id="105187">
                <section style="text-align: center;margin: 10px auto;">
                    <section style="background-color: #ebf4fe;padding: 10px 7px;">
                        <section style="background-color: #fff;border: 1px solid #c6e0fc;padding: 25px 7px 25px 12px;border-radius: 7px;">
                            <section style="display: flex;align-items: center;margin: 0 10px 30px 10px;">
                                <section class="assistant" style="box-sizing:border-box;flex: 1;width: 100%;height: 1px;background: linear-gradient(to right, #c6e0fc, #c6e0fc 3px, transparent 3px, transparent);background-size: 5px 100%;" data-width="100%"></section>
                                <section class="135brush" data-brushtype="text" style="font-size: 12px;letter-spacing: 1.5px;color: #c6e0fc;margin: 0 13px;">
                                    订阅与退订
                                </section>
                                <section class="assistant" style="box-sizing:border-box;flex: 1;width: 100%;height: 1px;background: linear-gradient(to right, #c6e0fc, #c6e0fc 3px, transparent 3px, transparent);background-size: 5px 100%;" data-width="100%"></section>
                            </section>
                            <section class="box-edit _135editor" style="display: flex;justify-content: space-between;align-items: flex-start;margin-top: 20px;">
                                <section style="box-sizing:border-box;width: 75%;" hm_fix="264:455" data-width="75%">
                                    <section class="135brush" data-brushtype="text" style="font-size: 14px;letter-spacing: 1.5px;color: #707475;text-align: left;margin-top: 4px;">
                                        修订或退订阅品种，请扫描或长按识别二维码
                                    </section>
                                    <section class="135brush" data-brushtype="text" style="font-size: 14px;letter-spacing: 1.5px;color: #707475;text-align: left;margin-top: 4px;">
                                        进入表单，重新选择提交即可！
                                    </section>
                                    <section class="assistant" style="box-sizing:border-box;margin-top: 4px;width: 100%;height: 1px;background: linear-gradient(to right, #c6e0fc, #c6e0fc 4px, transparent 4px, transparent);background-size: 7px 100%;" data-width="100%">
                                        <section class="135brush" data-brushtype="text" style="font-size: 14px; letter-spacing: 1.5px; color: #707475; text-align: left; margin-top: 4px;">
                                            咨询电话：010-68082722
                                        </section>
                                    </section>
                                </section>
                                <section style="box-sizing:border-box;width: 20%;" data-width="20%">
                                    <section style="box-sizing:border-box;width: 100%;" data-width="100%">
                                        <img style="box-sizing:border-box;width: 100%; display: block;" href="https://cloud.seatable.cn/dtable/forms/a19e64b6-4cc6-4575-82fc-d00eba2490f5/" src="https://ebfcnbj-1255758508.cos.ap-beijing.myqcloud.com/img/subsribe.png" data-ratio="1" data-w="85" data-width="100%"/><span style="caret-color: red;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</span>
                                    </section>
                                </section>
                            </section>
                        </section>
                    </section>
                </section>
            </section>
        </section>
        </body>
        </html>
        """

    return s,risk_information

def round_partial (value, resolution):
    """返回任意取整方式包含0.2,0.25等格式的价格数据"""
    return round (value / resolution) * resolution

def read_authorized_clients():
    """由seatable读出信号订阅的授权列表，通过手机号授权"""
    server_url = 'https://cloud.seatable.cn'
    api_token = custom_setting['seatable_api_token']
    base = Base(api_token, server_url)
    base.auth()
    rows = base.list_rows('Authorized', view_name='默认视图')
    AuthorizedClients = pd.Series(rows[0]).to_frame().T
    for i in rows[1:]:
        AuthorizedClients=AuthorizedClients.append(i,ignore_index=True)
    AuthorizedClientsPhone = list(set(AuthorizedClients['手机号'].tolist()))
    return AuthorizedClientsPhone

def read_subscribe_online():
    """读出订阅信号中的列表，手机号排重"""
    server_url = 'https://cloud.seatable.cn'
    api_token = custom_setting['seatable_api_token']
    base = Base(api_token, server_url)
    base.auth()
    rows = base.list_rows('SubScribe', view_name='默认视图')
    SubScribeOnline = pd.Series(rows[0]).to_frame().T
    for i in rows[1:]:
        SubScribeOnline=SubScribeOnline.append(i,ignore_index=True)
    SubScribeOnline.sort_values(by='_mtime',inplace=True,ascending=False)
    SubScribeOnline.drop_duplicates(subset='手机号', inplace=True)
    return SubScribeOnline

def return_mail_list(symbol, strategy)->list:
    """get subscribe info from seatable, return mail_list in strategy"""
    SubScribeOnline = read_subscribe_online()
    AuthorizedClientsPhone = read_authorized_clients()
    strategy_conversion_dict = {'TrendStarStrategy':'趋势星','RBreakerStrategy':'5M星'}
    contractRule = get_contract_rule(symbol)
    symbolHead = contractRule['CNname']+''.join(re.findall(r'[A-Za-z]', symbol.upper()))
    strategy_cnname = symbolHead + '-' + strategy_conversion_dict[strategy]
    SubScribe_Clients = SubScribeOnline[(SubScribeOnline['订阅产品'].apply(lambda x: strategy_cnname in x))&(SubScribeOnline['手机号'].isin(AuthorizedClientsPhone))]
    return list(set(SubScribe_Clients['邮箱'].tolist()))

def return_domain_contract(symbol,domain_contracts_dict=domain_contracts_dict)->str:
    """get subscribe info from seatable, return mail_list in strategy"""
    # SubScribeOnline = read_subscribe_online()
    # AuthorizedClientsPhone = read_authorized_clients()
    symbolHead = ''.join(re.findall(r'[A-Za-z]', symbol.upper()))

    return domain_contracts_dict.get(symbolHead,None)

def return_last_holding_day(contract, expired_day_str):
    """return contract lastopenday
    大连、郑州、上海品种，到期前月末前二个交易日，股指第三周周三，原油10号前后"""
    symbolHead = get_symbol_head(contract)
    exchange = get_contract_rule(contract)['exchange']
    last_trading_day_in_the_before_month = datetime.strptime(get_last_trading_day_in_the_before_month(expired_day_str),'%Y-%m-%d')
    if last_trading_day_in_the_before_month:
        last_trading_day_before_a_month_and_2_days = datetime.strftime(last_trading_day_in_the_before_month - timedelta(days=2),'%Y-%m-%d')       
        if (exchange in ['DCE','CZCE','SHFE','INE']) and (symbolHead not in ['SC','LU','FU']):
            #品种为商品，且不为原油、燃油、低硫燃油，最后交易日为交割月前月最后交易日
            return get_nearest_trading_day(last_trading_day_before_a_month_and_2_days)

        elif symbolHead in ['SC','LU']:
            #商品期货，原油、燃油、低硫燃油，暂时设定为与其他商品一致
            return get_trading_nday_before(expired_day_str,10)

        elif symbolHead == 'FU':
            #商品期货，原油、燃油、低硫燃油，暂时设定为与其他商品一致
            return get_trading_nday_before(expired_day_str,6)
        elif symbolHead in ['T','TF','TS']:
            return get_nearest_trading_day(last_trading_day_before_a_month_and_2_days)
        elif symbolHead in ['IF','IH','IC']:
            return expired_day_str
        else:
            print("未找到相关品种")
            return None
    else:
        print("最后交易日数据未包含相关信息")
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

def setting_dict(row):
    """将从strategy_sheet中，读出的setting信息，进行格式化，用于由strategy_sheet，生成strategy_setting.json"""
    setting = {}
    setting['class_name'] = row['strategy_name']
    params=row['param'].split(',')
    for param_txt in params:
        param = param_txt.split(':')
        if param[1] == 'TRUE':
            setting[param[0].strip()] = True
        elif param[1] == 'FALSE':
            setting[param[0].strip()] = False
        elif '/' in param[1]:
            setting[param[0].strip()] = param[1]
        else:
            setting[param[0].strip()] = float(param[1])
    
    return setting

def send_trade_message(trade_str):
    """发送企业微信群机器人提示信息"""
    url = custom_setting['wechat_url']
    headers = {"Content-Type": "text/plain"}
    # s = "开始发言测试! "
    data = {
        "msgtype": "text",
        "text": {
            "content": trade_str,
        }
    }
    r = requests.post(url, headers=headers, json=data)
    send_status = json.loads(r.text)
    return send_status['errcode']

def send_emails(subject,content,receivers):
    """发送提示邮件"""
    mail_host = custom_setting['mail_host']
    mail_user = custom_setting['mail_user']
    mail_pass = custom_setting['mail_password']
    sender = mail_user  
    message = MIMEMultipart()
    #邮件主题       
    message['Subject'] = subject 
    #发送方信息
    message['From'] = sender 
    #接受方信息     
    message['To'] = receivers[0]
    textApart = MIMEText(content,'html')
    message.attach(textApart)

    #登录并发送邮件
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host,"465") 
        #连接到服务器
        # smtpObj.connect(mail_host,25)
        #登录到服务器
        smtpObj.login(mail_user,mail_pass) 
        #发送
        smtpObj.sendmail(
            sender,receivers,message.as_string()) 
        #退出
        smtpObj.quit() 
        print('success')
    except smtplib.SMTPException as e:
        print('error',e) #打印错误

def tradingdays_from_now(symbol:str, tradingdays:int):
    """返回N个交易日前的日期，用于load_bar"""
    #转化symbol至symbol99, tradingdays <100
    endday=datetime.now(get_localzone())
    symbolHead = get_symbol_head(symbol)
    contractRule = get_contract_rule(symbol)
    contract99 = symbolHead+'99'
    exchange = Exchange(contractRule['exchange'])
    download_startd = endday - timedelta(days=2*tradingdays)
    if download_startd:
        daily_sheet = get_daily_sheet(symbol = contract99,exchange = exchange,start = download_startd)
        
    if len(daily_sheet) > tradingdays:
        return daily_sheet.index[-tradingdays]
    else:
        return

def to_CTP_symbol(symbol: str) -> str:
    """
    CZCE product of RQData has symbol like "TA1905" while
    vt symbol is "TA905.CZCE" so need to add "1" in symbol.
    """
    contractRule = get_contract_rule(symbol)
    exchange = contractRule['exchange']
    product = ''.join(re.findall(r'[A-Za-z]', symbol))
    time_str = ''.join(re.findall(r'[0-9]', symbol))
    
    if time_str in ["88", "888", "99", "889"]:
        return symbol.upper()
    elif (len(time_str) < 3) or (len(time_str)>4):
        print('symbol错误')
        return None
    
    if exchange == 'CFFEX':
        return symbol.upper()
    elif exchange == 'CZCE':
        if len(time_str)==4:
            year = time_str[1]
            month = time_str[2:]
        else:
            year = time_str[:2]
            month = time_str[2:]

        return f"{product}{year}{month}".upper()
    else:
        return symbol.lower()

#汇总工具
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()
        
class NewBarGenerator(BarGenerator):

    def __init__(self, 
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable = None,
        interval: Interval = Interval.MINUTE):
        """"""
        super().__init__(on_bar,window,on_window_bar,interval)
        self.day_bar: BarData = None
        self.week_bar: BarData = None
        self.last_bar: BarData = None

    def update_bar(self, bar: BarData) -> None:
        """
        Update 1 minute bar into generator
        """
        if self.interval == Interval.MINUTE:
            self.update_bar_minute_window(bar)
        elif self.interval == Interval.HOUR:
            self.update_bar_hour_window(bar)
        elif self.interval == Interval.DAILY:
            self.update_bar_day_window(bar) #处理日线
        else:
            self.update_bar_week_window(bar) #处理周线

    def update_bar_day_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.day_bar:
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.day_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
            return

        finished_bar = None
        temp_datetime = (bar.datetime + timedelta(hours=4))

        # If datetime is 14:59, update day bar into window bar and push
        if bar.datetime.minute == 59 and bar.datetime.hour == 14:
            self.day_bar.high_price = max(
                self.day_bar.high_price,
                bar.high_price
            )
            self.day_bar.low_price = min(
                self.day_bar.low_price,
                bar.low_price
            )

            self.day_bar.close_price = bar.close_price
            self.day_bar.volume += bar.volume
            self.day_bar.turnover += bar.turnover
            self.day_bar.open_interest = bar.open_interest

            finished_bar = self.day_bar
            self.day_bar = None #day_bar已经存入finish_bar，clear day_bar

        # night trading start
        # 
        elif temp_datetime.day != (self.day_bar.datetime+  timedelta(hours=5)).day and  (self.day_bar.datetime+  timedelta(hours=5)).weekday() != 5:

            finished_bar = self.week_bar
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.day_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
        # Otherwise only update minute bar
        else:
            self.day_bar.high_price = max(
                self.day_bar.high_price,
                bar.high_price
            )
            self.day_bar.low_price = min(
                self.day_bar.low_price,
                bar.low_price
            )

            self.day_bar.close_price = bar.close_price
            self.day_bar.volume += int(bar.volume)
            self.day_bar.turnover += bar.turnover
            self.day_bar.open_interest = bar.open_interest

        # Push finished window bar
        if finished_bar:
            self.on_hour_bar(finished_bar)

        # Cache last bar object
        self.last_bar = bar

    def update_bar_week_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object

        if not self.week_bar:
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.week_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
            return

        finished_bar = None

        # If time is Firday 14:59, update day bar into window bar and push
        if bar.datetime.minute == 59 and bar.datetime.hour == 14 and bar.datetime.weekday() == 4:
            self.week_bar.high_price = max(
                self.week_bar.high_price,
                bar.high_price
            )
            self.week_bar.low_price = min(
                self.week_bar.low_price,
                bar.low_price
            )

            self.week_bar.close_price = bar.close_price
            self.week_bar.volume += int(bar.volume)
            self.week_bar.open_interest = bar.open_interest
            self.week_bar.turnover += bar.turnover

            finished_bar = self.week_bar
            self.week_bar = None

        # isocalendar() 返回多少年的第几周的第几天 格式如（2018， 27， 5）
        # 周数不相同肯定是新的一周，可以推送出一根完整周k线了

        elif  (bar.datetime + timedelta(days=2,hours=5)).isocalendar()[1] != (self.week_bar.datetime + timedelta(days=2,hours=5)).isocalendar()[1]:
            # print(bar.datetime.isocalendar())
            finished_bar = self.week_bar

            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.week_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
        # Otherwise only update minute bar
        else:
            self.week_bar.high_price = max(
                self.week_bar.high_price,
                bar.high_price
            )
            self.week_bar.low_price = min(
                self.week_bar.low_price,
                bar.low_price
            )
            self.week_bar.close_price = bar.close_price
            self.week_bar.volume += int(bar.volume)
            self.week_bar.turnover += bar.turnover
            self.week_bar.open_interest = bar.open_interest

        # Push finished window bar
        if finished_bar:
            self.on_hour_bar(finished_bar) #on_window_bar只关心bar的数量，不关心bar的类型，所以可以直接调用

        # Cache last bar object
        self.last_bar = bar

class NewArrayManager(ArrayManager):
    """
    For:
    1. time series container of bar data
    2. calculating technical indicator value
    """
    def __init__(self, size: int = 100):
        super().__init__(size)
        self.datetime_array: np.ndarray = np.arange(str(datetime(2020,1,1)),\
            str(datetime(2020,1,1)+timedelta(minutes=size)),dtype='datetime64[m]')

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            self.inited = True

        self.datetime_array[:-1] = self.datetime_array[1:]
        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        self.close_array[:-1] = self.close_array[1:]
        self.volume_array[:-1] = self.volume_array[1:]
        self.turnover_array[:-1] = self.turnover_array[1:]
        self.open_interest_array[:-1] = self.open_interest_array[1:]

        self.datetime_array[-1] = np.datetime64(bar.datetime.replace(tzinfo=None))
        self.open_array[-1] = bar.open_price
        self.high_array[-1] = bar.high_price
        self.low_array[-1] = bar.low_price
        self.close_array[-1] = bar.close_price
        self.volume_array[-1] = bar.volume
        self.turnover_array[-1] = bar.turnover
        self.open_interest_array[-1] = bar.open_interest

    @property
    def datetime(self) -> np.ndarray:
        """
        Get open price time series.
        """
        return self.datetime_array

    @property
    def up_down(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        result = self.close_array[1:] - self.close_array[:-1]
        return result
       
    def diff(self, prices: np.ndarray, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        n_window price change.
        """
        n = int(n)
        result = prices[n:] - prices[:-n]
        if array:
            return result
        return result[-1]

    def up_down_percentile(self,n: int) -> float:
        """
        n_window price percentile. n must between 0~100
        """
        result = np.percentile(abs(self.up_down),n)
        return result

    def roc_percentile(self,n: int) -> float:
        """
        n_window price percentile. n must between 0~100
        """
        n = int(n)
        result = np.percentile(abs(self.roc(1,array = True)[1:]),n)
        return result

    def kama(self, prices: np.ndarray,n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        KAMA.
        """
        n = int(n)
        result = talib.KAMA(prices, n)
        if array:
            return result
        return result[-1]

    def mts(self,n: int) -> float:
        """
        Market Strength.
        """
        n = int(n)
        result = min(1,abs(self.close[-1]-self.open[-n])/max((max(self.high[-n:]) - min(self.low[-n:])),0.1))
        # if self.close_array[-1] > self.close_array[-n-1]:
        #     result = (self.close_array[-1]-self.close_array[-n-1])/max((max(self.high_array[-n:]) - min(self.low_array[-n:])),0.1)
        # else:
        #     result = (self.close_array[-n-1]-self.close_array[-1])/max((max(self.high_array[-n:]) - min(self.low_array[-n:])),0.1)
        # if array:
        #     return result
        return result

    def efcount(self,n: int,roc) -> float:
        """
        Market Strength.
        """
        n = int(n)
        result = sum(np.array(abs(self.roc(1,array = True)[-n:]))>roc)
        # if array:
        #     return result
        return result

    def dailyvar(self, n: int)-> float:
        """cal daily max up or down"""
        daily_var = np.maximum((self.high_array - self.open_array), (self.open_array - self.low_array))/ self.open_array
        result = np.percentile(daily_var,n)
        return result

    def trend_now(self, trend_var: float=4)-> Tuple[bool, float, float, float,float]:
        """根据var，define trend"""
        #data 没有nan数据,为ret*100
        #收盘价array
        close_array = self.close_array.tolist()
        high_array = self.high_array.tolist()
        low_array = self.low_array.tolist()
        #交易时间array
        datetime_array = self.datetime_array.tolist()
        #当前趋势涨跌幅
        # temp_trend = datas[0]
        #趋势反转幅度
        trend_now = (close_array[1]+0.01 - close_array[0]) / abs((close_array[1]+0.01 - close_array[0]))
        trend_max_verse = 0
        #趋势最大值
        trend_max = 0
        #趋势极值点收盘价
        trend_init_price = close_array[0]
        turn_price = close_array[0]
        trend_max_price = high_array[0] if trend_now > 0 else low_array[0]
        #趋势极值点点时间
        trend_max_datetime = datetime_array[0]
        trend_record = []
        temp_trend = 0
        change_now = False
        
        for high,low,close,trend_datetime \
            in zip(high_array[1:],low_array[1:],\
                close_array[1:], datetime_array[1:]):
            #累加每日涨跌幅
            if trend_now == 1:
                temp_trend = ((high - trend_init_price) / trend_init_price) * 100
                if temp_trend > trend_max:
                    trend_max = temp_trend
                    trend_max_price = high
                    trend_max_datetime = trend_datetime
                trend_max_verse = (min(trend_max_verse,(low - trend_max_price) / trend_max_price*100))
                if trend_max_verse < - trend_var:
                    turn_price = close
                    trend_record.append((trend_max_datetime,trend_datetime,turn_price,trend_max_price,trend_max))
                    trend_now = -1.0
                    trend_init_price = trend_max_price
                    trend_max_price = low
                    trend_max_datetime = trend_datetime
                    trend_max_verse = 0
                    temp_trend = ((low - trend_init_price) / trend_init_price) * 100
                    
            else:
                temp_trend = ((low - trend_init_price) / trend_init_price) * 100
                if temp_trend < trend_max:
                    trend_max = temp_trend
                    trend_max_price = low
                    trend_max_datetime = trend_datetime
                trend_max_verse = (max(trend_max_verse, (high - trend_max_price) / trend_max_price* 100)) 
                if trend_max_verse > trend_var:
                    turn_price = close
                    trend_record.append((trend_max_datetime,trend_datetime,turn_price,trend_max_price,trend_max))
                    trend_now = 1.0
                    trend_init_price = trend_max_price
                    trend_max_price = high
                    trend_max_datetime = trend_datetime
                    trend_max_verse = 0
                    temp_trend = ((high - trend_init_price) / trend_init_price) * 100

        
        if not trend_record:
                #         print('拐点数量少于1')
            if abs(trend_max) > abs(trend_max_verse):
                return False,0,temp_trend,turn_price,trend_max_price
            else:
                return False,0,temp_trend,turn_price,trend_max_price
        else:
            #返回当前趋势方向与距拐点的涨跌幅
            return change_now, trend_now, temp_trend,turn_price,trend_max_price


    def trend_turn(self, upvar: float=4, downvar:float=4)-> Tuple[bool, float, float, float,float]:
        """根据var，define trend"""
        #data 没有nan数据,为ret*100
        datas = self.roc(1,array = True)[1:].tolist()
        close_array = self.close_array[1:].tolist()
        temp_trend = datas[0]
        temp_verse = 0
        temp_verse_pre = 0
        temp_verse_max_price = close_array[0]
        trend_max = datas[0]
        trend_max_price = close_array[0]
        trend_turn = 0
        turn_price = close_array[0]
        change_now = False
        
        for value, close in zip(datas[1:],close_array[1:]):
            temp_trend += value
            if ((trend_max > 0) and (temp_trend > trend_max)) or ((trend_max < 0) and (temp_trend < trend_max)):
                trend_max = temp_trend
                trend_max_price = close

            temp_verse_pre = temp_verse
            temp_verse = temp_trend - trend_max
            if abs(temp_verse) > abs(temp_verse_pre):
                temp_verse_max_price = close
    #             print(f"i:{i},ret:{round(rets[i],2)},temp_trend:{round(temp_trend,2)},trend_max:{round(trend_max,2)},temp_verse:{round(temp_verse,2)}")

            if temp_verse > upvar:
                trend_turn = 1.0
                turn_price = close
                temp_trend = temp_verse
                trend_max = temp_verse
                trend_max_price = close
                temp_verse = 0
                change_now = True
                temp_verse_max_price = close
            
            elif temp_verse < -downvar:
                trend_turn = -1.0
                turn_price = close
                temp_trend = temp_verse
                trend_max = temp_verse
                trend_max_price = close
                temp_verse = 0
                change_now = True
                temp_verse_max_price = close
        
        if not trend_turn:
                #         print('拐点数量少于1')
            if abs(trend_max) > abs(temp_verse):
                return False,0,temp_trend,turn_price,trend_max_price
            else:
                return False,0,temp_trend,turn_price,temp_verse_max_price
        else:
            #返回当前趋势方向与距拐点的涨跌幅
            return change_now, trend_turn, temp_trend,turn_price,trend_max_price
  
    def trend_feature(self,inflection_point_start:str,inflection_point_end:str='2029-12-31')-> Tuple[int,float,float, int,int,int]:
        """根据var，拐点时间，拐点价格。返回期间，周期数，涨跌幅，振幅，持仓差，阳线数，阴线数"""
        #data 返回趋势方向，拐点
        #确定序列定位
        date_range, = np.where((self.datetime >= np.datetime64(inflection_point_start,'m')) & \
            (self.datetime <= np.datetime64(inflection_point_end,'m')))
        #统计K线数量
        daily_count = len(date_range)
        date_range_close_roc = None
        date_range_up_down_roc = None
        up_candle_nums = None
        down_candle_nums = None

        if daily_count>1:
            #收益率
            date_range_close_roc=(self.close[date_range[-1]] / self.close[date_range[0]]) -1
            #震幅
            range_max_price = max(max(self.close[date_range]),max(self.high[date_range]))
            range_min_price = min(min(self.close[date_range]),min(self.low[date_range]))
            date_range_up_down_roc = range_max_price / range_min_price -1
            #持仓差
            range_holding_change = (self.open_interest[date_range[-1]] - self.open_interest[date_range[0]])
            range_holding_change_rate = (self.open_interest[date_range[-1]] - self.open_interest[date_range[0]]) / self.open_interest[date_range[0]]

            #成交额
            turnover = self.turnover_array[date_range].sum()
            #阳线数
            if self.open[date_range[-1]]:
                up_candle_nums = np.count_nonzero((self.close[date_range] \
                    - self.open[date_range]) > 0)
                down_candle_nums = np.count_nonzero((self.close[date_range] \
                    - self.open[date_range]) < 0)
            else:
                up_candle_nums = np.count_nonzero((self.close[date_range[1]:date_range[-1]] \
                    - self.close[date_range[0]:date_range[-2]]) > 0)
                down_candle_nums = np.count_nonzero((self.close[date_range[1]:date_range[-1]] \
                    - self.close[date_range[0]:date_range[-2]]) < 0)

        return daily_count,date_range_close_roc,date_range_up_down_roc,up_candle_nums,down_candle_nums,turnover,range_holding_change_rate

    def trend_record(self, upvar: float=4, downvar:float=4)-> list:
        """根据var，define trend"""
        #data 没有nan数据,为ret*100
        #收盘价array
        close_array = self.close_array.tolist()
        high_array = self.high_array.tolist()
        low_array = self.low_array.tolist()
        #交易时间array
        datetime_array = self.datetime_array.tolist()
        #当前趋势涨跌幅
        # temp_trend = datas[0]
        #趋势反转幅度
        trend_now = (close_array[1]+0.01 - close_array[0]) / abs((close_array[1]+0.01 - close_array[0]))
        trend_max_verse = 0
        #趋势最大值
        trend_max = 0
        #趋势极值点收盘价
        trend_init_price = close_array[0]
        trend_max_price = high_array[0] if trend_now > 0 else low_array[0]
        #趋势极值点点时间
        trend_max_datetime = datetime_array[0]
        trend_record = []
        temp_trend = 0
        
        for high,low,trend_datetime \
            in zip(high_array[1:],low_array[1:],\
                datetime_array[1:]):
            #累加每日涨跌幅
            if trend_now == 1:
                temp_trend = ((high - trend_init_price) / trend_init_price) * 100
                if temp_trend > trend_max:
                    trend_max = temp_trend
                    trend_max_price = high
                    trend_max_datetime = trend_datetime
                trend_max_verse = (min(trend_max_verse,(low - trend_max_price) / trend_max_price*100))
                if trend_max_verse < - downvar:
                    trend_record.append((trend_max_datetime,trend_datetime,trend_max_price,trend_max))
                    trend_now = -1
                    trend_init_price = trend_max_price
                    trend_max_price = low
                    trend_max_datetime = trend_datetime
                    trend_max_verse = 0
                    temp_trend = ((low - trend_init_price) / trend_init_price) * 100
                    
            else:
                temp_trend = ((low - trend_init_price) / trend_init_price) * 100
                if temp_trend < trend_max:
                    trend_max = temp_trend
                    trend_max_price = low
                    trend_max_datetime = trend_datetime
                trend_max_verse = (max(trend_max_verse, (high - trend_max_price) / trend_max_price* 100)) 
                if trend_max_verse > upvar:
                    trend_record.append((trend_max_datetime,trend_datetime,trend_max_price,trend_max))
                    trend_now = 1
                    trend_init_price = trend_max_price
                    trend_max_price = high
                    trend_max_datetime = trend_datetime
                    trend_max_verse = 0
                    temp_trend = ((high - trend_init_price) / trend_init_price) * 100
        
        return trend_record

class TrendPoint(object):
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