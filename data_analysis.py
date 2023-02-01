from datetime import datetime, timedelta

import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf


import talib

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.utility import ArrayManager
from vnpy.trader.object import BarData



warnings.filterwarnings("ignore")
database = get_database()

def random_test(close_price):
    """"""
    acorr_result = acorr_ljungbox(close_price, lags=1)
    p_value = acorr_result[1]
    if p_value < 0.05:
        output("第二步：随机性检验：非纯随机性")
    else:
        output("第二步：随机性检验：纯随机性")
    output(f"白噪声检验结果:{acorr_result}\n")


def stability_test(close_price):
    """"""
    statitstic = ADF(close_price)
    t_s = statitstic[1]
    t_c = statitstic[4]["10%"]

    if t_s > t_c:
        output("第三步：平稳性检验：存在单位根，时间序列不平稳")
    else:
        output("第三步：平稳性检验：不存在单位根，时间序列平稳")

    output(f"ADF检验结果：{statitstic}\n")


def autocorrelation_test(close_price):
    """"""
    output("第四步：画出自相关性图，观察自相关特性")

    plot_acf(close_price, lags=60)
    plt.show()

    plot_pacf(close_price, lags=60).show()
    plt.show()


def statitstic_info(df):
    """"""
    mean = round(df.mean(), 4)
    median = round(df.median(), 4)
    std = round(df.std(),4)
    output(f"样本平均数：{mean}, 中位数: {median}, 标准差:{std}")

    skew = round(df.skew(), 4)
    kurt = round(df.kurt(), 4)

    if skew == 0:
        skew_attribute = "对称分布"
    elif skew > 0:
        skew_attribute = "分布偏左"
    else:
        skew_attribute = "分布偏右"

    if kurt == 0:
        kurt_attribute = "正态分布"
    elif kurt > 0:
        kurt_attribute = "分布陡峭"
    else:
        kurt_attribute = "分布平缓"

    output(f"偏度为：{skew}，属于{skew_attribute}；峰度为：{kurt}，属于{kurt_attribute}\n")


def output(msg):
    """
    Output message of backtesting engine.
    """
    print(f"{datetime.now()}\t{msg}")


class DataAnalysis:

    def __init__(self):
        """"""
        self.symbol = ""
        self.exchange = None
        self.interval = None
        self.start = None
        self.end = None
        self.rate = 0.0

        self.window_volatility = 20
        self.window_index = 20

        self.orignal = pd.DataFrame()

        self.index_1to1 = ["STDDEV","SMA"]
        self.index_2to2 = []
        self.index_3to1 = []
        self.index_2to1 = []
        self.index_4to1 = []
        self.intervals = []

        self.results = {}

    def load_history(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
        rate: float = 0.0,
        index_1to1: list = None,
        index_2to2: list = None,
        index_3to1: list = None,
        index_2to1: list = None,
        index_4to1: list = None,
        window_index: int = 20,
        window_volatility: int = 20,

    ):
        """"""
        output("开始加载历史数据")

        self.window_volatility = window_volatility
        self.window_index = window_index
        self.rate = rate
        self.index_1to1 = index_1to1
        self.index_2to2 = index_2to2
        self.index_3to1 = index_3to1
        self.index_2to1 = index_2to1
        self.index_4to1 = index_4to1

        # Load history data from database
        bars = database.load_bar_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end,

        )

        output(f"历史数据加载完成，数据量：{len(bars)}")

        # Generate history data in DataFrame
        t = []
        o = []
        h = []
        l = []  # noqa
        c = []
        v = []

        for bar in bars:
            time = bar.datetime.replace(tzinfo=None)
            open_price = bar.open_price
            high_price = bar.high_price
            low_price = bar.low_price
            close_price = bar.close_price
            volume = bar.volume

            t.append(time)
            o.append(open_price)
            h.append(high_price)
            l.append(low_price)
            c.append(close_price)
            v.append(volume)

        self.orignal["open"] = o
        self.orignal["high"] = h
        self.orignal["low"] = l
        self.orignal["close"] = c
        self.orignal["volume"] = v
        self.orignal.index = t
    
    def plot_close(self, df: DataFrame = None):
        if df is None:
            df = self.orignal

        if df is None:
            output("数据为空，请输入数据")

        close_price = df["close"]
        close_diff = talib.ROC(df["close"]).fillna(0)

        output("第一步:画出行情图，检查数据断点")

        close_price.plot(figsize=(20, 8), title="close_price")
        plt.show()
        close_diff.plot(figsize=(20, 8), title="close_diff")
        plt.show()

    def base_analysis(self, df: DataFrame = None):
        """"""
        if df is None:
            df = self.orignal

        if df is None:
            output("数据为空，请输入数据")

        close_price = df["close"]
        close_diff = talib.ROC(df["close"]).fillna(0)

        output("第一步:画出行情图，检查数据断点")

        close_price.plot(figsize=(20, 8), title="close_price")
        plt.show()
        close_diff.plot(figsize=(20, 8), title="close_diff")
        plt.show()

        random_test(close_diff)
        stability_test(close_diff)
        autocorrelation_test(close_diff)

        self.relative_volatility_analysis(df)
        self.growth_analysis(df)

        self.calculate_index(df)

        return df

    def relative_volatility_analysis(self, df: DataFrame = None):
        """
        相对波动率
        """
        output("第五步：相对波动率分析")
        df["volatility"] = talib.ATR(
            np.array(df["high"]),
            np.array(df["low"]),
            np.array(df["close"]),
            self.window_volatility
        )

        df["fixed_cost"] = df["close"] * self.rate
        df["relative_vol"] = df["volatility"] - df["fixed_cost"]

        df["relative_vol"].plot(figsize=(20, 6), title="relative volatility")
        plt.show()

        df["relative_vol"].hist(bins=200, figsize=(20, 6), grid=False)
        plt.show()

        statitstic_info(df["relative_vol"])

    def growth_analysis(self, df: DataFrame = None):
        """
        百分比K线变化率
        """
        output("第六步：变化率分析")
        # df["pre_close"] = df["close"].shift(1).fillna(0)
        df["pre_close"] = df["close"].shift(1).fillna(df["close"])

        df["g%"] = 100 * (df["close"] - df["pre_close"]) / df["close"]

        df["g%"].plot(figsize=(20, 6), title="growth", ylim=(-5, 5))
        plt.show()

        df["g%"].hist(bins=200, figsize=(20, 6), grid=False)
        plt.show()

        statitstic_info(df["g%"])

    def calculate_index(self, df: DataFrame = None):
        """"""
        output("第七步：计算相关技术指标，返回DataFrame\n")

        if self.index_1to1:
            for i in self.index_1to1:
                func = getattr(talib, i)
                df[i] = func(
                    np.array(df["close"]),
                    self.window_index
                )

        if self.index_3to1:
            for i in self.index_3to1:
                func = getattr(talib, i)
                df[i] = func(
                    np.array(df["high"]),
                    np.array(df["low"]),
                    np.array(df["close"]),
                    self.window_index
                )

        if self.index_2to2:
            for i in self.index_2to2:
                func = getattr(talib, i)
                result_down, result_up = func(
                    np.array(df["high"]),
                    np.array(df["low"]),
                    self.window_index
                )
                up = i + "_UP"
                down = i + "_DOWN"
                df[up] = result_up
                df[down] = result_down

        if self.index_2to1:
            for i in self.index_2to1:
                func = getattr(talib, i)
                df[i] = func(
                    np.array(df["high"]),
                    np.array(df["low"]),
                    self.window_index
                )

        if self.index_4to1:
            for i in self.index_4to1:
                func = getattr(talib, i)
                df[i] = func(
                    np.array(df["open"]),
                    np.array(df["high"]),
                    np.array(df["low"]),
                    np.array(df["close"]),
                )

        return df

    def custom_resampler(self,df):
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


    def multi_time_frame_analysis(self, intervals: list = None, df: DataFrame = None):
        """"""
        if not intervals:
            output("请输入K线合成周期")
            return

        if df is None:
            df = self.orignal

        if df is None:
            output("请先加载数据")
            return

        for interval in intervals:
            output("------------------------------------------------")
            output(f"合成{interval}周期K先并开始数据分析")

            data = pd.DataFrame()
   
            data["open"] = df["open"].resample(interval).apply(self.custom_resampler).dropna()
            data["high"] = df["high"].resample(interval).apply(self.custom_resampler).dropna()
            data["low"] = df["low"].resample(interval).apply(self.custom_resampler).dropna()
            data["close"] = df["close"].resample(interval).apply(self.custom_resampler).dropna()
            data["volume"] = df["volume"].resample(interval).apply(self.custom_resampler).dropna()

            # data= df.resample(interval).agg(ohlc_dict)
            result = self.base_analysis(data)
            self.results[interval] = result

    def multi_time_frame_compare(self, intervals: list = None, df: DataFrame = None):
        """"""
        if not intervals:
            output("请输入K线合成周期")
            return

        if df is None:
            df = self.orignal

        if df is None:
            output("请先加载数据")
            return

        compare_infos = []
        compare_head = ['interval','roc_average','roc_median','roc_std','roc_skew','roc_kurt','acf','random_p_05','stability_test']

        for interval in intervals:
            output("------------------------------------------------")
            output(f"合成{interval}周期K先并开始数据分析")

            data = pd.DataFrame()
   
            data["open"] = df["open"].resample(interval).apply(self.custom_resampler).dropna()
            data["high"] = df["high"].resample(interval).apply(self.custom_resampler).dropna()
            data["low"] = df["low"].resample(interval).apply(self.custom_resampler).dropna()
            data["close"] = df["close"].resample(interval).apply(self.custom_resampler).dropna()
            data["volume"] = df["volume"].resample(interval).apply(self.custom_resampler).dropna()

            # data= df.resample(interval).agg(ohlc_dict)
            data["roc"] = talib.ROC(data["close"]).fillna(0)
            data_return = data["roc"]
            compare_info = {}
            # acorr_result = acorr_ljungbox(data_return, lags=1)
            statitstic = ADF(data_return)
            acf_result = acf(data_return,nlags=10,alpha=0.05)
            compare_info['interval'] = interval
            compare_info['roc_average'] = round(data_return.mean(), 4)
            compare_info['roc_median'] = round(data_return.median(), 4)
            compare_info['roc_std'] = round(data_return.std(), 4)
            compare_info['std/median'] = round(compare_info['roc_std']/compare_info['roc_median'],-1)
            compare_info['roc_skew'] = round(data_return.skew(), 4)
            compare_info['roc_kurt'] = round(data_return.skew(), 4)
            compare_info['stability_test'] = statitstic[1] > statitstic[4]['10%']
            # compare_info['acf'] = acf_result
            compare_infos.append(compare_info)
        return pd.DataFrame(compare_infos)


    def show_chart(self, data, boll_wide):
        """"""
        data["boll_up"] = data["SMA"] + data["STDDEV"] * boll_wide
        data["boll_down"] = data["SMA"] - data["STDDEV"] * boll_wide

        up_signal = []
        down_signal = []
        len_data = len(data["close"])
        for i in range(1, len_data):
            if data.iloc[i]["close"] > data.iloc[i]["boll_up"]and data.iloc[i - 1]["close"] < data.iloc[i - 1]["boll_up"]:
                up_signal.append(i)

            elif data.iloc[i]["close"] < data.iloc[i]["boll_down"] and data.iloc[i - 1]["close"] > data.iloc[i - 1]["boll_down"]:
                down_signal.append(i)
        print(up_signal)
        plt.figure(figsize=(20, 8))
        close = data["close"]
        plt.plot(close, lw=1)
        plt.plot(close, '^', markersize=5, color='r',
                 label='UP signal', markevery=up_signal)
        plt.plot(close, 'v', markersize=5, color='g',
                 label='DOWN signal', markevery=down_signal)
        plt.plot(data["boll_up"], lw=0.5, color="r")
        plt.plot(data["boll_down"], lw=0.5, color="g")
        plt.legend()
        plt.show()

        # data["ATR"].plot(figsize=(20, 3), title="ATR")
        plt.show()

