"""
DolphinDB脚本，用于在DolphinDB中创建数据库和数据表。
"""


# 创建tick overview表
CREATE_TICK_OVERVIEW_TABLE_SCRIPT = """
dataPath = "dfs://vnpy"
db = database(dataPath)

tick_overview_columns = ["symbol", "exchange", "count", "start", "end", "datetime"]
tick_overview_type = [SYMBOL, SYMBOL, INT, NANOTIMESTAMP, NANOTIMESTAMP, NANOTIMESTAMP]
tick_overview = table(1:0, tick_overview_columns, tick_overview_type)

db.createPartitionedTable(
    tick_overview,
    "tick_overview",
    partitionColumns=["datetime"],
    sortColumns=["symbol", "exchange", "datetime"],
    keepDuplicates=LAST)
"""

# 创建dailybar表
CREATE_DAILY_BAR_TABLE_SCRIPT = """
dataPath = "dfs://vnpy"
db = database(dataPath)

bar_columns = ["symbol", "exchange", "datetime", "interval", "volume", "turnover", "open_interest", "open_price", "high_price", "low_price", "close_price", "settlement", "prev_settlement", "limit_up", "limit_down"]
bar_type = [SYMBOL, SYMBOL, NANOTIMESTAMP, SYMBOL, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE]
dailybar = table(1:0, bar_columns, bar_type)

db.createPartitionedTable(
    dailybar,
    "dailybar",
    partitionColumns=["datetime"],
    sortColumns=["symbol", "exchange", "interval", "datetime"],
    keepDuplicates=LAST)
"""

# 创建daily_overview表
CREATE_DAILY_OVERVIEW_TABLE_SCRIPT = """
dataPath = "dfs://vnpy"
db = database(dataPath)

dailybar_overview_columns = ["symbol", "exchange", "interval", "count", "start", "end", "datetime"]
dailybar_overview_type = [SYMBOL, SYMBOL, SYMBOL, INT, NANOTIMESTAMP, NANOTIMESTAMP, NANOTIMESTAMP]
dailybar_overview = table(1:0, dailybar_overview_columns, dailybar_overview_type)

db.createPartitionedTable(
    dailybar_overview,
    "dailybar_overview",
    partitionColumns=["datetime"],
    sortColumns=["symbol", "exchange", "interval", "datetime"],
    keepDuplicates=LAST)
"""

# 创建trade_data表
CREATE_TRADE_DATA_TABLE_SCRIPT = """
dataPath = "dfs://vnpy"
db = database(dataPath)

trade_data_columns = ["strategy_class", "strategy_name", "strategy_num", "strategy_period","datetime", "symbol", "exchange","orderid","tradeid", "direction", "offset","price","volume","display","calculate"]
trade_data_type = [STRING, STRING, STRING,INT,NANOTIMESTAMP, SYMBOL, SYMBOL, STRING,STRING, SYMBOL,SYMBOL,DOUBLE,DOUBLE,BOOL,BOOL]
trade_data = table(1:0, trade_data_columns, trade_data_type)

db.createPartitionedTable(
    trade_data,
    "tradedata",
    partitionColumns=["datetime"],
    sortColumns=["strategy_num","symbol", "tradeid","datetime"],
    keepDuplicates=LAST)
"""

# 创建sign_data表
CREATE_SIGN_DATA_TABLE_SCRIPT = """
dataPath = "dfs://vnpy"
db = database(dataPath)

sign_data_columns = ["tradingday", "order_time", "strategy_group", "strategy_id", "instrument", "period","sign","remark", "insert_time"]
sign_data_type = [STRING, NANOTIMESTAMP, STRING, STRING,SYMBOL,INT,STRING,STRING,NANOTIMESTAMP]
sign_data = table(1:0, sign_data_columns, sign_data_type)

db.createPartitionedTable(
    sign_data,
    "signdata",
    partitionColumns=["order_time"],
    sortColumns=["instrument","period","strategy_id","order_time"],
    keepDuplicates=LAST)
"""

# 创建signal表
CREATE_SIGNAL_TABLE_SCRIPT = """
dataPath = "dfs://vnpy"
db = database(dataPath)

Signal_Columns = ["symbol", "datetime", "interval", "strategy_num", "pos"]
Signal_Type = [SYMBOL, NANOTIMESTAMP, SYMBOL, SYMBOL, DOUBLE]
Signal = table(1:0, Signal_Columns, Signal_Type)
db.createPartitionedTable(
    Signal,
    "Signal",
    partitionColumns=["datetime"],
    sortColumns=["symbol", "interval", "strategy_num", "datetime"],
    keepDuplicates=LAST)

"""

# 创建memberrank表
CREATE_MEMBERRANK_TABLE_SCRIPT = """
dataPath = "dfs://vnpy"
db = database(dataPath)

member_rank_columns = ["symbol", "datetime", "member_name", "rank", "volume", "volume_change","rank_by"]
member_rank_type = [SYMBOL, NANOTIMESTAMP, SYMBOL, INT, DOUBLE, DOUBLE,SYMBOL]
memberrank = table(1:0, member_rank_columns, member_rank_type)

db.createPartitionedTable(
    memberrank,
    "memberrank",
    partitionColumns=["datetime"],
    sortColumns=["datetime","symbol","rank_by","rank"],
    keepDuplicates=LAST)
"""

# 创建trend_features表
CREATE_TRENDFEATURES_TABLE_SCRIPT = """
dataPath = "dfs://vnpy"
db = database(dataPath)

trend_features_columns = ["symbol", "exchange", "interval", "datetime", "close_price", "index_name","index_trend_var","index_trend_now", "trend_point_date", "trend_point_price", "trend_temp_point_price", "trend_cum_rate", "trend_up_down_range","trend_cum_revers","trend_period_days", "trend_up_nums", "trend_down_nums", "trend_linear_coef", "trend_linear_r2", "trend_linear_score"]
trend_features_type = [SYMBOL, SYMBOL, SYMBOL, NANOTIMESTAMP, DOUBLE, SYMBOL, SYMBOL, INT, NANOTIMESTAMP,DOUBLE, DOUBLE,DOUBLE,DOUBLE,DOUBLE,INT,INT,INT,DOUBLE, DOUBLE, DOUBLE]
trendfeatures = table(1:0, trend_features_columns, trend_features_type)

db.createPartitionedTable(
    trendfeatures,
    "trendfeatures",
    partitionColumns=["datetime"],
    sortColumns=["symbol","interval","index_name","index_trend_var","datetime"],
    keepDuplicates=LAST)
"""