使用说明:
vnpy_custom是使用vnpy进行交易，研究的工具包，主要用于cta策略与数据库的对接，回撤，生成分析报告
1. cta策略相关工具，主要解决cta策略在运行过程中的工具 cta_utility, 其中与交易规则相关的数据，在包中/data/contractRule.csv，与/data/trading_days.json。
2. 数据库数据下载，更新，使用的工具，主要依赖vt_setting.json中的数据库配置与数据源配置，默认使用rqdata。cta_utility中的trading_datas的更新工具也在此文件中
3. trading_signal_tools是发送交易数据，邮件通知，企业微信通知的工具，其中依据于vt_setting中的邮件、企业微信，seatable等配置
4. 其他工具文件是重构vnpy相关工具的文件

