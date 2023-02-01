"""
主要提供seatable信号订阅列表工具，以及邮箱发送工具。相关配置文件，存在.vntrader中的signal_info信息中
外部使用时，主要引入return_mail_list  send_trade_message   send_emails
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import ssl
import json

import pandas as pd
import requests
from seatable_api import Base, context
import re

from vnpy.trader.utility import load_json
from vnpy_custom.cta_utility import get_contract_rule

#
#
# # 导入邮箱密码等设置文件
vnpy_home_path = Path.home().joinpath(".vntrader")
custom_setting_filename = vnpy_home_path.joinpath("vt_setting.json")
custom_setting = load_json(custom_setting_filename)

def read_authorized_clients():
    """由seatable读出信号订阅的授权列表，通过手机号授权"""
    server_url = 'https://cloud.seatable.cn'
    api_token = custom_setting['seatable_api_token']
    base = Base(api_token, server_url)
    base.auth()
    rows = base.list_rows('Authorized', view_name='默认视图')
    AuthorizedClients = pd.Series(rows[0]).to_frame().T
    for i in rows[1:]:
        AuthorizedClients = AuthorizedClients.append(i, ignore_index=True)
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
        SubScribeOnline = SubScribeOnline.append(i, ignore_index=True)
    SubScribeOnline.sort_values(by='_mtime', inplace=True, ascending=False)
    SubScribeOnline.drop_duplicates(subset='手机号', inplace=True)
    return SubScribeOnline


def return_mail_list(symbol, strategy) -> list:
    """get subscribe info from seatable, return mail_list in strategy"""
    SubScribeOnline = read_subscribe_online()
    AuthorizedClientsPhone = read_authorized_clients()
    strategy_conversion_dict = {'TrendStarStrategy': '趋势星', 'RBreakerStrategy': '5M星'}
    contractRule = get_contract_rule(symbol)
    symbolHead = contractRule['CNname'] + ''.join(re.findall(r'[A-Za-z]', symbol.upper()))
    strategy_cnname = symbolHead + '-' + strategy_conversion_dict[strategy]
    SubScribe_Clients = SubScribeOnline[(SubScribeOnline['订阅产品'].apply(lambda x: strategy_cnname in x)) & (
        SubScribeOnline['手机号'].isin(AuthorizedClientsPhone))]
    return list(set(SubScribe_Clients['邮箱'].tolist()))


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


def send_emails(subject, content, receivers):
    """发送提示邮件"""
    mail_host = custom_setting['mail_host']
    mail_user = custom_setting['mail_user']
    mail_pass = custom_setting['mail_password']
    sender = mail_user
    message = MIMEMultipart()
    # 邮件主题
    message['Subject'] = subject
    # 发送方信息
    message['From'] = sender
    # 接受方信息
    message['To'] = receivers[0]
    textApart = MIMEText(content, 'html')
    message.attach(textApart)

    # 登录并发送邮件
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(mail_host, 587) as server:
            server.ehlo()  # Can be omitted
            server.starttls(context=context)
            server.ehlo()  # Can be omitted
            server.login(mail_user, mail_pass)
            server.sendmail(sender, receivers, message.as_string())

        # 退出
        print('success')
    except smtplib.SMTPException as e:
        print('error', e)  # 打印错误



