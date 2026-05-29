# weibo_hot_mail_modified.py
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import schedule
import time

# ================= 配置项 =================
FROM_ADDR = '1625220574@qq.com'        # 发件邮箱
PASSWORD = 'qpzxhlnvdjwccdaf'          # SMTP 授权码（不是邮箱密码）
TO_ADDR = '1625220574@qq.com'          # 接收邮箱
SMTP_HOST = 'smtp.qq.com'
SMTP_PORT = 587                         # 使用 TLS 端口
COOKIE_VALUE = 'SUB=_2AkMec2l0f8NxqwFRmv4Vz2_qaoVzzQ7EieKoL5ivJRMxHRl-yT8XqkYDtRB6NfNHm6sO3ofHJ1oRuRpE8Nb0jRjOMf0f; SUBP=0033WrSXqPxfM72-Ws9jqgMF55529P9D9WhooAzb6bcWvSTXJoY1iNkk; _s_tentry=-; Apache=3396300863548.0205.1764765948328; SINAGLOBAL=3396300863548.0205.1764765948328; ULV=1764765948329:1:1:1:3396300863548.0205.1764765948328:'

HEADER_TEMPLATE = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh-Hans;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Cookie': COOKIE_VALUE
}

# ================= 请求网页 =================
def page_request(url, header):
    try:
        r = requests.get(url=url, headers=header, timeout=10)
        r.encoding = 'utf-8'
        return r.text
    except Exception as e:
        print('网页请求失败：', e)
        return ''

# ================= 解析网页 =================
def page_parse(html):
    soup = BeautifulSoup(html, 'lxml')
    news = []
    urls_title = soup.select('#pl_top_realtimehot > table > tbody > tr > td.td-02 > a')
    hotness = soup.select('#pl_top_realtimehot > table > tbody > tr > td.td-02 > span')

    for i in range(len(urls_title)):
        new = {}
        title = urls_title[i].get_text().strip()
        url = urls_title[i].get('href', '').strip()
        if url == 'javascript:void(0);' or not url:
            url = urls_title[i].get('href_to') or ''
        hot = 'top' if i == 0 else (hotness[i - 1].get_text().strip() if i - 1 < len(hotness) else '')
        new['title'] = title
        new['url'] = "https://s.weibo.com" + url if url.startswith('/') else url
        new['hot'] = hot
        news.append(new)

    print(f'抓取到热搜条数：{len(news)}，时间：{time.strftime("%Y-%m-%d %X")}')
    for element in news[:5]:  # 调试输出前5条
        print(element['title'], element['hot'], element['url'])

    sendMail(news)

# ================= 发送邮件 =================
def sendMail(news):
    content = ''
    for i, item in enumerate(news):
        content += f"{i+1}、\t{item['title']}\t热度:{item['hot']}\t链接:{item['url']}\n"
    content += '\n获取事件时间为 ' + time.strftime('%Y-%m-%d %X') + '\n'

    msg = MIMEText(content, 'plain', 'utf-8')
    msg['From'] = FROM_ADDR           # 必须设置完整发件人
    msg['To'] = TO_ADDR               # 必须设置接收人
    msg['Subject'] = Header('微博热搜', 'utf-8')

    try:
        qqmail = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        qqmail.ehlo()
        qqmail.starttls()
        qqmail.login(FROM_ADDR, PASSWORD)
        qqmail.sendmail(FROM_ADDR, [TO_ADDR], msg.as_string())  # 注意这里 TO_ADDR 需要用列表
        qqmail.quit()
        print('邮件发送成功')
    except Exception as e:
        print('邮件发送失败：', e)# ================= 定时任务 =================
def job():
    print('**************开始爬取微博热搜**************')
    url = 'https://s.weibo.com/top/summary'
    html = page_request(url=url, header=HEADER_TEMPLATE)
    if html:
        page_parse(html)

if __name__ == "__main__":
    INTERVAL_SECONDS = 20
    print(f'已启动定时任务：每 {INTERVAL_SECONDS} 秒抓取并发送微博热搜（请确保 Cookie 与 SMTP 配置正确）')
    schedule.every(INTERVAL_SECONDS).seconds.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)
