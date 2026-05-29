# downloadTop250.py
import requests
from bs4 import BeautifulSoup
import re
import docx
from docx.oxml.ns import qn
import os
import time

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
RESULT_DIR = 'result'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def page_request(url, headers, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.encoding = 'utf-8'
            return r.text
        except Exception as e:
            print(f'请求 {url} 失败（尝试 {attempt+1}/{retries}）：{e}')
            time.sleep(2)
    return None

def save_word(data, filename):
    file = docx.Document()
    # 设置中文字体（Times New Roman 兼容做示例，若要显示中文，请按环境改为 '宋体' 等）
    file.styles['Normal'].font.name = u'Times New Roman'
    file.styles['Normal'].element.rPr.rFonts.set(qn('w:eastAsia'), u'Times New Roman')
    for element in data:
        file.add_paragraph(element)
    path = os.path.join(RESULT_DIR, filename)
    file.save(path)

def sub_page_requests(url, headers, data):
    html = page_request(url, headers)
    if not html:
        data.append('子页面无法访问：' + url)
        save_word(data, f"{data[0]}、{data[1]}.docx")
        return
    soup = BeautifulSoup(html, 'lxml')
    # 影片信息区（导演/编剧/主演/类型/上映/片长等）
    info_tag = soup.find(attrs={'id': 'info'})
    if info_tag:
        info_text = info_tag.get_text(separator=' ', strip=True)
        data.append('影片信息：' + info_text)
    else:
        data.append('影片信息：未找到 id=info 区域')
    # 获取简介（v:summary）
    summary_tag = soup.find(attrs={'property': 'v:summary'})
    if summary_tag:
        summary = summary_tag.get_text()
        summary = summary.replace('\n', '').strip()
        data.append(data[1] + ' 影片简介:\n' + summary)
    else:
        data.append(data[1] + ' 影片简介: 无')
    # 保存
    filename = f"{data[0]}、{data[1]}.docx"
    # 防止文件名中有非法字符
    filename = "".join(c for c in filename if c not in r'\/:*?"<>|')
    save_word(data, filename)

def page_parse(html, headers):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup.find_all(attrs={'class': 'item'}):
        data = []
        try:
            num = tag.find('em').get_text().strip()
        except:
            num = '未知序号'
        data.append(num)
        try:
            name = tag.find_all(attrs={'class': 'title'})[0].get_text().strip()
        except:
            name = '未知片名'
        data.append(name)
        # 豆瓣链接
        href = tag.find(attrs={'class': 'hd'}).a
        url = href.attrs.get('href', '')
        data.append('豆瓣链接:' + url)
        # 评分与评论数（使用正则抽数字）
        info = tag.find(attrs={'class': 'star'}).get_text() if tag.find(attrs={'class': 'star'}) else ''
        info = info.replace('\n', '').strip()
        mode = re.compile(r'\d+\.?\d*')
        found = mode.findall(info)
        if len(found) >= 1:
            data.append('豆瓣评分:' + found[0])
        else:
            data.append('豆瓣评分:无')
        if len(found) >= 2:
            data.append('评分人数:' + found[1])
        else:
            data.append('评分人数:无')
        # 进入子页面获取详情
        sub_page_requests(url, headers, data)
        print(f'第 {num} 部电影信息爬取完成：{name}')
        time.sleep(1)  # 子页间隔，降低风险

if __name__ == "__main__":
    print('**************开始爬取豆瓣电影 Top250（详细信息并保存为 Word）**************')
    headers = UA
    for startNum in range(0, 251, 25):
        url = f"https://movie.douban.com/top250?start={startNum}"
        html = page_request(url, headers)
        if html:
            page_parse(html, headers)
        else:
            print(f'页面 {url} 请求失败，跳过')
        time.sleep(2)
    print('**************全部爬取完成**************')
