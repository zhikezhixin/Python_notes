import os
import re
import time
from urllib.request import urlopen

# 创建用来存放爬取结果文件的文件夹
dstDir = 'YuanShi'
if not os.path.isdir(dstDir):
    os.mkdir(dstDir)

# 爬取起始页面
startUrl = r'http://www.cae.cn/cae/html/main/col48/column_48_1.html'
# 读取网页内容
with urlopen(startUrl) as fp:
    content = fp.read().decode()

# 提取并遍历每位院士的链接
pattern = r'<li class="name_list"><a href="(.+)" target="_blank">(.+)</a></li>'
result = re.findall(pattern, content)
print(f"Found {len(result)} academicians.")

# 爬取每位院士的简介和照片
for item in result:
    perUrl, name = item
    print(f'正在爬取{name} ({perUrl})...')
    name = os.path.join(dstDir, name)
    perUrl = r'http://www.cae.cn/' + perUrl
    with urlopen(perUrl) as fp:
        content = fp.read().decode()

    # 抓取照片并保存为本地图片文件
    pattern = r'<img src="/cae/admin/upload/(.+?)" style='
    result = re.findall(pattern, content, re.I)
    if result:
        picUrl = r'http://www.cae.cn/cae/admin/upload/' + result[0].replace(' ', r'%20')
        print(f'抓取照片链接: {picUrl}')
        with open(name + '.jpg', 'wb') as pic:
            pic.write(urlopen(picUrl).read())

    # 抓取简介并写入本地文本文件
    pattern = r'<p>(.+?)</p>'
    result = re.findall(pattern, content)
    if result:
        intro = re.sub('(<a.+?</a>)|(&ensp;)|(&nbsp;)', '', '\n'.join(result))
        with open(name + '.txt', 'w', encoding='utf8') as fp:
            fp.write(intro)
    print(f'完成爬取{name}')

print("所有院士信息爬取完成。")
