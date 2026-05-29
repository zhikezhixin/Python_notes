import requests
from bs4 import BeautifulSoup
import time


def page_request(url, ua):
    """
    请求网页内容
    :param url: 目标网页URL
    :param ua: 请求头User-Agent
    :return: 网页HTML内容
    """
    try:
        response = requests.get(url=url, headers=ua, timeout=10)
        response.raise_for_status()
        html = response.content.decode('utf-8')
        return html
    except Exception as e:
        print(f"请求网页失败: {e}")
        return None


def page_parse(html):
    """
    解析网页内容，提取电影基本信息
    :param html: 网页HTML内容
    """
    soup = BeautifulSoup(html, 'lxml')

    # 获取每部电影的排名
    position = soup.select('#content > div > div.article > ol > li > div > div.pic > em')

    # 获取豆瓣电影名称
    name = soup.select('#content > div > div.article > ol > li > div > div.info > div.hd > a > span:nth-child(1)')

    # 获取电影评分
    rating = soup.select('#content > div > div.article > ol > li > div > div.info > div.bd > div > span.rating_num')

    # 获取电影链接
    href = soup.select('#content > div > div.article > ol > li > div > div.info > div.hd > a')

    # 输出电影信息
    print(f"\n本页共有 {len(name)} 部电影:")
    print("-" * 120)
    print(f"{'排名':<6}{'片名':<40}{'评分':<10}{'链接'}")
    print("-" * 120)

    for i in range(len(name)):
        rank = position[i].get_text()
        movie_name = name[i].get_text()
        score = rating[i].get_text()
        link = href[i].get('href')
        print(f"{rank:<6}{movie_name:<40}{score:<10}{link}")

    print("-" * 120)


if __name__ == "__main__":
    print('=' * 60)
    print('豆瓣电影Top250爬虫程序')
    print('=' * 60)

    # 设置请求头，模拟浏览器访问
    ua = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # 豆瓣电影Top250每页有25部电影，start就是每页电影的开头
    total_movies = 0
    for page_num, startNum in enumerate(range(0, 250, 25), 1):
        print(f"\n{'*' * 60}")
        print(f"正在爬取第 {page_num} 页 (第 {startNum + 1}-{min(startNum + 25, 250)} 部电影)")
        print(f"{'*' * 60}")

        url = f"https://movie.douban.com/top250?start={startNum}"
        html = page_request(url=url, ua=ua)

        if html:
            page_parse(html=html)
            total_movies += 25
            print(f"\n第 {page_num} 页爬取完成！")
        else:
            print(f"\n第 {page_num} 页爬取失败！")

        # 礼貌性延迟，避免对服务器造成压力
        if startNum < 225:  # 最后一页不需要延迟
            time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"爬取完成！共爬取 {total_movies} 部电影信息")
    print(f"{'=' * 60}")