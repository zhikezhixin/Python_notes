# import urllib.request
# response = urllib.request.urlopen('http://www.baidu.com')
# print(response.read())
#
# import urllib.parse
# url = 'http://fanyi.baidu.com/sug'
# data = {'wd' : '苹果'}
# data = urllib.parse.urlencode(data).encode()
# response = urllib.request.urlopen(url,data)
# print(response.read())

import urllib3
http = urllib3.PoolManager()
response = http.request('POST','http://fanyi.baidu.com/sug',fields = {'kw' : '苹果'})
print(response.read())


