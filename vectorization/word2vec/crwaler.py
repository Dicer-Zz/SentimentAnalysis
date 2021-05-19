import requests
import random
import time
from bs4 import BeautifulSoup

def crawler():
    '''
    爬取红楼梦
    :url
    :return
    '''
    path = 'http://www.purepen.com/hlm/'
    file = open('./data/红楼梦.txt', 'w+')
    for page in range(1, 121):
        url = path + ('000'+str(page))[-3:] + '.htm'
        print(url)
        html = requests.get(url)
        html.encoding = html.apparent_encoding
        soup = BeautifulSoup(html.text, 'lxml')
        title = soup.find(align = 'center').text
        print(title)
        content = soup.find(face = '宋体').text
        file.write(title   + '\t\n')
        file.write(content + '\t\n')
        sec = random.randint(0, 3)
        print("Sleep %d sconds." % sec)
        time.sleep(sec)

if __name__ == '__main__':
    crawler()