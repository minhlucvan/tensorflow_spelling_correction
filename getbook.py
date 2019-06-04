
# Import libraries
import shutil

import requests
import urllib.request
import time
from bs4 import BeautifulSoup

print('Starting request https://sachvui.com...')

for pageNumber in range(1, 50):
    print('request page {}'.format(pageNumber))
    # Set the URL you want to webscrape from
    url = 'https://sachvui.com/the-loai/tat-ca.html/{}'.format(pageNumber)
    # url = 'https://sachvui.com/the-loai/van-hoc-viet-nam.html/{}'.format(pageNumber)

    # Connect to the URL
    response = requests.get(url)

    print('response from page {}'.format(pageNumber))
    # Parse HTML and save to BeautifulSoup object¶
    page = BeautifulSoup(response.text, "html.parser")
    # To download the whole data set, let's do a for loop through all a tags
    for detail_anchor in page.select('a[href^="https://sachvui.com/ebook/"]'):
        print('request ebook {}'.format(detail_anchor['href']))

        detail_url = detail_anchor['href']
        detail_response = requests.get(detail_url)
        print('response ebook {}'.format(detail_anchor['href']))

        detail = BeautifulSoup(detail_response.text, "html.parser")

        epub_anchor = detail.select_one('a[href^="https://sachvui.com/download/epub/"]')
        if epub_anchor == None:
            print('no epub found')
        else:
            download_url = epub_anchor['href']
            print('start download {}'.format(download_url))
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent',
                                  'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
            urllib.request.install_opener(opener)
            filename = detail_url.split("/")[-1].split('.')[-1]
            urllib.request.urlretrieve(download_url, './books/raw/{}.epub'.format(filename))
            print('downloaded {}'.format(download_url))
            time.sleep(1)