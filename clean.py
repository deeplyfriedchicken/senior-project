from urllib.request import urlopen
from urllib.request import HTTPError

from sys import argv

from bs4 import BeautifulSoup

def scrapeTweets():
    with open(argv[1], 'r') as fp:
        fh = open("combined.txt".format(argv[1]),"w")
        line = fp.readline()
        cnt = 1
        while line:
            print('scraping')
            tweet_id = line.split()
            quote_page = 'https://twitter.com/anyuser/status/' + tweet_id[0]
            
            try:
                page = urlopen(quote_page)
                soup = BeautifulSoup(page, 'html.parser')

                name_box = soup.find('p', attrs={'class': 'TweetTextSize TweetTextSize--jumbo js-tweet-text tweet-text'})
                if name_box:
                    tweet = name_box.text.replace('\n', ' ').replace('\r', '')
                    fh.write('{}\t{}\t{}\n'.format(tweet_id[0], tweet_id[1], tweet))
                else:
                    print('N/A')
            except HTTPError:
                print('HTTPError')

            line = fp.readline()
            cnt += 1
        fh.close()

def organizeData(param):
    with open('combined.txt'.format(argv[1]), 'r') as fp:
        cnt = 1
        line = fp.readline()
        while line:
            data = line.split('\t')
            if (len(data) == 3):
                if (data[1] == param):
                    path = param + "/data{}.txt".format(data[0])
                    fh = open(path, "w")
                    fh.write(data[2])
                    fh.close()
            line = fp.readline()
            cnt += 1


scrapeTweets()

# organizeData('negative')
# organizeData('positive')
# organizeData('neutral')