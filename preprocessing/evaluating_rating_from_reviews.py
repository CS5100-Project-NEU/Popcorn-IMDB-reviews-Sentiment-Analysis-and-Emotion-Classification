import requests
import re
from bs4 import BeautifulSoup


ratings = []


def get_ratings(imdb_id):

    url = 'http://www.imdb.com/title/' + imdb_id + '/reviews'

    html_content = requests.get(url)
    print url
    html_data = str(html_content.text.encode('ascii', 'ignore').decode('utf-8', 'ignore'))

    soup = BeautifulSoup(html_data, 'html.parser')
    html_body = soup.find_all('div', {'class': 'lister-item-content'})

    for i in range(0, len(html_body)):

        try:
            rating_fetcher = html_body[i].find('span')
            rating = str(rating_fetcher.find('span'))
            rating = rating.split('<span>')[1]
            rating = rating.split('</span>')[0]
            ratings.append(rating)

        except: pass


def main():

    global ratings
    file_read = open('C:/***/movie_list.txt', 'r')
    movies_list = file_read.readlines()

    for i in range(0, len(movies_list)):
        movies_list[i] = movies_list[i].split('\n')[0]

    for i in range(0, len(movies_list)):
        get_ratings(movies_list[i])

    rater = 0
    for i in ratings:
        rater += int(i)

    avg_rating = float(rater) / float(len(ratings))

    print avg_rating                                    # found average ratings of the movies as 6.5904


main()
