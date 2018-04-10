import requests
import re
from bs4 import BeautifulSoup


review_dict = {}


def extract_review_content(imdb_id):

    url = 'http://www.imdb.com/title/' + imdb_id + '/reviews'

    html_content = requests.get(url)
    html_data = str(html_content.text.encode('ascii', 'ignore').decode('utf-8', 'ignore'))

    soup = BeautifulSoup(html_data, 'html.parser')
    html_body = soup.find_all('div', {'class': 'lister-item-content'})

    for i in range(0, len(html_body)):

        try:
            rating_fetcher = html_body[i].find('span')
            rating = str(rating_fetcher.find('span'))
            rating = rating.split('<span>')[1]
            rating = rating.split('</span>')[0]

            review_fetcher = html_body[i].find('div', {'class': 'text show-more__control'})

            review = str(review_fetcher).split('>')[1].split('<')[0]

            review_dict[review] = rating

        except: pass


def main():

    global review_dict
    file_read = open('C:/***/movie_list.txt', 'r')
    movies_list = file_read.readlines()

    for i in range(0, len(movies_list)):
        movies_list[i] = movies_list[i].split('\n')[0]

    for i in range(0, len(movies_list)):
        extract_review_content(movies_list[i])

        create_review_files(i + 1)
        review_dict = {}


def create_review_files(movie_count):

    review_count_per_movie = 20
    pos_path = 'C:/Users/Srivardhan/Desktop/NeU/FAI/project/code/res/pos/'
    neg_path = 'C:/Users/Srivardhan/Desktop/NeU/FAI/project/code/res/neg/'

    pos_counter = 0
    neg_counter = 0

    review_count = 0

    for review in review_dict:

        if review_count < review_count_per_movie:
            review_count += 1

            # since average rating of the movies in the corpus is 6.5904
            # considering reviews with rating 7 as neutral
            if int(review_dict[review]) > 7:
                pos_counter += 1
                dest_path = pos_path + str(movie_count) + str('_') + str(pos_counter) + '.txt'
                file_write = open(dest_path, 'w')
                file_write.write(str(review))
                file_write.close()

            if int(review_dict[review]) < 7:
                neg_counter += 1
                dest_path = neg_path + str(movie_count) + str('_') + str(neg_counter) + '.txt'
                file_write = open(dest_path, 'w')
                file_write.write(str(review))
                file_write.close()


main()
