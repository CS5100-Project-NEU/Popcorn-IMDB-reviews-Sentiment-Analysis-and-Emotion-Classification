import requests
import re
from bs4 import BeautifulSoup


imdb_list = []
full_movie_list = []


def extract_movies_list(imdb_id):

    url = 'http://www.imdb.com/title/' + imdb_id
    html_content = requests.get(url)
    html_data = str(html_content.text.encode('ascii', 'ignore').decode('utf-8', 'ignore'))

    soup = BeautifulSoup(html_data, 'html.parser')
    html_body = soup.find('div', {'class': 'rec_const_picker'})
    href_regex = {'href': re.compile("^/title")}

    try:
        for link in html_body.find_all('a', href_regex):
            href_value = link.get('href')
            new_movie = str(href_value).split('/')[2]

            print new_movie

            if new_movie not in full_movie_list:
                imdb_list.append(new_movie)
                full_movie_list.append(new_movie)
                break

    except:
        pass


def next_main():

    global imdb_list
    file_write = open('C:/***/movie_list.txt', 'a')

    for i in range(0, 100):
        if i < len(imdb_list):
            extract_movies_list(imdb_list[i])
            file_write.write(imdb_list[i] + '\n')

    print imdb_list
    imdb_list = []


def main():

    file_read = open('C:/***/super_movies.txt', 'r')
    super_movies_list = file_read.readlines()

    for movie in super_movies_list:
        imdb_list.append(movie.split('\n')[0])
        next_main()


main()
