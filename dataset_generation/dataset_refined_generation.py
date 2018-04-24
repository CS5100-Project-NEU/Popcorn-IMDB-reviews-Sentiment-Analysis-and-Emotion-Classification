import requests
from bs4 import BeautifulSoup

from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas
import string
import operator

from os import listdir
from os.path import isfile, join


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


def create_review_files(movie_count):

    review_count_per_movie = 20
    pos_path = 'C:/***/pos/'
    neg_path = 'C:/***/neg/'

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


def generate_review_data():

    global review_dict
    file_read = open('C:/***/movie_list.txt', 'r')
    movies_list = file_read.readlines()

    for i in range(0, len(movies_list)):
        movies_list[i] = movies_list[i].split('\n')[0]

    for i in range(0, len(movies_list)):
        extract_review_content(movies_list[i])

        create_review_files(i + 1)
        review_dict = {}


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


def generate_movies_list():

    file_read = open('C:/***/super_movies.txt', 'r')
    super_movies_list = file_read.readlines()

    for movie in super_movies_list:
        imdb_list.append(movie.split('\n')[0])
        next_main()


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


def ratings_evaluator():

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


pn = 'pos'                                      # 'neg' to generate negative corpus
data_path = 'C:/***/input/'
out_path = 'C:/***/output'

files_list = [f for f in listdir(data_path) if isfile(join(data_path, f))]
stop_words = set


def file_data_list():

    files_data = []
    for i in range(0, len(files_list)):
        file_open = open(data_path + files_list[i], 'r')
        files_data.append(file_open.read().decode('utf-8').encode('ascii', 'ignore'))

    for i in range(0, len(files_data)):
        files_data[i] = str(files_data[i]).lower()

    punc_list = string.punctuation.replace('_', '')
    for i in range(0, len(files_data)):
        for s in punc_list:
            files_data[i] = str(files_data[i]).replace(s, '')

    return files_data


def stopping_words(data_cluster, re_occur_val):

    global stop_words
    # stop_words = set(stopwords.words('english'))
    if re_occur_val is not 'again':
        stop_words = set(stop_words_in_corpus(data_cluster))       # to use custom generated stop words

    for i in range(0, len(data_cluster)):

        word_tokens = word_tokenize(data_cluster[i])
        filtered_sentence = [w for w in word_tokens if w not in stop_words]

        data_stopped = ''

        for word in filtered_sentence:
            data_stopped = data_stopped + word + ' '

        data_cluster[i] = data_stopped[:-1]
    return data_cluster


def stemming(data_cluster):

    stem_remover = PorterStemmer()

    for i in range(0, len(data_cluster)):
        data = data_cluster[i]
        data_stemmed = ''
        for plural in data.split():
            data_stemmed += stem_remover.stem(plural) + ' '
        data_cluster[i] = data_stemmed[:-1]

    return data_cluster


def create_bigram_data(data):

    for s in range(0, len(data)):
        line = data[s]
        data[s] = ''
        words = line.split()
        for w in range(0, len(words) - 1):
            data[s] += words[w] + '_' + words[w + 1] + ' '
        data[s] = data[s][:-1]

    return data


def create_csv(data):

    row_nums = []
    polarity = []

    for k in range(0, len(data)):
        row_nums.append(k + 1)
        if pn is 'pos':
            polarity.append(1)
        else:
            polarity.append(0)

    corpus = list(zip(row_nums, data, polarity))

    df = pandas.DataFrame(data=corpus, columns=['Row_Num', 'Processed_Text', 'Polarity'])
    df.to_csv(out_path + '/corpus_bigram_' + pn + '.csv', index=False, header=True)

    counter = 1

    for i in data:
        file_write = open(out_path + '/' + pn + '/' + str(counter) + '.txt', 'w')
        file_write.write(i)
        counter += 1


def stop_words_in_corpus(data):
    terms_dict = {}
    terms_count = 0
    for i in data:
        for j in i.split():
            terms_count += 1
            if j in terms_dict:
                terms_dict[j] += 1
            else:
                terms_dict[j] = 1

    t_list = sorted(terms_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    file_write = open(out_path + '/' + 'stops' + '.txt', 'w')

    stop_list = []
    current_t_count = 0
    for i in range(0, len(t_list)):
        if current_t_count < (0.01 * len(t_list)):
            file_write.write(str(t_list[i][0]) + ' ' + str(t_list[i][1]) + '\n')
            current_t_count += 1
            stop_list.append(t_list[i][0])

    return stop_list


def _pre_process_():

    raw_data = file_data_list()
    # raw_data = create_bigram_data(raw_data)     # uncomment to generate bigrams
    stopped_data = stopping_words(raw_data, '')
    stopped_stemmed_data = stemming(stopped_data)
    refined_data = stopping_words(stopped_stemmed_data, 'again')
    create_csv(refined_data)


def _main_():

    generate_movies_list()
    generate_review_data()
    ratings_evaluator()
    _pre_process_()
