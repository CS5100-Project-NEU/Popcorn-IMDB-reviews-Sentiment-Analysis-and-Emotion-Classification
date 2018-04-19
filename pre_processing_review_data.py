from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas
import string
import operator

from os import listdir
from os.path import isfile, join


pn = 'pos'                                      # 'neg' to generate negative corpus
data_path = 'C:/***/'
out_path = 'C:/***/unigrams/'      # 'unigrams' to generate unigrams

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


_pre_process_()
