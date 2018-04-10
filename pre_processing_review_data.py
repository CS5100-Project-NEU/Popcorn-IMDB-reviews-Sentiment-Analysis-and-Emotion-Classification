from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas
import string

from os import listdir
from os.path import isfile, join


pn = 'pos'                                      # 'neg' to generate negative corpus
data_path = 'C:/***/'
out_path = 'C:/***/bigrams/'                    # 'unigrams' to generate unigrams

files_list = [f for f in listdir(data_path) if isfile(join(data_path, f))]


def file_data_list():

    files_data = []
    for i in range(0, len(files_list)):
        file_open = open(data_path + files_list[i], 'r')
        files_data.append(file_open.read().decode('utf-8').encode('ascii', 'ignore'))

    return files_data


def stemming(data_cluster):

    stem_remover = PorterStemmer()

    for i in range(0, len(data_cluster)):
        data = data_cluster[i]
        data_stemmed = ''
        for plural in data.split():
            data_stemmed += stem_remover.stem(plural) + ' '
        data_cluster[i] = data_stemmed[:-1]

    return data_cluster


def stopping_words(data_cluster):

    stop_words = set(stopwords.words('english'))
    punc_list = string.punctuation

    for p in punc_list:
        stop_words.add(p)

    for i in range(0, len(data_cluster)):
        for s in punc_list:
            data_cluster[i] = str(data_cluster[i]).replace(s, '').lower()

        word_tokens = word_tokenize(data_cluster[i])
        filtered_sentence = [w for w in word_tokens if w not in stop_words]

        data_stopped = ''

        for word in filtered_sentence:
            data_stopped = data_stopped + word + ' '

        data_cluster[i] = data_stopped[:-1]
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

    for i in range(0, len(data)):
        row_nums.append(i + 1)
        if pn is 'pos':
            polarity.append(1)
        else:
            polarity.append(0)

    corpus = list(zip(row_nums, data, polarity))

    df = pandas.DataFrame(data=corpus, columns=['Row_Num', 'Processed_Text', 'Polarity'])
    df.to_csv(out_path + 'corpus_bigram_' + pn + '.csv', index=False, header=True)

    counter = 1

    for i in data:
        file_write = open(out_path + '/' + pn + '/' + str(counter) + '.txt', 'w')
        file_write.write(i)
        counter += 1


raw_data = file_data_list()
stemmed_data = stemming(raw_data)
stopped_stemmed_data = stopping_words(stemmed_data)
bigram_data = create_bigram_data(stemmed_data)              # comment to generate unigrams
create_csv(bigram_data)
