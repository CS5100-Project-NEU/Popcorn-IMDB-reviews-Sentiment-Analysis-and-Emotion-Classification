from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas

from os import listdir
from os.path import isfile, join


data_path = 'C:/Users/Srivardhan/Desktop/NeU/FAI/project/code/aclImdb/train/neg/'
# data_path = 'C:/Users/Srivardhan/Desktop/NeU/FAI/project/code/test/data_pre/'
out_path = 'C:/Users/Srivardhan/Desktop/NeU/FAI/project/code/'

files_list = [f for f in listdir(data_path) if isfile(join(data_path, f))]


def file_data_list():

    files_data = []
    for i in range(0, len(files_list)):
        file_open = open(data_path + files_list[i], 'r')
        # print file_open.read().decode('utf-8')
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
        # print(data_stemmed + '\n')

    return data_cluster


def stopping_words(data_cluster):

    stop_words = set(stopwords.words('english'))
    punc_list = ['.', ',', '`', '`', '\'', '(', ')', '*', '&', '~', '<br />', '?']
    for p in punc_list:
        stop_words.add(p)

    for i in range(0, len(data_cluster)):
        for s in punc_list:
            data_cluster[i] = str(data_cluster[i]).replace(s, '')

        word_tokens = word_tokenize(data_cluster[i])
        filtered_sentence = [w for w in word_tokens if w not in stop_words]

        data_stopped = ''

        for word in filtered_sentence:
            data_stopped = data_stopped + word + ' '

        data_cluster[i] = data_stopped[:-1]
    return data_cluster


def create_csv(data):

    row_nums = []
    polarity = []

    for i in range(0, len(data)):
        row_nums.append(i)
        polarity.append(0)

    corpus = list(zip(row_nums, data, polarity))

    df = pandas.DataFrame(data=corpus, columns=['Row_Num', 'Processed_Text', 'Polarity'])
    df.to_csv(out_path + 'corpus_neg.csv', index=False, header=True)


raw_data = file_data_list()
stemmed_data = stemming(raw_data)
stopped_stemmed_data = stopping_words(stemmed_data)
create_csv(stopped_stemmed_data)
