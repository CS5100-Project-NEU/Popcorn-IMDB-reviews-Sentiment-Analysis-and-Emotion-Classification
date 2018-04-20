import copy
import random
import csv
import nltk
from collections import Counter


# splitting the dataSet into 90:10 ratio for 10-fold cross validation
def k_fold_data(all_data, K):
    test_data = copy.deepcopy(all_data)
    size = len(test_data)
    training_data = []
    separation_factor = (K - 1) / float(K)
    while len(training_data) < int(separation_factor * size):
        i = random.randrange(len(test_data))
        training_data.append(test_data.pop(i))

    return [training_data, test_data]


def create_classification_classes(pos_data, neg_data):
    data_by_class = dict()
    data_by_class[0] = neg_data
    data_by_class[1] = pos_data
    return data_by_class


def concatenate_reviews(reviews):
    return " ".join([r.lower() for r in reviews])


def count_text(text, size):
    words = list(nltk.bigrams(text.split()))
    word_dict = dict(Counter(words))
    for key in word_dict.keys():
        word_dict[key] = word_dict[key] / float(size)
    return word_dict


def word_count(reviews):
    size = len(reviews)
    review_text = concatenate_reviews(reviews)
    return count_text(review_text, size)


def get_review_list(reviews):
    review_list = list()
    for r in reviews:
        review_list.append(r[1])
    return review_list


def make_prediction(text, word_freq, class_prob):
    prediction = class_prob
    # text_counts = text.split()
    text_counts = list(nltk.bigrams(text.split()))
    for word in text_counts:
        if word in word_freq.keys():
            relative_word_occurrence = word_freq[word]
            prediction *= relative_word_occurrence
        # else:
        #     prediction *= 0
    return prediction


def split_data_by_classes(reviews):
    pos_data = list()
    neg_data = list()
    size = len(reviews)

    for i in range(size):
        text = reviews[i][1]
        if reviews[i][2] == '0':
            neg_data.append(text)
        else:
            pos_data.append(text)

    return pos_data, neg_data


def main(n, k):
    pos_data = list(csv.reader(open('training/pos/corpus_pos.csv', "r")))
    neg_data = list(csv.reader(open('training/neg/corpus_neg.csv', "r")))

    all_data = pos_data + neg_data

    acc = 0
    for i in range(n):
        all_data = k_fold_data(all_data, k)

        pos_data, neg_data = split_data_by_classes(all_data[0])

        pos_total = len(pos_data)
        neg_total = len(neg_data)

        total = pos_total + neg_total

        pos_prob = pos_total / float(total)
        neg_prob = neg_total / float(total)

        neg_counts = word_count(neg_data)
        pos_counts = word_count(pos_data)

        test_data = all_data[1]

        neg = make_prediction(test_data[0][1], neg_counts, neg_prob)
        pos = make_prediction(test_data[0][1], pos_counts, pos_prob)

        acc = max(neg, pos)

    print "Average accuracy for k-fold cross validation split over n iterations is " + str(float(acc) / n) + "%"

    # accuracy = looc(all_data)
    # print "accuracy for multiclass classification with LOOCV is " + str(accuracy * 100) + "%"


main(1000, 10)
