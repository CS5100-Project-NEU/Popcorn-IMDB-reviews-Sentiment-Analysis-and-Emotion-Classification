import copy
import random
import csv
import math
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# splitting the dataSet into 90:10 ratio for 10-fold cross validation
def k_fold_data(all_data, k):
    test_data = copy.deepcopy(all_data)
    size = len(test_data)
    training_data = []
    separation_factor = (k - 1) / float(k)
    while len(training_data) < int(separation_factor * size):
        i = random.randrange(len(test_data))
        training_data.append(test_data.pop(i))

    return [training_data, test_data]


def create_classification_classes(pos_data, neg_data):
    data_by_class = dict()
    data_by_class[0] = neg_data
    data_by_class[1] = pos_data
    return data_by_class


tokenize = lambda doc: doc.lower().split(" ")


def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values


def tfidf(documents):

    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    word_count = dict()
    for document in tokenized_documents:
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)

            if term in word_count:
                word_count[term] += tf
            else:
                word_count[term] = tf

    return word_count


def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)


def get_review_list(reviews):
    review_list = list()
    for r in reviews:
        review_list.append(r[1])
    return review_list


def word_probabilities(term_tfidfs, size, vocab_size):
    word_probability = dict()
    for key in term_tfidfs:
        word_probability[key] = float(term_tfidfs[key] + 1) / (size + vocab_size)

    return word_probability


def total_size(term_tfidfs):
    total = 0
    for key in term_tfidfs:
        total += term_tfidfs[key]

    return total


def make_prediction(text, word_freq, class_prob):
    prediction = class_prob
    text_counts = text.split()
    for word in text_counts:
        if word in word_freq.keys():
            relative_word_occurrence = word_freq[word]
            prediction *= relative_word_occurrence
    return prediction


def split_data_by_classes(reviews):
    pos_data = list()
    neg_data = list()
    size = len(reviews)

    for i in range(size):
        text = reviews[i][1]
        text = ''.join([x for x in text if not x.isdigit()])
        if reviews[i][2] == '0':
            neg_data.append(text)
        else:
            pos_data.append(text)

    return pos_data, neg_data


def main(n, k):
    pos_data = list(csv.reader(open('training/pos/corpus_pos.csv', "r")))
    neg_data = list(csv.reader(open('training/neg/corpus_neg.csv', "r")))

    all_data = pos_data + neg_data
    actual_labels = list()
    predicted_labels = list()

    correct_classification_count = 0
    incorrect_classification_count = 0
    for i in range(n):
        data = k_fold_data(all_data, k)

        pos_data, neg_data = split_data_by_classes(data[0])

        pos_total = len(pos_data)
        neg_total = len(neg_data)

        total = pos_total + neg_total

        pos_prob = pos_total / float(total)
        neg_prob = neg_total / float(total)

        neg_tfidfs = tfidf(neg_data)
        pos_tfidfs = tfidf(pos_data)

        vocab_counts = dict(Counter(neg_tfidfs) + Counter(pos_tfidfs))
        vocab_size = len(vocab_counts)

        pos_total_size = total_size(pos_tfidfs)
        neg_total_size = total_size(neg_tfidfs)

        neg_probabilities = word_probabilities(neg_tfidfs, pos_total_size, vocab_size)
        pos_probabilities = word_probabilities(pos_tfidfs, neg_total_size, vocab_size)

        test_data = data[1]

        test_total = len(test_data)

        for x in range(100):

            j = random.randrange(len(test_data))

            test_review = test_data[j][1]
            test_class = test_data[j][2]

            actual_labels.append(test_class)

            neg = make_prediction(test_review, neg_probabilities, neg_prob)
            pos = make_prediction(test_review, pos_probabilities, pos_prob)

            if pos > neg:
                output_class = '1'
                predicted_labels.append('1')
            else:
                output_class = '0'
                predicted_labels.append('0')

            if output_class == test_class:
                correct_classification_count += 1
            else:
                incorrect_classification_count += 1

    evaluation(actual_labels, predicted_labels)

    percentage = (correct_classification_count
                  / float(correct_classification_count + incorrect_classification_count)) * 100
    print "Average accuracy for k-fold cross validation split over n iterations is " + str(percentage) + "%"


def evaluation(actual_labels, predicted_labels):
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, [0, 1], '1', 'binary')
    recall = recall_score(actual_labels, predicted_labels, [0, 1], '1')
    f1 = f1_score(actual_labels, predicted_labels, [0, 1], '1')
    str_accuracy = "%.9f" % accuracy
    str_precision = "%.9f" % precision
    str_recall = "%.9f" % recall
    str_f1 = "%.9f" % f1
    print "Accuracy: " + str_accuracy
    print "Precision: " + str_precision
    print "Recall: " + str_recall
    print "F1: " + str_f1


main(1, 10)
