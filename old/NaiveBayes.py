import copy
import random
import csv
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math


tokenize = lambda doc: doc.lower().split(" ")


def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values


def word_tfidf(documents):

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
    try:
        return " ".join([r.lower() for r in reviews])
    except:
        return ""


def count_text(text, size):
    words = text.split()
    word_dict = dict(Counter(words))
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


def word_probabilities(word_counts, vocab_size):
    word_probability = dict()
    size = len(word_counts)
    for key in word_counts:
        word_probability[key] = float(word_counts[key] + 1) / (size + vocab_size)

    return word_probability


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


def naive_bayes(n, k, pos_file_path, neg_file_path, is_tfidf):
    pos_data = list(csv.reader(open(pos_file_path, "r")))
    neg_data = list(csv.reader(open(neg_file_path, "r")))

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

        if is_tfidf:
            neg_counts = word_tfidf(neg_data)
            pos_counts = word_tfidf(pos_data)
        else:
            neg_counts = word_count(neg_data)
            pos_counts = word_count(pos_data)

        vocab_counts = dict(Counter(neg_counts) + Counter(pos_counts))

        vocab_size = len(vocab_counts)

        neg_probabilities = word_probabilities(neg_counts, vocab_size)
        pos_probabilities = word_probabilities(pos_counts, vocab_size)

        test_data = data[1]

        # test_total = len(test_data)

        for x in range(10):

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


def evaluation(actual_labels, predicted_labels):
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, [0, 1], '1', 'binary')
    recall = recall_score(actual_labels, predicted_labels, [0, 1], '1')
    f1 = f1_score(actual_labels, predicted_labels, [0, 1], '1')
    str_accuracy = "%.2f" % (accuracy * 100)
    str_precision = "%.2f" % (precision * 100)
    str_recall = "%.2f" % (recall * 100)
    str_f1 = "%.2f" % (f1 * 100)
    print "Accuracy: " + str_accuracy + "%"
    print "Precision: " + str_precision + "%"
    print "Recall: " + str_recall + "%"
    print "F1: " + str_f1 + "%"


def main():
    print "Naive Bayes for Unigrams"
    naive_bayes(100, 10, 'training/pos/corpus_pos.csv', 'training/neg/corpus_neg.csv', False)

    print "Naive Bayes for Bigrams"
    naive_bayes(100, 10, 'training/pos/corpus_bigram_pos.csv', 'training/neg/corpus_bigram_neg.csv', False)

    print "Naive Bayes for Unigrams with TF-IDF"
    naive_bayes(100, 10, 'training/pos/corpus_pos.csv', 'training/neg/corpus_neg.csv', True)

    print "Naive Bayes for Bigrams with TF-IDF"
    naive_bayes(100, 10, 'training/pos/corpus_bigram_pos.csv', 'training/neg/corpus_bigram_neg.csv', True)


main()
