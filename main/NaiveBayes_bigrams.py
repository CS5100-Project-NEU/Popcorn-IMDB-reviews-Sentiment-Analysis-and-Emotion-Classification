import copy
import random
import csv
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def main(n, k):
    pos_data = list(csv.reader(open('training/pos/corpus_bigram_pos.csv', "r")))
    neg_data = list(csv.reader(open('training/neg/corpus_bigram_neg.csv', "r")))

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

        neg_counts = word_count(neg_data)
        pos_counts = word_count(pos_data)

        vocab_counts = dict(Counter(neg_counts) + Counter(pos_counts))

        vocab_size = len(vocab_counts)

        neg_probabilities = word_probabilities(neg_counts, vocab_size)
        pos_probabilities = word_probabilities(pos_counts, vocab_size)

        test_data = data[1]

        j = random.randrange(len(test_data))

        test_total = len(test_data)

        # for j in range(test_total):

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

    percentage = (correct_classification_count / float(correct_classification_count + incorrect_classification_count)) * 100
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


main(100, 10)
