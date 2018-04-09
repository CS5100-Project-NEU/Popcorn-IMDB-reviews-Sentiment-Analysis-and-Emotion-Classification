# the code is written in python 2.7

import math
import copy
import random
import csv
from collections import Counter



# splitting the dataSet into 90:10 ratio for 10-fold cross validation
def k_fold_data(all_data, K):
    test_data = copy.deepcopy(all_data)
    size = len(test_data)
    training_data = []
    separation_factor = (K-1)/float(K)
    while len(training_data) < int(separation_factor * size):
        i = random.randrange(len(test_data))
        training_data.append(test_data.pop(i))

    return [training_data, test_data]


# creating dictionary based on review polarity
def create_classification_classes(pos_data, neg_data):
    data_by_class = dict()
    data_by_class[0] = neg_data
    data_by_class[1] = pos_data
    return data_by_class


# # mean of the values
# def mu(values):
#     sumOfvalues = float(reduce(lambda first, rest: first + rest, values, 0))
#     count = len(values)
#     return sumOfvalues / count
#
#
# # variance of the values
# def sigma(values):
#     avg = mu(values)
#
#     xsquare = sum(x * x for x in values)
#     xbyN = xsquare / len(values)
#     return math.sqrt(xbyN - (avg * avg))
#
#
# #  mean and variance tuples
# def muSigma(values):
#     muSigmaValues = []
#     numOfColumns = len(values[0])
#     for i in range(numOfColumns - 1):
#         sample = []
#         for j in range(len(values)):
#             sample.append(values[j][i])
#         muSigmaValues.append((mu(sample), sigma(sample)))
#     return muSigmaValues
#
#
# def muSigmaOfLabels(pos_data, neg_data):
#     classes = create_classification_classes(pos_data, neg_data)
#     muSigmaValues = {}
#     for key in classes.keys():
#         muSigmaValues[key] = muSigma(classes[key])
#     return muSigmaValues
#
#
# # gaussian probability distribution function
# def gaussianProbability(feature, mu, sigma):
#     e = math.exp(-(math.pow(feature - mu, 2) / (math.pow(sigma, 2) * 2)))
#     return e * (1 / (math.sqrt(2 * math.pi) * sigma))
#
#
# # probabilty of each label
# def probabiltyOfEachLabel(muSigmaValues, testDataPoint):
#     probabilities = {}
#     for key in muSigmaValues.keys():
#         prob = 0.25
#         numOfColumns = len(muSigmaValues[key])
#         for index in range(numOfColumns):
#             mu, sigma = muSigmaValues[key][index]
#             feature = testDataPoint[index]
#             prob = prob * gaussianProbability(feature, mu, sigma)
#         probabilities[key] = prob
#     return probabilities
#
#
# # predicted label based on argmax
# def predictedLabel(muSigmaValues, testDataPoint):
#     dictOfProbs = probabiltyOfEachLabel(muSigmaValues, testDataPoint)
#     return max(dictOfProbs, key=dictOfProbs.get)
#
#
# # accuracy for 70 30 split dataset
# def splitAccuray(muSigmaValues, testData):
#     numOfRows = len(testData)
#     count = 0
#     for index in range(numOfRows):
#         if testData[index][-1] == predictedLabel(muSigmaValues, testData[index]):
#             count += 1
#
#     return (float(count) / numOfRows) * 100


# leave one out cross validation testing
# def looc(observations):
#     count = 0
#     rowsInobservations = len(observations)
#     # print "Predicted      Actual"
#     for i in range(rowsInobservations):
#         copyOfData = copy.deepcopy(observations)
#         copyOfData.pop(i)
#         train = copyOfData
#         test = observations[i]
#         muSigmaValues = muSigmaOfLabels(train)
#         result = predictedLabel(muSigmaValues, test)
#         # print (str(result) + "           " + str(test[-1]))
#         if result == test[-1]:
#             count += 1
#     return float(count) / rowsInobservations


def concatenate_reviews(reviews):
    return " ".join([r[1].lower() for r in reviews])


def count_text(text):
    words = text.split()
    return Counter(words)


def word_count(reviews):
    review_text = concatenate_reviews(reviews)
    return count_text(review_text)


def make_prediction(text, word_freq, class_prob, class_total):
    prediction = 1.0
    text_counts = count_text(text)
    for word in text_counts:
        word_count1 = text_counts.get(word)
        word_freq_value = (word_freq.get(word, 0)) + 1
        sum_all_word_freq = sum(word_freq.values())
        prediction *= float(word_count1) * (word_freq_value / float(sum_all_word_freq + class_total))
    return prediction * float(class_prob)


def main(n, k):
    pos_data = list(csv.reader(open('training/pos/corpus_pos.csv', "r")))
    neg_data = list(csv.reader(open('training/neg/corpus_neg.csv', "r")))

    all_data = pos_data + neg_data

    acc = 0
    for i in range(n):
        pos_data = k_fold_data(pos_data, k)
        neg_data = k_fold_data(neg_data, k)

        pos_total = len(pos_data[0])
        neg_total = len(neg_data[0])

        total = pos_total + neg_total

        pos_prob = pos_total / float(total);
        neg_prob = neg_total / float(total);

        neg_counts = word_count(neg_data[0])
        pos_counts = word_count(pos_data[0])

        test_data = pos_data[1] + neg_data[1];

        neg = make_prediction(test_data[0][1], neg_counts, neg_prob, neg_total)
        pos = make_prediction(test_data[0][1], pos_counts, pos_prob, pos_total)

        acc = max(neg, pos)

    print "Average accuracy for k-fold cross validation split over n iterations is " + str(float(acc) / 10000) + "%"

    # accuracy = looc(all_data)
    # print "accuracy for multiclass classification with LOOCV is " + str(accuracy * 100) + "%"


main(1000, 10)
