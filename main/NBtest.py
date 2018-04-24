import sys
import pickle
import os
import io
import math
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# removing stop words and non-alaphanumeric weeds,
# calculating the frequency of occurances of each word
# storing the result in word_dict
def parse_review(file, stopwords_list,stemmer):
  word_dict = {}
  negation_list = ["no","not","never","can't","won't","cannot","didn't","couldn't"]
  negation_flag = False

  # To avoid error in parsing special characters
  with io.open(file, encoding="utf-8") as f:
    word_list = word_tokenize(f.read())

  for word in word_list:
    
    # Check for negation
    if word in negation_list:
      if negation_flag:
        negation_flag = False
      else:
        negation_flag = True
        continue

    # is the token a character (or string of characters)?
    if not word.isalnum():
      negation_flag = False


    if word.isalnum() and word not in stopwords_list:

      word = stemmer.stem(word)

      if negation_flag:
        word = "!"+word
        negation_flag = False

      if word not in word_dict:
        word_dict[word] = 1
      else:
        word_dict[word] += 1

  return word_dict

def classify_review(document_word_dict, positive_dict, negative_dict):

  positive_dict_count = sum(positive_dict.values())
  negative_dict_count = sum(negative_dict.values())
  logp_doc_pos = 0
  logp_doc_neg = 0

  total_word_count = positive_dict_count + negative_dict_count

  for word in document_word_dict:
    # if the token has already been processed, skip it
    if word not in positive_dict and word not in negative_dict:
      continue

    # frequency of occurence
    if word in positive_dict and word in negative_dict:
      word_count = positive_dict[word] + negative_dict[word]
    elif word in positive_dict:
      word_count = positive_dict[word]
    else: word_count = negative_dict[word]

    # probaility of occurance in documents
    prob_word = float(word_count)/total_word_count

    # probabily of word's occurance in pos/neg document
    if word in positive_dict:
      prob_word_pos = float(positive_dict[word])/positive_dict_count
    else: prob_word_pos = 0
    if word in negative_dict:
      prob_word_neg = float(negative_dict[word])/negative_dict_count
    else: prob_word_neg = 0

    # Apply Naive Bayes
    prob_word_pos = float(prob_word_pos)/prob_word
    prob_word_neg = float(prob_word_neg)/prob_word

    if prob_word_pos != 0:
      logp_doc_pos += math.log(prob_word_pos)
    if prob_word_neg !=0:
      logp_doc_neg += math.log(prob_word_neg)

  # 1: POSITIVE, 0:NEGATIVE
  if logp_doc_pos > logp_doc_neg:
    return 1

  return 0

# loading the training data
with open('pos.pickle', 'rb') as handle:
  positive_dict = pickle.load(handle)
with open('neg.pickle', 'rb') as handle:
  negative_dict = pickle.load(handle)

# intialization
stopwords_list = set(stopwords.words("english"))
stemmer = PorterStemmer()
os.chdir(sys.argv[1])

pos_list = os.listdir(os.getcwd() + "/pos")
neg_list = os.listdir(os.getcwd() + "/neg")

positive_count = 0

# processing
for review in pos_list:
  document_word_dict = parse_review(os.getcwd() + "/pos/" + review, stopwords_list, stemmer)
  positive_count += classify_review(document_word_dict, positive_dict, negative_dict)

positive_accuracy = (float(positive_count)/len(pos_list))*100
print("Accuracy of positive classification [positive as positive]: " + str(positive_accuracy) +"%")

negative_count = 0
for review in neg_list:
  document_word_dict = parse_review(os.getcwd() + "/neg/" + review, stopwords_list, stemmer)
  negative_count += (not classify_review(document_word_dict, positive_dict, negative_dict))

neg_accuracy = (float(negative_count)/len(neg_list))*100
print("Accuracy of negative classification [neagtive as negative]: " + str(neg_accuracy) +"%")

print "Confusion Matrix:"
print "[ " + str(positive_count) + ", " + str(len(pos_list)) + " ]"
print "[ " + str(negative_count) + ", " + str(len(neg_list)) + " ]"