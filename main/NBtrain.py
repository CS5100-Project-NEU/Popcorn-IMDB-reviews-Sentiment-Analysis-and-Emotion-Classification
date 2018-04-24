import sys
import os
import pickle
import re
import io
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def parse_file(file_name):  
  word_list = []

  with open(file_name,"r") as f:
    for line in f:
      for word in line.split():
        word_list.append(word)

  return word_list

# removing stop words and non-alaphanumeric weeds,
# calculating the frequency of occurances of each word
# storing the result in word_dict
def parse_reviews(file_path):

  file_list = os.listdir(file_path) 
  word_dict = {}
  negation_list = ["no","not","never","can't","won't","cannot","didn't","couldn't"]
  negation_flag = False
  stopword_list = set(stopwords.words("english"))
  stemmer = PorterStemmer()

  for file in file_list:
    # To avoid error in parsing special characters
    with io.open(file_path + "/" + file,"r", encoding="utf-8" ) as f:
      word_list = word_tokenize(f.read())

    for word in word_list:
      if word in negation_list:
        # checking for double negatives
        if negation_flag:
          negation_flag = False
        else:
          negation_flag = True
        continue

      if not word.isalnum():
        negation_flag = False

      if word.isalnum() and word not in stopword_list:
        word = stemmer.stem(word)
        if negation_flag:
          word = "!" + word
          negation_flag = False
        if word not in word_dict:
          word_dict[word] = 1
        else:
          word_dict[word] += 1

  return word_dict

current_directory = os.getcwd()
os.chdir(sys.argv[1])
positive_dict = parse_reviews(os.getcwd() + "/pos")
negative_dict = parse_reviews(os.getcwd() + "/neg")
os.chdir(current_directory)

with open('pos.pickle', 'wb') as handle:
  pickle.dump(positive_dict, handle)
with open('neg.pickle', 'wb') as handle:
  pickle.dump(negative_dict, handle)
