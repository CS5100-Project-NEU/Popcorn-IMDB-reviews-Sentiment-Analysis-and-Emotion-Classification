import csv
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
import plotly.plotly as py
from collections import Counter


class Imdb:
    def __init__(self,csv_file):
        self.csv_file=csv_file
        
    def get_data(self):
        with open(self.csv_file) as f:
                reader = csv.reader(f, delimiter="\t")
                imdb_data = list(reader)
                imdb_data=imdb_data[1:]
        return imdb_data



    def create_dicts(self):  
        dictionaries=[] 
        dict_plus={}
        dict_minus={}
        # data=self.get_data()
        data = list(csv.reader(open(self.csv_file, "r")))
        for line in data:
            for word in line[2].split():
                word = re.sub(r'[^\w\s]','', word)
                word = word.lower()
                if line[1]=="1":
                    # change
                    if word in dict_plus:
                        dict_plus[word] += 1
                    else:
                        dict_plus[word] = 1
                    if word not in dict_minus:
                        dict_minus[word] = 0
                if line[1]=="0":
                    if word in dict_minus:
                        dict_minus[word] += 1
                    else:
                        dict_minus[word] = 1
                    if word not in dict_plus:
                        dict_plus[word] = 0
        dictionaries = [dict_plus,dict_minus]        
        return dictionaries

    def full_dict(self):   
        full_dict={}
        data=self.get_data()
        for line in data:
            for word in line[2].split():
                word = re.sub(r'[^\w\s]','', word)
                word = word.lower()
            if word in full_dict:
                full_dict[word]+=1
            else:
                full_dict[word]=1
        return full_dict

    def create_array(self): 
        dictionaries = self.create_dicts()
        dict_plus = dictionaries[0]
        dict_minus = dictionaries[1]
        vec_plus = []
        for word, counts in dict_plus.items():
            vec_plus.append(counts)
        vec_minus = []
        for word, counts in dict_minus.items():
            vec_minus.append(counts)
        vec_plus_array = np.array(vec_plus)
        vec_minus_array = np.array(vec_minus)
        return vec_plus_array, vec_minus_array

    def common_words(self,n,polarity):
        if polarity == "plus":
            polarity=0
        elif polarity == "minus":
            polarity=1
        else:
            print "Error, you must enter plus or minus"
        # add try except
        dictionary=self.create_dicts()[polarity]
        return sorted(dictionary, key=dictionary.get, reverse=True)[:n]

    def plot_dicts(self):
        dictionaries = self.create_dicts()
        dict_plus = dictionaries[0]
        dict_minus = dictionaries[1]
        plt.figure(1)
        y = dict_plus.values()
        x = np.arange(len(y))
        plt.subplot(2,1,1)
        #plt.xticks([])
        #plt.yticks([])
        plt.title('Positive reviews', fontweight='bold')
        plt.plot(x,y,"k")
        y = dict_minus.values()
        x = np.arange(len(y))
        plt.subplot(2,1,2)
        #plt.xticks([])
        #plt.yticks([])
        plt.title('Negative reviews', fontweight='bold')
        plt.plot(x,y,'k')
        plt.matplotlib.rcParams.update({'font.size': 36})
        plt.show()

    def u_test(self):
        vec_plus_array, vec_minus_array  = self.create_array()
        u, p_value = mannwhitneyu(vec_plus_array, vec_minus_array, use_continuity=False, alternative="two-sided")
        print "two-sample Mann Whitney U-test p-value =", p_value

    def plot_significant(self):
        dictionaries = self.create_dicts()
        dict_plus = dictionaries[0]
        dict_minus = dictionaries[1]
        too_common = list(set(sorted(dict_plus, key=dict_plus.get, reverse=True)[:65] + sorted(dict_plus, key=dict_plus.get, reverse=True)[:65]))
        too_common_neg = list(set(sorted(dict_minus, key=dict_minus.get, reverse=True)[:65] + sorted(dict_minus, key=dict_minus.get, reverse=True)[:65]))
        """
        #we used this to check how well plus most common overlaps with minus most common
        discrepancy=0
        for item in sorted(dict_plus, key=dict_plus.get, reverse=True)[:65]:
            if item not in sorted(dict_minus, key=dict_minus.get, reverse=True)[:65]:
                discrepancy += 1
        discrepancy = float(discrepancy)/float(len(sorted(dict_plus, key=dict_plus.get, reverse=True)[:65]))

        print "The discrepancy is: ", discrepancy
        """
        vec_plus_b = []
        for word, counts in dict_plus.items():
            vec_plus_b.append([word, counts])
        vec_minus_b = []
        for word, counts in dict_minus.items():
            vec_minus_b.append([word, counts])

        most_diff=[]
        for i, item in enumerate(vec_plus_b):
            if item[0] in too_common:
                continue
            else:
                gap = abs(vec_plus_b[i][1]-vec_minus_b[i][1])
                if gap > 80:
                    most_diff.append([vec_plus_b[i][0],gap])
        most_dict_plus={}
        most_dict_minus={}
        for item in most_diff:
            most_dict_plus[item[0]]=dict_plus[item[0]]
            most_dict_minus[item[0]]=dict_minus[item[0]]

        y1 = most_dict_plus.values()
        x1 = np.arange(len(y1))
        plt.bar(x1, y1, alpha=0.5, color="green", label="positive reviews")
        plt.xticks(x1 + 0.4, most_dict_plus.keys(), rotation='horizontal')
        y2=most_dict_minus.values()
        x2=np.arange(len(y2))
        plt.bar(x2, y2, alpha=0.5, color="yellow", label="negative reviews")
        plt.xticks(x2 + 0.4, most_dict_minus.keys(), rotation='horizontal')
        plt.suptitle("Sentiment Analysis of IMDB reviews", fontsize=36, fontweight='bold')
        plt.legend(loc='upper right')
        plt.matplotlib.rcParams.update({'font.size': 36})
        plt.show()

    def total_words(self, dictionary):
        total=0
        for word, counts in dictionary.items():
            add=counts
            total+=add
        return total

    def naive_bayes(self, string, sentiment):
        full_dict = self.full_dict()
        dictionaries = self.create_dicts()
        dict_plus = dictionaries[0]
        dict_minus = dictionaries[1]
        full_dict_words = self.total_words(full_dict)
        print "full_dict_words ", full_dict_words
        total_plus_words = self.total_words(dict_plus)
        print "total_plus_words ", total_plus_words
        total_minus_words = self.total_words(dict_minus)
        print "total_minus_words ", total_minus_words
        string_list = []
        p_total = float(1)
        p_string = float(1)
        p_dict = float(1)
        denominator = float(1)
        for word in string.split():
            word = re.sub(r'[^\w\s]','', word)
            word = word.lower()
            string_list.append(word)
        if sentiment == "plus":
            for word in string_list:
                try:
                    denominator = float(full_dict[word])/float(full_dict_words)
                    p_string *= float(dict_plus[word]+1.0) / float(total_plus_words) / denominator
                except KeyError:
                    continue
            p_dict = float(total_plus_words)/(float(total_plus_words)+float(total_minus_words))
            p_total = p_dict*p_string

        elif sentiment == "minus":
            for word in string_list:
                try:
                    denominator = float(full_dict[word])/float(full_dict_words)
                    p_string *= float(dict_minus[word]+1.0) / float(total_minus_words) / denominator
                except KeyError:
                    continue
            p_dict = float(total_minus_words)/(float(total_plus_words)+float(total_minus_words))
            p_total = p_dict*p_string

        else:
            print "You must specify 'plus' or 'minus'"
            sys.exit()
        return p_total

    def classify(self,string):
        prob_plus=self.naive_bayes(string, "plus")
        prob_minus=self.naive_bayes(string, "minus")
        if prob_plus > prob_minus:
            confidence = prob_plus/(prob_plus + prob_minus)*100
            if confidence > 90.0:
                print "We have a", float("{0:.1f}".format(confidence)), "confidence that you entered a positive review."
            else:
                print "Sorry, we can't really tell."
        elif prob_plus < prob_minus:
            confidence = prob_minus/(prob_plus + prob_minus)*100
            if confidence>90.0:
                print "We have a", float("{0:.1f}".format(confidence)), "confidence that you entered a negative review."
            else:
                print "Sorry, we can't really tell."
        else:
            print "Sorry, we can't really tell."



