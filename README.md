# IMDB-reviews-Sentiment-Analysis-Emotion-Classification
An Artificial Intelligence (AI) project for course CS5100 at Northeastern University

___
In this project we developed machine learning models that use movie reviews by users to classify the sentiment of reviews.

___


### How to execute:
  * Extract `dataset/raw_reviews.zip` and `dataset/dataset.zip` in `main` directory.
  * Execute `python NBtrain.py '../main/train'` and then `python NBtest.py '../main/test'` for the main implementation
  * Execute `python NaiveBayes_bigrams.py' and `python NaiveBayes_TFIDF.py' respectively

### For running the evaluations
  * Execute `python review_polarity.py` and `python review_polarity.py`


### [Dataset](https://github.com/CS5100-Project-NEU/Popcorn-IMDB-reviews-Sentiment-Analysis-and-Emotion-Classification/tree/master/dataset)
  We have extracted our custom datasets by implementing the DFS crawler. Refer `dataset_generation/` for the code and `dataset/` for the extracted dataset.

### Results
  | n = 12000 | Predicted: Positive | Predicted: Negative |
  |:-:|:-:|:-:|
  | Actual: Positive | 4803 | 5348 |
  | Actual: Negative | 1197 | 652 |
  **True Positives:** 10,151

**F1 Score:** 0.8242 | **Accuracy:** 84.59%

___

Additional Datasets compatible with the project: [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
