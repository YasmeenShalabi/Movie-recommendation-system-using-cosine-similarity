#Sentiment analysis on text data using NLTK (natural language toolkit)

import nltk #provides tools for text processing and classificaion
from nltk.corpus import movie_reviews #corpus dataset used for training nd testing =>contains labeled movie reviews (pos and neg)
from nltk.classify import NaiveBayesClassifier #naive bayes classifier-commonly used for text classification tasks like sentiment analysis
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords #stopwords Ex: the, and ,etc..
import random

#Download nltk data files
nltk.download('movie_reviews') #movie_reviews =>the corpus used
nltk.download('punkt') #punkt=> tokenizer for splitting sentences into words
nltk.download('stopwords')

#Preprocess the dataset and extract features
def extract_features(words):
    return {word: True for word in words} #converts a list of words into a dictionary where keys are the words and values are True; This format is required by nltk classifiers to process the features

#Load the movie_reviews dataset from NLTK
#creates a list of tuples where each tuple contains a list of movie reviews fielId and its cateogry(+ or -)
documents= [(list(movie_reviews.words(fileid)), category) #loads the words form each movie review in the dataset
                for category in movie_reviews.categories() #returns 2 categories (pos and neg reviews)
                for fileid in movie_reviews.fileids(category)] #fileid represents a specific movie review file

#Shuffle the dataset to ensure random distribution
random.shuffle(documents)

#Prepare the dataset for training and testing
featuresets= [(extract_features(d), c) for (d,c) in documents] #converts each review into set of features
train_set, test_set= featuresets[:1600], featuresets[1600:] #training has 1600(80% of the data, test_set has 400 (20% of the data)

#Train the Naive Bayes Classifier
classifier= NaiveBayesClassifier.train(train_set) #learns to distinguis between pos and neg sentiments based on words in reviews

#Evaluate the classifier on the test set
accuracy= nltk_accuracy(classifier, test_set) #compares classifiers predicted labels with actual labels in test set
print(f"Accuracy: {accuracy * 100:.2f}%") #calculates percent of correct predictions

#Show the most informative features
classifier.show_most_informative_features(10) #displays top 10 words that provide most information for distinguishing between + and - reviews ==>  words that the classifier finds most useful in making its prediction

#Test on new input sentences
def analyze_sentiment(text):
    #Tokenize and remove stopwords
    words= nltk.word_tokenize(text)
    words= [word for word in words if word.lower() not in stopwords.words('english')]

    #Predict sentiment
    features= extract_features(words)
    return classifier.classify(features) #returns pos or neg

#Test the classifier with some custom text inputs
test_sentences= [
    "This movie was really amazing!",
    "I hated this movie, it was such a waste of time.",
    "The plot was dull but the performance was great overall.",
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {analyze_sentiment(sentence)}")
    print()