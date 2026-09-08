#Movie recommendation system using cosine similarity

import pandas as pd #to handle movie data as a dataframe
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer #tfidf vectorizer(term frequency inverse document frequency) converts text data(movie genres) into numerical vectors => helps weigh words by importance in a document

#sample movie dataset
data = {'movie_id': [1, 2, 3, 4, 5],
        'title': ['The Matrix', 'John Wick', 'The Godfather', 'Pulp Fiction', 'The Dark Knight'],
        'genre': ['Action, Sci-Fi', 'Action, Thriller', 'Crime, Drama', 'Crime, Drama', 'Action, Crime, Drama']}

#convert dataset into dataframe
df= pd.DataFrame(data)

#display dataset
print("Movie Data: ")
print(df)

#define a TF-IDF Vectorizer to transform genre text into vectors
tfidf= TfidfVectorizer(stop_words='english') #stop_words removes useless words like 'the', 'and', etc..

#fit and transform the genre column into a matrix of TF-IDF features
tfidf_matrix= tfidf.fit_transform(df['genre']) #each row reps a movie, each col reps a genre-related word; values are the tfidf scores (weight/importance of the word for each genre)

#compute the cosine similarity matrix between all movies based on genre request
cosine_sim= cosine_similarity(tfidf_matrix, tfidf_matrix) #each tuple i and j reps similarity btw movie i and movie j

#function to recommend movies based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    #get index of movie that matches title
    idx= df[df['title'] == title].index[0]

    #get the pairwise similarity scores of all movies with that movie
    sim_scores= list(enumerate(cosine_sim[idx])) #result is a list of tuples where each tuple contains a movie index and its similarity score

    #sort movies based on similarity scores
    sim_scores= sorted(sim_scores, key=lambda x: x[1], reverse=True) #sorts the movies by similarity in desc order of most similar first

    #get indices of 2 most similar movies
    sim_scores= sim_scores[1:3] #excluse movie itself(index 0) and get movie indexes 1 and 2

    #get movie indices
    movie_indices= [i[0] for i in sim_scores] #taking first element for i in sim_scores

    #return titles of the most similar/recommended movies
    return df['title'].iloc[movie_indices]

#test the recommendation system with an example
movie_title= 'The Matrix'
recommended_movies= get_recommendations(movie_title)

print(f"Movie recommended for '{movie_title}':")
for movie in recommended_movies:
    print(movie)