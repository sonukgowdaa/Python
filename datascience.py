import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Convert genres into vectors
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genre'])

# Calculate similarity
similarity = cosine_similarity(genre_matrix)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()

    # Find movie index
    movie_index = None
    for index, title in enumerate(movies['title']):
        if title.lower() == movie_name:
            movie_index = index
            break

    if movie_index is None:
        print("Movie not found!")
        return

    # Get similarity scores
    scores = list(enumerate(similarity[movie_index]))

    # Sort movies by similarity
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nMovies similar to '{movies.iloc[movie_index]['title']}':\n")

    count = 0
    for i in sorted_scores[1:]:
        print(movies.iloc[i[0]]['title'])
        count += 1
        if count == 5:
            break

# User Input
movie = input("Enter movie name: ")
recommend(movie)