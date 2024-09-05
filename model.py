import pandas as pd
import re

# Function to remove year from movie titles
def remove_year_from_title(title):
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()

# Load datasets
movies = pd.read_csv(r'D:\\python files\advanced projects\flask\02\movies.csv')
ratings = pd.read_csv(r'D:\\python files\advanced projects\flask\02\ratings.csv')


# Preprocess movie titles
movies['clean_title'] = movies['title'].apply(remove_year_from_title)

# Save the cleaned dataset
movies.to_csv('cleaned_movies_dataset.csv', index=False)
cleaned_movies_dataset = pd.read_csv(r'D:\python files\advanced projects\flask\cleaned_movies_dataset.csv')


# Merge datasets to get movie titles and their ratings
movie_data = pd.merge(ratings, cleaned_movies_dataset, on='movieId')

# Create a pivot table with users as rows and movies as columns
user_movie_matrix = movie_data.pivot_table(index='userId', columns='clean_title', values='rating')

# Fill NaN values with 0
user_movie_matrix.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity


# Compute similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)

# Convert to a DataFrame for easier manipulation
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Function to recommend movies
def recommend_movies(movie_name, num_recommendations=5):
    # Sort the movies based on similarity scores
    similar_movies = movie_similarity_df[movie_name].sort_values(ascending=False)
    
    # Return the top 'n' similar movies (excluding the input movie)
    return similar_movies.iloc[1:num_recommendations+1]

recommended = recommend_movies("Toy Story", num_recommendations=5)
print(recommended)
