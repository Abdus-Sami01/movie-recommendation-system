from flask import Flask, render_template, request
import pandas as pd
import re

app = Flask(__name__,template_folder='templates')
def remove_year_from_title(title):
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()

# Load data
movies = pd.read_csv(r'D:\python files\advanced projects\flask\02\movies.csv')
ratings = pd.read_csv(r'D:\python files\advanced projects\flask\02\ratings.csv')
movies['clean_title'] = movies['title'].apply(remove_year_from_title)

# Save the cleaned dataset
movies.to_csv('cleaned_movies_dataset.csv', index=False)
cleaned_movies_dataset = pd.read_csv(r'D:\python files\advanced projects\flask\cleaned_movies_dataset.csv')

movie_data = pd.merge(ratings,cleaned_movies_dataset , on='movieId')

user_movie_matrix = movie_data.pivot_table(index='userId', columns='clean_title', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Compute similarity between movies
from sklearn.metrics.pairwise import cosine_similarity
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Recommendation function
def recommend_movies(movie_name, num_recommendations=5):
    similar_movies = movie_similarity_df[movie_name].sort_values(ascending=False)
    return similar_movies.iloc[1:num_recommendations+1].index

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        movie_name = request.form.get('movie_name')
        if movie_name in movie_similarity_df.index:
            recommendations = recommend_movies(movie_name)
        else:
            recommendations = ["Movie not found. Please try another movie."]
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
