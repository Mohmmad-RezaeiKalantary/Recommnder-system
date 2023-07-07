#!/usr/bin/env python
# coding: utf-8


import re
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ast import literal_eval
from datasets import load_dataset
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans

###################################Movies##################################

Movies = pd.read_csv('movies_metadata.csv')

# Define the cleaning function
def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

Movies = Movies.drop([19730, 29503, 35587])

# Clean the 'genres', 'production_companies', 'production_countries', and 'spoken_languages' columns
Movies['genres'] = Movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
Movies['production_companies'] = Movies['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
Movies['production_countries'] = Movies['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
Movies['spoken_languages'] = Movies['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# Clean the movie titles
Movies["clean_title"] = Movies["original_title"].apply(clean_title)

Movies = Movies.rename(columns={'id': 'movieId'})

Movies["movieId"] = pd.to_numeric(Movies["movieId"])

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(Movies["clean_title"])

###################################Rating##################################
ratings = pd.read_csv('ratings.csv')
ratings = ratings.drop('timestamp', axis=1)
sorted_ratings = ratings.sort_values(by=ratings.columns[1])
links=pd.read_csv('links.csv')
links=links.drop('imdbId',axis=1)
ratings = pd.merge(sorted_ratings, links, on='movieId')
ratings.rename(columns={'movieId': 'tmdbId', 'tmdbId': 'movieId'}, inplace=True)

###################################Collaborative##################################
# Define the search function
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = Movies.iloc[indices].iloc[::-1]
    return results

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 3)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 3)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 3)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    # Merge with Movies DataFrame to get additional movie information
    rec_movies = rec_percentages.merge(Movies, left_index=True, right_on="movieId")

    # Extract the year from the release_date column
    rec_movies["year"] = pd.to_datetime(rec_movies["release_date"], errors="coerce").dt.year

    # Sort by the year column in descending order
    rec_movies = rec_movies.sort_values("year", ascending=False)

    return rec_movies.head(2)[["year", "title", "genres"]]

# Define the recommender function
def recommend_movies(title):
    results = search(title)
    movie_id = results.iloc[0]["movieId"]
    similar_movies = find_similar_movies(movie_id)
    return similar_movies    



###################################Cluster##################################

# Step 1: Preprocessing
genre_matrix = pd.get_dummies(Movies['genres'].apply(pd.Series).stack()).groupby(level=0).sum()

# Step 2: Cluster Generation
num_clusters =50  # Set the desired number of clusters
kmeans = KMeans(n_clusters=num_clusters,n_init = 10)
cluster_labels = kmeans.fit_predict(genre_matrix)

def search_cluster(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf[:43779 ]).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = Movies.iloc[indices].iloc[::-1]
    return results

# Step 3: Recommendation Generation
def cluster_based_recommender(title1, title2, title3):
    input_movie_indices = Movies.index[Movies['title'].isin([title1, title2, title3])]

    input_movie_cluster_labels = cluster_labels[input_movie_indices]

    similar_movies_indices = []
    for cluster_label in input_movie_cluster_labels:
        cluster_movies_indices = np.where(cluster_labels == cluster_label)[0]
        similar_movies_indices.extend(cluster_movies_indices)

    similar_movies_indices = list(set(similar_movies_indices) - set(input_movie_indices))
    similar_movies_indices = [index for index in similar_movies_indices if index in Movies.index]

    similar_movies = Movies.loc[similar_movies_indices, ["release_date", "title", "genres"]]
    if len(similar_movies) >= 6:
        recommended_movies = similar_movies.sample(n=6, random_state=25)
    else:
        recommended_movies = similar_movies

    # Extract the year from the release_date column
    recommended_movies["year"] = pd.to_datetime(recommended_movies["release_date"], errors="coerce").dt.year
    # Sort by the year column in descending order
    recommended_movies = recommended_movies.sort_values("year", ascending=False)

    return recommended_movies


###################################Content##################################
# Step 1: Preprocessing
# Combine relevant features into a single string
Movies['features'] = Movies['genres']
Movies['features'] = Movies['features'].apply(lambda x: ' '.join(x))

# Step 2: Create a TF-IDF matrix
tfidf_content = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_content.fit_transform(Movies['features'])
tfidf_matrix = tfidf_matrix[:25000]
# Step 3: Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def search_content(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf[:20000]).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = Movies.iloc[indices].iloc[::-1]
    return results


### modified for searh
def content_based_recommendation(title1, title2, title3, num_recommendations=6):
    # Get the indices of the movies with the given titles
    matching_movies = Movies[Movies['title'].isin([title1, title2, title3])]
    if matching_movies.empty:
        return pd.DataFrame(columns=['year', 'title', 'genres'])  # Return an empty DataFrame if no match is found

    indices = matching_movies.index

    # Compute the average cosine similarity scores for the given movies
    avg_sim_scores = np.mean(cosine_sim[indices], axis=0)

    # Sort the movies based on the similarity scores
    top_indices = np.argsort(avg_sim_scores)[::-1][:num_recommendations]

    # Return the top recommended movies
    recommended_movies = Movies.iloc[top_indices][['title', 'genres', 'release_date']]

    # Extract the year from the release_date column
    recommended_movies['year'] = pd.to_datetime(recommended_movies['release_date'], errors='coerce').dt.year

    # Sort by the year column in descending order
    recommended_movies = recommended_movies.sort_values('year', ascending=False)

    # Keep only the desired columns in the final output
    recommended_movies = recommended_movies[['year', 'title', 'genres']]

    return recommended_movies.head(num_recommendations)


###################################Preidict##################################


import pandas as pd

def predict(title1, title2, title3, method):
    if method == "collaborative":
        results1 = search(title1)
        results2 = search(title2)
        results3 = search(title3)
        similar_movies1 = find_similar_movies(results1.iloc[0]["movieId"])
        similar_movies2 = find_similar_movies(results2.iloc[0]["movieId"])
        similar_movies3 = find_similar_movies(results3.iloc[0]["movieId"])

        # Concatenate the similar movies from all three titles
        similar_movies = pd.concat([similar_movies1, similar_movies2, similar_movies3], ignore_index=True)
    elif method == "cluster":
        results1 = search_cluster(title1)
        results2 = search_cluster(title2)
        results3 = search_cluster(title3)

        similar_movies = cluster_based_recommender(results1.iloc[0]['original_title'],
                                                   results2.iloc[0]['original_title'],
                                                   results3.iloc[0]['original_title'])

        similar_movies = similar_movies.head(6)[["title", "year", "genres"]]
    elif method == "content":
        results1 = search_content(title1)
        results2 = search_content(title2)
        results3 = search_content(title3)

        similar_movies = content_based_recommendation(results1.iloc[0]['original_title'],
                                                      results2.iloc[0]['original_title'],
                                                      results3.iloc[0]['original_title'])
        similar_movies = similar_movies.head(6)[["title", "year", "genres"]]
    else:
        return pd.DataFrame({"Error": ["Invalid recommendation method. Choose either 'collaborative', 'cluster', or 'content'."]})

    return similar_movies


# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=["text", "text", "text", gr.Radio(["collaborative", "cluster", "content"], label="Method")],
    outputs='dataframe',
    examples=[['Captain America','Avengers: Infinity War','Ant-Man', "collaborative"],['Shrek','The Smurfs','Up', "content"]],
    title = "Recommender System",
    description="Experience the ultimate recommendation journey with our cutting-edge recommender system! Utilizing collaborative, content, and cluster methods, we cater to your unique preferences. While computational limitations may impact results, maximize your experience by exploring our content-based or collaborative-based method. Please note that due to limitations, results may vary. Choose the path of personalized discovery and unlock a world of possibilities!",
    flagging_options=["Good Prediction", "Bad Prediction"],
    theme='abidlabs/banana'
)

# Launch the interface
interface.launch(debug=False)

