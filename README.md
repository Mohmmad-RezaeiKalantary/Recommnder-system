# Recommender System

This recommender system utilizes collaborative filtering, content-based filtering, and clustering methods to provide personalized movie recommendations.

![Alt Text](Image.png)


## Overview

This recommender system combines multiple approaches to provide accurate and diverse movie recommendations. The system utilizes the following methods:

- Collaborative Filtering: This method analyzes user behavior and preferences to identify similar users and recommend movies based on their ratings.

- Content-Based Filtering: This method examines the content and characteristics of movies to recommend similar movies based on genres and other features.

- Clustering: This method groups movies into clusters based on their genre information and recommends movies from the same cluster.

The system allows users to input movie titles and choose a recommendation method to generate personalized recommendations.

## Hugging Face Model

You can access the pre-trained recommender system model on Hugging Face Model Hub.

- [Recommender System Model](https://huggingface.co/spaces/Mahziar/Mahziar-Recommender-System): Access the pre-trained recommender system model on Hugging Face Model Hub.

## How to Use the code

To use the recommender system, follow these steps:

1. Install the required dependencies by running the following command:

2. Download the movie dataset from the following link: [Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Place the dataset files (`movies_metadata.csv`, `ratings.csv`, `links.csv`) in the same directory as the recommender system code.

3. Run the recommender system code by executing the following command:

4. The system will launch a Gradio interface where you can enter movie titles and choose a recommendation method to get personalized recommendations.

5. Explore the recommendations and enjoy personalized movie suggestions!


## Acknowledgments

- The movie dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
- The recommender system code utilizes the Gradio library for the user interface.

