import pandas as pd
import graphlab

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
rating_training_data = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
# print rating_training_data.shape

training_data = graphlab.SFrame(rating_training_data)

# Get recommendations for first 4 users and print them
# users = range(1,5) specifies user ID of first 4 users
# k=6 specifies top 6 recommendations to be given
# Similar to popular news system
popularity_movie_model = graphlab.popularity_recommender.create(training_data, user_id='user_id', item_id='movie_id', target='rating')
popular_movies = popularity_movie_model.recommend(users=range(1,5),k=6)
popular_movies.print_rows(num_rows=24)

#Train Model and Make Recommendations
recommender_movie_model = graphlab.item_similarity_recommender.create(training_data,
	user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')
recommend_movie = recommender_movie_model.recommend(users=range(1,5),k=6)
recommend_movie.print_rows(num_rows=24)

# To check for user's recommendation
recommend_movie = recommender_movie_model.recommend(users=[946],k=6) # Replace "946" with your user_id
recommend_movie.print_rows(num_rows=6)


# Evaluation of our model
rating_test_data = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
test_data = graphlab.SFrame(rating_test_data)
# print rating_test_data.shape

evaluation = graphlab.compare(test_data, [popularity_movie_model, recommender_movie_model])
graphlab.show_comparison(evaluation, [popularity_movie_model, recommender_movie_model])

"""
Movie recommendation system
# Reference: https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/
"""
