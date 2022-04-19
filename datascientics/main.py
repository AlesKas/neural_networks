from __future__ import annotations
import operator
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

# Load data
RATINGS = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', encoding='cp1252')
BOOKS = pd.read_csv('data/BX-Books.csv', sep=';', encoding='cp1252')
USERS = pd.read_csv('data/BX-Users.csv', sep=';', encoding='cp1252')

# Compute mean of books rated by user
ratings_per_user = RATINGS.groupby('User-ID')['Book-Rating'].count()
rating_per_user_mean = statistics.mean(ratings_per_user.to_list())
print(f"Mean of given ratings per user: {rating_per_user_mean:.4f}")

# Show distribution of ratings per users
# We can see exponential distribution here
# ratings_per_user.hist(bins=20, range=(0,100))
# plt.show()

# Compute mean of ratings per book
ratings_per_book = RATINGS.groupby('ISBN')['Book-Rating'].count()
rating_per_book_mean = statistics.mean(ratings_per_book.tolist())
print(f"Mean of recieved ratings per book: {rating_per_book_mean:.4f}")

# Again we can see exponential distribution
# ratings_per_book.hist(bins=20, range=(0,50))
# plt.show()

ratings_per_book_df = pd.DataFrame(ratings_per_book)
filtered_ratings_per_book = ratings_per_book_df[ratings_per_book_df['Book-Rating'] > rating_per_book_mean]
popular_books = filtered_ratings_per_book.index.to_list()

ratings_per_user_df = pd.DataFrame(ratings_per_user)
filtered_ratings_per_user = ratings_per_user_df[ratings_per_user_df['Book-Rating'] > rating_per_user_mean]
active_users = filtered_ratings_per_user.index.to_list()

# Filter out books with number of ratings below average
# and users with number of given ratings below average
filtered_ratings = RATINGS[RATINGS['ISBN'].isin(popular_books)]
filtered_ratings = RATINGS[RATINGS['User-ID'].isin(active_users)]
print(f"Before: {len(RATINGS)}, after: {len(filtered_ratings)}")

# Dataframe was so big, that Int overflow, so I decided to use HDFS store and 
# store pivot table on hard drive
number_of_subparts = 5
dataframes = np.array_split(filtered_ratings, number_of_subparts)
rating_matrix = pd.HDFStore('store.h5')

if 'df' not in rating_matrix:
    for index in range(number_of_subparts):
        chunk = dataframes[index].pivot_table(index='User-ID', columns='ISBN', values='Book-Rating')
        chunk = chunk.fillna(0)
        if index == 0:
            rating_matrix['df'] = chunk
        else:
            pd.concat([rating_matrix['df'], chunk])

def similar_users(user_id : int, matrix : pd.DataFrame, k=3):
    # Get users ratings
    user = matrix.loc[[user_id]]
    # Get ratings of other users
    other_users = matrix[matrix.index != user_id]
    # Compute cosine similarity
    similarities = cosine_similarity(user, other_users)[0].tolist()
    # Create dict of user_index : similarity
    index_similarity = dict(zip(other_users.index.tolist(), similarities))
    # Sort
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    # Take top K users
    top_users = index_similarity_sorted[:k]
    users = [u[0] for u in top_users]
    return users

def recommend_item(user_index : int, similar_user_indicies : list[int], matrix : pd.DataFrame, items=5):
    # Get simililar users by their index
    similar_users = matrix[matrix.index.isin(similar_user_indicies)]
    # Calculate mean for every rating
    similar_users = similar_users.mean(axis=0)
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])

    # Get users row, transpose it and select books with rating 0, which is probably unseen book
    user_df = matrix[matrix.index == user_index]
    user_df = user_df.transpose()
    user_df.columns = ['rating']
    user_df = user_df[user_df['rating'] == 0]
    unseen = user_df.index.tolist()

    # Select books from similar users that are so far unseen by current user, and sort them by
    # their average rating
    similar_users_df = similar_users_df[similar_users_df.index.isin(unseen)]
    similar_users_df = similar_users_df.sort_values(by=['mean'], ascending=False)

    top_n = similar_users_df.head(items)
    # Finally, return list of ISBN books that could current user like
    return(top_n.index.tolist())

current_user = 8
similar_users_indicies = similar_users(current_user, rating_matrix['df'])
recommended_books = recommend_item(current_user, similar_users_indicies, rating_matrix['df'])
print(similar_users_indicies)
print(recommended_books)

final_book_list = recommended_books.copy()
# Some of the ISBN in the BX-Books.csv are missing first digit
for item in recommended_books:
    final_book_list.append(item[1:])
    
recommended = BOOKS[BOOKS['ISBN'].isin(final_book_list)].dropna(axis='columns')

print(recommended)