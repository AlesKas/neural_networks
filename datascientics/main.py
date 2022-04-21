from __future__ import annotations
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

# Load data
RATINGS = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', encoding='cp1252')
BOOKS = pd.read_csv('data/BX-Books.csv', sep=';', encoding='cp1252')
USERS = pd.read_csv('data/BX-Users.csv', sep=';', encoding='cp1252')

# Perform some basic data analysis
# Compute mean of books rated by user
ratings_per_user = RATINGS.groupby('User-ID')['Book-Rating'].count()
user_given_ratings_mean = statistics.mean(ratings_per_user.to_list())
print(f"Mean of given ratings per user: {user_given_ratings_mean:.4f}")

# Show distribution of ratings per users
# We can see exponential distribution here
# ratings_per_user.hist(bins=20, range=(0,100))
# plt.show()

# Compute mean of ratings per book
ratings_per_book = RATINGS.groupby('ISBN')['Book-Rating'].count()
book_rating_mean = statistics.mean(ratings_per_book.tolist())
print(f"Mean of recieved ratings per book: {book_rating_mean:.4f}")

# Again we can see exponential distribution
# ratings_per_book.hist(bins=20, range=(0,50))
# plt.show()

ratings_per_book_df = pd.DataFrame(ratings_per_book)
ratings_per_book_df = ratings_per_book_df[ratings_per_book_df['Book-Rating'] > book_rating_mean]
popular_books = ratings_per_book_df.index.to_list()

ratings_per_user_df = pd.DataFrame(ratings_per_user)
ratings_per_user_df = ratings_per_user_df[ratings_per_user_df['Book-Rating'] > user_given_ratings_mean]
active_users = ratings_per_user_df.index.to_list()

# Filter out books with number of ratings below average
# and users with number of given ratings below average
filtered_ratings = RATINGS[RATINGS['ISBN'].isin(popular_books)]
filtered_ratings = RATINGS[RATINGS['User-ID'].isin(active_users)]
print(f"Before: {len(RATINGS)}, after: {len(filtered_ratings)}")

# Check if I have deleted most of the irrelevant users and books
# ratings_per_user_df.hist(bins=20, range=(0,100))
# plt.show()

# ratings_per_book_df.hist(bins=20, range=(0,50))
# plt.show()

# Dataframe was so big, that Int overflow, so I decided to use HDFS store and 
# store pivot table on hard drive
number_of_subparts = 5
dataframe_parts = np.array_split(filtered_ratings, number_of_subparts)
rating_matrix = pd.HDFStore('store.h5')

if 'df' not in rating_matrix:
    for index in range(number_of_subparts):
        chunk = dataframe_parts[index].pivot_table(index='User-ID', columns='ISBN', values='Book-Rating')
        chunk = chunk.fillna(0)
        if index == 0:
            rating_matrix['df'] = chunk
        else:
            pd.concat([rating_matrix['df'], chunk])

def similar_users(user_id : int, matrix : pd.DataFrame, k=3):
    # Get users ratings
    user = matrix.loc[matrix.index == user_id]
    # Get ratings of other users
    other_users = matrix[matrix.index != user_id]
    # Compute cosine similarity
    similarities = cosine_similarity(user, other_users)[0].tolist()
    # Create dict of user_index : similarity
    similarities = dict(zip(other_users.index.tolist(), similarities))
    similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    # Take top K users
    top_users = similarities[:k]
    users = [u[0] for u in top_users]
    return users

def recommend_item(user_index : int, similar_users_ids : list[int], matrix : pd.DataFrame, items=5):
    # Get similar users by their index
    similar_users = matrix[matrix.index.isin(similar_users_ids)]
    # Calculate mean for every rating
    similar_users = similar_users.mean(axis=0)
    similar_users = pd.DataFrame(similar_users, columns=['mean'])

    # Get users row, transpose it and select books with rating 0, which is probably unseen book
    user_df = matrix[matrix.index == user_index]
    user_df = user_df.transpose()
    user_df.columns = ['rating']
    user_df = user_df[user_df['rating'] == 0]
    unseen = user_df.index.tolist()

    # Select books from similar users that are so far unseen by current user, and sort them by
    # their average rating
    similar_users = similar_users[similar_users.index.isin(unseen)]
    similar_users = similar_users.sort_values(by=['mean'], ascending=False)

    top_n = similar_users.head(items)
    # Finally, return list of ISBN books that could current user like
    return(top_n.index.tolist())

while True:
    try:
        user = list(ratings_per_user_df.sample(n=1).index)[0]
        similar_users_ids = similar_users(user, rating_matrix['df'])
        recommended_books = recommend_item(user, similar_users_ids, rating_matrix['df'])
        print(f"Similar users for user {user} are: {similar_users_ids}.")
        break
    except ValueError:
        print(f"Cannot sompute recommendation list for user {user}.")

final_book_list = recommended_books.copy()
# Some of the ISBN in the BX-Books.csv are missing first digit
for item in recommended_books:
    final_book_list.append(item[1:])
    
recommended = BOOKS[BOOKS['ISBN'].isin(final_book_list)].dropna(axis='columns')

print(recommended)