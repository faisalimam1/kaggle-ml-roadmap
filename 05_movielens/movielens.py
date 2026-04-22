# ============================================================
# MovieLens 100K — Recommendation System
# 30-Day Kaggle ML Roadmap
# Author: Faisal Imam
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# DAY 25 — EDA + Popularity-Based Recommender
# ============================================================

# ── Load Data ─────────────────────────────────────────────
ratings = pd.read_csv(
    '/kaggle/input/datasets/faisalimam19/movie-lense-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

movies = pd.read_csv(
    '/kaggle/input/datasets/faisalimam19/movie-lense-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    usecols=[0, 1],
    names=['item_id', 'title']
)

print("Ratings shape:", ratings.shape)
print("Movies shape: ", movies.shape)
print(ratings.head())

# ── Basic Statistics ───────────────────────────────────────
print("\nTotal ratings:   ", len(ratings))
print("Unique users:    ", ratings['user_id'].nunique())
print("Unique movies:   ", ratings['item_id'].nunique())
print("Rating scale:    ", ratings['rating'].min(), "to", ratings['rating'].max())
print("\nRating breakdown:")
print(ratings['rating'].value_counts().sort_index())

# ── Rating Distribution ────────────────────────────────────
plt.figure(figsize=(8, 5))
ratings['rating'].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ── Ratings per User ───────────────────────────────────────
user_activity = ratings.groupby('user_id')['item_id'].count()

plt.figure(figsize=(10, 5))
user_activity.hist(bins=50, color='coral', edgecolor='black')
plt.title('Number of Ratings per User')
plt.xlabel('Number of Movies Rated')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()

print("Average ratings per user:", user_activity.mean().round(1))
print("Min ratings per user:    ", user_activity.min())
print("Max ratings per user:    ", user_activity.max())

# ── Ratings per Movie ──────────────────────────────────────
movie_popularity = ratings.groupby('item_id')['user_id'].count()

plt.figure(figsize=(10, 5))
movie_popularity.hist(bins=50, color='mediumseagreen', edgecolor='black')
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings Received')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

print("Average ratings per movie:", movie_popularity.mean().round(1))
print("Movies with < 10 ratings: ", (movie_popularity < 10).sum())

# ── Sparsity ───────────────────────────────────────────────
n_users   = ratings['user_id'].nunique()
n_movies  = ratings['item_id'].nunique()
n_ratings = len(ratings)

total_possible = n_users * n_movies
sparsity = 1 - (n_ratings / total_possible)

print(f"\nUsers:            {n_users}")
print(f"Movies:           {n_movies}")
print(f"Total ratings:    {n_ratings}")
print(f"Total possible:   {total_possible}")
print(f"Sparsity:         {sparsity:.2%}")

# ── Top 10 Most Active Users ───────────────────────────────
top_users = user_activity.sort_values(ascending=False).head(10)
print("\nTop 10 most active users:\n", top_users)

# ── Movie Stats ────────────────────────────────────────────
movie_stats = ratings.groupby('item_id').agg(
    num_ratings=('rating', 'count'),
    avg_rating=('rating', 'mean')
).reset_index()

movie_stats = movie_stats.merge(movies, on='item_id')
movie_stats = movie_stats.sort_values('num_ratings', ascending=False)

print("\nTop 10 Most Rated Movies:")
print(movie_stats[['title', 'num_ratings', 'avg_rating']].head(10).to_string(index=False))

# ── Popularity-Based Recommender (Weighted Rating) ─────────
C = ratings['rating'].mean()
m = movie_stats['num_ratings'].quantile(0.50)

print(f"\nGlobal mean rating: {C:.3f}")
print(f"Minimum ratings threshold (50th percentile): {m:.0f}")

qualified = movie_stats[movie_stats['num_ratings'] >= m].copy()
print(f"Movies above threshold: {len(qualified)} out of {len(movie_stats)}")

def weighted_rating(row, C=C, m=m):
    v = row['num_ratings']
    R = row['avg_rating']
    return (v / (v + m)) * R + (m / (v + m)) * C

qualified['weighted_score'] = qualified.apply(weighted_rating, axis=1)

top_movies = qualified.sort_values('weighted_score', ascending=False).head(10)

print("\n🎬 Top 10 Popularity-Based Recommendations:")
print(top_movies[['title', 'num_ratings', 'avg_rating', 'weighted_score']]
      .to_string(index=False))

def get_popular_recommendations(n=5):
    return top_movies[['title', 'weighted_score']].head(n)

print("\nSame list for every user (no personalization):")
print(get_popular_recommendations())
