# ============================================================
# MovieLens 100K — Recommendation System
# 30-Day Kaggle ML Roadmap
# Author: Faisal Imam
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

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


# ============================================================
# DAY 26 — Content-Based Filtering
# ============================================================

# ── Genre column names ─────────────────────────────────────
genre_cols = [
    'unknown', 'Action', 'Adventure', 'Animation',
    'Childrens', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'FilmNoir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'SciFi',
    'Thriller', 'War', 'Western'
]

# ── Reload movies with all genre columns ───────────────────
movies_full = pd.read_csv(
    '/kaggle/input/datasets/faisalimam19/movie-lense-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    names=['item_id', 'title', 'release_date', 'video_date',
           'imdb_url'] + genre_cols
)

movies_full = movies_full[['item_id', 'title'] + genre_cols]

print("Movies full shape:", movies_full.shape)
print(movies_full.head())

# ── Genre Matrix ───────────────────────────────────────────
genre_matrix = movies_full[genre_cols].values
print("Genre matrix shape:", genre_matrix.shape)

# Quick check — genres of Toy Story
toy_story = movies_full[movies_full['title'].str.contains('Toy Story')]
print("\nToy Story genres:")
print(toy_story[genre_cols].T[toy_story.index[0]])

# ── Cosine Similarity Matrix ───────────────────────────────
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
print("Similarity matrix shape:", cosine_sim.shape)
print("Movie 0 vs itself:      ", cosine_sim[0][0])
print("Movie 0 vs movie 1:     ", cosine_sim[0][1].round(4))

# ── Movie-to-Movie Recommender ─────────────────────────────
indices = pd.Series(movies_full.index, index=movies_full['title'])

def get_similar_movies(title, n=10):
    """
    Given a movie title, return the n most similar movies
    based on genre cosine similarity.
    """
    matches = movies_full[movies_full['title'].str.contains(title, case=False)]
    if matches.empty:
        print(f"No movie found matching '{title}'")
        return

    movie_title = matches.iloc[0]['title']
    idx = indices[movie_title]

    print(f"\nFinding movies similar to: {movie_title}")
    print(f"Genres: {list(movies_full.loc[idx, genre_cols][movies_full.loc[idx, genre_cols] == 1].index)}")

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    movie_indices     = [i[0] for i in sim_scores]
    similarity_values = [round(i[1], 4) for i in sim_scores]

    results = movies_full.iloc[movie_indices][['title']].copy()
    results['similarity_score'] = similarity_values
    results['genres'] = results.index.map(
        lambda i: list(movies_full.loc[i, genre_cols][movies_full.loc[i, genre_cols] == 1].index)
    )

    return results.reset_index(drop=True)

print(get_similar_movies('Toy Story'))
print(get_similar_movies('Star Wars'))
print(get_similar_movies('Fargo'))

# ── User Taste Profile ─────────────────────────────────────
def get_user_taste_profile(user_id, min_rating=4):
    """
    Build a genre taste profile for a user based on
    movies they rated >= min_rating.
    """
    liked = ratings[
        (ratings['user_id'] == user_id) &
        (ratings['rating'] >= min_rating)
    ]

    if liked.empty:
        print(f"User {user_id} has no ratings >= {min_rating}")
        return None

    print(f"User {user_id} liked {len(liked)} movies (rating >= {min_rating})")

    liked_movies  = movies_full[movies_full['item_id'].isin(liked['item_id'])]
    genre_vectors = liked_movies[genre_cols].values
    taste_profile = genre_vectors.mean(axis=0)

    return taste_profile

profile      = get_user_taste_profile(user_id=1)
taste_series = pd.Series(profile, index=genre_cols).sort_values(ascending=False)
print("\nUser 1 taste profile (top genres):")
print(taste_series[taste_series > 0].round(3))

# ── Personalized Content-Based Recommender ────────────────
def content_based_recommend(user_id, n=10, min_rating=4):
    """
    Recommend movies to a user based on their genre taste profile.
    Excludes movies they have already rated.
    """
    taste_profile = get_user_taste_profile(user_id, min_rating)
    if taste_profile is None:
        return

    sim_scores    = cosine_similarity([taste_profile], genre_matrix)[0]
    rated_items   = ratings[ratings['user_id'] == user_id]['item_id'].values
    rated_indices = movies_full[movies_full['item_id'].isin(rated_items)].index.tolist()

    results = movies_full[['item_id', 'title']].copy()
    results['similarity_score'] = sim_scores
    results = results[~results.index.isin(rated_indices)]
    results = results.sort_values('similarity_score', ascending=False).head(n)
    results['genres'] = results.index.map(
        lambda i: list(movies_full.loc[i, genre_cols][movies_full.loc[i, genre_cols] == 1].index)
    )

    print(f"\n🎬 Top {n} Content-Based Recommendations for User {user_id}:")
    return results[['title', 'similarity_score', 'genres']].reset_index(drop=True)

print(content_based_recommend(user_id=1))
print(content_based_recommend(user_id=50))
print(content_based_recommend(user_id=200))

# ── Compare Popularity vs Content-Based ───────────────────
print("=" * 60)
print("POPULARITY-BASED (same for everyone):")
print("=" * 60)
print(get_popular_recommendations(n=5))

print("\n" + "=" * 60)
print("CONTENT-BASED for User 1:")
print("=" * 60)
print(content_based_recommend(user_id=1, n=5))

print("\n" + "=" * 60)
print("CONTENT-BASED for User 200:")
print("=" * 60)
print(content_based_recommend(user_id=200, n=5))

# ── Genre Distribution — Filter Bubble Analysis ───────────
genre_totals = movies_full[genre_cols].sum().sort_values(ascending=False)
print("\nGenre distribution across all movies:")
print(genre_totals)
