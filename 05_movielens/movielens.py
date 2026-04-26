# ============================================================
# MovieLens 100K — Recommendation System
# Author: Faisal Imam
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split as sklearn_split
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DAY 15 — EDA + Popularity-Based Recommender
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
# DAY 16 — Content-Based Filtering
# ============================================================

genre_cols = [
    'unknown', 'Action', 'Adventure', 'Animation',
    'Childrens', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'FilmNoir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'SciFi',
    'Thriller', 'War', 'Western'
]

movies_full = pd.read_csv(
    '/kaggle/input/datasets/faisalimam19/movie-lense-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    names=['item_id', 'title', 'release_date', 'video_date', 'imdb_url'] + genre_cols
)

movies_full  = movies_full[['item_id', 'title'] + genre_cols]
genre_matrix = movies_full[genre_cols].values
cosine_sim   = cosine_similarity(genre_matrix, genre_matrix)
indices      = pd.Series(movies_full.index, index=movies_full['title'])

print("Movies full shape:      ", movies_full.shape)
print("Genre matrix shape:     ", genre_matrix.shape)
print("Similarity matrix shape:", cosine_sim.shape)

def get_similar_movies(title, n=10):
    matches = movies_full[movies_full['title'].str.contains(title, case=False)]
    if matches.empty:
        print(f"No movie found matching '{title}'")
        return
    movie_title = matches.iloc[0]['title']
    idx         = indices[movie_title]
    print(f"\nFinding movies similar to: {movie_title}")
    print(f"Genres: {list(movies_full.loc[idx, genre_cols][movies_full.loc[idx, genre_cols] == 1].index)}")
    sim_scores        = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
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

def get_user_taste_profile(user_id, min_rating=4):
    liked = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= min_rating)]
    if liked.empty:
        print(f"User {user_id} has no ratings >= {min_rating}")
        return None
    print(f"User {user_id} liked {len(liked)} movies (rating >= {min_rating})")
    liked_movies  = movies_full[movies_full['item_id'].isin(liked['item_id'])]
    genre_vectors = liked_movies[genre_cols].values
    return genre_vectors.mean(axis=0)

profile      = get_user_taste_profile(user_id=1)
taste_series = pd.Series(profile, index=genre_cols).sort_values(ascending=False)
print("\nUser 1 taste profile (top genres):")
print(taste_series[taste_series > 0].round(3))

def content_based_recommend(user_id, n=10, min_rating=4):
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

genre_totals = movies_full[genre_cols].sum().sort_values(ascending=False)
print("\nGenre distribution across all movies:")
print(genre_totals)


# ============================================================
# DAY 27 — User-Based Collaborative Filtering
# ============================================================

user_item_matrix = ratings.pivot_table(
    index='user_id', columns='item_id', values='rating'
)

print("User-Item Matrix shape:", user_item_matrix.shape)

ratings_per_user = user_item_matrix.notna().sum(axis=1)
print(f"Min ratings per user:  {ratings_per_user.min()}")
print(f"Max ratings per user:  {ratings_per_user.max()}")
print(f"Mean ratings per user: {ratings_per_user.mean().round(1)}")

def pearson_similarity(user1_id, user2_id, matrix):
    u1 = matrix.loc[user1_id]
    u2 = matrix.loc[user2_id]
    both_rated    = u1.notna() & u2.notna()
    common_movies = both_rated.sum()
    if common_movies < 2:
        return 0, 0
    r1, r2  = u1[both_rated].values, u2[both_rated].values
    corr, _ = pearsonr(r1, r2)
    if np.isnan(corr):
        return 0, common_movies
    return corr, common_movies

corr, common = pearson_similarity(1, 2, user_item_matrix)
print(f"\nUser 1 vs User 2:   correlation = {corr:.4f}, common = {common}")
corr, common = pearson_similarity(1, 200, user_item_matrix)
print(f"User 1 vs User 200: correlation = {corr:.4f}, common = {common}")

def get_similar_users(target_user_id, matrix, n=10, min_common=5):
    similarities = []
    for user_id in [u for u in matrix.index.tolist() if u != target_user_id]:
        corr, common = pearson_similarity(target_user_id, user_id, matrix)
        if common >= min_common:
            similarities.append({'user_id': user_id, 'similarity': corr, 'common_movies': common})
    sim_df = pd.DataFrame(similarities).sort_values('similarity', ascending=False)
    return sim_df.head(n).reset_index(drop=True)

print("\nFinding similar users to User 1...")
similar_users = get_similar_users(1, user_item_matrix, n=10, min_common=5)
print(similar_users.to_string(index=False))
print(f"\nUsers with similarity > 0.5 to User 1: {len(similar_users[similar_users['similarity'] > 0.5])}")

def user_based_recommend(target_user_id, matrix, movies_df,
                          n_similar=20, n_recommendations=10, min_common=5):
    similar      = get_similar_users(target_user_id, matrix, n=n_similar, min_common=min_common)
    if similar.empty:
        return None
    target_rated = matrix.loc[target_user_id].dropna().index.tolist()
    predictions  = {}
    for _, row in similar.iterrows():
        neighbor_id, similarity = row['user_id'], row['similarity']
        if similarity <= 0:
            continue
        for movie_id, rating in matrix.loc[neighbor_id].dropna().items():
            if movie_id in target_rated:
                continue
            if movie_id not in predictions:
                predictions[movie_id] = {'weighted_sum': 0, 'sim_sum': 0}
            predictions[movie_id]['weighted_sum'] += similarity * rating
            predictions[movie_id]['sim_sum']      += similarity
    if not predictions:
        return None
    pred_ratings  = {mid: v['weighted_sum'] / v['sim_sum']
                     for mid, v in predictions.items() if v['sim_sum'] > 0}
    pred_series   = pd.Series(pred_ratings).sort_values(ascending=False)
    top_movie_ids = pred_series.head(n_recommendations).index
    results = movies_df[movies_df['item_id'].isin(top_movie_ids)].copy()
    results['predicted_rating'] = results['item_id'].map(pred_series)
    results = results.sort_values('predicted_rating', ascending=False)
    print(f"\n🎬 Top {n_recommendations} User-Based CF Recommendations for User {target_user_id}:")
    return results[['title', 'predicted_rating']].reset_index(drop=True)

print(user_based_recommend(1,   user_item_matrix, movies))
print(user_based_recommend(50,  user_item_matrix, movies))
print(user_based_recommend(200, user_item_matrix, movies))


# ============================================================
# DAY 28 — Item-Based CF + Matrix Factorization (SVD)
# ============================================================

# ── Item-Item Similarity ───────────────────────────────────
item_user_matrix  = user_item_matrix.T
item_user_filled  = item_user_matrix.fillna(0)
item_similarity   = cosine_similarity(item_user_filled)
item_sim_df       = pd.DataFrame(
    item_similarity,
    index=item_user_matrix.index,
    columns=item_user_matrix.index
)
print("Item similarity matrix shape:", item_sim_df.shape)

def get_similar_movies_cf(movie_id, n=10):
    if movie_id not in item_sim_df.index:
        return None
    sim_scores  = item_sim_df[movie_id].sort_values(ascending=False).drop(movie_id)
    top_similar = sim_scores.head(n).reset_index()
    top_similar.columns = ['item_id', 'similarity']
    return top_similar.merge(movies, on='item_id')[['title', 'similarity']]

star_wars_id = movies[movies['title'].str.contains('Star Wars')].iloc[0]['item_id']
print(f"\nMovies similar to Star Wars (rating-based):")
print(get_similar_movies_cf(star_wars_id))

def item_based_recommend(user_id, n_recommendations=10, n_similar_items=20):
    user_ratings   = user_item_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return None
    unrated_movies = set(user_item_matrix.columns) - set(user_ratings.index)
    predictions    = {}
    for candidate_movie in unrated_movies:
        if candidate_movie not in item_sim_df.index:
            continue
        sims = item_sim_df[candidate_movie][user_ratings.index]
        sims = sims[sims > 0].sort_values(ascending=False).head(n_similar_items)
        if sims.empty:
            continue
        w = (sims * user_ratings[sims.index]).sum()
        s = sims.sum()
        if s > 0:
            predictions[candidate_movie] = w / s
    if not predictions:
        return None
    pred_series   = pd.Series(predictions).sort_values(ascending=False)
    top_movie_ids = pred_series.head(n_recommendations).index
    results = movies[movies['item_id'].isin(top_movie_ids)].copy()
    results['predicted_rating'] = results['item_id'].map(pred_series)
    results = results.sort_values('predicted_rating', ascending=False)
    print(f"\n🎬 Top {n_recommendations} Item-Based CF Recommendations for User {user_id}:")
    return results[['title', 'predicted_rating']].reset_index(drop=True)

print(item_based_recommend(1))
print(item_based_recommend(50))
print(item_based_recommend(200))

# ── SVD — Matrix Factorization ─────────────────────────────
user_ratings_mean = np.nanmean(user_item_matrix.values, axis=1, keepdims=True)
matrix_demeaned   = user_item_matrix.fillna(0).values - user_ratings_mean

k = 50
U, sigma, Vt      = svds(csr_matrix(matrix_demeaned), k=k)
predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_ratings_mean
predicted_df      = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)
print(f"SVD complete. Predicted matrix shape: {predicted_df.shape} | Latent factors: {k}")

def svd_recommend(user_id, n=10):
    rated      = user_item_matrix.loc[user_id].dropna().index.tolist()
    user_preds = predicted_df.loc[user_id].drop(index=rated, errors='ignore')
    top_ids    = user_preds.sort_values(ascending=False).head(n).index
    results    = movies[movies['item_id'].isin(top_ids)].copy()
    results['predicted_rating'] = results['item_id'].map(user_preds.to_dict())
    results    = results.sort_values('predicted_rating', ascending=False)
    print(f"\n🎬 Top {n} SVD Recommendations for User {user_id}:")
    return results[['title', 'predicted_rating']].reset_index(drop=True)

print(svd_recommend(1))
print(svd_recommend(50))
print(svd_recommend(200))

# ── 4-Way Comparison ──────────────────────────────────────
for label, fn in [
    ("POPULARITY-BASED", lambda: get_popular_recommendations(n=5)),
    ("CONTENT-BASED User 1", lambda: content_based_recommend(user_id=1, n=5)),
    ("USER-BASED CF User 1", lambda: user_based_recommend(1, user_item_matrix, movies, n_recommendations=5)),
    ("ITEM-BASED CF User 1", lambda: item_based_recommend(1, n_recommendations=5)),
    ("SVD User 1",           lambda: svd_recommend(1, n=5)),
]:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(fn())


# ============================================================
# DAY 29 — Evaluation + Full Comparison
# ============================================================

# ── Train/Test Split ───────────────────────────────────────
train_ratings, test_ratings = sklearn_split(ratings, test_size=0.2, random_state=42)
train_matrix = train_ratings.pivot_table(
    index='user_id', columns='item_id', values='rating'
)
print(f"Train: {len(train_ratings)} | Test: {len(test_ratings)}")
print("Train matrix shape:", train_matrix.shape)

# ── Baseline ───────────────────────────────────────────────
global_mean   = train_ratings['rating'].mean()
baseline_rmse = np.sqrt(mean_squared_error(test_ratings['rating'], [global_mean] * len(test_ratings)))
baseline_mae  = mean_absolute_error(test_ratings['rating'], [global_mean] * len(test_ratings))
print(f"\nBaseline — RMSE: {baseline_rmse:.4f} | MAE: {baseline_mae:.4f}")

# ── SVD on Train ───────────────────────────────────────────
urt         = np.nanmean(train_matrix.values, axis=1, keepdims=True)
mat_d       = train_matrix.fillna(0).values - urt
U2, s2, Vt2 = svds(csr_matrix(mat_d), k=50)
pred_tr     = np.dot(np.dot(U2, np.diag(s2)), Vt2) + urt
predicted_train_df = pd.DataFrame(pred_tr, index=train_matrix.index, columns=train_matrix.columns)

svd_preds, svd_actual = [], []
for _, row in test_ratings.iterrows():
    uid, iid, actual = row['user_id'], row['item_id'], row['rating']
    if uid in predicted_train_df.index and iid in predicted_train_df.columns:
        svd_preds.append(np.clip(predicted_train_df.loc[uid, iid], 1, 5))
        svd_actual.append(actual)

svd_rmse = np.sqrt(mean_squared_error(svd_actual, svd_preds))
svd_mae  = mean_absolute_error(svd_actual, svd_preds)
print(f"SVD — RMSE: {svd_rmse:.4f} | MAE: {svd_mae:.4f} | Evaluated on {len(svd_preds)} ratings")

# ── Item-Based CF on Train ─────────────────────────────────
train_item_sim_df = pd.DataFrame(
    cosine_similarity(train_matrix.T.fillna(0)),
    index=train_matrix.columns, columns=train_matrix.columns
)

item_preds, item_actual = [], []
for _, row in test_ratings.iterrows():
    uid, iid, actual = row['user_id'], row['item_id'], row['rating']
    if uid not in train_matrix.index or iid not in train_item_sim_df.index:
        continue
    utr  = train_matrix.loc[uid].dropna()
    sims = train_item_sim_df[iid][utr.index]
    sims = sims[sims > 0].sort_values(ascending=False).head(20)
    if sims.empty:
        continue
    w, s = (sims * utr[sims.index]).sum(), sims.sum()
    if s > 0:
        item_preds.append(np.clip(w / s, 1, 5))
        item_actual.append(actual)

item_rmse = np.sqrt(mean_squared_error(item_actual, item_preds))
item_mae  = mean_absolute_error(item_actual, item_preds)
print(f"Item-Based CF — RMSE: {item_rmse:.4f} | MAE: {item_mae:.4f} | Evaluated on {len(item_preds)} ratings")

# ── Precision@K and Recall@K ──────────────────────────────
def precision_recall_at_k(model_fn, test_df, k=10, relevant_threshold=4):
    precisions, recalls = [], []
    np.random.seed(42)
    sample_users = np.random.choice(test_df['user_id'].unique(),
                                     size=min(100, test_df['user_id'].nunique()),
                                     replace=False)
    for uid in sample_users:
        relevant = test_df[(test_df['user_id'] == uid) &
                           (test_df['rating'] >= relevant_threshold)]['item_id'].tolist()
        if not relevant:
            continue
        try:
            recs = model_fn(uid)
            if not recs:
                continue
        except:
            continue
        hits = len(set(recs[:k]) & set(relevant))
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
    return (np.mean(precisions) if precisions else 0,
            np.mean(recalls)    if recalls    else 0)

def svd_rec_ids(user_id, n=10):
    if user_id not in predicted_train_df.index:
        return []
    rated = train_ratings[train_ratings['user_id'] == user_id]['item_id'].tolist()
    up    = predicted_train_df.loc[user_id].drop(
        index=[i for i in rated if i in predicted_train_df.columns], errors='ignore')
    return up.sort_values(ascending=False).head(n).index.tolist()

def item_rec_ids(user_id, n=10):
    if user_id not in train_matrix.index:
        return []
    utr     = train_matrix.loc[user_id].dropna()
    unrated = set(train_matrix.columns) - set(utr.index)
    preds   = {}
    for cand in unrated:
        if cand not in train_item_sim_df.index:
            continue
        sims = train_item_sim_df[cand][utr.index]
        sims = sims[sims > 0].sort_values(ascending=False).head(20)
        if sims.empty:
            continue
        w, s = (sims * utr[sims.index]).sum(), sims.sum()
        if s > 0:
            preds[cand] = w / s
    if not preds:
        return []
    return pd.Series(preds).sort_values(ascending=False).head(n).index.tolist()

def pop_rec_ids(user_id, n=10):
    rated = train_ratings[train_ratings['user_id'] == user_id]['item_id'].tolist()
    return [i for i in top_movies['item_id'].tolist() if i not in rated][:n]

print("\nComputing Precision@10 and Recall@10...")
pop_p,  pop_r  = precision_recall_at_k(pop_rec_ids,  test_ratings, k=10)
item_p, item_r = precision_recall_at_k(item_rec_ids, test_ratings, k=10)
svd_p,  svd_r  = precision_recall_at_k(svd_rec_ids,  test_ratings, k=10)

print(f"Popularity    — Precision@10: {pop_p:.4f}  | Recall@10: {pop_r:.4f}")
print(f"Item-Based CF — Precision@10: {item_p:.4f}  | Recall@10: {item_r:.4f}")
print(f"SVD           — Precision@10: {svd_p:.4f}  | Recall@10: {svd_r:.4f}")

# ── Final Comparison Table ─────────────────────────────────
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Model':          ['Global Mean Baseline', 'Item-Based CF', 'SVD (Matrix Factorization)'],
    'RMSE':           [round(baseline_rmse, 4), round(item_rmse, 4), round(svd_rmse, 4)],
    'MAE':            [round(baseline_mae, 4),  round(item_mae, 4),  round(svd_mae, 4)],
    'Precision@10':   [round(pop_p, 4),          round(item_p, 4),   round(svd_p, 4)],
    'Recall@10':      [round(pop_r, 4),           round(item_r, 4),   round(svd_r, 4)],
    'Personalized':   ['No', 'Yes', 'Yes'],
    'Handles Sparsity': ['N/A', 'Partially', 'Best']
})
print(comparison.to_string(index=False))

# ── Hybrid Summary ─────────────────────────────────────────
print("\n" + "=" * 70)
print("HYBRID APPROACH — How Production Systems Work")
print("=" * 70)
print("""
STAGE 1 — Candidate Generation: CF or SVD (~500 candidates fast)
STAGE 2 — Ranking: Content-Based + Context + Business Rules
STAGE 3 — Fallback: Popularity-Based (cold start)

Key insight: low RMSE != good recommendations
Optimize for the metric that matches the business goal.
""")

print("=" * 70)
print(f"COMPLETE JOURNEY — Days 25-29")
print("=" * 70)
print(f"""
Day 25: Sparsity 93.70% | Popularity baseline built
Day 26: Content-based filtering | Genre cosine similarity
Day 27: User-based CF | Pearson correlation | 10/943 strong neighbors
Day 28: Item-based CF (RMSE {item_rmse:.4f}) | SVD (k=50 latent factors)
Day 29: SVD wins on ranking — Precision@10: {svd_p:.4f}, Recall@10: {svd_r:.4f}
""")
