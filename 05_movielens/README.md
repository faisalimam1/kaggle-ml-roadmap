# 🎬 MovieLens 100K — Recommendation System

> **Project Type:** Recommendation System  
> **Dataset:** MovieLens 100K (University of Minnesota)  
> **Status:** ✅ Complete  

---

## 📌 What This Project Is About

Every time Netflix suggests a show or Spotify generates a playlist, a recommendation system is at work. The core problem is simple to state but hard to solve:

> *"Given what I know about this user, what should I show them next?"*

This project builds a complete recommendation system from scratch — starting from a raw ratings file, progressing through four different algorithmic approaches, and ending with a rigorous evaluation that reveals a counterintuitive result: **the model with the worst error score gave the best recommendations.**

---

## 📂 File Structure

```
05_movielens/
│
├── movielens.py       ← Complete pipeline: Days 25–29
└── README.md          ← This file
```

The entire project lives in a single growing Python file. Each day's work is clearly separated by a section header so you can follow the progression from baseline to advanced.

---

## 🗂️ Dataset

**Source:** [MovieLens 100K on Kaggle](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)

| Property | Value |
|----------|-------|
| Total ratings | 100,000 |
| Unique users | 943 |
| Unique movies | 1,682 |
| Rating scale | 1 to 5 (integer) |
| Sparsity | **93.70%** |

**Files used:**

| File | Format | Contents |
|------|--------|----------|
| `u.data` | Tab-separated | user_id, item_id, rating, timestamp |
| `u.item` | Pipe-separated | item_id, title + 19 binary genre flags |

> **What sparsity means in practice:** For every 100 possible user-movie pairs, only 6 have any rating at all. The matrix is almost completely empty — and this is the central challenge that every algorithm in this project is designed to address.

---

## 🗺️ Project Roadmap

```
Day 25 → Understand the problem space (EDA + Popularity Baseline)
Day 26 → First personalization (Content-Based Filtering)
Day 27 → Community-driven recommendations (User-Based CF)
Day 28 → Stable item patterns + latent factors (Item-Based CF + SVD)
Day 29 → Honest evaluation + full comparison
```

---

## 📅 Day-by-Day Breakdown

### EDA + Popularity-Based Recommender

**Goal:** Understand the data before building anything. Establish a baseline every future model must beat.

**Key findings from EDA:**

| Analysis | Finding | Implication |
|----------|---------|-------------|
| Rating distribution | Ratings 3 and 4 dominate, very few 1s | Positivity bias — users pre-select movies they expect to like |
| Ratings per user | Right-skewed, some users rated 700+ | Low-activity users are hard to personalize for (cold start) |
| Ratings per movie | Most movies rated by very few users | Long tail problem — obscure films have almost no signal |
| Sparsity | **93.70%** | Core challenge — 6 in 100 pairs have any rating |

**The Weighted Rating Formula (IMDb method):**

```
WR = (v / (v + m)) × R  +  (m / (v + m)) × C

v = number of votes for the movie
m = minimum votes threshold (50th percentile)
R = movie's average rating
C = global mean rating across all movies
```

Why not raw average? A movie with 5 ratings all at 5★ should not beat a movie with 500 ratings averaging 4.5★. The formula pulls low-vote scores toward the global mean — rewarding confidence, not just enthusiasm.

**Top 3 results:**

| Rank | Movie | Weighted Score |
|------|-------|---------------|
| 🥇 | Schindler's List (1993) | 4.39 |
| 🥈 | The Shawshank Redemption (1994) | 4.37 |
| 🥉 | Casablanca (1942) | 4.36 |

**Ceiling:** Every user gets this exact same list. Zero personalization.

---

### Content-Based Filtering

**Goal:** Personalize recommendations using item features — without looking at any other users.

**The idea:** Represent every movie as a 19-dimensional genre vector. Build a user taste profile by averaging the genre vectors of movies they liked. Find movies most similar to that taste profile using cosine similarity.

**Why cosine similarity, not euclidean distance?**  
Euclidean distance measures magnitude — a movie with 8 genres would always seem far from a movie with 2 genres even if they share the same ones. Cosine only measures direction (genre overlap), not how many genres a movie has.

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
Result: 1.0 = identical, 0.0 = no overlap
```

**User taste profiles (real results):**

| User | Drama | Comedy | Action | Top Recommendation |
|------|-------|--------|--------|--------------------|
| User 1 | 0.479 | 0.301 | 0.239 | Something to Talk About (Comedy/Drama/Romance) |
| User 200 | low | low | high | Ben-Hur (Action/Adventure/Drama) |

Same dataset. Same number of liked movies. Completely different lists. First real personalization.

**Limitation — The Filter Bubble:**  
Content-based can only recommend things similar to what the user already liked. A user who only rated action movies will never see a documentary — even if they'd love it. This is the primary motivation for collaborative filtering.

---

### User-Based Collaborative Filtering

**Goal:** Find users who think like you — then recommend what they loved that you haven't seen.

**The idea:** No item features needed. Just rating patterns. If User A and User B have rated 50 movies almost identically, they likely share taste. Recommend to User A what User B loved but User A hasn't seen.

**Why Pearson correlation, not cosine similarity?**  
User A gives everything 4-5 stars. User B gives everything 1-2 stars. They may have identical taste (both prefer the same movies relative to their own scale) but cosine similarity would call them different. Pearson centers ratings around each user's mean before comparing — it captures relative preference, not absolute scores.

```
pearson(A, B) = Σ[(Ra - R̄a)(Rb - R̄b)] / √[Σ(Ra - R̄a)² × Σ(Rb - R̄b)²]
Result: +1 = perfect agreement, 0 = no correlation, -1 = perfect disagreement
```

**Weighted prediction formula:**
```
predicted_rating(u, m) = Σ[sim(u,n) × rating(n,m)] / Σsim(u,n)

Neighbors with higher similarity get proportionally more influence.
Neighbors with negative similarity are excluded entirely.
```

**Real results for User 1:**
- Only **10 out of 943 users** had Pearson similarity > 0.5
- Top recommendation: Schindler's List (predicted rating: 5.0)

**Limitation:** Computing similarity for 943 users is fine. At 10 million users it becomes computationally impossible. User tastes also shift over time — old ratings may no longer reflect current preferences.

---

### Item-Based CF + Matrix Factorization (SVD)

**Goal:** Build more stable, scalable collaborative filtering. Then graduate to learned latent factors.

#### Part A — Item-Based Collaborative Filtering

**The idea:** Instead of finding users similar to you, find movies that are consistently rated similarly across all users. Then use your own ratings on similar movies to predict how you'd rate a new one.

**Key difference from Day 26 (content-based):**
- Day 26 cosine similarity: computed on 19-dimensional **genre vectors**
- Day 28 cosine similarity: computed on 943-dimensional **user rating vectors**

Two movies are similar not because they share genres, but because the **same users rated them similarly**. Completely different signal — and often a better one.

**Why item-based beats user-based in production:**

| Criterion | User-Based | Item-Based |
|-----------|-----------|-----------|
| Stability | User tastes change | Item relationships are stable |
| Scalability | Millions of users = slow | Fewer items than users typically |
| Precomputation | Hard to cache | Compute once, reuse forever |

**RMSE: 0.9678 | MAE: 0.7573** (evaluated on 19,969 test ratings)

#### Part B — Matrix Factorization (SVD)

**The idea:** Don't compare rows or columns directly. Decompose the entire rating matrix into hidden (latent) factors that explain why users rate things the way they do.

```
R ≈ U × Σ × Vᵀ

R = original (943 × 1682) sparse rating matrix
U = user latent factors  (943 × k)
Σ = singular values       (k × k diagonal)
V = item latent factors  (1682 × k)
k = 50 latent factors used
```

**What are latent factors?** Hidden taste dimensions that emerge from the data — never explicitly named. One factor might capture "preference for slow artistic films," another "love of action blockbusters." A predicted rating is the dot product of a user's factor scores and a movie's factor scores.

**Implementation note:** The `scikit-surprise` library conflicts with NumPy 2.0 on Kaggle. This project implements SVD using `scipy.sparse.linalg.svds` — fully compatible and mathematically equivalent.

---

### Evaluation + Full Comparison

**Goal:** Honestly measure how good each model actually is. Reveal what the numbers really mean.

**Train/Test split:** 80,000 training ratings | 20,000 held-out test ratings

#### Rating Prediction Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| RMSE | √(mean((predicted − actual)²)) | Prediction accuracy — penalizes large errors more |
| MAE | mean(\|predicted − actual\|) | Prediction accuracy — equal weight to all errors |

#### Ranking Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Precision@K | Relevant in top K / K | Quality of your recommendation list |
| Recall@K | Relevant in top K / Total relevant | Completeness — how many good items did you surface |

**"Relevant" = movies the user actually rated ≥ 4 in the test set**

#### Final Results

| Model | RMSE | MAE | Precision@10 | Recall@10 | Personalized |
|-------|------|-----|-------------|----------|--------------|
| Global Mean Baseline | 1.1239 | 0.9420 | 0.0724 | 0.0516 | No |
| Item-Based CF | 0.9678 | 0.7573 | 0.0347 | 0.0111 | Yes |
| **SVD (Matrix Factorization)** | 2.4980 | 2.2418 | **0.2306** | **0.2162** | Yes |

#### The Counterintuitive Finding

SVD had the **worst RMSE** but **dominated every ranking metric**.

How? RMSE measures how close your predicted rating number is to the actual rating. Precision@K measures whether the movies you recommend are actually ones the user loves. A model can be mediocre at predicting exact star ratings yet excellent at identifying which movies belong at the top of the list.

**The lesson:** The metric you optimize must match the business goal. Netflix doesn't care if it predicted 3.8 instead of 4.2. It cares if you watched the movie.

---

## 🏗️ How Production Systems Actually Work

No single algorithm wins everything. Real systems (Netflix, Spotify, Amazon) use a hybrid:

```
STAGE 1 — Candidate Generation
  └── SVD or Collaborative Filtering
      Retrieve ~500 broadly relevant candidates fast

STAGE 2 — Ranking
  └── Content-Based + User Context + Business Rules
      Score and rank the 500 candidates → show top 10

STAGE 3 — Cold Start Fallback
  └── Popularity-Based
      New user with no history? Show what everyone loves.
```

Each weakness of one approach is covered by another's strength:
- CF alone → filter bubble, cold start problem
- Content alone → never discovers new genres  
- Popularity alone → zero personalization
- SVD alone → cold start problem
- **Combined → each gap is filled**

---

## 📊 Algorithm Summary

| Algorithm | Personalized | Needs Ratings | Needs Features | Cold Start | Filter Bubble |
|-----------|-------------|--------------|----------------|-----------|--------------|
| Popularity-Based | No | Aggregate only | No | ✅ Handles it | Extreme |
| Content-Based | Yes | For taste profile | Yes (critical) | New items: ✅ | Yes |
| User-Based CF | Yes | Yes (critical) | No | ❌ Problem | Less severe |
| Item-Based CF | Yes | Yes (critical) | No | ❌ Problem | Less severe |
| SVD | Yes | Yes (critical) | No | ❌ Problem | Least severe |

---

## ⚙️ How to Run

**Requirements:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

No special installation needed. All libraries are standard and compatible with NumPy 2.0.

**Update the data paths** at the top of `movielens.py` to match your Kaggle dataset location:
```python
ratings = pd.read_csv('path/to/u.data', ...)
movies  = pd.read_csv('path/to/u.item', ...)
```

**Run the full pipeline:**
```bash
python movielens.py
```

Each day's section runs sequentially. Variables from earlier days (like `ratings`, `movies`, `user_item_matrix`) are reused by later days — run the file top to bottom.

---

## 🎯 Key Takeaways

**On sparsity:**
93.70% of the rating matrix is empty. Every algorithm in this project exists to make good predictions despite that emptiness.

**On metric choice:**
RMSE and Precision@K measure different things. SVD had the worst RMSE and the best Precision@10. Always choose the metric that matches the actual business goal — not the one that looks best on paper.

**On similarity metrics:**
Use cosine similarity for genre vectors (direction matters, not magnitude). Use Pearson correlation for user ratings (adjusts for individual rating scale bias).

**On production reality:**
No single algorithm wins in production. A hybrid of popularity-based (cold start) + collaborative filtering (candidate generation) + content-based (ranking) outperforms any individual approach.

**On complexity:**
Item-Based CF outperforms User-Based CF in production not because it's more mathematically sophisticated, but because item relationships are stable over time and the similarity matrix can be precomputed once and reused indefinitely.

---

## 🔗 Links

- 📁 [Full Repository](https://github.com/faisalimam1/kaggle-ml-roadmap)
- 📊 [Dataset on Kaggle](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
- 💼 [LinkedIn](https://www.linkedin.com/in/faisalimam19)

---

*Part of the 30-Day Kaggle ML Roadmap by Faisal Imam*
