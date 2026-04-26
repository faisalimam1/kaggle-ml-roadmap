# 🤖 30-Day Kaggle ML Roadmap

> A structured, hands-on machine learning journey — from raw data to real Kaggle submissions.  
> **Status:** 🟢 Active

---

## 👨‍💻 About This Repository

I'm **Faisal Imam**, a final-year Computer Science student documenting my complete ML learning journey through real Kaggle competitions and datasets.

This is not a collection of copied tutorials. Every notebook here represents:
- Original exploratory data analysis
- Hand-crafted feature engineering
- Model building from scratch with full reasoning
- Real Kaggle submissions with public leaderboard scores

Each project follows a deliberate learning sequence — simpler concepts first, advanced techniques built on top.

---

## 📊 Roadmap Progress

| # | Dataset | Type | Concepts Covered | Best Score | Status |
|---|---------|------|-----------------|-----------|--------|
| 01 | [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) | Classification | EDA, Data Cleaning, Feature Engineering, Logistic Regression | - | ✅ Complete |
| 02 | [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand) | Regression | Linear Regression, Ridge, Lasso, XGBoost, RMSLE | **0.40794** | ✅ Complete |
| 03 | [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | NLP Classification | Text Processing, TF-IDF, Naive Bayes, Threshold Tuning, Cross Validation | **F1: 0.9343** | ✅ Complete |
| 04 | [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | Clustering | K-Means, Distance Metrics, Elbow Method | - | ✅ Complete |
| 05 | [MovieLens 100K](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset) | Recommendation | EDA, Sparsity Analysis, Popularity-Based, Content-Based, Collaborative Filtering, SVD | **Precision@10: 0.2306** | ✅ Complete |

---

## 🏆 Results & Scores

| Dataset | Best Model | Metric | Score |
|---------|-----------|--------|-------|
| Titanic | Logistic Regression | Accuracy | - |
| Bike Sharing | XGBoost | RMSLE | **0.40794** |
| SMS Spam | Naive Bayes (threshold=0.3) | F1 Score | **0.9343** |
| SMS Spam | Naive Bayes | Cross-Val F1 | **0.9099 (5-fold)** |
| MovieLens | SVD (Matrix Factorization) | Precision@10 | **0.2306** |

---

## 🚀 Live Demos

| Project | Live App | Description |
|---------|----------|-------------|
| SMS Spam Classifier | [▶️ Try it on Hugging Face](https://huggingface.co/spaces/faisalimam19/sms-spam-classifier) | Type any SMS message — get SPAM or HAM instantly |

---

## 📁 Repository Structure

```
kaggle-ml-roadmap/
│
├── 01_titanic/
│   ├── titanic.ipynb                  ← Full notebook: EDA → Feature Engineering → Model
│   └── titanic_submission.csv         ← Kaggle submission file
│
├── 02_bike_sharing/
│   ├── bikesharing.ipynb              ← Full notebook: EDA → XGBoost → Submission
│   └── bike_submission.csv            ← Kaggle submission (RMSLE: 0.40794)
│
├── 03_sms_spam/
│   └── sms_spam_complete.py           ← Complete pipeline: EDA → TF-IDF → NB → CV → Deployment
│
├── 04_customer_segmentation/          ← Check folder README for complete details
│
└── 05_movielens/
    └── movielens.py                   ← Complete pipeline: Days 25–29 | EDA → Content-Based → CF → SVD → Evaluation
```

---

## 🧠 What Each Project Covers

### 01 · Titanic — Classification Fundamentals
- Full EDA: survival rates by sex, class, age
- Data cleaning: median imputation, mode imputation
- Feature engineering: Title extraction, FamilySize, IsAlone, HasCabin
- Binary encoding and one-hot encoding
- Logistic Regression with reasoning

### 02 · Bike Sharing Demand — Regression + XGBoost
- Datetime parsing → extracted hour, month, year, day
- Target skewness analysis: 1.24 → -0.85 after log transform
- Caught and removed data leakage (`casual` + `registered` sum to target)
- Engineered `is_rush_hour`, `time_of_day`, `workday_rush` interaction feature
- Linear Regression baseline: **0.93 RMSLE**
- Ridge & Lasso: identified underfitting — regularization doesn't fix non-linearity
- XGBoost: **0.2882 validation RMSLE → 0.40794 public leaderboard**
- `workday_rush` (hand-engineered) ranked **#1 in XGBoost feature importance**

### 03 · SMS Spam Detection — NLP Classification ✅ Complete

**Dataset:** 5,572 real SMS messages | 86.6% ham | 13.4% spam

**🔗 Live Demo:** https://huggingface.co/spaces/faisalimam19/sms-spam-classifier

**Full Pipeline:**
- EDA — identified class imbalance (accuracy is misleading here)
- Key insight: spam messages are ~2x longer than ham — length is a signal
- Text preprocessing: lowercasing → punctuation removal → stopword removal → stemming
- Example: `"Congratulations! You've WON a FREE iPhone!!!"` → `"congratul won free iphon call now"`
- TF-IDF vectorization → each message converted into 3000 numerical features
- Stratified 80/20 train-test split (4,457 train | 1,115 test)
- Naive Bayes trained and threshold tuned (0.5 → 0.3)
- Logistic Regression compared — Naive Bayes wins (F1: 0.9343 vs 0.8593)
- 5-fold stratified cross validation confirms consistency
- Model deployed live on Hugging Face Spaces using Gradio

**Model Comparison:**

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Naive Bayes (default threshold = 0.5) | 99.20% | 83.22% | 0.9051 |
| **Naive Bayes (tuned threshold = 0.3)** | — | **90.60%** | **0.9343** |
| Logistic Regression | — | 75.84% | 0.8593 |

**Cross Validation:** Mean F1 = 0.9099 | Std Dev = 0.013 (5-fold stratified)

### 04 · Mall Customer Segmentation — Clustering ✅ Complete
- K-Means, Elbow Method, customer segment analysis
- Check the folder README for complete details

### 05 · MovieLens 100K — Recommendation System ✅ Complete

**Dataset:** 100,000 ratings | 943 users | 1,682 movies | Rating scale: 1–5 | Sparsity: 93.70%

**EDA + Popularity-Based Recommender**
- Analyzed rating distribution — identified positivity bias (ratings skewed toward 4 and 3)
- Studied user activity — identified cold-start risk for low-activity users
- Computed sparsity: **93.70%** — only 6 in every 100 user-movie pairs have any rating
- Built Weighted Rating Recommender using IMDb-style formula
- Top result: Schindler's List (1993) — weighted score: 4.39
- Key limitation: every user gets the same list — zero personalization

**Content-Based Filtering**
- Built 1,682 × 19 genre matrix — every movie as a numerical vector
- Computed 1,682 × 1,682 cosine similarity matrix
- Built user taste profiles — average genre vector of liked movies
- User 1 taste profile: Drama 0.479 | Comedy 0.301 | Action 0.239
- First real personalization: User 1 (drama/romance) vs User 200 (action/sci-fi) got completely different lists
- Key limitation: filter bubble — never recommends outside existing taste

**User-Based Collaborative Filtering**
- Built 943 × 1,682 user-item rating matrix
- Implemented Pearson correlation (adjusts for rating scale bias vs cosine)
- Only 10 out of 943 users had similarity > 0.5 with User 1
- Weighted prediction: neighbor similarity × neighbor rating / sum of similarities
- Top recommendation for User 1: Schindler's List (predicted 5.0)
- Key limitation: doesn't scale to millions of users, user taste drift over time

**Item-Based CF + Matrix Factorization (SVD)**
- Built item-item similarity on rating patterns (not genres) — captures community-driven similarity
- Item-Based CF RMSE: **0.9678** | MAE: **0.7573**
- SVD: decomposed rating matrix into 50 latent factors using scipy svds
- Predicted full rating matrix from U × Σ × Vᵀ decomposition
- Key insight: item relationships are stable over time — preferred over user-based in production

**Evaluation + Full Comparison**
- Held-out 20% test set for honest evaluation
- Implemented RMSE, MAE, Precision@10, Recall@10

| Model | RMSE | MAE | Precision@10 | Recall@10 | Personalized |
|-------|------|-----|-------------|----------|--------------|
| Global Mean Baseline | 1.1239 | 0.9420 | 0.0724 | 0.0516 | No |
| Item-Based CF | 0.9678 | 0.7573 | 0.0347 | 0.0111 | Yes |
| **SVD (Matrix Factorization)** | 2.4980 | 2.2418 | **0.2306** | **0.2162** | Yes |

**Key finding:** SVD had the worst RMSE but dominated every ranking metric. Low RMSE ≠ good recommendations. Optimize for the metric that matches the business goal.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightgrey?style=flat&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat)
![NLTK](https://img.shields.io/badge/NLTK-3.8-yellow?style=flat)
![Gradio](https://img.shields.io/badge/Gradio-Deployed-orange?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?style=flat&logo=huggingface)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-9cf?style=flat)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?style=flat&logo=kaggle)

---

## 💡 Key Learnings So Far

**On Feature Engineering:**
> The #1 most important feature in the Bike Sharing model (`workday_rush`) didn't exist in the raw data. It was created from an EDA observation. Models reward good features more than model complexity.

**On Model Selection:**
> Linear Regression, Ridge, and Lasso all scored ~0.93 RMSLE on Bike Sharing. XGBoost scored 0.28. The problem wasn't the features — it was the model class. Non-linear patterns need non-linear models.

**On Data Leakage:**
> `casual` and `registered` columns sum exactly to `count` (the target). Keeping them would have given a perfect training score and a completely broken real-world model. Always inspect your columns before training.

**On Overfitting to Validation:**
> Tuning improved validation RMSLE (0.2882 → 0.2765) but worsened Kaggle score (0.4079 → 0.4121). Validation on randomly split data doesn't simulate future time periods. Time-series cross-validation is the real fix.

**On Class Imbalance:**
> In the SMS Spam dataset, 86.6% of messages are ham. A model predicting ham every time scores 86.6% accuracy and catches zero spam. This is why F1 score is the right metric for imbalanced classification problems.

**On Text Preprocessing:**
> Raw text is noise. FREE, Free, and free are the same word. winning and winner carry the same meaning. Cleaning text before modeling isn't optional — it's what separates signal from noise.

**On Threshold Tuning:**
> Default Naive Bayes at threshold 0.5 scored F1 = 0.9051 and missed 25 spam messages. Lowering threshold to 0.3 pushed F1 to 0.9343 and recall from 83.22% to 90.60%. Default model settings are rarely optimal — tuning based on business context is what separates a working model from a great one.

**On Algorithm Fit:**
> Logistic Regression underperformed Naive Bayes (F1: 0.8593 vs 0.9343) on SMS Spam. The best-known algorithm isn't always the best fit — Naive Bayes was designed for word frequencies, which is exactly what TF-IDF produces.

**On Cross Validation:**
> One test split is never enough. 5-fold cross validation with Mean F1 = 0.9099 and Std Dev = 0.013 proves the model is consistent across all data — not just lucky on one split.

**On Model Deployment:**
> Building a model is only half the job. Wrapping it in a Gradio app and deploying it on Hugging Face Spaces makes it accessible to anyone in the world — no code, no installation required. A live demo is 10x more impressive than a GitHub link alone.

**On Sparsity in Recommendation Systems:**
> The MovieLens matrix is 93.70% empty — only 6 in every 100 user-movie pairs have a rating. This is the central challenge of recommendation systems: making confident predictions from almost no data. Every technique from Day 26 onward exists specifically to solve this.

**On Popularity Bias:**
> Raw average rating is not a reliable popularity metric. A movie with 5 ratings all at 5★ should not outrank a movie with 500 ratings at 4.5★. The Weighted Rating formula (used by IMDb) corrects for this by pulling low-vote scores toward the global mean — rewarding confidence, not just enthusiasm.

**On Evaluation Metrics:**
> SVD had the worst RMSE (2.50) but the best Precision@10 (0.2306) and Recall@10 (0.2162). RMSE measures rating prediction accuracy. Precision@K measures recommendation quality. They are not the same thing. Always optimize for the metric that matches the actual business goal.

**On Choosing the Right Similarity Metric:**
> Cosine similarity measures direction — right for genre vectors where magnitude doesn't matter. Pearson correlation adjusts for rating scale bias — right for user ratings where one user rates everything 5 and another rates everything 2 but they share identical taste. Picking the wrong metric gives misleading similarity scores.

**On Production Recommendation Systems:**
> No single algorithm wins everything. Production systems use a hybrid approach: SVD or CF for candidate generation, content-based for ranking and diversity, popularity-based as a cold-start fallback. Each weakness of one approach is covered by another's strength.

---

## 📝 Follow My Journey

I document each day's learning on LinkedIn with specific insights, code snippets, and results.

👉 [Connect with me on LinkedIn](https://www.linkedin.com/in/faisalimam19)

---

## 🎯 Goal
- Hands-on Kaggle experience across 5 problem types
- Strong intuition for EDA, feature engineering, and model selection
- Real leaderboard scores as proof of work
- Live deployed models anyone can interact with

---

*Updated regularly as the roadmap progresses. Star ⭐ the repo to follow along.*
