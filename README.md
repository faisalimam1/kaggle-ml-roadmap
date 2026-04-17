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
| 03 | [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | NLP Classification | Text Processing, TF-IDF, Naive Bayes, Threshold Tuning | **F1: 0.9343** | 🟡 In Progress |
| 04 | [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | Clustering | K-Means, Distance Metrics, Elbow Method | - | ⏳ Upcoming |
| 05 | [MovieLens](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) | Recommendation | Collaborative Filtering, Similarity Metrics | - | ⏳ Upcoming |

---

## 🏆 Results & Scores

| Dataset | Best Model | Metric | Score |
|---------|-----------|--------|-------|
| Titanic | Logistic Regression | Accuracy | - |
| Bike Sharing | XGBoost | RMSLE | **0.40794** |
| SMS Spam | Naive Bayes (threshold=0.3) | F1 Score | **0.9343** |

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
│   └── sms_spam_classifier.py         ← End-to-end pipeline: EDA → TF-IDF → NB → Tuning
│
├── 04_customer_segmentation/          ← Coming soon
└── 05_movielens/                      ← Coming soon
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

### 03 · SMS Spam Detection — NLP Classification (In Progress)

**Dataset:** 5,572 real SMS messages | 86.6% ham | 13.4% spam

**Pipeline built:**
- EDA — identified class imbalance (accuracy is a misleading metric here)
- Key insight: spam messages are ~2x longer than ham — length is a signal
- Text preprocessing pipeline:
  - Lowercasing → removes case inconsistency
  - Punctuation & number removal → eliminates noise
  - Stopword removal → strips words with zero signal
  - Stemming → reduces words to root form
- Example: `"Congratulations! You've WON a FREE iPhone!!!"` → `"congratul won free iphon call now"`
- TF-IDF vectorization → converted each message into 3000 numerical features
- Stratified 80/20 train-test split (4,457 train, 1,115 test)

**Model Comparison:**

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes (default threshold = 0.5) | 97.67% | 99.20% | 83.22% | 0.9051 |
| **Naive Bayes (tuned threshold = 0.3)** | — | — | **90.60%** | **0.9343** |
| Logistic Regression | — | — | 75.84% | 0.8593 |

**Key Finding:**
Threshold tuning from 0.5 → 0.3 improved F1 from 0.9051 → **0.9343** and caught significantly more spam. Default thresholds are rarely optimal for imbalanced business problems where recall matters more than precision.

**Coming up:**
- Day 10: Cross validation + model persistence + testing on custom SMS messages

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightgrey?style=flat&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat)
![NLTK](https://img.shields.io/badge/NLTK-3.8-yellow?style=flat)
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

**On Class Imbalance (NLP):**
> In the SMS Spam dataset, 86.6% of messages are ham. A model predicting "ham" every time scores 86.6% accuracy — and catches zero spam. This is why F1 score is the right metric for imbalanced classification problems.

**On Text Preprocessing:**
> Raw text is noise. "FREE", "Free", and "free" are the same word. "winning" and "winner" carry the same meaning. Cleaning text before modeling isn't optional — it's what separates signal from noise.

**On Threshold Tuning:**
> Default Naive Bayes at threshold 0.5 scored F1 = 0.9051 and missed 25 spam messages. Lowering threshold to 0.3 pushed F1 to 0.9343 and recall from 83.22% to 90.60%. Default model settings are rarely optimal for real business problems. Tuning the decision threshold based on business context (missing spam > false alarms) is what separates a working model from a great one.

**On Algorithm Fit:**
> Logistic Regression underperformed Naive Bayes (F1: 0.8593 vs 0.9343) on SMS Spam. The best-known algorithm isn't always the best fit — Naive Bayes was designed for word frequencies, which is exactly what TF-IDF produces.

---

## 📝 Follow My Journey

I document each day's learning on LinkedIn with specific insights, code snippets, and results.

👉 [Connect with me on LinkedIn](https://www.linkedin.com/in/faisalimam19)

---

## 🎯 Goal

Become job-ready for **AI/ML roles by June 2026** through:
- Hands-on Kaggle experience across 5 problem types
- Strong intuition for EDA, feature engineering, and model selection
- Real leaderboard scores as proof of work

---

*Updated regularly as the roadmap progresses. Star ⭐ the repo to follow along.*
