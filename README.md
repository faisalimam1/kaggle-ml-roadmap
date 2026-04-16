# 🤖 30-Day Kaggle ML Roadmap

> A structured, hands-on machine learning journey, from raw data to real Kaggle submissions.  
> **Status:** 🟢 Active

---

## 👨‍💻 About This Repository

I'm **Faisal Imam**, a final-year Computer Science student documenting my complete ML learning journey through real Kaggle competitions and datasets.

This is not a collection of copied tutorials. Every notebook here represents:
- Original exploratory data analysis
- Hand-crafted feature engineering
- Model building from scratch with full reasoning
- Real Kaggle submissions with public leaderboard scores

Each project follows a deliberate learning sequence, simpler concepts first, advanced techniques built on top.

---

## 📊 Roadmap Progress

| # | Dataset | Type | Concepts Covered | Kaggle Score | Status |
|---|---------|------|-----------------|-------------|--------|
| 01 | [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) | Classification | EDA, Data Cleaning, Feature Engineering, Logistic Regression | - | ✅ Complete |
| 02 | [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand) | Regression | Linear Regression, Ridge, Lasso, XGBoost, RMSLE | **0.40794** | ✅ Complete |
| 03 | [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | NLP Classification | Text Processing, TF-IDF, Naive Bayes | - | 🟡 In Progress |
| 04 | [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | Clustering | K-Means, Distance Metrics, Elbow Method | - | ⏳ Upcoming |
| 05 | [MovieLens](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) | Recommendation | Collaborative Filtering, Similarity Metrics | - | ⏳ Upcoming |

---

## 🏆 Results & Scores

| Dataset | Best Model | Metric | Score |
|---------|-----------|--------|-------|
| Titanic | Logistic Regression | Accuracy | - |
| Bike Sharing | XGBoost | RMSLE | **0.40794** |
| SMS Spam | In Progress | F1 Score | - |

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
│   └── day8_preprocessing.py          ← EDA + full text preprocessing pipeline
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

**Day 8 — EDA + Text Preprocessing**
- Dataset: 5,572 real SMS messages | 86.6% ham | 13.4% spam
- Identified class imbalance problem → accuracy is a misleading metric here
- Key insight: spam messages are ~2x longer than ham messages — length is a signal
- Built complete text preprocessing pipeline from scratch:
  - Lowercasing → removes case inconsistency ("FREE" = "free")
  - Punctuation & number removal → eliminates noise
  - Stopword removal → strips words with zero signal ("the", "is", "a")
  - Stemming → reduces words to root form ("winning", "winner" → "win")
- Result: `"Congratulations! You've WON a FREE iPhone!!!"` → `"congratul won free iphon call now"`

**Coming up:**
- Day 9: TF-IDF vectorization + train-test split
- Day 10: Naive Bayes classifier + F1 evaluation

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
