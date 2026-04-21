# 🛍️ Mall Customer Segmentation
### Unsupervised Machine Learning | K-Means Clustering | Business Intelligence

> *"The algorithm didn't just find the 5 groups. The data always had them. The algorithm just confirmed what careful EDA already revealed."*

---

## 📌 Project Overview

A mall's marketing team has data on 200 customers but no way to target them meaningfully, sending the same campaign to a broke college student and a wealthy executive is wasted spend.

This project solves that.

Using **K-Means Clustering**, I segmented 200 mall customers into **5 distinct personas** based on their Annual Income and Spending Score — each with a tailored marketing strategy that the business can act on immediately.

This is not a classification problem. There are no labels, no correct answers, no ground truth. The challenge is discovering hidden structure in raw data and translating it into business decisions.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| Rows | 200 customers |
| Features | CustomerID, Gender, Age, Annual Income (k$), Spending Score (1–100) |
| Missing Values | None |
| Target | None — Unsupervised Learning |

**Spending Score** is a mall-assigned metric (1–100) based on purchase behavior and frequency. Higher = more active spender.

---

## 🧠 The Approach

```
Step 1: EDA          → Understand distributions, spot visual clusters
        ↓
Step 2: Preprocessing → Select features, scale with StandardScaler
        ↓
Step 3: Elbow Method  → Find optimal K mathematically
        ↓
Step 4: K-Means       → Train model, assign cluster labels
        ↓
Step 5: Interpretation → Name segments, generate business recommendations
```

---

## 🔍 Exploratory Data Analysis

Before running any algorithm, I performed a thorough EDA to understand the data and visually identify potential clusters.

**Key observations from EDA:**

- Most customers fall in the **₹40k–₹60k income** and **40–60 spending score** range — the "average" zone
- The **highest spenders** (score 80–99) appeared at both **low income (20–40k)** AND **high income (70–90k)** — two very different customer types with the same behavior for different reasons
- Surprisingly, **lowest spenders** also appeared in the same income zones — meaning income alone does not predict spending behavior
- Very high income customers (120k+) showed **bimodal spending** — either very high or very low, with almost no middle ground
- Cross-referencing **Age**: younger customers tend to spend more regardless of income; older customers with lower income tend to spend conservatively

> **Critical insight from EDA:** After staring at the Income vs Spending Score scatter plot, I visually identified **5 natural clusters** before running a single line of ML code. The algorithm's job was simply to confirm and formalize what the data was already showing.

---

## ⚙️ Preprocessing

### Why only Income and Spending Score?

| Feature | Decision | Reason |
|---|---|---|
| CustomerID | ❌ Dropped | Just an identifier, no signal |
| Gender | ❌ Dropped | Categorical — K-Means uses Euclidean distance, not meaningful for categories |
| Age | ❌ Dropped (for primary clustering) | Used for cross-analysis post-clustering |
| Annual Income | ✅ Used | Core business variable |
| Spending Score | ✅ Used | Core behavior variable |

### Why StandardScaler?

Income ranges from ~15k to ~137k. Spending Score ranges from 1 to 99.

Without scaling, income would **dominate every distance calculation** purely because its numbers are larger — not because it carries more information. StandardScaler converts both features to mean=0, std=1.

This is not optional preprocessing hygiene in K-Means. It's correctness.


## 📐 Finding Optimal K — The Elbow Method

K-Means requires you to specify K (number of clusters) upfront. The Elbow Method finds the optimal K by running the algorithm for K = 1 to 10 and plotting **WCSS (Within-Cluster Sum of Squares)** — a measure of how tightly packed clusters are.


**Result:** The elbow appears sharply at **K = 5** — WCSS drops significantly up to K=5, then flattens. Adding more clusters beyond 5 yields diminishing returns.

This mathematically confirmed the 5 groups I identified visually during EDA. When your intuition and your algorithm agree — that's a strong signal.

---

## 🤖 Model — K-Means Clustering


**Why k-means++?**
Standard K-Means places initial centroids randomly — bad luck can lead to poor final clusters. `k-means++` spreads initial centroids intelligently, giving consistently better results.

---

## 🎯 Results — The 5 Customer Segments

| Cluster | Segment Name | Avg Income | Avg Spending Score | Size |
|---|---|---|---|---|
| 0 | 😐 Average Joes | Mid (~55k) | Mid (~50) | Largest group |
| 1 | 💎 Target Customers | High (~87k) | High (~82) | Medium |
| 2 | 🔥 Impulsive Spenders | Low (~26k) | High (~79) | Medium |
| 3 | 💰 Cautious Rich | High (~87k) | Low (~18) | Medium |
| 4 | 💸 Budget Conscious | Low (~26k) | Low (~20) | Medium |

### Cluster Visualization

Each segment occupies a distinct region in Income vs Spending Score space — clean, non-overlapping, interpretable groups.

- **Top-right:** Target Customers (High/High)
- **Top-left:** Impulsive Spenders (Low/High)
- **Bottom-right:** Cautious Rich (High/Low)
- **Bottom-left:** Budget Conscious (Low/Low)
- **Center:** Average Joes (Mid/Mid)

---

## 💼 Business Recommendations

| Segment | Profile | Strategy |
|---|---|---|
| 💎 **Target Customers** | High income, high spending — already engaged | VIP loyalty programs, premium brand partnerships, exclusive early access |
| 💰 **Cautious Rich** ⭐ | High income, low spending — biggest opportunity | Luxury experiences, quality-over-quantity campaigns, personalized concierge offers |
| 🔥 **Impulsive Spenders** | Low income, high spending — emotionally driven | Flash sales, limited-time offers, FOMO marketing. Monitor for credit risk |
| 💸 **Budget Conscious** | Low income, low spending — value seekers | Combo deals, loyalty points, value bundles, discount events |
| 😐 **Average Joes** | Mid income, mid spending — the broad majority | Seasonal promotions, mid-range products, referral programs |

### ⭐ The Highest-Value Insight

**The Cautious Rich (Cluster 3) is the single most valuable segment for the business.**

They have high disposable income but low spending scores. The money is there — they're just not spending it in this mall. This is not an income problem. It's a marketing and experience problem.

Understanding *why* high earners don't spend — and removing that barrier — is potentially worth millions in additional revenue. No new customers needed.

---

## 🔬 Bonus Analysis — Age Cross-Reference

After labeling clusters, I independently cross-referenced **Age** against segments to find deeper behavioral patterns:

| Segment | Age Pattern | Interpretation |
|---|---|---|
| Target Customers | Younger (~32) | Young professionals, earning well, spending freely |
| Impulsive Spenders | Youngest (~25) | Lifestyle spending beyond income — FOMO-driven |
| Cautious Rich | Middle-aged (~41) | Wealthier but more conservative — experience over product |
| Budget Conscious | Older (~45) | Near retirement, fixed income mindset |
| Average Joes | Mixed (~43) | Broad working-age population |

This was not part of the original clustering — it was discovered by thinking beyond the model output. Age validates and enriches the segment profiles.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| Scikit-learn | StandardScaler, KMeans |
| Matplotlib | Elbow plot, scatter plots |
| Seaborn | Distribution plots, styled visualizations |
| Kaggle Notebooks | Development environment |

---

## 🧠 Key Learnings

**On Unsupervised Learning:**
> Without labels, the algorithm has no correct answer to chase. The goal shifts from prediction to discovery. Visual EDA is the most important step — it's the only way to sanity-check whether your clusters reflect real structure or random noise.

**On Feature Scaling:**
> StandardScaler is not optional preprocessing in K-Means. It's correctness. Unscaled features with larger numerical ranges will always dominate distance calculations — and your clusters will reflect the scale of your data, not the patterns in it.

**On the Elbow Method:**
> The optimal K is where adding one more cluster stops being worth the cost. Not the highest K (that just describes individuals), not the lowest K (that just creates chaos). The elbow is the last point where the trade-off is still in your favor.

**On Business Thinking:**
> The most valuable output of this project is not the cluster plot. It's the identification of the Cautious Rich — a segment with high income and low spending. That single insight reframes the marketing problem entirely: stop trying to attract new customers and start understanding why existing high-earners aren't spending.

---

## 👤 Author

**Faisal Imam** 

🔗 [LinkedIn](https://www.linkedin.com/in/faisalimam19) | 🏆 [Kaggle](https://www.kaggle.com) | 📁 [Full 30-Day Roadmap Repo](../)

---

*Part of the [30-Day Kaggle ML Roadmap](../) — building ML intuition through real data, real problems, real insights.*
