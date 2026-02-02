# Project Roadmap: Customer Segmentation & Recommendation System

[![GitHub repo size](https://img.shields.io/github/repo-size/wikiepeidia/Predict-customer-engagement-)](https://github.com/wikiepeidia/Predict-customer-engagement-)
[![GitHub last commit](https://img.shields.io/github/last-commit/wikiepeidia/Predict-customer-engagement-)](https://github.com/wikiepeidia/Predict-customer-engagement-/commits)
![Commit activity](https://img.shields.io/github/commit-activity/w/wikiepeidia/Predict-customer-engagement-)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Phase 1: Project Setup & Data Preparation

- [x] **Repository Init**
  - [x] Initialize Git repository
  - [x] Create `.gitignore` (add data/ folder, .env, .DS_Store)
  - [x] Create `README.md` with project scope
- [x] **Data Ingestion**
  - [x] Load `portfolio.csv`, `profile.csv`, and `transcript.csv`
  - [x] Check for missing values and duplicates
  - [x] Handle null values (especially in `profile.csv` income/gender)
- [x] **Data Merging & Cleaning**
  - [x] Transform `transcript.csv` to extract offer events vs. transaction events
  - [x] Merge datasets to create a single `customer_360_view` dataframe (One row per customer)

## Phase 2: Feature Engineering (Crucial for Clustering)

- [x] **Demographic Features**
  - [x] Encode categorical variables (Gender)
  - [x] Normalize/Bin numerical variables (Age, Income)
  - [x] Calculate "Membership Days" from `became_member_on`
- [x] **Behavioral Features (The "Preferences")**
  - [x] Calculate Total Transaction Amount
  - [x] Calculate Average Transaction Value
  - [x] Calculate Offer Completion Rate
  - [x] Calculate preference for specific channels (Mobile, Web, Social)
  - [x] Identify favorite offer types (BOGO vs. Discount)

## Phase 3: The Core Model - Clustering (Unsupervised Learning)

*Goal: Identify "Communities" (Groups A, B, C) based on behavior.*

- [x] **Preprocessing**
  - [x] Apply Feature Scaling (StandardScaler or MinMaxScaler)
  - [x] Dimensionality Reduction (PCA) - *Optional, if features are too noisy*
- [x] **Model Selection**
  - [x] Determine optimal K (Elbow Method & Silhouette Score)
  - [x] Train K-Means Clustering model
- [x] **Cluster Analysis (The "Understanding")**
  - [x] Append `cluster_label` to the main dataframe
  - [x] Visualise clusters (Scatter plots / T-SNE)
  - [x] **Profile the Clusters**:
    - [x] Interpret Group A (e.g., "High Spend, Low Frequency")
    - [x] Interpret Group B (e.g., "Discount Seekers")
    - [x] Interpret Group C (e.g., "Casuals")

## Phase 4: The Operational Model - Classification (Supervised Learning)

*Goal: The "Small Model" to assign NEW customers to existing clusters.*

- [x] **Dataset Preparation**
  - [x] Define X (Customer Features) and y (Cluster Label from Phase 3)
  - [x] Train/Test Split
- [x] **Model Training**
  - [x] Train Classifier (Random Forest, Gradient Boosting, or Logistic Regression)
  - [x] Optimize for Accuracy/F1-Score
- [x] **Save Models**
  - [x] Pickle/Save the Scaler
  - [x] Pickle/Save the Classifier

## Phase 5: Recommendation Logic & Inference

*Goal: "Give me a customer -> Get a Segment -> Get an Offer"*

- [ ] **Define Strategy**
  - [ ] Map Top Offer ID to each Cluster ID (e.g., Cluster 0 gets Offer X, Cluster 1 gets Offer Y)
- [ ] **Build Inference Pipeline**
  - [ ] Create function `predict_segment(new_customer_data)`
  - [ ] Create function `get_recommendation(segment_id)`
- [ ] **Final Output**
  - [ ] Demonstrate flow: New User Input -> Classifier -> Predicted Cluster -> Targeted Offer

## Phase 6: Documentation & Reporting

- [ ] Write final report on Cluster characteristics
- [ ] Document the "Business Questions" answered
- [ ] Finalize code cleanup
- [ ] Make a new notebook with Graphs for slides and report