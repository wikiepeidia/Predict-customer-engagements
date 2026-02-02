# GitHub Copilot Instructions for Predict-customer-engagements

## Project Context
This is a Customer Segmentation & Recommendation System project. The goal is to use K-Means Clustering to identify customer communities based on behavior data (likely Starbucks dataset).

## Architecture & Data Flow
- **Data Source**: `data/` directory containing:
  - `portfolio.csv`: Offer metadata (id, reward, difficulty, duration).
  - `profile.csv`: Customer demographics (age, gender, income).
  - `transcript.csv`: Transaction logs (event type, value).
- **Target Structure**: Create a `customer_360_view` dataframe (one row per customer) before modeling.
- **Phases**: Follow the roadmap in `README.md`:
  1. Data Prep (Cleaning, Merging).
  2. Feature Engineering (Demographics, Behavioral).
  3. Clustering (K-Means, Analysis).
  4. Recommendation System.

## Development Standards
- **Language**: Python 3.10+
- **Key Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
- **Code Structure**:
  - Prefer functional programming for data transformations to ensure reproducibility.
  - Use Jupyter Notebooks (`.ipynb`) for exploration and visualization.
  - Refactor stable logic into python scripts if the project grows (e.g., `src/preprocessing.py`).

## Specific Conventions
- **Data Cleaning**:
  - Pay special attention to null values in `profile.csv` (income, gender).
  - Clean `became_member_on` to calculate "Membership Days".
- **Feature Engineering**:
  - Create features for: Total Transaction Amount, Average Transaction Value, Offer Completion Rate.
  - One-hot encode categorical variables.
  - Normalize numerical variables before clustering.
- **Modeling**:
  - Use the Elbow Method and Silhouette Score to determine optimal K.
  - Visualize clusters using PCA or t-SNE 2D projections.

## User Interaction
- When generating code, reference the specific CSV file names located in `data/`.
- If suggesting a new step, check if it aligns with the current Phase in `README.md`.
