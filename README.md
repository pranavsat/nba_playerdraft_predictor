# 🏀 NBA Draft Prediction 

<p align="center">
  <img src="https://cdn.nba.com/manage/2024/01/GettyImages-2158847399-scaled-e1719459595813.jpg"
       alt="NBA Draft banner" width="700">
</p>

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## 📌 System Overview
This project develops machine learning models to predict the likelihood of a college basketball player being drafted into the NBA. It simulates a real-world ML workflow by covering **data cleaning, feature engineering, model training, evaluation, and reporting**, with a strong focus on handling **extreme class imbalance**.  

The primary business objective is to support **NBA scouts, analysts, and fans** with a data-driven tool for identifying high-potential draft candidates.  

Key Highlights:
- **End-to-End ML Pipeline**: From preprocessing to prediction.  
- **Multiple Models**: Logistic Regression, Random Forest, XGBoost, LightGBM.  
- **Imbalance Handling**: Comparative experiments with tuned models.  
- **Explainability**: Feature importance and coefficient analysis.  
- **Reproducibility**: Models stored as `.pkl` files with unified config paths.  

---

## 📑 Table of Contents
1. [⚡ Quickstart](#quickstart)  
2. [📊 Data Preparation](#data-preparation)  
3. [🧪 Model Evaluation](#model-evaluation)  
4. [💼 Business Impact](#business-impact)  
5. [📂 Repository Structure](#repository-structure)  
7. [🤝 Collaboration & Contributions](#collaboration--contributions)  
8. [🔮 Future Work](#future-work)  

---

## ⚡ Quickstart

### 1. Clone the repository and install dependencies: 
```bash
git clone https://github.com/tuannm3812/advmla_assignment_1
cd advmla_assignment_1
pip install -r requirements.txt
```

### 2. Install dependencies with Poetry
``` bash
poetry install
```

### 3. Run Jupyter notebooks
```bash
poetry run jupyter lab
```

### 4. Train models and generate results

Each model will:
- Preprocess the dataset
- Train default and tuned models
- Save trained models into the models/ folder
- Generate evaluation metrics and figures

### 5. Export predictions for Kaggle

After running the best model notebook, a `submission.csv` file will be created:
```bash
ls submission.csv
```

--- 
## 📊 Data Preparation
The dataset comes from the Kaggle “NBA Draft Prediction” competition, containing **14,774 training rows** (62 features) and **1,297 test rows** (61 features). The target variable is `drafted`, where 1 indicates a player was selected in the NBA draft and 0 otherwise.  

Key steps included:  
- **Handling Missing Values**:  
  - Median imputation for numerical features (e.g., height, weight).  
  - Mode imputation for categorical variables (e.g., team, conference).  
- **Outlier Treatment**:  
  - Quantile-based capping applied to features with extreme skew (e.g., rebounds, assists, shooting percentages).  
- **Feature Transformation**:  
  - One-hot encoding for categorical variables in Logistic Regression.  
  - Ordinal encoding for categorical variables in tree-based models.  
  - Standard scaling for numerical features in regression models.  
- **Data Splitting**:  
  - Stratified train-validation-test splits to preserve rare drafted class (~0.8%).  
- **Imbalanced Data**:  
  - Severe imbalance motivated emphasis on **recall, F1-score, and AUROC**, instead of accuracy.  

---

## 🧪 Model Evaluation
We implemented four modeling approaches, each with default and tuned configurations:  

1. **Logistic Regression**  
   - Establishes an interpretable baseline.  
   - Tuned via regularization parameter `C` using cross-validation.  

2. **Random Forest**  
   - Captures non-linear relationships and interactions.  
   - Tuned on `n_estimators`, `max_depth`, and `min_samples_split`.  

3. **XGBoost**  
   - Gradient boosting optimized for tabular data.  
   - Tuned hyperparameters: `learning_rate`, `max_depth`, `n_estimators`.  

4. **LightGBM**  
   - Efficient boosting model, faster than XGBoost.  
   - Tuned with **Hyperopt** for parameters such as `num_leaves`, `max_depth`, and `learning_rate`.  

Each model was saved in the `/models` folder (`.joblib`) for reproducibility.

---

## 📈 Achieved Results
**Key validation outcomes across models:**  

- **Logistic Regression (Tuned)**:  
  - ROC-AUC: **0.9979** | Precision: **0.9231** | Recall: **0.6316**  
  - Strong baseline with interpretability.  

- **Random Forest (Tuned)**:  
  - ROC-AUC: **0.9967** | Precision: **0.7778** | Recall: **0.3684**  
  - Conservative recall, better for ranking features than predicting drafted players.  

- **XGBoost (Tuned)**:  
  - ROC-AUC: **0.9953** | Precision: **0.7143** | Recall: **0.5263**  
  - Balanced trade-off between recall and precision.  

- **LightGBM (Tuned)**:  
  - ROC-AUC: **0.9970** | Precision: **0.5152** | Recall: **0.8947**  
  - Best recall, but higher false positives.  

**Unified Validation Comparison:**  
- Logistic Regression = Best **precision**.  
- XGBoost = Best **precision/recall balance**.  
- LightGBM = Best **recall**, but costly in terms of scouting resources.  

---

## 💼 Business Impact
- High recall ensures few talented players are missed, aligning with **NBA scouting priorities**.  
- False positives (lower precision) risk inefficient use of scouting resources.  
- **Recommended workflow**:  
  - Use **Logistic Regression** for transparency and shortlist reliability.  
  - Use **XGBoost** for balanced candidate identification.  
  - Use **LightGBM** as an early “broad net” filter.  

This layered approach supports scouts in efficiently prioritizing talent evaluation.

---

## 📂 Repository Structure
```plaintext
nba-draft-prediction/
│── data_raw/               # Original files (from Kaggle / course)
│── models/                 # Saved individual best models (.pkl / .joblib)
│── notebooks/              # Individual experiments
│   ├── ManhTuan_Nguyen  
│   ├── Ashley_Wang
│   ├── Pranav_Sathyababu
│   ├── Weishu_Sun
│── requirements.txt        # Dependencies
│── README.md               # Project overview
```
---

## 🤝 Collaboration & Contributions

- **Manh Tuan Nguyen (25739083):** Implemented Logistic Regression, Random Forest, XGBoost, and LightGBM notebooks. Led evaluation, tuning, and unified performance comparison.
- **Weishu Sun (25721325):** Developed Logistic Regression, XGBoost, and LightGBM models. Proposed SMOTEENN with XGBoost to improve recall.
- **Ashley Wang (14121545):** Built full pipeline with SMOTE + Logistic Regression, achieving highest Kaggle score (0.99914).
- **Pranav Sathyababu (25588726):** Worked on Logistic Regression, Random Forest, XGBoost, and LightGBM models. Focused on feature engineering and interpretability.

---

## 🔮 Future Work
- **Threshold Tuning:** Optimize cut-off for drafted classification.
- **Ensemble Models:** Stack Logistic Regression, XGBoost, and LightGBM.
- **Resampling:** Explore advanced methods beyond SMOTE/SMOTEENN.
- **Explainability:** Use SHAP/LIME for clearer insights.
- **Deployment:** Build a scouting dashboard for probability-based player rankings.
