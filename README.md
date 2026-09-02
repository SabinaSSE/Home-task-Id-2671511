# Disease Prediction using Machine Learning

## Overview
This project applies machine learning classification techniques to predict whether a tumor is malignant or benign, using the Breast Cancer Wisconsin (Diagnostic) dataset. It was completed as part of the Special Study in ICT coursework.

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, loaded directly via `sklearn.datasets.load_breast_cancer()`.
- **Size:** 569 patient records, 30 numerical features per record (e.g. mean radius, mean texture, mean smoothness, etc.).
- **Target classes:**
  - 0 = Malignant (212 cases)
  - 1 = Benign (357 cases)

## Preprocessing Steps
1. **Loading & inspection** — Loaded the dataset into a pandas DataFrame and checked its shape and class distribution.
2. **Missing value check** — Confirmed there were no missing values in the dataset.
3. **Train/test split** — Split the data into 80% training and 20% testing, using stratified sampling to preserve the original class ratio in both sets.
4. **Feature scaling** — Applied `StandardScaler` to standardize features (mean = 0, standard deviation = 1). This was necessary because raw feature values vary widely in scale (e.g. "mean area" is in the hundreds/thousands, while "mean smoothness" is a small decimal), which can bias distance- and margin-based models. Scaling was applied for Logistic Regression and SVM; Random Forest used unscaled data since tree-based models are not sensitive to feature scale.

## Models Used
Three classification algorithms were implemented and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Trained on scaled features; default hyperparameters |
| Random Forest | 100 trees (`n_estimators=100`); trained on unscaled features |
| SVM (RBF kernel) | Trained on scaled features; default hyperparameters |

No hyperparameter tuning (e.g. GridSearchCV) was performed — all models used default or minimally adjusted scikit-learn settings. Tuning is noted as a possible future improvement.

## Evaluation Metrics
Models were evaluated on the held-out 20% test set (114 samples) using Accuracy, Precision, Recall, F1-score, and Specificity.

| Model | Accuracy | Precision | Recall | F1-score | Specificity |
|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9762 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9286 |
| SVM (RBF) | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9762 |

Recall measures how well a model catches actual benign cases; Specificity measures how well it catches actual malignant cases — an important complementary metric in a medical context, where missing a real malignant case (a false negative) is especially costly.

**Observation:** Logistic Regression and SVM produced identical metrics because they made identical predictions on every test sample (verified by direct array comparison). This is plausible given the relatively small test set (114 samples) and the fact that both are margin/boundary-based methods trained on the same scaled features. Random Forest performed slightly lower but remained competitive.

### ROC Curve and AUC
An ROC curve was plotted for all three models to evaluate performance across all classification thresholds, not just the default 0.5 cutoff. The Area Under the Curve (AUC) summarizes each model's ability to separate the two classes into a single score (1.0 = perfect, 0.5 = random guessing):

| Model | AUC |
|---|---|
| Logistic Regression | 0.995 |
| Random Forest | 0.994 |
| SVM (RBF) | 0.995 |

All three models achieved very high AUC scores, confirming strong class separability in this dataset regardless of the specific algorithm used.

### Cross-Validation
To confirm the single train/test split result wasn't a fluke, 5-fold cross-validation was performed on all three models:

| Model | Mean Accuracy | Std. Deviation |
|---|---|---|
| Logistic Regression | 0.9807 | ±0.0065 |
| Random Forest | 0.9561 | ±0.0228 |
| SVM (RBF) | 0.9736 | ±0.0147 |

The low standard deviations show these results are stable across different data splits, not dependent on one lucky split. Note that SVM scores slightly lower than Logistic Regression under cross-validation, whereas on the original single split they tied exactly — this is expected minor variation, not a contradiction.

## Results & Figures
All plots are saved in the `results/` folder:
- `confusion_matrix_Logistic_Regression.png`
- `confusion_matrix_Random_Forest.png`
- `confusion_matrix_SVM.png`
- `model_comparison.png` — bar chart comparing Accuracy, Precision, Recall, F1-score, and Specificity across all three models.
- `roc_curve.png` — ROC curves for all three models with AUC scores, showing performance across all classification thresholds.

## Conclusion
All three models performed well on this dataset, with Logistic Regression and SVM achieving the highest scores (98.25% accuracy, AUC ~0.995). Cross-validation confirmed these results are stable rather than a product of one favorable train/test split. This suggests the dataset's classes are well-separated and that relatively simple models can achieve strong performance without extensive tuning. Future work could explore hyperparameter tuning and additional models (e.g. Gradient Boosting, k-NN) for further comparison.

## How to Run
Models were trained using default scikit-learn hyperparameters. Hyperparameter tuning (e.g., via GridSearchCV) was not performed but could further improve results.
1. Clone this repository.
2. Install dependencies: `pip install scikit-learn pandas matplotlib`
3. Open and run the Jupyter Notebook (`.ipynb`) file.
4. Generated plots will be saved automatically to the `results/` folder.
