# Home-task-Id-2671511
Breast cancer prediction using Logistic Regression, Random Forest, and SVM in scikit-learn.
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
Models were evaluated on the held-out 20% test set (114 samples) using Accuracy, Precision, Recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 |
| SVM (RBF) | 0.9825 | 0.9861 | 0.9861 | 0.9861 |

**Observation:** Logistic Regression and SVM produced identical metrics because they made identical predictions on every test sample (verified by direct array comparison). This is plausible given the relatively small test set (114 samples) and the fact that both are margin/boundary-based methods trained on the same scaled features. Random Forest performed slightly lower but remained competitive.

## Results & Figures
All plots are saved in the `results/` folder:
- `confusion_matrix_Logistic_Regression.png`
- `confusion_matrix_Random_Forest.png`
- `confusion_matrix_SVM.png`
- `model_comparison.png` — bar chart comparing Accuracy, Precision, Recall, and F1-score across all three models.

## Conclusion
All three models performed well on this dataset, with Logistic Regression and SVM achieving the highest scores (98.25% accuracy). This suggests the dataset's classes are well-separated and that relatively simple models can achieve strong performance without extensive tuning. Future work could explore hyperparameter tuning and additional models (e.g. Gradient Boosting, k-NN) for further comparison.

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install scikit-learn pandas matplotlib`
3. Open and run the Jupyter Notebook (`.ipynb`) file.
4. Generated plots will be saved automatically to the `results/` folder.
