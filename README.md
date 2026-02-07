# Breast Cancer Dataset - Machine Learning Model Comparison

A comprehensive comparison of six machine learning classification models on the breast cancer dataset, evaluating their performance in diagnosing malignant and benign tumors.

## Dataset Overview

- **Total Samples**: 569
- **Features**: 30 numerical features
- **Target Variable**: Diagnosis (Binary Classification)
  - **B (Benign)**: 357 samples (62.7%)
  - **M (Malignant)**: 212 samples (37.3%)
- **Data Split**:
  - Training Set: 455 samples (80%)
  - Test Set: 114 samples (20%)

## Models Evaluated

Six popular machine learning algorithms were trained and evaluated:

1. Support Vector Machine (SVM)
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. Random Forest
5. Gradient Boosting
6. XGBoost

## Results Summary

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|--------|----------|
| 1 | **SVM** | **98.25%** | **100.00%** | **95.35%** | **97.62%** |
| 2 | Logistic Regression | 97.37% | 97.62% | 95.35% | 96.47% |
| 3 | Random Forest | 96.49% | 97.56% | 93.02% | 95.24% |
| 4 | Gradient Boosting | 95.61% | 95.24% | 93.02% | 94.12% |
| 5 | XGBoost | 95.61% | 95.24% | 93.02% | 94.12% |
| 6 | KNN | 94.74% | 93.02% | 93.02% | 93.02% |

## Key Findings

- **Best Overall Performance**: SVM achieved the highest accuracy (98.25%) with perfect precision (100%)
- **Strong Runner-up**: Logistic Regression performed nearly as well with 97.37% accuracy
- **All Models Effective**: Even the lowest-performing model (KNN) achieved 94.74% accuracy
- **Precision vs Recall**: SVM's perfect precision indicates zero false positives, crucial in medical diagnosis
- **Ensemble Methods**: Random Forest, Gradient Boosting, and XGBoost showed strong but slightly lower performance

## Performance Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of all positive predictions, how many were actually positive (important to minimize false positives)
- **Recall**: Of all actual positives, how many were correctly identified (important to minimize false negatives)
- **F1-Score**: Harmonic mean of precision and recall, providing a balanced measure

## Medical Context

In breast cancer diagnosis:
- **High Precision** reduces false alarms (healthy patients incorrectly diagnosed as having cancer)
- **High Recall** ensures actual cancer cases are not missed
- SVM's perfect precision makes it particularly valuable in this context

## Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Jupyter Notebook

## Usage

The Jupyter notebook contains the complete analysis including:
- Data loading and exploration
- Data preprocessing
- Model training
- Performance evaluation
- Results visualization

## Conclusion

All six models demonstrated strong performance on the breast cancer dataset, with SVM emerging as the top performer. The high accuracy across all models (>94%) suggests that this dataset is well-suited for machine learning classification, and these models could serve as valuable decision-support tools in medical diagnosis.

## License

This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset, which is publicly available.

## Author

Model comparison analysis on breast cancer dataset for binary classification.
