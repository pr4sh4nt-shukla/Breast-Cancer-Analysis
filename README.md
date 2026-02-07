# 🩺 Breast Cancer Classification - ML Model Comparison

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Boosting-XGBoost-red)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Overview

This project performs a comprehensive comparison of six machine learning classification models on the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The analysis evaluates model performance in diagnosing malignant and benign tumors, focusing on real-world medical diagnostic accuracy where precision and recall are critical.

The project demonstrates that **SVM achieves 98.25% accuracy with perfect precision (100%)**, making it the optimal model for minimizing false positives in cancer diagnosis—a crucial requirement in healthcare applications.

## 🛠️ Tech Stack

- **Machine Learning:** `scikit-learn`, `xgboost`
- **Data Processing:** `pandas`, `numpy`
- **Model Evaluation:** Classification metrics (Accuracy, Precision, Recall, F1-Score)
- **Development:** Jupyter Notebook

## 📊 Dataset Overview

- **Total Samples:** 569 patient records
- **Features:** 30 numerical features (mean, standard error, and worst values of cell characteristics)
- **Target Variable:** Binary diagnosis classification
  - **Benign (B):** 357 samples (62.7%)
  - **Malignant (M):** 212 samples (37.3%)
- **Train/Test Split:** 80/20 (455 training, 114 test samples)

## 🤖 Models Evaluated

Six state-of-the-art classification algorithms were trained and benchmarked:

1. **Support Vector Machine (SVM)** - Kernel-based classifier
2. **Logistic Regression** - Linear probabilistic model
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Random Forest** - Ensemble decision tree method
5. **Gradient Boosting** - Sequential boosting algorithm
6. **XGBoost** - Optimized gradient boosting framework

## 🏆 Performance Results

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|:----:|:------|:--------:|:---------:|:------:|:--------:|
| 🥇 | **SVM** | **98.25%** | **100.00%** | **95.35%** | **97.62%** |
| 🥈 | Logistic Regression | 97.37% | 97.62% | 95.35% | 96.47% |
| 🥉 | Random Forest | 96.49% | 97.56% | 93.02% | 95.24% |
| 4 | Gradient Boosting | 95.61% | 95.24% | 93.02% | 94.12% |
| 5 | XGBoost | 95.61% | 95.24% | 93.02% | 94.12% |
| 6 | KNN | 94.74% | 93.02% | 93.02% | 93.02% |

## 💡 Key Insights

### Model Performance Analysis

* **Champion (SVM):** Achieved **perfect precision (100%)**—zero false positives, meaning no healthy patients were incorrectly diagnosed with cancer.
* **Reliability Across Board:** All six models exceeded **94% accuracy**, demonstrating the dataset's suitability for ML classification.
* **Linear vs Ensemble:** Surprisingly, **Logistic Regression** (linear model) outperformed complex ensemble methods, suggesting the data is linearly separable.
* **Ensemble Performance:** Random Forest, Gradient Boosting, and XGBoost showed strong but slightly lower results, indicating potential overfitting on this specific dataset.

### Medical Diagnostic Context

In cancer diagnosis, two metrics are critical:

* **Precision (PPV):** Minimizes false alarms—important for patient mental health and healthcare cost reduction.
* **Recall (Sensitivity):** Ensures actual cancer cases aren't missed—critical for patient survival.

**SVM's perfect precision** makes it the ideal model for preliminary screening, while maintaining high recall (95.35%) ensures minimal missed diagnoses.

## 📈 Performance Metrics Explained

- **Accuracy:** Percentage of correct predictions (both benign and malignant)
- **Precision:** Of all positive predictions, how many were truly malignant (reduces false positives)
- **Recall:** Of all actual malignant cases, how many were correctly identified (reduces false negatives)
- **F1-Score:** Harmonic mean of precision and recall—balances both concerns

## 📂 Repository Structure

```
breast-cancer-ml-comparison/
│
├── model_comparison_on_breast_cancer_dataset.ipynb   # Main analysis notebook
├── requirements.txt                                   # Python dependencies
└── README.md                                          # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.11+
pip or conda package manager
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/breast-cancer-ml-comparison.git
cd breast-cancer-ml-comparison
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook
```bash
jupyter notebook model_comparison_on_breast_cancer_dataset.ipynb
```

## 📝 Usage

The notebook contains a complete ML pipeline:

1. **Data Loading:** Import breast cancer dataset from sklearn
2. **Exploratory Analysis:** Examine class distribution and feature statistics
3. **Data Preprocessing:** Feature scaling with StandardScaler
4. **Model Training:** Train all six classifiers
5. **Evaluation:** Compare models using multiple metrics
6. **Results Visualization:** Generate performance comparison tables

## 🎓 Conclusions

* **SVM is the optimal model** for this medical diagnostic task, achieving the best balance of accuracy and precision.
* **All models demonstrate >94% accuracy**, validating that machine learning is highly effective for breast cancer classification.
* **Perfect precision (SVM)** makes it deployment-ready for real-world medical screening applications.
* **Simple models (Logistic Regression) compete with complex ensembles**, suggesting data quality matters more than model complexity.

## 🔮 Future Enhancements

- [ ] Implement cross-validation for more robust performance estimates
- [ ] Add ROC-AUC curves and confusion matrices
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Feature importance analysis
- [ ] Deploy model as a web application using Flask/FastAPI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is publicly available and free for research purposes.

## 📧 Contact

**Prashant Shukla**  
📧 Email: prashantshukla8851@gmail.com
💼 LinkedIn: [Prashant Shukla](https://www.linkedin.com/in/prashant-shukla-58ba19373) 

**Project Link:** [https://github.com/pr4sh4nt-shukla/Breast-Cancer-Analysis](https://github.com/pr4sh4nt-shukla/Breast-Cancer-Analysis)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
