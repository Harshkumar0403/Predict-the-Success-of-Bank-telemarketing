# Predict-the-Success-of-Bank-telemarketing
The data is related with direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed

This project explores the performance of three machine learning models—**Logistic Regression**, **Random Forest**, and **CatBoost**—for a binary classification task. It includes preprocessing, feature selection, and evaluation using metrics like **precision**, **recall**, and **F1 score**. The project demonstrates how feature selection and hyperparameter tuning can enhance model performance.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Implementation Steps](#implementation-steps)
5. [Models and Hyperparameters](#models-and-hyperparameters)
6. [Results](#results)
7. [Usage](#usage)
8. [Future Enhancements](#future-enhancements)
9. [License](#license)

---

## **Project Overview**

The project focuses on binary classification and aims to:
- Compare the performance of **Logistic Regression**, **Random Forest**, and **CatBoost** classifiers.
- Implement feature selection using CatBoost to identify the most important features.
- Evaluate the models using classification metrics and confusion matrices.
- Improve model performance through class weighting and hyperparameter optimization.

---

## **Dependencies**

To run this project, ensure the following libraries are installed:

- **Python 3.8+**
- **Pandas**
- **Numpy**
- **Scikit-learn**
- **CatBoost**
- **Matplotlib**

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## **Dataset**

The dataset should include:
- **Features (`X`)**: A combination of numerical and categorical columns.
- **Target (`y`)**: Binary labels (`0` or `1`).

### **Preprocessing Steps:**
1. **Handling Missing Values:** 
   - Replace missing values in categorical columns with `'Unknown'`.
   - Fill numerical columns with the mean value.
2. **Encoding:** 
   - One-hot encoding for categorical columns.
   - Scaling numerical features using `MinMaxScaler`.
3. **Feature Selection:** 
   - Use CatBoost to identify the top 25 most important features.

---

## **Implementation Steps**

1. **Data Splitting:** 
   - Split the data into training (`80%`) and validation (`20%`) sets.
2. **Model Training:** 
   - Train **Logistic Regression**, **Random Forest**, and **CatBoost** models using specific hyperparameters.
3. **Feature Selection:** 
   - Use CatBoost's `get_feature_importance()` to select the top 25 features.
4. **Evaluation:** 
   - Evaluate all models using **classification reports**, **F1 scores**, and **confusion matrices**.

---

## **Models and Hyperparameters**

### **1. Logistic Regression**
- **Class Weights:** `[0.25, 0.75]`
- **Maximum Iterations:** `2000`

### **2. Random Forest**
- **Class Weights:** `{0: 0.25, 1: 0.75}`
- **Number of Estimators:** `100`
- **Maximum Depth:** `10`
- **Minimum Samples Split:** `35`
- **Minimum Samples Leaf:** `20`

### **3. CatBoost**
- **Class Weights:** `[0.25, 0.75]`
- **Learning Rate:** `0.05`
- **L2 Regularization:** `7`
- **Iterations:** `200`
- **Depth:** `8`
- **Bagging Temperature:** `0.5`

---

## **Results**

### **Evaluation Metrics**

| Model               | Precision (Class 1) | Recall (Class 1) | F1 Score | Accuracy |
|---------------------|---------------------|------------------|----------|----------|
| **Logistic Regression** | 0.73                | 0.78             | 0.75     | 86%      |
| **Random Forest**       | 0.73                | 0.81             | 0.76     | 86%      |
| **CatBoost**            | 0.74                | 0.84             | 0.77     | 86%      |

### **Confusion Matrices**

#### Logistic Regression
```
[[7392  925]
 [ 488  998]]
```

#### Random Forest
```
[[7312 1005]
 [ 403 1083]]
```

#### CatBoost
```
[[7237 1080]
 [ 304 1182]]
```

### **Feature Selection with CatBoost**
- After selecting the  CatBoost achieved:
  - **Precision (Class 1):** 0.74
  - **Recall (Class 1):** 0.84
  - **F1 Score:** 0.77

---

## **Usage**

### Clone the Repository:
```bash
git clone https://github.com/Harshkumar0403/Predict-the-Success-of-Bank-telemarketing.git
cd Predict-the-Success-of-Bank-telemarketing
```

### Run the Code:
- Update the dataset path in the script.
- Execute the main script:
```bash
python main.py
```

---

## **Key Takeaways**

1. **Logistic Regression**: Performs well as a baseline but struggles with recall for the minority class.
2. **Random Forest**: Balances precision and recall better than Logistic Regression but slightly underperforms compared to CatBoost.
3. **CatBoost**: Delivers the best recall and F1 score for the minority class, especially after feature selection.

---

## **Future Enhancements**

1. **Hyperparameter Tuning:** Use automated tools like Optuna for advanced tuning.
2. **Feature Engineering:** Explore interaction terms and polynomial features for Logistic Regression.
3. **Cross-Validation:** Implement cross-validation to ensure robust performance metrics.
4. **Explainability:** Use SHAP values to understand feature contributions better.

---

## **Acknowledgements**

- [CatBoost Documentation](https://catboost.ai/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- The open-source community for continuous inspiration and support.
