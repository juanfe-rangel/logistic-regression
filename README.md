# Heart Disease Prediction with Logistic Regression

A machine learning project implementing logistic regression from scratch for heart disease prediction, featuring comprehensive exploratory data analysis, custom gradient descent optimization, decision boundary visualization, and L2 regularization techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Analysis](#running-the-analysis)
- [Model Performance](#model-performance)
- [Built With](#built-with)
- [Authors](#authors)

## Project Overview

This project implements a complete machine learning pipeline for cardiovascular disease prediction, including:

- **Exploratory Data Analysis (EDA)**: Statistical summaries, class distribution analysis, outlier detection using IQR method, and missing value handling
- **Data Preprocessing**: 70/30 stratified train-test split and StandardScaler normalization for feature consistency
- **Model Implementation**: Custom logistic regression built from scratch using NumPy (sigmoid function, binary cross-entropy cost, gradient descent optimization)
- **Visualization**: Decision boundary plots for 3 feature pairs showing linear separability and model behavior
- **Regularization**: L2 regularization (Ridge) with hyperparameter tuning across 5 lambda values to prevent overfitting
- **Model Interpretation**: Weight coefficient analysis and feature importance visualization

**Final Model Performance:**
- **Test Accuracy**: 85.19%
- **Test Precision**: 0.853
- **Test Recall**: 0.806
- **Test F1-Score**: 0.829
- **Training Iterations**: 1000 with learning rate α=0.01

## Dataset Description

**Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

**Dataset Characteristics:**
- **Samples**: 270 patients (after preprocessing)
- **Original Features**: 13 clinical and diagnostic attributes
- **Selected Features**: 6 (for model training)
- **Target**: Binary classification (1 = Heart Disease Presence, 0 = Absence)
- **Class Distribution**: 
  - Absence (0): 150 samples (55.6%)
  - Presence (1): 120 samples (44.4%)

**Selected Features for Modeling:**

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Age | Continuous | 29-77 years | Patient age |
| Cholesterol | Continuous | 126-564 mg/dL | Serum cholesterol level |
| BP | Continuous | 94-200 mmHg | Resting blood pressure |
| Max HR | Continuous | 71-202 bpm | Maximum heart rate achieved during exercise |
| ST depression | Continuous | 0-6.2 | Exercise-induced ST segment depression |
| Number of vessels fluro | Discrete | 0-3 | Number of major vessels colored by fluoroscopy |

**Additional Features (in dataset but not used for final model):**
- Sex (Gender: 0=Female, 1=Male)
- Chest pain type (4 categories)
- FBS over 120 (Fasting blood sugar > 120 mg/dL)
- EKG results (Resting electrocardiographic results)
- Exercise angina (Exercise-induced angina)
- Slope of ST (ST segment slope during peak exercise)
- Thallium (Thallium stress test results)

**Data Quality:**
- **Missing Values**: None detected in the dataset
- **Outliers**: Detected using IQR method across all numerical features
- **Data Balance**: Relatively balanced with 55.6% / 44.4% split

## Getting Started

These instructions will help you set up and run the project on your local machine for development, analysis, and experimentation.

### Prerequisites

Required software:
```
Python 3.8 or higher
Jupyter Notebook or JupyterLab
```

Required Python libraries:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-logistic-regression.git
   cd heart-disease-logistic-regression
   ```

2. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```

3. **Download the dataset**
   - Create a `data/` folder in the project directory
   - Download the heart disease dataset (CSV format)
   - Place it as `data/Heart_Disease_Prediction.csv`

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook heart_disease_lr_analysis.ipynb
   ```

## Project Structure

```
logistic-regression/
│
├── data/
│   └── Heart_Disease_Prediction.csv    # Dataset file (270 samples)
│
├── heart_disease_lr_analysis.ipynb     # Main Jupyter notebook
│
└── README.md                            # This file
```

## Running the Analysis

### Step 1: Data Exploration and Preprocessing

**Exploratory Data Analysis:**
- Load dataset and display basic statistics (mean, std, min, max for all features)
- Analyze class distribution with bar charts and pie charts
- Detect outliers using IQR method for all numerical features
- Check and handle missing values (median imputation for numerical, mode for categorical)

**Results:**
- Total samples: 270
- No missing values found
- Outliers detected in multiple features using IQR method

**Data Preparation:**
```python
# Feature selection
selected_features = ['Age', 'Cholesterol', 'BP', 'Max HR', 'ST depression', 'Number of vessels fluro']

# Train-test split (70/30, stratified)
Training set: 189 samples (70%)
Test set: 81 samples (30%)

# Feature normalization (StandardScaler)
Mean ≈ 0, Std ≈ 1 for all features
```

### Step 2: Model Training (Logistic Regression from Scratch)

**Implementation Components:**

```python
class LogisticRegression:
    def sigmoid(z):
        # Activation function: 1 / (1 + e^(-z))
    
    def compute_cost(y_true, y_pred):
        # Binary cross-entropy: -mean(y*log(ŷ) + (1-y)*log(1-ŷ))
    
    def fit(X, y):
        # Gradient descent optimization
        # For each iteration:
        #   1. Forward pass: ŷ = sigmoid(X·w + b)
        #   2. Compute cost
        #   3. Compute gradients: dw, db
        #   4. Update parameters: w -= α·dw, b -= α·db
    
    def predict(X, threshold=0.5):
        # Binary predictions using 0.5 threshold
```

**Training Configuration:**
- Learning rate (α): 0.01
- Iterations: 1000
- Cost reduction: 0.6931 → 0.4887 (29.5% decrease)

**Training Progress:**
```
Iteration 100/1000, Cost: 0.5945
Iteration 200/1000, Cost: 0.5507
...
Iteration 1000/1000, Cost: 0.4887
```

### Step 3: Model Evaluation

**Performance Metrics:**

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| Accuracy | 77.25% | **85.19%** |
| Precision | 0.789 | **0.853** |
| Recall | 0.667 | **0.806** |
| F1 Score | 0.723 | **0.829** |

**Confusion Matrix (Test Set):**
- True Positives (TP): 29
- False Positives (FP): 5
- True Negatives (TN): 40
- False Negatives (FN): 7

**Key Observation:** Test performance exceeds training performance, indicating excellent generalization without overfitting.

### Step 4: Feature Importance Analysis

**Learned Weights (Ranked by Absolute Magnitude):**

| Feature | Weight | Impact on Disease Risk |
|---------|--------|----------------------|
| Number of vessels fluro | +0.7631 | ↑ Increases (strongest predictor) |
| Max HR | -0.5774 | ↓ Decreases (protective factor) |
| ST depression | +0.5613 | ↑ Increases (risk factor) |
| BP | +0.1039 | ↑ Increases (mild risk) |
| Cholesterol | +0.0933 | ↑ Increases (mild risk) |
| Age | -0.0719 | ↓ Decreases (weak protective) |

**Bias (Intercept):** -0.1879

**Clinical Interpretation:**
- **Number of vessels colored by fluoroscopy** is the strongest predictor (weight = 0.76)
- **Maximum heart rate** is a protective factor (negative weight = -0.58)
- Higher ST depression correlates with disease presence (positive weight = 0.56)

### Step 5: Decision Boundary Visualization (2D Feature Pairs)

Three feature pairs were analyzed to visualize model separability:

**Pair 1: Age vs Cholesterol**
- Test Accuracy: 62.96%
- Separability: Moderate overlap between classes
- Observation: These two features alone provide limited discriminatory power

**Pair 2: BP vs Max HR**
- Test Accuracy: 72.84%
- Separability: Better clustering with clearer separation
- Observation: Max HR shows strong discriminative power

**Pair 3: ST Depression vs Number of Vessels Fluro**
- Test Accuracy: 77.78%
- Separability: **Strongest separation** among all pairs
- Observation: Both features are top predictors in the full model

**Key Findings:**
- All decision boundaries are **linear** (straight lines)
- 2D models have lower accuracy than full 6D model (85.19%)
- Demonstrates value of using multiple features together
- Logistic regression imposes linear constraint; nonlinear patterns cannot be captured

### Step 6: Regularization (L2 Ridge Regression)

**Regularization Implementation:**

```python
class LogisticRegressionL2:
    def compute_cost(y_true, y_pred, weights):
        # Cost = Cross-Entropy + (λ/(2m)) * ||w||²
        # L2 penalty term added to prevent overfitting
    
    def fit(X, y):
        # Gradient with regularization:
        # dw = normal_gradient + (λ/m) * w
        # Weight decay pulls weights toward zero
```

**Hyperparameter Tuning (Lambda Values Tested):**

| λ | Train Acc | Test Acc | Final Cost | Weight Norm ||w|| |
|---|-----------|----------|------------|-------------------|
| 0.000 | 0.7725 | **0.8519** | 0.4887 | 1.1205 |
| 0.001 | 0.7725 | **0.8519** | 0.4887 | 1.1205 |
| 0.010 | 0.7725 | **0.8519** | 0.4887 | 1.1203 |
| 0.100 | 0.7725 | **0.8519** | 0.4891 | 1.1183 |
| 1.000 | 0.7725 | **0.8519** | 0.4928 | 1.0988 |

**Regularization Effects:**
1. **Weight Magnitude Control:** As λ increases, ||w|| decreases (1.1205 → 1.0988, 1.9% reduction)
2. **Cost Increase:** Higher λ → higher final cost due to added L2 penalty
3. **Accuracy Stability:** Test accuracy remains stable at 85.19% across all λ values
4. **Optimal λ:** Any value in [0, 0.1] range performs well; model is robust to regularization

**Decision Boundary Comparison (BP vs Max HR):**
- Unregularized (λ=0): Test Acc = 72.84%, ||w|| = 0.842
- Regularized (λ=0.1): Test Acc = 72.84%, ||w|| = 0.840
- Boundaries are visually very similar; regularization has minimal impact on this dataset

## Model Performance

### Final Model Summary

**Best Configuration:** 
- Full 6D model with features: Age, Cholesterol, BP, Max HR, ST depression, Number of vessels
- Learning rate: 0.01
- Iterations: 1000
- Regularization: λ=0 or λ=0.01 (both achieve same performance)

**Test Set Performance:**
- **Accuracy**: 85.19%
- **Precision**: 85.3% (when model predicts disease, it's correct 85% of the time)
- **Recall**: 80.6% (model detects 80.6% of actual disease cases)
- **F1-Score**: 82.9% (harmonic mean of precision and recall)

**Confusion Matrix Interpretation:**
- **True Positives (29)**: Correctly identified heart disease patients
- **True Negatives (40)**: Correctly identified healthy patients
- **False Positives (5)**: Healthy patients incorrectly flagged as at-risk
- **False Negatives (7)**: At-risk patients missed by the model

**Clinical Implications:**
- High precision (85.3%) minimizes unnecessary patient anxiety from false alarms
- High recall (80.6%) ensures most at-risk patients are identified for intervention
- F1-score (82.9%) demonstrates balanced performance across both classes


## Built With

* **[Python 3.11](https://www.python.org/)**
* **[NumPy 2.4.1](https://numpy.org/)** 
* **[Pandas 3.0.0](https://pandas.pydata.org/)**
* **[Matplotlib 3.10.8](https://matplotlib.org/)** 
* **[Scikit-learn 1.6.0+](https://scikit-learn.org/)** 
* **[Jupyter Notebook](https://jupyter.org/)**

## Deployment


AWS SageMaker Deployment (AWS Academy Lab Environment)
This project includes preparation steps for deploying the model to AWS SageMaker to enable scalable, production-ready inference. However, the deployment was not executed due to restrictions imposed by the AWS Academy Learner Lab environment, which limits permissions for creating and managing SageMaker resources in educational settings.

Planned Deployment Workflow:

Model Serialization
The trained logistic regression model was intended to be exported, including weights, bias, and the feature scaler, and packaged in a format compatible with SageMaker.

Training Job Setup
A training job was planned using a custom scikit-learn estimator, with the instance type specified (e.g., ml.m5.large) and hyperparameters configured, such as learning rate, regularization parameter, and number of training iterations.

Endpoint Deployment
The process included defining the model endpoint configuration, deploying the model to a SageMaker endpoint, and setting up auto-scaling policies.

Inference Validation
The endpoint was planned to be tested using sample patient data, for example: input [Age=60, Cholesterol=300, …] producing an output probability of 0.68, indicating high risk.

Deployment Constraints
Status: Not deployed due to AWS Academy Lab restrictions.

## Authors

Juan Felipe Rangel Rodriguez
