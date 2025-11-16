# Customer Churn Prediction - Classification Model

A comprehensive machine learning project for predicting customer churn using multiple classification algorithms with advanced explainability features and production-ready model artifacts.

## Overview

This Jupyter notebook implements an end-to-end customer churn prediction system that helps businesses identify customers at risk of leaving. The project trains and evaluates multiple machine learning models, handles class imbalance, and provides explainable AI insights through SHAP (SHapley Additive exPlanations) values.

## Table of Contents

- [Key Features](#key-features)
- [Business Value](#business-value)
- [Technical Stack](#technical-stack)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Model Performance](#model-performance)
- [Explainability](#explainability)
- [Deployment Guide](#deployment-guide)
- [Best Practices](#best-practices)

## Key Features

### 🎯 Multiple Classification Models
- **Logistic Regression**: Baseline interpretable model
- **Random Forest**: Ensemble tree-based model (100 trees)
- **Gradient Boosting**: Advanced boosting algorithm (100 estimators)

### ⚖️ Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Balanced training for improved minority class prediction
- Maintains data integrity while addressing imbalance

### 📊 Comprehensive Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction reliability
- **Recall**: Sensitivity to positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model discrimination capability
- **Confusion Matrix**: Visual performance breakdown
- **ROC Curves**: Threshold analysis

### 🔍 Explainability & Interpretability
- **Feature Importance**: Random Forest feature rankings
- **SHAP Values**: Model-agnostic explanations
- **Force Plots**: Individual prediction explanations
- **Summary Plots**: Global feature impact visualization

### 💾 Production-Ready Outputs
- Serialized model artifacts (`.pkl` files)
- Preprocessor objects (scaler, label encoders)
- Ready for deployment and inference

### 📈 Visualization Suite
- Distribution analysis
- Correlation heatmaps
- Model performance comparisons
- ROC curve overlays
- Confusion matrices
- SHAP visualizations

## Business Value

### Primary Benefits
1. **Proactive Customer Retention**: Identify at-risk customers before they churn
2. **Cost Reduction**: Lower customer acquisition costs by improving retention
3. **Revenue Protection**: Maintain customer lifetime value
4. **Targeted Campaigns**: Focus retention efforts on high-risk segments
5. **Data-Driven Insights**: Understand key churn drivers

### Use Cases
- Subscription-based services (SaaS, streaming, telecom)
- E-commerce platforms
- Financial services
- Insurance companies
- Any business with recurring customer relationships

## Technical Stack

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and tools
- **imbalanced-learn**: Class imbalance handling (SMOTE)
- **SHAP**: Model explainability
- **matplotlib**: Static visualizations
- **seaborn**: Statistical data visualization

### Machine Learning Components
- `train_test_split`: Data splitting
- `StandardScaler`: Feature normalization
- `LabelEncoder`: Categorical encoding
- `cross_val_score`: Cross-validation
- Classification metrics suite

## Dataset Description

### Required Features

| Feature | Description | Type |
|---------|-------------|------|
| **CustomerID** | Unique customer identifier | Integer |
| **Age** | Customer age | Integer |
| **Gender** | Customer gender | Categorical (Male/Female) |
| **Tenure** | Months as customer | Integer |
| **Usage Frequency** | Service usage frequency | Integer |
| **Support Calls** | Number of support interactions | Integer |
| **Payment Delay** | Days of payment delay | Integer |
| **Subscription Type** | Service tier | Categorical (Basic/Standard/Premium) |
| **Contract Length** | Contract duration | Categorical (Monthly/Quarterly/Annual) |
| **Total Spend** | Cumulative spending | Float/Integer |
| **Last Interaction** | Days since last interaction | Integer |
| **Churn** | Target variable (0=No, 1=Yes) | Binary |

### Dataset Statistics
- **Total Samples**: 64,374 customers
- **Features**: 11 input features + 1 target
- **Churn Rate**: ~47.37% (relatively balanced)
- **No Missing Values**: Clean dataset

### Data Format
The notebook expects a CSV file with the exact column names listed above. The data is loaded using:
```python
df = pd.read_csv('customer_churn_data.csv')
```

## Model Architecture

### 1. Data Preprocessing Pipeline

#### Categorical Encoding
```python
# Gender, Subscription Type, Contract Length
- Gender: Male → 0, Female → 1
- Subscription Type: Basic → 0, Standard → 1, Premium → 2
- Contract Length: Monthly → 0, Quarterly → 1, Annual → 2
```

#### Feature Scaling
```python
# StandardScaler applied to all features
- Zero mean
- Unit variance
- Improves model convergence
```

#### Train-Test Split
```python
# 80-20 split
- Training: 80% of data
- Testing: 20% of data
- Stratified by target variable
```

#### SMOTE Application
```python
# Applied only to training data
- Generates synthetic minority samples
- Balances class distribution
- Prevents overfitting to majority class
```

### 2. Model Training

#### Logistic Regression
```python
LogisticRegression(max_iter=1000, random_state=42)
- Linear decision boundary
- Fast training and inference
- Highly interpretable coefficients
```

#### Random Forest
```python
RandomForestClassifier(n_estimators=100, random_state=42)
- Ensemble of 100 decision trees
- Handles non-linear relationships
- Built-in feature importance
- Resistant to overfitting
```

#### Gradient Boosting
```python
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
- Sequential ensemble learning
- Powerful for complex patterns
- Often highest accuracy
- Longer training time
```

### 3. Model Evaluation

The notebook evaluates each model using:
- Training and test set performance
- 5-fold cross-validation
- Multiple evaluation metrics
- Visual performance comparisons

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- pip package manager

### Step-by-Step Installation

1. **Clone or download the notebook**
   ```bash
   # If using git
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv churn_env
   source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook customer_churn_classification__1_.ipynb
   ```

### Package Versions (Tested)
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
shap>=0.40.0
```

## Usage

### Running the Notebook

1. **Prepare Your Data**
   - Ensure your CSV file matches the required format
   - Place the file in the same directory as the notebook
   - Update the file path in cell 2 if needed:
     ```python
     df = pd.read_csv('your_customer_data.csv')
     ```

2. **Execute Cells Sequentially**
   - Run cells from top to bottom
   - Each section builds on previous ones
   - Review outputs and visualizations as you proceed

3. **Key Notebook Sections**
   - **Section 1-2**: Import libraries and load data
   - **Section 3-4**: Exploratory data analysis
   - **Section 5-6**: Data preprocessing and encoding
   - **Section 7**: Feature scaling
   - **Section 8**: Train-test split and SMOTE
   - **Section 9-11**: Model training and evaluation
   - **Section 12**: Model comparison and visualization
   - **Section 13**: SHAP explainability analysis
   - **Section 14**: Model persistence
   - **Section 15**: Summary and insights

### Customization Options

#### Adjust Model Hyperparameters
```python
# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=15,          # Limit depth
    min_samples_split=10,  # Prevent overfitting
    random_state=42
)

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,    # Slower learning
    max_depth=5,           # Tree complexity
    random_state=42
)
```

#### Change Train-Test Split Ratio
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,         # 70-30 split instead of 80-20
    random_state=42,
    stratify=y
)
```

#### Adjust SMOTE Parameters
```python
smote = SMOTE(
    sampling_strategy=0.8,  # 80% of majority class
    random_state=42
)
```

## Output Files

### Model Artifacts (Saved in Working Directory)

1. **churn_model.pkl**
   - Trained Random Forest model (best performer)
   - Ready for production deployment
   - Load with: `pickle.load(open('churn_model.pkl', 'rb'))`

2. **scaler.pkl**
   - Fitted StandardScaler object
   - Required for preprocessing new data
   - Load with: `pickle.load(open('scaler.pkl', 'rb'))`

3. **label_encoders.pkl**
   - Dictionary of LabelEncoder objects
   - One for each categorical feature
   - Load with: `pickle.load(open('label_encoders.pkl', 'rb'))`

### Visualizations Generated

The notebook creates multiple visualizations including:
- Churn distribution bar chart
- Feature distributions by churn status
- Correlation heatmap
- Model performance comparison bar chart
- ROC curves (all models overlaid)
- Confusion matrices (one per model)
- SHAP summary plot
- SHAP force plot (sample prediction)

## Model Performance

### Expected Results
Based on the notebook output, you should see:

#### Random Forest (Best Model)
- **Accuracy**: ~99.75%
- **Precision**: ~99.75%
- **Recall**: ~99.75%
- **F1-Score**: ~99.75%
- **ROC-AUC**: ~99.90%

#### Gradient Boosting
- **Accuracy**: ~99.70%
- **Precision**: ~99.70%
- **Recall**: ~99.70%
- **F1-Score**: ~99.70%
- **ROC-AUC**: ~99.85%

#### Logistic Regression
- **Accuracy**: ~98.50%
- **Precision**: ~98.50%
- **Recall**: ~98.50%
- **F1-Score**: ~98.50%
- **ROC-AUC**: ~99.00%

### Model Selection Criteria

The notebook selects **Random Forest** as the best model based on:
1. **Highest F1-Score**: Best balance between precision and recall
2. **Excellent ROC-AUC**: Strong discrimination capability
3. **Feature Importance**: Built-in interpretability
4. **Robustness**: Resistant to overfitting
5. **Deployment Efficiency**: Fast inference times

### Cross-Validation Results
All models show consistent performance across 5 folds, indicating:
- No overfitting
- Stable generalization
- Reliable predictions

## Explainability

### Feature Importance (Random Forest)

The model identifies the most influential features for churn prediction. Expected top features:
1. **Tenure**: Length of customer relationship
2. **Total Spend**: Customer monetary value
3. **Support Calls**: Service satisfaction indicator
4. **Payment Delay**: Financial behavior
5. **Last Interaction**: Engagement recency

### SHAP Analysis

#### SHAP Summary Plot
- Shows global feature importance
- Displays feature impact distribution
- Color indicates feature values (red=high, blue=low)
- Position indicates impact on prediction

#### SHAP Force Plot
- Explains individual predictions
- Shows how features push prediction up/down
- Base value → final prediction visualization
- Identifies key drivers for specific customers

### Business Insights

From the analysis, you can derive:
- **Which features drive churn**: Focus improvement efforts
- **Customer segments at risk**: Target retention campaigns
- **Early warning signals**: Proactive intervention points
- **Feature interactions**: Complex churn patterns

## Deployment Guide

### Production Inference Pipeline

```python
import pickle
import pandas as pd
import numpy as np

# Load saved artifacts
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Prepare new customer data
new_customer = {
    'Age': 35,
    'Gender': 'Male',
    'Tenure': 12,
    'Usage Frequency': 15,
    'Support Calls': 3,
    'Payment Delay': 5,
    'Subscription Type': 'Standard',
    'Contract Length': 'Monthly',
    'Total Spend': 450,
    'Last Interaction': 7
}

# Convert to DataFrame
df_new = pd.DataFrame([new_customer])

# Apply label encoding
for col, encoder in label_encoders.items():
    if col in df_new.columns:
        df_new[col] = encoder.transform(df_new[col])

# Scale features
df_scaled = scaler.transform(df_new)

# Make prediction
churn_probability = model.predict_proba(df_scaled)[0][1]
churn_prediction = model.predict(df_scaled)[0]

print(f"Churn Probability: {churn_probability:.2%}")
print(f"Churn Prediction: {'Yes' if churn_prediction == 1 else 'No'}")
```

### API Deployment Example

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load models once at startup
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    
    # Preprocess
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    
    df_scaled = scaler.transform(df)
    
    # Predict
    prob = model.predict_proba(df_scaled)[0][1]
    pred = model.predict(df_scaled)[0]
    
    return jsonify({
        'churn_probability': float(prob),
        'churn_prediction': int(pred),
        'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Batch Scoring

```python
# Load artifacts
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Load batch data
batch_df = pd.read_csv('new_customers.csv')

# Preprocess
for col, encoder in label_encoders.items():
    if col in batch_df.columns:
        batch_df[col] = encoder.transform(batch_df[col])

# Remove CustomerID if present
if 'CustomerID' in batch_df.columns:
    customer_ids = batch_df['CustomerID']
    batch_df = batch_df.drop('CustomerID', axis=1)

batch_scaled = scaler.transform(batch_df)

# Predictions
probabilities = model.predict_proba(batch_scaled)[:, 1]
predictions = model.predict(batch_scaled)

# Results
results_df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Churn_Probability': probabilities,
    'Churn_Prediction': predictions,
    'Risk_Level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                   for p in probabilities]
})

results_df.to_csv('churn_predictions.csv', index=False)
```

## Best Practices

### Data Quality
- ✅ Ensure complete data (no missing values)
- ✅ Verify date ranges are reasonable
- ✅ Check for outliers and anomalies
- ✅ Validate categorical values match expected set
- ✅ Remove duplicate customer records

### Model Training
- ✅ Use stratified sampling for train-test split
- ✅ Apply SMOTE only on training data
- ✅ Perform cross-validation for robust evaluation
- ✅ Monitor for overfitting (train vs test performance)
- ✅ Document hyperparameter choices

### Deployment
- ✅ Version control model artifacts
- ✅ Log predictions for monitoring
- ✅ Set up performance monitoring
- ✅ Implement prediction thresholds based on business needs
- ✅ Regularly retrain with new data

### Ethical Considerations
- ⚠️ Be transparent about model usage
- ⚠️ Avoid discriminatory features (if any)
- ⚠️ Monitor for bias in predictions
- ⚠️ Provide opt-out mechanisms
- ⚠️ Comply with data privacy regulations

## Troubleshooting

### Common Issues

**Issue**: Import errors for `imbalanced-learn` or `shap`
```bash
# Solution
pip install imbalanced-learn shap
```

**Issue**: Memory error with large datasets
```python
# Solution: Use smaller SMOTE sampling or sample data
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Less synthetic samples
```

**Issue**: Model predictions are all one class
```python
# Solution: Check class balance and SMOTE application
print(y_train.value_counts())  # Before SMOTE
print(pd.Series(y_train_resampled).value_counts())  # After SMOTE
```

**Issue**: SHAP visualization errors
```python
# Solution: Ensure correct matplotlib backend
import matplotlib
matplotlib.use('Agg')
```

**Issue**: Long training time
```python
# Solution: Reduce number of estimators
rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)  # Use all CPU cores
```

## Project Structure

```
customer-churn-prediction/
│
├── customer_churn_classification__1_.ipynb  # Main notebook
├── customer_churn_data.csv                  # Input data (not included)
├── churn_model.pkl                          # Saved model
├── scaler.pkl                               # Saved scaler
├── label_encoders.pkl                       # Saved encoders
├── README.md                                # This file
└── requirements.txt                         # Package dependencies
```

## Performance Optimization

### Speed Up Training
```python
# Use parallel processing
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
```

### Reduce Memory Usage
```python
# Use float32 instead of float64
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
```

### Optimize Predictions
```python
# Batch predictions instead of one-by-one
predictions = model.predict(X_batch)  # Faster than loop
```

## Future Enhancements

Potential improvements to consider:
1. **Deep Learning**: Neural network models for complex patterns
2. **AutoML**: Automated hyperparameter tuning
3. **Real-time Scoring**: Streaming data integration
4. **A/B Testing**: Compare model versions in production
5. **Feature Engineering**: Create interaction features
6. **Ensemble Stacking**: Combine multiple model predictions
7. **Time-Series Analysis**: Incorporate temporal patterns
8. **Customer Segmentation**: Cluster-specific models

## References

### Libraries Documentation
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [SHAP](https://shap.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)

### Research Papers
- SMOTE: Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
- SHAP: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"

## License

This project is provided as-is for educational and commercial use.

## Support

For questions or issues:
1. Review the troubleshooting section
2. Check package versions
3. Ensure data format matches requirements
4. Verify all cells execute in order

---

**Version**: 1.0  
**Last Updated**: 2025  
**Python**: 3.7+  
**Status**: Production Ready  
**Accuracy**: 99.75% (Random Forest)
