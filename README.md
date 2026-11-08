# Titanic Survival Prediction using Logistic Regression

A Machine Learning project that predicts whether a passenger survived the Titanic disaster using Logistic Regression. This project explores the complete ML workflow, from data exploration and preprocessing to model training, evaluation, and interpretation.

## Project Overview

The objective of this project is to build a binary classification model that predicts passenger survival based on selected passenger attributes.

The project uses the Titanic dataset and explores how features such as passenger class, sex, age, fare, and family information relate to survival.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Dataset

The project initially uses the Titanic dataset available through Seaborn.

```python
import seaborn as sns

df = sns.load_dataset('titanic')
```

The dataset contains passenger information, including age, sex, passenger class, fare, embarkation port, and survival status.

## Project Workflow

### 1. Data Exploration and Cleaning
- Inspected the dataset using `head()`, `info()`, and `describe()`.
- Identified missing values.
- Examined feature distributions and relationships.

### 2. Exploratory Data Analysis (EDA)

Used Seaborn and Matplotlib to visualize:
- Passenger survival distribution.
- Survival rates based on sex and passenger class.
- Age distribution among survivors and non-survivors.
- Relationships between numerical features.

### 3. Data Preprocessing

- Separated input features (X) and target variable (y).
- Handled missing numerical values using median imputation.
- Handled missing categorical values using most-frequent imputation.
- Applied StandardScaler to numerical features.
- Used OneHotEncoder for categorical features.
- Implemented ColumnTransformer and Pipeline to organize preprocessing and model training.

### 4. Model Training

Implemented Logistic Regression using Scikit-learn.

Two feature configurations were explored:

- **Full Model:** Used 7 selected input features.
- **Reduced Model:** Used only 3 features: `pclass`, `sex`, and `age`.

### 5. Model Evaluation

Evaluated model performance using:
- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision, Recall, and F1-score

## Results

| Model | Number of Features | Accuracy |
|---|---:|---:|
| Full Model | 7 | 80.45% |
| Reduced Model | 3 | 78.77% |

The reduced model achieved comparable accuracy despite using fewer features.

## Key Learnings

Through this project, I gained practical experience with:

- Data manipulation and exploration using Pandas.
- Exploratory Data Analysis using Seaborn.
- Handling missing values and categorical variables.
- Feature selection and preprocessing.
- Logistic Regression for binary classification.
- Building ML pipelines using Scikit-learn.
- Evaluating models beyond accuracy.
- Understanding Logistic Regression coefficients.

## Repository Structure

```text
Titanic-survival-prediction/
│
├── Titanic_Survival_Prediction_Logistic_Regression.ipynb
│
└── README.md
```

## Future Improvements

- Experiment with additional feature combinations.
- Perform hyperparameter tuning.
- Compare Logistic Regression with other classification algorithms.
- Test the workflow using the official Kaggle Titanic dataset.

## Conclusion

This project served as a practical application of supervised learning concepts, helping me understand the end-to-end process of building, evaluating, and interpreting a Machine Learning classification model.

**Built as part of my journey towards becoming an AI/ML Engineer.**
