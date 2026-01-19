# Iris Flower Classification using Machine Learning


---

## Project Overview

This project presents an end-to-end **machine learning pipeline** built on the classic **Iris flower dataset**.  
The work systematically covers **data loading, preprocessing, exploratory data analysis (EDA), statistical testing, outlier handling, model training, evaluation, and model persistence**.  
The primary objective is to understand feature behavior across species and identify the **best-performing classification model** based on quantitative evaluation metrics.

---

## Dataset Description

The Iris dataset contains **150 observations** equally distributed among three flower species: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*.  
Each observation consists of four numerical features representing flower morphology:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The dataset is well-balanced, contains **no missing values**, and is suitable for both statistical analysis and supervised learning tasks.

---

## Data Preprocessing

The dataset was extracted from a compressed archive and loaded into a Pandas DataFrame.  
The identifier column (`Id`) was removed as it does not contribute to predictive modeling.  
Data types were verified, and the dataset was confirmed to be clean with **zero null values**.

Descriptive statistics including **mean, standard deviation, minimum, maximum, and quartiles** were computed to understand feature distributions and variability.

---

## Exploratory Data Analysis (EDA)

Exploratory analysis was conducted to study the distribution and relationships of features.  
Species counts confirmed a perfectly balanced dataset.  
Pairwise relationships between numerical features revealed that **petal-related features exhibit strong positive correlations**, while sepal width shows weaker or negative correlations with other features.

Histogram and kernel density estimation (KDE) plots demonstrated that **petal features provide strong separability between species**, whereas sepal features show partial overlap.

---

## Outlier Detection and Data Cleaning

Outliers were identified using the **Interquartile Range (IQR) method**.  
A small number of outliers were detected primarily in the **Sepal Width** feature.  
These observations were removed to improve data consistency, and the DataFrame index was reset.

Post-cleaning analysis confirmed stable feature distributions with minimal impact on class balance.

---

## Statistical Analysis

Several statistical tests were applied to validate assumptions and compare group means:

- **Shapiro–Wilk Test** was used to assess normality, and results indicated that the data does not significantly deviate from a normal distribution.
- **Levene’s Test** revealed unequal variances across species, suggesting heteroscedasticity.
- **Independent t-Test** showed a statistically significant difference in petal length between *Iris-setosa* and *Iris-versicolor*.
- **One-Way ANOVA** confirmed highly significant differences in petal width across all three species.

These results support the hypothesis that species are distinguishable based on morphological features.

---

## Machine Learning Models

Multiple supervised classification algorithms were trained and evaluated using a standardized pipeline with feature scaling:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  

Each model was evaluated using **accuracy, precision, recall, and F1-score** on a held-out test set.

---

## Model Evaluation and Selection

Among all tested models, the **Support Vector Machine (SVM)** achieved the highest performance with an accuracy of approximately **96.7%**.  
It demonstrated strong generalization capability and balanced performance across all species classes.

## Classification Results

The final classification report shows:

- Perfect precision and recall for *Iris-setosa*
- High predictive performance for *Iris-versicolor* and *Iris-virginica*
- Minimal misclassification and strong overall consistency

Macro and weighted averages further confirm the robustness of the selected model.

---

## Technologies Used

- Python  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- SciPy  
- Scikit-learn  
- Joblib  

---

## How to Run the Project

Install dependencies and run the notebook:

```bash
pip install -r requirements.txt
jupyter notebook
The trained SVM pipeline was serialized and saved as:

