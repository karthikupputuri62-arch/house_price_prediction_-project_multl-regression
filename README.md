# 🏠 House Price Prediction Using Linear Regression

**Author:** Karthik

---

## 📌 Project Overview

This project is a **Machine Learning regression project** that predicts the price of a house based on different property-related features.

The project uses the **Linear Regression** algorithm from Scikit-learn.

The complete project is implemented using **Object-Oriented Programming (OOP)** with a Python class named `MINI`.

The project performs the following major tasks:

1. Load the house dataset.
2. Perform date preprocessing.
3. Convert categorical columns into numerical values.
4. Separate input features (`X`) and target (`y`).
5. Split the dataset into training and testing data.
6. Train a Linear Regression model.
7. Generate training and testing predictions.
8. Calculate model accuracy manually using the R² formula.
9. Calculate model loss manually using Mean Squared Error.
10. Save the trained model using Pickle.
11. Load the saved model.
12. Predict the price of a new house.

---

# 🎯 Project Objective

The main objective of this project is to build a machine learning model that can predict **house prices** based on features such as:

* Number of bedrooms
* Number of bathrooms
* Living area
* Lot area
* Number of floors
* Waterfront
* View
* House condition
* Above-ground area
* Basement area
* Year built
* Year renovated
* City
* Country
* Day
* Month
* Year

The trained model can then be saved and reused to predict the price of a new property.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Pickle**
* **Object-Oriented Programming**

---

# 📦 Libraries Used

```python
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle
```

### Explanation

| Library            | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| `os`               | Used for operating-system related operations                 |
| `sys`              | Used for exception handling and obtaining error line numbers |
| `warnings`         | Used to suppress warning messages                            |
| `numpy`            | Numerical operations                                         |
| `pandas`           | Data loading and data manipulation                           |
| `matplotlib`       | Data visualization                                           |
| `sklearn`          | Machine learning library                                     |
| `train_test_split` | Splits data into training and testing sets                   |
| `LinearRegression` | Machine learning regression algorithm                        |
| `pickle`           | Saves and loads the trained model                            |

---

# 🏗️ Project Architecture

The project is designed using a Python class:

```python
class MINI:
```

The class contains several functions that perform different stages of the machine learning workflow.

```text
                    MINI CLASS
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   Data Loading     Model Training    Evaluation
        │                │                │
   __init__()      model_prediction()   results()
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                       manul_accuracy()       manual_loss()
                              │                     │
                              └──────────┬──────────┘
                                         │
                                  Model Persistence
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                         save_model()         load_model()
                                                   │
                                                   ↓
                                              sample()
```

---

# 📊 Dataset

The project expects a CSV file:

```text
data .csv
```

The dataset contains information about houses and their prices.

The first column is treated as the **target variable**, which is the house price.

The remaining columns are treated as **input features**.

---

# 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Read CSV File
     ↓
Date Conversion
     ↓
Extract Day / Month / Year
     ↓
Drop Original Date Column
     ↓
Convert Country to Numerical Value
     ↓
Convert City to Numerical Values
     ↓
Separate X and y
     ↓
Train-Test Split
     ↓
Linear Regression
     ↓
Training Prediction
     ↓
Testing Prediction
     ↓
Manual R² Calculation
     ↓
Manual MSE Calculation
     ↓
Save Model
     ↓
Load Model
     ↓
Predict New House Price
```

---

# 1️⃣ `__init__()` Function

The constructor is automatically executed when an object of the `MINI` class is created.

```python
def __init__(self,path):
```

It receives the path of the dataset.

Example:

```python
obj = MINI('data .csv')
```

---

## Step 1: Store Dataset Path

```python
self.path = path
```

The path provided by the user is stored inside the object.

For example:

```text
data .csv
```

---

## Step 2: Read CSV File

```python
self.df = pd.read_csv(self.path)
```

Pandas reads the CSV file and stores it as a DataFrame.

The DataFrame is stored in:

```python
self.df
```

---

# 📅 Step 3: Convert Date Column

```python
self.df['date'] = pd.to_datetime(
    self.df['date'],
    dayfirst=True
)
```

The `date` column is converted from text/string format into Pandas datetime format.

For example:

```text
22/04/2026
```

becomes a datetime value.

`dayfirst=True` means that the date is interpreted as:

```text
Day / Month / Year
```

instead of:

```text
Month / Day / Year
```

---

# 📅 Step 4: Extract Day

```python
self.df['day'] = self.df['date'].dt.day
```

This extracts the day from the date.

Example:

```text
22/04/2026
```

becomes:

```text
22
```

---

# 📅 Step 5: Extract Month

```python
self.df['month'] = self.df['date'].dt.month
```

This extracts the month.

Example:

```text
22/04/2026
```

becomes:

```text
4
```

---

# 📅 Step 6: Extract Year

```python
self.df['year'] = self.df['date'].dt.year
```

This extracts the year.

Example:

```text
22/04/2026
```

becomes:

```text
2026
```

Therefore:

```text
date = 22/04/2026
```

is transformed into:

```text
day   = 22
month = 4
year  = 2026
```

---

# 🗑️ Step 7: Drop Original Date Column

```python
self.df = self.df.drop(['date'], axis=1)
```

The original `date` column is removed because the model will use:

```text
day
month
year
```

instead of the original date.

`axis=1` means we are dropping a **column**.

---

# 🌎 Step 8: Convert Country to Numerical Value

```python
self.df['country'] = self.df['country'].map({
    'USA': 0
})
```

Machine learning algorithms require numerical values.

Therefore:

```text
USA → 0
```

is used.

---

# 🏙️ Step 9: Convert City to Numerical Values

The city column contains categorical values.

For example:

```text
Shoreline
Seattle
Kent
Bellevue
Redmond
```

The code converts them into numbers.

Example:

```text
Shoreline → 1
Seattle → 2
Kent → 3
Bellevue → 4
Redmond → 5
```

and so on.

This is done using:

```python
self.df['city'] = self.df['city'].map({...})
```

The purpose is to convert categorical text into numerical values so that Linear Regression can process them.

---

# 🔀 Step 10: Separate Input and Target

```python
self.X = self.df.iloc[:,1:]
self.y = self.df.iloc[:,0]
```

This is an important step.

### `X`

```python
self.X = self.df.iloc[:,1:]
```

takes all rows and all columns starting from the second column.

Therefore:

```text
X = Input Features
```

### `y`

```python
self.y = self.df.iloc[:,0]
```

takes the first column.

Therefore:

```text
y = Target Variable
```

In this project:

```text
X → House Features
y → House Price
```

---

# ✂️ Step 11: Train-Test Split

```python
self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    self.X,
    self.y,
    test_size=0.2,
    random_state=42
)
```

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

### `test_size=0.2`

Means 20% of the data is used for testing.

### `random_state=42`

Ensures that the same train-test split is obtained every time the program runs.

---

# 2️⃣ `model_prediction()` Function

```python
def model_prediction(self):
```

This function creates and trains the Linear Regression model.

---

## Create Model

```python
self.reg = LinearRegression()
```

A Linear Regression object is created.

---

## Train Model

```python
self.reg.fit(self.X_train, self.y_train)
```

The model learns the relationship between:

```text
X_train → House Features
y_train → House Prices
```

The model learns coefficients for the different features.

---

# 📈 Training Predictions

```python
self.y_train_predictions = self.reg.predict(self.X_train)
```

The trained model predicts prices for the training dataset.

The result is stored in:

```python
self.y_train_predictions
```

---

# 📈 Testing Predictions

```python
self.y_test_predictions = self.reg.predict(self.X_test)
```

The model predicts house prices for previously unseen testing data.

The result is stored in:

```python
self.y_test_predictions
```

---

# 3️⃣ `manul_accuracy()` Function

```python
def manul_accuracy(self,actual,predictions):
```

This function manually calculates the **R² score**.

Although the function is named `manul_accuracy`, technically this is **not classification accuracy**.

It calculates:

```text
R² = 1 - (Sum of Squared Errors / Total Sum of Squares)
```

---

## Step 1: Calculate Mean

```python
total = 0

for i in range(len(actual)):
    total = total + actual.iloc[i]

mean = total / len(actual)
```

This calculates the average of the actual house prices.

---

## Step 2: Calculate Numerator

```python
numerator = 0

for i in range(len(actual)):
    error = actual.iloc[i] - predictions[i]
    numerator = numerator + (error ** 2)
```

This calculates the **Sum of Squared Errors (SSE)**.

The error is:

```text
Actual Price - Predicted Price
```

Then the error is squared.

---

## Step 3: Calculate Denominator

```python
denominator = 0

for i in range(len(actual)):
    square = actual.iloc[i] - mean
    denominator = denominator + (square ** 2)
```

This calculates the **Total Sum of Squares (SST)**.

---

## Step 4: Calculate R²

```python
accuracy = 1 - (numerator / denominator)
```

This is the R² formula:

```text
R² = 1 - SSE / SST
```

A value closer to `1` indicates that the model explains a larger proportion of the variation in house prices.

---

# 4️⃣ `manual_loss()` Function

```python
def manual_loss(self,actual,predictions):
```

This function manually calculates the model's **Mean Squared Error (MSE)**.

---

## Calculate Prediction Error

```python
loss = actual.iloc[i] - predictions[i]
```

This calculates the difference between actual and predicted prices.

---

## Square the Error

```python
loss ** 2
```

Squaring ensures that negative and positive errors do not cancel each other.

---

## Add All Squared Errors

```python
numerator = numerator + (loss ** 2)
```

This gives the total squared error.

---

## Calculate Mean

```python
loss = numerator / len(actual - 1)
```

The intention of this function is to calculate the Mean Squared Error.

Conceptually:

```text
MSE = Sum of Squared Errors / Number of Observations
```

### ⚠️ Recommended improvement

For a standard MSE calculation, it would be clearer and safer to use:

```python
loss = numerator / len(actual)
```

Or use Scikit-learn:

```python
from sklearn.metrics import mean_squared_error

loss = mean_squared_error(actual, predictions)
```

---

# 5️⃣ `results()` Function

```python
def results(self):
```

This function calculates and displays the model's training and testing performance.

It calls:

```python
train_accuracy = self.manul_accuracy(
    self.y_train,
    self.y_train_predictions
)

test_accuracy = self.manul_accuracy(
    self.y_test,
    self.y_test_predictions
)
```

and:

```python
train_loss = self.manual_loss(
    self.y_train,
    self.y_train_predictions
)

test_loss = self.manual_loss(
    self.y_test,
    self.y_test_predictions
)
```

Finally, it prints:

```text
Train Accuracy
Train Loss
Test Accuracy
Test Loss
```

---

# 📊 Training vs Testing

The project evaluates the model using two datasets.

### Training Data

The model has already seen this data during training.

```text
X_train → Model → y_train_predictions
```

### Testing Data

The model has not seen this data during training.

```text
X_test → Model → y_test_predictions
```

Testing performance is particularly important because it gives an indication of how well the model may perform on unseen data.

---

# 6️⃣ `save_model()` Function

```python
def save_model(self):
```

This function saves the trained Linear Regression model into a Pickle file.

```python
with open("house.pkl", "wb") as f:
    pickle.dump(self.reg, f)
```

The file created is:

```text
house.pkl
```

### `wb`

Means:

```text
Write Binary
```

The trained model is serialized and stored in binary format.

---

# 💾 Why Save the Model?

Without saving the model, we would need to retrain it every time we wanted to make a prediction.

By saving the model:

```text
Train Once
    ↓
Save Model
    ↓
Load Model Later
    ↓
Make Predictions
```

This is useful when deploying machine learning applications.

---

# 7️⃣ `load_model()` Function

```python
def load_model(self):
```

This function loads the previously saved model.

```python
with open("house.pkl", "rb") as f:
    self.m = pickle.load(f)
```

### `rb`

Means:

```text
Read Binary
```

The loaded model is stored in:

```python
self.m
```

After loading, it can be used for prediction.

---

# 8️⃣ `sample()` Function

```python
def sample(
    self,
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
    city,
    country,
    day,
    month,
    year
):
```

This function accepts information about a new house and predicts its price.

---

## Input Features

The function accepts 17 values:

```text
1. bedrooms
2. bathrooms
3. sqft_living
4. sqft_lot
5. floors
6. waterfront
7. view
8. condition
9. sqft_above
10. sqft_basement
11. yr_built
12. yr_renovated
13. city
14. country
15. day
16. month
17. year
```

These features must be supplied in the **same order used during model training**.

---

# 🔮 Prediction

The following code performs the prediction:

```python
result = self.m.predict([[
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
    city,
    country,
    day,
    month,
    year
]])
```

The model returns the predicted house price.

Then:

```python
print(f'Result: {result[0]}')
```

displays the predicted result.

---

# 🧪 Example Prediction

The project calls:

```python
obj.sample(
    3,
    1.5,
    1340,
    7912,
    1.5,
    0,
    0,
    3,
    1340,
    0,
    1945,
    2005,
    2,
    0,
    22,
    4,
    2026
)
```

The values represent:

| Feature        | Value |
| -------------- | ----: |
| Bedrooms       |     3 |
| Bathrooms      |   1.5 |
| Sqft Living    |  1340 |
| Sqft Lot       |  7912 |
| Floors         |   1.5 |
| Waterfront     |     0 |
| View           |     0 |
| Condition      |     3 |
| Sqft Above     |  1340 |
| Sqft Basement  |     0 |
| Year Built     |  1945 |
| Year Renovated |  2005 |
| City           |     2 |
| Country        |     0 |
| Day            |    22 |
| Month          |     4 |
| Year           |  2026 |

The model then generates a predicted house price.

---

# 9️⃣ Main Function

The following code controls the complete workflow:

```python
if __name__ == '__main__':
```

This means the following code runs when the Python file is executed directly.

---

## Create Object

```python
obj = MINI('data .csv')
```

This creates an object of the `MINI` class.

The constructor:

```python
__init__()
```

is automatically called.

---

## Train Model

```python
obj.model_prediction()
```

This:

1. Creates Linear Regression.
2. Trains the model.
3. Generates training predictions.
4. Generates testing predictions.

---

## Display Results

```python
obj.results()
```

This calculates:

```text
Training R²
Testing R²
Training Loss
Testing Loss
```

and prints them.

---

## Save Model

```python
obj.save_model()
```

This creates:

```text
house.pkl
```

---

## Load Model

```python
obj.load_model()
```

This loads:

```text
house.pkl
```

into:

```python
self.m
```

---

## Make Prediction

```python
obj.sample(
    3, 1.5, 1340, 7912, 1.5,
    0, 0, 3, 1340, 0,
    1945, 2005, 2, 0,
    22, 4, 2026
)
```

This predicts the price of a new house.

---

# 📁 Recommended Project Structure

```text
House-Price-Prediction/
│
├── data .csv
│
├── house_price_prediction.py
│
├── house.pkl
│
├── requirements.txt
│
└── README.md
```

---

# 📦 requirements.txt

The project can use the following dependencies:

```text
numpy
pandas
matplotlib
scikit-learn
```

Pickle is part of Python's standard library, so it does not need to be separately installed.

---

# ▶️ How to Run the Project

## Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

## Step 2: Open the Project

```bash
cd House-Price-Prediction
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Make Sure the Dataset Exists

Place:

```text
data .csv
```

inside the project directory.

## Step 5: Run the Python Program

```bash
python house_price_prediction.py
```

---

# 📌 Important Data Processing Requirement

The same preprocessing used during training must also be applied to new prediction data.

For example:

```text
USA → 0
Seattle → 2
```

Therefore, when making a prediction, the model should receive the numerical representation expected during training.

The order of the 17 features must also remain unchanged.

---

# ⚠️ Important Note About City Encoding

The project currently uses:

```python
self.df['city'].map({...})
```

This manually assigns numbers to cities.

For example:

```text
Shoreline → 1
Seattle → 2
Kent → 3
```

This works technically, but Linear Regression may interpret these numbers as having an ordered numerical relationship.

For a more robust production machine learning project, **One-Hot Encoding** is generally preferable for nominal city categories.

For example:

```text
Seattle → [0,1,0,...]
Kent    → [0,0,1,...]
```

This prevents the model from assuming that:

```text
Seattle (2) > Shoreline (1)
```

has a meaningful mathematical relationship.

---

# ⚠️ Important Note About the `country` Feature

The code maps:

```python
USA → 0
```

Since the dataset contains only USA, this feature has no variation.

A feature with only one unique value does not provide useful information to Linear Regression and can generally be removed.

---

# 📈 Model Evaluation

The project manually calculates two important regression metrics.

### R² Score

```text
R² = 1 - SSE/SST
```

It measures how well the model explains the variation in house prices.

### Mean Squared Error

```text
MSE = Σ(Actual - Predicted)² / N
```

It measures the average squared prediction error.

---

# 🔍 Why Both Training and Testing Metrics?

The project calculates metrics on both datasets to identify possible **overfitting**.

For example:

```text
Training R²  → Very High
Testing R²   → Much Lower
```

This can indicate that the model performs well on training data but does not generalize well to unseen data.

A smaller difference between training and testing performance generally indicates better generalization.

---

# 🚀 Future Improvements

The project can be improved by:

* Using One-Hot Encoding for city.
* Removing constant features.
* Using `Pipeline` for preprocessing and modeling.
* Using `ColumnTransformer`.
* Adding proper missing-value handling.
* Detecting and handling outliers.
* Using cross-validation.
* Comparing multiple regression algorithms.
* Using Random Forest Regression.
* Using Gradient Boosting.
* Using XGBoost.
* Performing hyperparameter optimization.
* Adding visualizations.
* Building a Flask web application.
* Deploying the model online.
* Adding automated model evaluation.

---

# 💡 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Python programming
* Object-Oriented Programming
* Pandas DataFrame operations
* Date preprocessing
* Feature engineering
* Categorical encoding
* Train-test splitting
* Linear Regression
* Model training
* Prediction
* R² calculation
* Mean Squared Error
* Model serialization
* Pickle
* Exception handling
* Machine learning workflow

---

# 🏆 Conclusion

This **House Price Prediction** project demonstrates a complete basic machine learning workflow using **Linear Regression**.

The project starts with loading and preprocessing the dataset, converts categorical and date information into numerical features, separates independent and dependent variables, and divides the data into training and testing sets.

A Linear Regression model is then trained using the training data. Predictions are generated for both training and testing datasets, and model performance is evaluated using manually implemented **R² and Mean Squared Error calculations**.

Finally, the trained model is saved as:

```text
house.pkl
```

and loaded again to make predictions for new house information.

This project provides a strong foundation for developing a complete **machine learning-based house price prediction application**.

---

## 👨‍💻 Author

**Karthik**

### Project: House Price Prediction Using Linear Regression

---
