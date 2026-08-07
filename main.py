# ============================================
# WEEK 2 - LINEAR REGRESSION
# California Housing Dataset
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("housing.csv")

print("Dataset loaded successfully!")

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())


# ============================================
# 2. UNDERSTAND THE DATASET
# ============================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# ============================================
# 3. HANDLE MISSING VALUES
# ============================================

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing numerical values with median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================
# 4. SELECT FEATURES AND TARGET
# ============================================

# Features
X = df[
    [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]
]

# Target
y = df["median_house_value"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================
# 5. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================
# 6. CREATE LINEAR REGRESSION MODEL
# ============================================

model = LinearRegression()


# ============================================
# 7. TRAIN MODEL
# ============================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================
# 8. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)


# ============================================
# 9. EVALUATE MODEL
# ============================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n================================")
print("MODEL EVALUATION")
print("================================")

print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)


# ============================================
# 10. MODEL COEFFICIENTS
# ============================================

print("\nModel Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)


# ============================================
# 11. ACTUAL VS PREDICTED VALUES
# ============================================

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))


# ============================================
# 12. VISUALIZATION
# ============================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")

plt.title("Actual vs Predicted House Values")

plt.show()


# ============================================
# 13. PREDICT A NEW HOUSE
# ============================================

new_house = pd.DataFrame({
    "longitude": [-122.23],
    "latitude": [37.88],
    "housing_median_age": [25],
    "total_rooms": [5000],
    "total_bedrooms": [1000],
    "population": [2000],
    "households": [800],
    "median_income": [5.0]
})

predicted_price = model.predict(new_house)

print("\n================================")
print("NEW HOUSE PREDICTION")
print("================================")

print(
    "Predicted House Value:",
    predicted_price[0]
)