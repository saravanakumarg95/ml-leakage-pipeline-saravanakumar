import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Task 1: Create a synthetic dataset with required features and target.
np.random.seed(42)
n_records = 80

area_sqft = np.random.randint(600, 3501, size=n_records)
num_bedrooms = np.random.randint(1, 6, size=n_records)
age_years = np.random.randint(0, 41, size=n_records)
noise = np.random.normal(0, 8, size=n_records)

price_lakhs = (
    0.06 * area_sqft
    + 7.5 * num_bedrooms
    - 0.5 * age_years
    + noise
)

housing_df = pd.DataFrame(
    {
        "area_sqft": area_sqft,
        "num_bedrooms": num_bedrooms,
        "age_years": age_years,
        "price_lakhs": price_lakhs,
    }
)

X = housing_df[["area_sqft", "num_bedrooms", "age_years"]]
y = housing_df["price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Total records in synthetic dataset: {len(housing_df)}")
print(f"Intercept: {model.intercept_:.4f}")
print("Feature coefficients:")
for feature_name, coefficient in zip(X.columns, model.coef_):
    print(f"  {feature_name}: {coefficient:.4f}")

y_pred = model.predict(X_test)

actual_vs_predicted = pd.DataFrame(
    {
        "actual_price_lakhs": y_test.values,
        "predicted_price_lakhs": y_pred,
    }
)

print("\nFirst five actual vs predicted values:")
print(actual_vs_predicted.head())


# Task 2: Evaluate model performance using MAE, RMSE, and R2.
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# MAE is the average absolute prediction error in lakhs; lower values mean closer predictions on average.
# RMSE also measures error in lakhs but penalizes large mistakes more strongly than MAE.
# R2 tells how much variance in price is explained by the model; values closer to 1 indicate better fit.


# Task 3: Compute residuals and plot their histogram.
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=12, edgecolor="black", color="#4C78A8")
plt.axvline(0, color="red", linestyle="--", linewidth=1)
plt.title("Histogram of Residuals")
plt.xlabel("Residual (Actual - Predicted) in Lakhs")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# A residual is the difference between actual and predicted value (actual - predicted).
# A roughly centered and symmetric histogram around zero suggests the model errors are mostly random.
# Strong skewness or heavy tails would suggest bias, outliers, or missing nonlinear relationships.