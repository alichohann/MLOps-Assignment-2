import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing Dataset
boston_housing = datasets.load_boston()
data = pd.DataFrame(data=boston_housing.data, columns=boston_housing.feature_names)
target = pd.Series(boston_housing.target, name='target')

# Select the feature and target variable
X = data.drop(['MEDV'], axis=1).values
y = target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest model
model = RandomForestRegressor()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared (Coefficient of Determination): {r2:.2f}')

# Visualize the results (optional)
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='black')
plt.plot([0, 50], [0, 50], color='blue', linewidth=3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest Regression on Boston Housing Dataset')
plt.show()
