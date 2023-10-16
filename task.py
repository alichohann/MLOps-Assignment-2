import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing Dataset
boston_housing = datasets.load_boston()
data = pd.DataFrame(data=boston_housing.data,
                    columns=boston_housing.feature_names)
target = pd.Series(boston_housing.target, name='target')

# Select the feature and target variable
X = data.drop(['MEDV'], axis=1).values
y = target

# Split the data into training and testing sets
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest model
model = RandomForestRegressor()

# Train the model on the training data
model.fit(Xtr, ytr)

# Make predictions on the test data
y_pred = model.predict(Xts)

# Evaluate the model
mse = mean_squared_error(yts, y_pred)
r2 = r2_score(yts, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared (Coefficient of Determination): {r2:.2f}')

plt.scatter(yts, y_pred, color='black')
plt.plot([0, 50], [0, 50], color='blue', linewidth=3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest Regression on Boston Housing Dataset')
plt.show()
