import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

train_df = pd.read_csv('task_one/train.csv')
test_df = pd.read_csv('task_one/test.csv')

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
X = train_df[features].copy()
X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
X = X.drop(['FullBath', 'HalfBath'], axis=1)
y = train_df['SalePrice']

X_test = test_df[features].copy()
X_test['TotalBath'] = X_test['FullBath'] + 0.5 * X_test['HalfBath']
X_test = X_test.drop(['FullBath', 'HalfBath'], axis=1)

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X_test)

submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': predictions})
submission.to_csv('task_one/submission.csv', index=False)

print('Model trained and predictions saved to task_one/submission.csv')

train_preds = model.predict(X)
plt.figure(figsize=(8,6))
plt.scatter(y, train_preds, alpha=0.5)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice (Training Set)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.show() 