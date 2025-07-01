from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

# Load and train
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
pickle.dump(model, open("iris_model.pkl", "wb"))
print("✅ Model saved as 'iris_model.pkl'")
