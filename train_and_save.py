from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle

# Load built-in dataset (no download needed)
data = fetch_california_housing()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build & train pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knnregressor", KNeighborsRegressor(n_neighbors=10))
])
pipeline.fit(X_train, y_train)

# Save the model
with open("california_knn_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved successfully!")
print(f"   Test R² score: {pipeline.score(X_test, y_test):.3f}")