import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load a sample dataset(Iris Flower Set)
data = load_iris()
X = data.data #Features (measurements)
y = data.target #Labels (flower types)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#Check Accuracy
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
# Save the trained model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")



