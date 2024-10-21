pip install scikit-learn
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Sample dataset (You can replace this with your dataset)
data = {
    'Height': [5.1, 5.9, 5.5, 5.7, 6.0, 5.3, 5.8, 6.2, 6.1, 5.0],
    'Weight': [62, 72, 55, 68, 80, 50, 75, 90, 85, 52],
    'Gender': [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]  # 0: Female, 1: Male
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['Height', 'Weight']]  # Features (Height and Weight)
y = df['Gender']  # Target variable (Gender)

# Split dataset into training set and test set (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling: KNN uses distance calculations, so it's good to scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)  # k is the number of neighbors (here k=3)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = knn.predict(X_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Step 3: Display the predictions and actual values
print("Predicted labels:", y_pred)
print("Actual labels:", y_test.values)
