import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
data = pd.read_csv('../datasets/movie_reviews/Movie Reviews.csv')

# Convert labels: 'pos' = 1 (positive review), 'neg' = 0 (negative review)
data['label'] = data['tag'].apply(lambda x: 1 if x == 'pos' else 0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Convert texts to vector representation (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Function for training the model and printing metrics
def train_and_evaluate(alpha):
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Alpha: {alpha} - Model accuracy: {accuracy:.4f}")
    return accuracy, y_pred

# Tune the optimal alpha value
alphas = np.linspace(0.1, 1.0, 10)
accuracies = []
best_alpha = None
best_accuracy = 0

for alpha in alphas:
    accuracy, y_pred = train_and_evaluate(alpha)
    accuracies.append(accuracy)
    if accuracy > best_accuracy:
        best_alpha = alpha
        best_accuracy = accuracy

# Visualize accuracy by alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas, accuracies, marker='o')
plt.title("Accuracy by Alpha Parameter")
plt.xlabel("alpha")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Train the model with the best alpha and print the classification report
print(f"\nBest alpha parameter: {best_alpha}, Accuracy: {best_accuracy:.4f}")
model = MultinomialNB(alpha=best_alpha)
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)

print("Classification report:\n", classification_report(y_test, y_pred))

# Build and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
