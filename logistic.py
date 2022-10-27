import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from preprocess import load_datasets

def main():
    X_train, X_test, y_train, y_test = load_datasets()
    y_train = np.where(y_train == 0, 0, 1)
    y_test = np.where(y_test == 0, 0, 1)
    logistic = LogisticRegression(max_iter=5000).fit(X_train, y_train)

    accuracy = logistic.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    logistic_predictions = logistic.predict(X_test)
    cm = confusion_matrix(y_test, logistic_predictions)
    print(cm)

if __name__ == "__main__":
    main()
