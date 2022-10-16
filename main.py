from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from preprocess import load_datasets


def main():
    X_train, X_test, y_train, y_test = load_datasets()

    knn = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    knn_predictions = knn.predict(X_test)
    cm = confusion_matrix(y_test, knn_predictions)
    print(cm)


if __name__ == "__main__":
    main()
