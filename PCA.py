import numpy as np
import matplotlib.pyplot as plt
from preprocess_new import load_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    X_train, X_test, y_train, y_test = load_datasets()
    X_train = X_train.reshape(2870, 256*256)
    X_test = X_test.reshape(394, 256*256)
    X = np.vstack((X_train, X_test))
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    k = 0
    total = sum(pca.explained_variance_)
    current_sum = 0
    while current_sum / total < 0.95:
        current_sum += pca.explained_variance_[k]
        k += 1
    print(pca.explained_variance_ratio_[:50])
    print(k)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.show()
if __name__ == "__main__":
    main()
    
