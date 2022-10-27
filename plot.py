import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_string import load_datasets
from collections import Counter

# plot the distribution of each class
def plot(savepath):
    X_train, X_test, y_train, y_test = load_datasets()
    print('y_train:', y_train[0])
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(y_train)
    fig.savefig(savepath)


if __name__ == "__main__":
    plot('/Users/weianyin/Documents/Workspaces/cs229_MRI_image/class_barchart.png')