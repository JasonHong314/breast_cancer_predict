from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features.values
y = breast_cancer_wisconsin_diagnostic.data.targets.values.ravel()

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

class SVM:
    def __init__(self, lr=0.01, C=1.0, max_iter=1000):
        self.lr = lr
        self.C = C
        self.max_iter = max_iter
    def fit(self, X, y):
        y_bin = np.where(y == y[0], 1, -1)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.max_iter):
            for i in range(X.shape[0]):
                if y_bin[i] * (np.dot(X[i], self.w) + self.b) < 1:
                    self.w += self.lr * (self.C * y_bin[i] * X[i] - 2 * 1e-4 * self.w)
                    self.b += self.lr * self.C * y_bin[i]
                else:
                    self.w += self.lr * (-2 * 1e-4 * self.w)
    def predict_proba(self, X):
        scores = np.dot(X, self.w) + self.b
        proba = 1 / (1 + np.exp(-scores))
        return proba

if __name__ == '__main__':
    svm = SVM(lr=0.01, C=1.0, max_iter=1000)
    svm.fit(X_norm, y)
    svm_proba = svm.predict_proba(X_norm)
    svm_df = pd.DataFrame(X, columns=breast_cancer_wisconsin_diagnostic.data.features.columns)
    svm_df['label'] = y
    svm_df['svm_proba'] = svm_proba
    svm_df.to_csv('breast_cancer_svm.csv', index=False, encoding='utf-8-sig')

    # 用PCA降维到二维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)

    svm_proba = svm.predict_proba(X_norm)

    plt.figure(figsize=(10, 6))

    # 根据SVM预测的概率绘制散点图，颜色映射概率值
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=svm_proba, cmap='coolwarm', alpha=0.7)

    plt.colorbar(scatter, label='SVM positive class probability')

    plt.title('SVM Classification Visualization Using PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()