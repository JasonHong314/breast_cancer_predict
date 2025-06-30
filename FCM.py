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


class FCM:
    def __init__(self, n_clusters=2, m=2, max_iter=150, error=1e-5):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
    def fit(self, X):
        N = X.shape[0]
        C = self.n_clusters
        U = np.random.dirichlet(np.ones(C), size=N)
        for _ in range(self.max_iter):
            U_old = U.copy()
            centers = (U**self.m).T @ X / np.sum(U**self.m, axis=0)[:,None]
            dist = np.zeros((N, C))
            for k in range(C):
                dist[:,k] = np.linalg.norm(X - centers[k], axis=1)
            dist = np.fmax(dist, 1e-10)
            U = 1.0 / (dist[:,:,None] / dist[:,None,:])**(2/(self.m-1))
            U = U.sum(axis=2)
            U = 1.0 / U
            if np.linalg.norm(U - U_old) < self.error:
                break
        self.U = U
        self.centers = centers
    def predict_proba(self, X):
        N = X.shape[0]
        C = self.n_clusters
        dist = np.zeros((N, C))
        for k in range(C):
            dist[:,k] = np.linalg.norm(X - self.centers[k], axis=1)
        dist = np.fmax(dist, 1e-10)
        U = 1.0 / (dist[:,:,None] / dist[:,None,:])**(2/(self.m-1))
        U = U.sum(axis=2)
        U = 1.0 / U
        return U



if __name__ == "__main__":
    fcm = FCM(n_clusters=2)
    fcm.fit(X_norm)
    fcm_proba = fcm.predict_proba(X_norm)
    fcm_df = pd.DataFrame(X, columns=breast_cancer_wisconsin_diagnostic.data.features.columns)
    fcm_df['label'] = y
    fcm_df['fcm_proba_0'] = fcm_proba[:,0]
    fcm_df['fcm_proba_1'] = fcm_proba[:,1]
    fcm_df.to_csv('breast_cancer_fcm.csv', index=False, encoding='utf-8-sig')

    # 利用PCA降维至2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)

    # 选出FCM对各类的隶属度（概率）
    fcm_proba_0 = fcm_proba[:, 0]
    fcm_proba_1 = fcm_proba[:, 1]

    plt.figure(figsize=(10, 6))

    # 根据第一个聚类中心的隶属度绘制，颜色渐变显示隶属程度
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=fcm_proba_0, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='FCM cluster 0 membership degree')

    plt.title('FCM Clustering Visualization Using PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()


