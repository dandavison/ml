class Gaussian:

    def fit(self, X):
        n, d = X.shape
        self.mu = mu = X.mean(axis=0)
        self.Sigma = (X - mu).T @ (X - mu) / (n * d)
        return self
