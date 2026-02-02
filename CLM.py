import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class KMeansClustering:
	def __init__(self, K: int, max_iter: int = 300, tol: float = 1e-4, random_state: Optional[int] = None):
		self.K = int(K)
		self.max_iter = int(max_iter)
		self.tol = float(tol)
		self.random_state = random_state
		self.centroids = None
		self.labels_ = None

	def _euclidean_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
		# Returns squared distances (n_samples, K)
		# using (x - c)^2 = x^2 + c^2 - 2xc
		X_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]
		C_sq = np.sum(centroids ** 2, axis=1)[np.newaxis, :]
		cross = X.dot(centroids.T)
		dists = X_sq + C_sq - 2 * cross
		# numerical issues
		dists = np.maximum(dists, 0.0)
		return dists

	def _init_centroids_kmeanspp(self, X: np.ndarray) -> np.ndarray:
		rng = np.random.default_rng(self.random_state)
		n_samples = X.shape[0]
		centroids = np.empty((self.K, X.shape[1]), dtype=float)

		# pick first centroid randomly
		first_idx = rng.integers(0, n_samples)
		centroids[0] = X[first_idx]

		# pick remaining
		for k in range(1, self.K):
			dists = self._euclidean_distances(X, centroids[:k])
			closest_sq = np.min(dists, axis=1)
			probs = closest_sq / np.sum(closest_sq)
			if np.any(probs > 0):
				cumulative = np.cumsum(probs)
				r = rng.random()
				next_idx = np.searchsorted(cumulative, r)
			else:
				next_idx = rng.integers(0, n_samples)
			centroids[k] = X[next_idx]

		return centroids

	def fit(self, X: np.ndarray, use_kmeanspp: bool = True, verbose: bool = False):
		X = np.asarray(X, dtype=float)
		n_samples, n_features = X.shape

		if self.K <= 0 or self.K > n_samples:
			raise ValueError("K must be > 0 and <= number of samples")

		if use_kmeanspp:
			centroids = self._init_centroids_kmeanspp(X)
		else:
			rng = np.random.default_rng(self.random_state)
			indices = rng.choice(n_samples, size=self.K, replace=False)
			centroids = X[indices].astype(float)

		labels = np.full(n_samples, -1, dtype=int)

		for it in range(self.max_iter):
			old_centroids = centroids.copy()

			# Assignment step
			dists = self._euclidean_distances(X, centroids)
			labels = np.argmin(dists, axis=1)

			# Update step
			for k in range(self.K):
				members = X[labels == k]
				if members.shape[0] > 0:
					centroids[k] = np.mean(members, axis=0)
				else:
					# empty cluster: reinitialize to a random point
					rng = np.random.default_rng(self.random_state)
					centroids[k] = X[rng.integers(0, n_samples)]

			# convergence check
			shift = np.linalg.norm(centroids - old_centroids)
			if verbose:
				print(f"Iteration {it+1}, centroid shift {shift:.6f}")
			if shift <= self.tol:
				break

		self.centroids = centroids
		self.labels_ = labels
		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		if self.centroids is None:
			raise ValueError("Model not fitted yet. Call `fit` first.")
		X = np.asarray(X, dtype=float)
		dists = self._euclidean_distances(X, self.centroids)
		return np.argmin(dists, axis=1)

	def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
		self.fit(X, **kwargs)
		return self.labels_

	def plot_clusters(self, X: np.ndarray, show: bool = True, cmap: str = 'tab10'):
		X = np.asarray(X, dtype=float)
		if X.shape[1] < 2:
			raise ValueError("plot_clusters requires at least 2D data")
		if self.centroids is None or self.labels_ is None:
			raise ValueError("Model not fitted yet. Call `fit` first.")

		plt.figure(figsize=(6, 5))
		colors = plt.get_cmap(cmap)
		for k in range(self.K):
			members = X[self.labels_ == k]
			plt.scatter(members[:, 0], members[:, 1], s=20, color=colors(k), label=f'cluster {k}')

		plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=200, c='black', marker='X', label='centroids')
		plt.legend()
		plt.tight_layout()
		if show:
			plt.show()


if __name__ == "__main__":
	# simple sanity example
	rng = np.random.default_rng(0)
	# create 3 gaussian blobs
	X1 = rng.normal(loc=[0, 0], scale=0.5, size=(100, 2))
	X2 = rng.normal(loc=[5, 5], scale=0.5, size=(100, 2))
	X3 = rng.normal(loc=[0, 5], scale=0.5, size=(100, 2))
	X = np.vstack([X1, X2, X3])

	kmeans = KMeansClustering(K=3, max_iter=100, random_state=0)
	kmeans.fit(X, use_kmeanspp=True, verbose=True)
	kmeans.plot_clusters(X)

