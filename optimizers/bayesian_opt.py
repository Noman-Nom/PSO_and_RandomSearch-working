"""
Bayesian Optimization using Gaussian Processes
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from .base_optimizer import BaseOptimizer


class GaussianProcess:
    """Simple Gaussian Process implementation"""
    
    def __init__(self, kernel_lengthscale=1.0, kernel_variance=1.0, noise=1e-6):
        self.lengthscale = kernel_lengthscale
        self.variance = kernel_variance
        self.noise = noise
        self.X_train = []
        self.y_train = []
    
    def rbf_kernel(self, X1, X2):
        """RBF (Gaussian) kernel"""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.variance * np.exp(-0.5 / self.lengthscale**2 * sqdist)
    
    def fit(self, X, y):
        """Fit GP to data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y).reshape(-1, 1)
        
        # Compute kernel matrix
        K = self.rbf_kernel(self.X_train, self.X_train)
        self.K_inv = np.linalg.inv(K + self.noise * np.eye(len(X)))
    
    def predict(self, X):
        """Predict mean and std"""
        X = np.array(X)
        
        if len(self.X_train) == 0:
            return np.zeros(len(X)), np.ones(len(X))
        
        K_s = self.rbf_kernel(self.X_train, X)
        K_ss = self.rbf_kernel(X, X)
        
        # Mean
        mu = K_s.T.dot(self.K_inv).dot(self.y_train)
        
        # Variance
        var = np.diag(K_ss) - np.diag(K_s.T.dot(self.K_inv).dot(K_s))
        var = np.maximum(var, 1e-8)  # Numerical stability
        
        return mu.flatten(), np.sqrt(var)


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using Gaussian Processes
    
    Uses acquisition function (Expected Improvement) to balance
    exploration and exploitation.
    """
    
    def __init__(self, search_space, n_iterations=50, n_initial=5, 
                 xi=0.01, seed=42):
        """
        Args:
            search_space: Parameter search space
            n_iterations: Total number of iterations
            n_initial: Number of random initial points
            xi: Exploration parameter for EI
            seed: Random seed
        """
        super().__init__(search_space, n_iterations, seed)
        
        self.n_initial = n_initial
        self.xi = xi
        self.gp = GaussianProcess()
        
        # Store normalized parameters for GP
        self.X_normalized = []
        self.y_observed = []
        
        self.iteration = 0
    
    def _config_to_array(self, config):
        """Convert config dict to normalized array"""
        arr = []
        for param_name in sorted(self.search_space.keys()):
            param_type, lower, upper, scale = self.search_space[param_name]
            value = config[param_name]
            
            # Normalize to [0, 1]
            if scale == 'log':
                normalized = (np.log(value) - np.log(lower)) / (np.log(upper) - np.log(lower))
            else:
                normalized = (value - lower) / (upper - lower)
            
            arr.append(normalized)
        
        return np.array(arr)
    
    def _array_to_config(self, arr):
        """Convert normalized array to config dict"""
        config = {}
        for i, param_name in enumerate(sorted(self.search_space.keys())):
            param_type, lower, upper, scale = self.search_space[param_name]
            normalized = np.clip(arr[i], 0, 1)
            
            # Denormalize
            if scale == 'log':
                value = np.exp(np.log(lower) + normalized * (np.log(upper) - np.log(lower)))
            else:
                value = lower + normalized * (upper - lower)
            
            config[param_name] = self.clip_param(param_name, value)
        
        return config
    
    def _expected_improvement(self, X):
        """Expected Improvement acquisition function"""
        if len(self.y_observed) == 0:
            return np.ones(len(X))
        
        mu, sigma = self.gp.predict(X)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        best_y = np.max(self.y_observed)
        
        with np.errstate(divide='warn'):
            imp = mu - best_y - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei.flatten()
    
    def suggest_next_params(self):
        """Suggest next parameters using acquisition function"""
        if self.iteration >= self.n_iterations:
            return None
        
        # Random initialization phase
        if self.iteration < self.n_initial:
            self.iteration += 1
            return self.sample_random_config()
        
        # Bayesian optimization phase
        # Optimize acquisition function
        def neg_ei(x):
            return -self._expected_improvement(x.reshape(1, -1))[0]
        
        # Multi-start optimization
        n_dim = len(self.search_space)
        best_x = None
        best_ei = -np.inf
        
        for _ in range(10):  # 10 random starts
            x0 = np.random.rand(n_dim)
            res = minimize(neg_ei, x0, method='L-BFGS-B', 
                          bounds=[(0, 1)] * n_dim)
            
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x
        
        self.iteration += 1
        return self._array_to_config(best_x)
    
    def update(self, params, score):
        """Update GP with new observation"""
        # Store in history
        self.history.append({
            'params': params,
            'score': score
        })
        
        # Update normalized data
        x_norm = self._config_to_array(params)
        self.X_normalized.append(x_norm)
        self.y_observed.append(score)
        
        # Refit GP
        if len(self.X_normalized) > 0:
            self.gp.fit(self.X_normalized, self.y_observed)
        
        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
    
    def optimize(self, objective_func, verbose=True):
        """Run Bayesian optimization"""
        for i in range(self.n_iterations):
            # Suggest parameters
            params = self.suggest_next_params()
            
            # Evaluate
            score = objective_func(params)
            
            # Update
            self.update(params, score)
            
            if verbose and (i + 1) % 5 == 0:
                phase = "Random" if i < self.n_initial else "Bayesian"
                print(f"{phase} Iteration {i+1}/{self.n_iterations}: "
                      f"Score = {score:.4f}, Best = {self.best_score:.4f}")
        
        return self.best_params, self.best_score


# Test
if __name__ == "__main__":
    search_space = {
        'learning_rate': ('float', 1e-4, 1e-2, 'log'),
        'hidden_size': ('int', 64, 256, 'linear'),
    }
    
    def objective(params):
        # Peaked at lr=0.001, hidden=160
        x = np.log(params['learning_rate']) - np.log(0.001)
        y = (params['hidden_size'] - 160) / 100
        return -((x**2 + y**2) * 10)
    
    optimizer = BayesianOptimizer(search_space, n_iterations=20, n_initial=5)
    best_params, best_score = optimizer.optimize(objective, verbose=True)
    
    print(f"\nBest params: {best_params}")
    print(f"Best score: {best_score:.4f}")