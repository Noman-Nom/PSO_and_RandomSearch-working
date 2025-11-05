"""
Random Search for hyperparameter optimization
"""
import numpy as np
from .base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
    Random Search optimizer
    
    Samples hyperparameters uniformly at random from the search space.
    Simple but effective baseline for hyperparameter optimization.
    """
    
    def __init__(self, search_space, n_iterations=50, seed=42):
        super().__init__(search_space, n_iterations, seed)
        self.iteration = 0
    
    def suggest_next_params(self):
        """Sample random configuration"""
        if self.iteration >= self.n_iterations:
            return None
        
        self.iteration += 1
        return self.sample_random_config()
    
    def update(self, params, score):
        """Update with evaluation result"""
        # Store in history
        self.history.append({
            'params': params,
            'score': score
        })
        
        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
    
    def optimize(self, objective_func, verbose=True):
        """
        Run optimization
        
        Args:
            objective_func: Function that takes params dict and returns score
            verbose: Print progress
        
        Returns:
            best_params, best_score
        """
        for i in range(self.n_iterations):
            # Suggest parameters
            params = self.suggest_next_params()
            
            # Evaluate
            score = objective_func(params)
            
            # Update
            self.update(params, score)
            
            if verbose and (i + 1) % 5 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}: "
                      f"Score = {score:.4f}, Best = {self.best_score:.4f}")
        
        return self.best_params, self.best_score


# Test
if __name__ == "__main__":
    # Define search space
    search_space = {
        'learning_rate': ('float', 1e-4, 1e-2, 'log'),
        'batch_size': ('int', 16, 128, 'log'),
        'hidden_size': ('int', 64, 512, 'linear'),
    }
    
    # Dummy objective
    def objective(params):
        return np.random.rand()
    
    # Run random search
    optimizer = RandomSearch(search_space, n_iterations=10)
    best_params, best_score = optimizer.optimize(objective, verbose=True)
    
    print(f"\nBest params: {best_params}")
    print(f"Best score: {best_score:.4f}")