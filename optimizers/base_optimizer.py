"""
Abstract base class for hyperparameter optimizers
"""
from abc import ABC, abstractmethod
import numpy as np
import json
import os


class BaseOptimizer(ABC):
    """Base class for all hyperparameter optimization algorithms"""
    
    def __init__(self, search_space, n_iterations=50, seed=42):
        """
        Args:
            search_space: Dict defining parameter ranges
                Example: {
                    'learning_rate': ('float', 1e-4, 1e-2, 'log'),
                    'batch_size': ('int', 16, 128, 'log'),
                    'hidden_size': ('int', 64, 512, 'linear'),
                    'dropout': ('float', 0.1, 0.5, 'linear')
                }
            n_iterations: Number of optimization iterations
            seed: Random seed
        """
        self.search_space = search_space
        self.n_iterations = n_iterations
        self.seed = seed
        np.random.seed(seed)
        
        # Results storage
        self.history = []
        self.best_params = None
        self.best_score = -np.inf
        
    @abstractmethod
    def suggest_next_params(self):
        """Suggest next hyperparameter configuration to evaluate"""
        pass
    
    @abstractmethod
    def update(self, params, score):
        """Update optimizer with evaluation result"""
        pass
    
    def sample_param(self, param_name):
        """Sample a single parameter from search space"""
        param_type, lower, upper, scale = self.search_space[param_name]
        
        if scale == 'log':
            if param_type == 'float':
                value = np.exp(np.random.uniform(np.log(lower), np.log(upper)))
            else:  # int
                value = int(np.exp(np.random.uniform(np.log(lower), np.log(upper))))
        else:  # linear
            if param_type == 'float':
                value = np.random.uniform(lower, upper)
            else:  # int
                value = int(np.random.uniform(lower, upper + 1))
        
        return value
    
    def sample_random_config(self):
        """Sample random configuration from search space"""
        config = {}
        for param_name in self.search_space:
            config[param_name] = self.sample_param(param_name)
        return config
    
    def clip_param(self, param_name, value):
        """Clip parameter to valid range"""
        param_type, lower, upper, scale = self.search_space[param_name]
        
        # Clip to bounds
        value = np.clip(value, lower, upper)
        
        # Convert to correct type
        if param_type == 'int':
            value = int(round(value))
        
        return value
    
    def save_results(self, filepath):
        """Save optimization results to JSON"""
        results = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'history': [
                {
                    'iteration': i,
                    'params': h['params'],
                    'score': float(h['score'])
                }
                for i, h in enumerate(self.history)
            ]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {filepath}")
    
    def get_best_params(self):
        """Return best parameters found"""
        return self.best_params, self.best_score
    
    def get_history(self):
        """Return full optimization history"""
        return self.history


class ObjectiveFunction:
    """Wrapper for objective function to track evaluations"""
    
    def __init__(self, eval_func):
        """
        Args:
            eval_func: Function that takes params dict and returns score
        """
        self.eval_func = eval_func
        self.n_evaluations = 0
    
    def __call__(self, params):
        """Evaluate and track"""
        score = self.eval_func(params)
        self.n_evaluations += 1
        return score