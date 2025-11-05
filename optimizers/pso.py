"""
Particle Swarm Optimization for hyperparameter tuning
"""
import numpy as np
from .base_optimizer import BaseOptimizer


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO)
    
    Maintains a population of particles that explore the search space
    by following personal best and global best positions.
    """
    
    def __init__(self, search_space, n_iterations=50, population_size=10,
                 w=0.7, c1=1.5, c2=1.5, seed=42):
        """
        Args:
            search_space: Parameter search space
            n_iterations: Number of iterations
            population_size: Number of particles
            w: Inertia weight
            c1: Cognitive parameter (personal best influence)
            c2: Social parameter (global best influence)
            seed: Random seed
        """
        super().__init__(search_space, n_iterations, seed)
        
        self.population_size = population_size
        self.w = w  # Inertia
        self.c1 = c1  # Cognitive
        self.c2 = c2  # Social
        
        # Initialize particles
        self._initialize_particles()
        
        self.iteration = 0
        self.current_particle = 0
    
    def _initialize_particles(self):
        """Initialize particle positions and velocities"""
        self.particles = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_scores = []
        
        for _ in range(self.population_size):
            # Random initial position
            particle = self.sample_random_config()
            self.particles.append(particle)
            
            # Initialize velocity (as fraction of range)
            velocity = {}
            for param_name in self.search_space:
                param_type, lower, upper, scale = self.search_space[param_name]
                
                if scale == 'log':
                    lower_log = np.log(lower)
                    upper_log = np.log(upper)
                    range_size = upper_log - lower_log
                else:
                    range_size = upper - lower
                
                velocity[param_name] = np.random.uniform(-range_size * 0.1, 
                                                         range_size * 0.1)
            
            self.velocities.append(velocity)
            self.personal_best_positions.append(particle.copy())
            self.personal_best_scores.append(-np.inf)
    
    def suggest_next_params(self):
        """Suggest next particle to evaluate"""
        if self.iteration >= self.n_iterations:
            return None
        
        # Return current particle
        return self.particles[self.current_particle].copy()
    
    def update(self, params, score):
        """Update particle with evaluation result"""
        # Store in history
        self.history.append({
            'params': params,
            'score': score,
            'particle': self.current_particle,
            'iteration': self.iteration
        })
        
        # Update personal best
        if score > self.personal_best_scores[self.current_particle]:
            self.personal_best_scores[self.current_particle] = score
            self.personal_best_positions[self.current_particle] = params.copy()
        
        # Update global best
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
        
        # Move to next particle
        self.current_particle += 1
        
        # If all particles evaluated, update positions
        if self.current_particle >= self.population_size:
            self._update_particle_positions()
            self.current_particle = 0
            self.iteration += 1
    
    def _update_particle_positions(self):
        """Update all particle positions and velocities"""
        for i in range(self.population_size):
            particle = self.particles[i]
            velocity = self.velocities[i]
            personal_best = self.personal_best_positions[i]
            
            for param_name in self.search_space:
                param_type, lower, upper, scale = self.search_space[param_name]
                
                # Get current values in appropriate space
                if scale == 'log':
                    current = np.log(particle[param_name])
                    p_best = np.log(personal_best[param_name])
                    g_best = np.log(self.best_params[param_name])
                else:
                    current = particle[param_name]
                    p_best = personal_best[param_name]
                    g_best = self.best_params[param_name]
                
                # Random factors
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                # Update velocity
                cognitive = self.c1 * r1 * (p_best - current)
                social = self.c2 * r2 * (g_best - current)
                velocity[param_name] = (self.w * velocity[param_name] + 
                                       cognitive + social)
                
                # Update position
                new_value = current + velocity[param_name]
                
                # Convert back if log scale
                if scale == 'log':
                    new_value = np.exp(new_value)
                
                # Clip to bounds
                particle[param_name] = self.clip_param(param_name, new_value)
    
    def optimize(self, objective_func, verbose=True):
        """
        Run PSO optimization
        
        Args:
            objective_func: Function that takes params dict and returns score
            verbose: Print progress
        
        Returns:
            best_params, best_score
        """
        total_evaluations = self.n_iterations * self.population_size
        eval_count = 0
        
        for iteration in range(self.n_iterations):
            for particle_idx in range(self.population_size):
                # Suggest parameters
                params = self.suggest_next_params()
                
                # Evaluate
                score = objective_func(params)
                
                # Update
                self.update(params, score)
                
                eval_count += 1
                
                if verbose and eval_count % 10 == 0:
                    print(f"Evaluation {eval_count}/{total_evaluations}: "
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
    
    # Dummy objective (Rosenbrock-like function)
    def objective(params):
        x = params['learning_rate'] / 1e-2
        y = params['hidden_size'] / 512
        return -(100 * (y - x**2)**2 + (1 - x)**2)
    
    # Run PSO
    optimizer = ParticleSwarmOptimizer(search_space, n_iterations=5, 
                                      population_size=5)
    best_params, best_score = optimizer.optimize(objective, verbose=True)
    
    print(f"\nBest params: {best_params}")
    print(f"Best score: {best_score:.4f}")