"""
Main experiment runner for hyperparameter optimization research
"""
import sys
sys.path.append('..')

import torch
import yaml
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Import project modules
from data.loaders import DatasetLoader
from models.mlp import create_mlp_from_config
from models.cnn import create_cnn_from_config
from utils.trainer import train_model_with_config, ModelTrainer
from optimizers.random_search import RandomSearch
from optimizers.pso import ParticleSwarmOptimizer
from optimizers.bayesian_opt import BayesianOptimizer


class HyperparameterExperiment:
    """Main experiment orchestrator"""
    
    def __init__(self, config_path='config.yaml'):
        """Load configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = self.config['experiment']['device']
        if self.device == 'cuda':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("⚠️ CUDA not available, using CPU")
                self.device = 'cpu'
        else:
            print(f"✓ Using {self.device.upper()}")
        # Set seed
        # Create results directory----
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(
            self.config['experiment']['results_dir'],
            f"{self.config['dataset']['name']}_{self.config['model']['type']}_{timestamp}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"✓ Experiment directory: {self.results_dir}")
        
        # Save config
        with open(os.path.join(self.results_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def load_data(self, seed=42):
        """Load dataset"""
        dataset_config = self.config['dataset']
        
        loader = DatasetLoader(
            dataset_name=dataset_config['name'],
            data_dir=dataset_config['data_dir'],
            val_split=dataset_config['val_split'],
            test_split=dataset_config['test_split'],
            seed=seed
        )
        
        return loader
    
    def create_model(self, hyperparams, dataset_info):
        """Create model from hyperparameters"""
        model_type = self.config['model']['type']
        
        if model_type == 'MLP':
            model = create_mlp_from_config(
                input_shape=dataset_info['input_shape'],
                num_classes=dataset_info['num_classes'],
                config={
                    'hidden_size': hyperparams['hidden_size'],
                    'num_layers': hyperparams['num_layers'],
                    'dropout': hyperparams['dropout'],
                    'use_batchnorm': self.config['model']['use_batchnorm']
                }
            )
        elif model_type == 'CNN':
            model = create_cnn_from_config(
                input_shape=dataset_info['input_shape'],
                num_classes=dataset_info['num_classes'],
                config={
                    'conv_channels': self.config['model']['conv_channels'],
                    'fc_hidden': self.config['model']['fc_hidden'],
                    'dropout': hyperparams['dropout']
                }
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def convert_search_space(self):
        """Convert config search space to optimizer format"""
        search_space = {}
        
        for param_name, param_config in self.config['search_space'].items():
            search_space[param_name] = (
                param_config['type'],
                param_config['min'],
                param_config['max'],
                param_config['scale']
            )
        
        return search_space
    
    def objective_function(self, hyperparams, loader, dataset_info, seed=42, optimizer_name='random_search'):
        """
        Objective function to optimize
        Returns validation accuracy
        
        Args:
            hyperparams: Hyperparameter dictionary
            loader: Dataset loader
            dataset_info: Dataset information
            seed: Random seed
            optimizer_name: Name of optimizer (to determine training epochs)
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get data loaders
        train_loader, val_loader, _ = loader.get_loaders(
            batch_size=hyperparams['batch_size'],
            num_workers=self.config['experiment']['num_workers']
        )
        
        # Create model
        model = self.create_model(hyperparams, dataset_info)
        
        # Select epochs based on optimizer
        # PSO uses fewer epochs per particle (50) vs Random Search (200)
        if optimizer_name == 'pso' and 'pso_training' in self.config:
            max_epochs = self.config['pso_training']['max_epochs']
            patience = self.config['pso_training']['early_stopping_patience']
        else:
            max_epochs = self.config['training']['max_epochs']
            patience = self.config['training']['early_stopping_patience']
        
        # Training config
        train_config = {
            'learning_rate': hyperparams['learning_rate'],
            'weight_decay': hyperparams['weight_decay'],
            'optimizer': self.config['training']['optimizer_type'],
            'epochs': max_epochs,
            'patience': patience,
            'verbose': False
        }
        
        # Train model
        try:
            results, trainer = train_model_with_config(
                model, train_loader, val_loader, train_config, self.device
            )
            val_acc = results['best_val_acc']
            # Return with full precision (not rounded) so optimizer can differentiate
            # Even small differences matter for hyperparameter optimization
            return val_acc
            
        except Exception as e:
            print(f"⚠️ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def run_optimizer(self, optimizer_name, seed=42):
        """Run single optimizer"""
        print(f"\n{'='*60}")
        print(f"Running {optimizer_name.upper()} (seed={seed})")
        print(f"{'='*60}")
        
        # Load data
        loader = self.load_data(seed=seed)
        dataset_info = loader.get_info()
        
        # Convert search space
        search_space = self.convert_search_space()
        
        # Track evaluation count to create unique seeds per trial
        evaluation_counter = [0]  # Use list to allow modification in nested function
        
        # Create objective with optimizer name
        def objective(params):
            # Create unique seed for each evaluation: base_seed + evaluation_number
            # This ensures different random initializations for each hyperparameter config
            trial_seed = seed + evaluation_counter[0]
            evaluation_counter[0] += 1
            
            score = self.objective_function(params, loader, dataset_info, trial_seed, optimizer_name)
            # Show 4 decimal places to see small differences between configs
            print(f"  Evaluated: val_acc={score:.4f}%")
            return score
        
        # Create optimizer
        if optimizer_name == 'random_search':
            opt_config = self.config['optimizers']['random_search']
            optimizer = RandomSearch(
                search_space=search_space,
                n_iterations=opt_config['n_iterations'],
                seed=seed
            )
        
        elif optimizer_name == 'pso':
            opt_config = self.config['optimizers']['pso']
            optimizer = ParticleSwarmOptimizer(
                search_space=search_space,
                n_iterations=opt_config['n_iterations'],
                population_size=opt_config['population_size'],
                w=opt_config['w'],
                c1=opt_config['c1'],
                c2=opt_config['c2'],
                seed=seed
            )
        
        elif optimizer_name == 'bayesian':
            opt_config = self.config['optimizers']['bayesian']
            optimizer = BayesianOptimizer(
                search_space=search_space,
                n_iterations=opt_config['n_iterations'],
                n_initial=opt_config['n_initial'],
                xi=opt_config['xi'],
                seed=seed
            )
        
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Calculate expected total epochs for reporting
        if optimizer_name == 'pso':
            opt_config = self.config['optimizers']['pso']
            pso_epochs = self.config['pso_training']['max_epochs']
            training_epochs = pso_epochs  # For consistency
            expected_total_epochs = opt_config['n_iterations'] * opt_config['population_size'] * pso_epochs
        else:
            opt_config = self.config['optimizers'][optimizer_name]
            training_epochs = self.config['training']['max_epochs']
            pso_epochs = training_epochs  # For consistency
            expected_total_epochs = opt_config['n_iterations'] * training_epochs
        
        print(f"📊 Exploration budget: ~{expected_total_epochs} total epochs")
        if optimizer_name == 'pso':
            print(f"   {opt_config['n_iterations']} iterations × {opt_config['population_size']} particles × {pso_epochs} epochs/particle")
        else:
            print(f"   {opt_config['n_iterations']} trials × {training_epochs} epochs/trial")
        
        # Run optimization
        best_params, best_score = optimizer.optimize(objective, verbose=True)
        
        # Calculate actual epochs used (approximate)
        if optimizer_name == 'pso':
            actual_epochs = len(optimizer.history) * pso_epochs
        else:
            actual_epochs = len(optimizer.history) * training_epochs
        
        print(f"\n✓ {optimizer_name} completed!")
        print(f"  Best validation accuracy: {best_score:.2f}%")
        print(f"  Best parameters: {best_params}")
        print(f"  Evaluations performed: {len(optimizer.history)}")
        
        # Save results
        result_dir = os.path.join(self.results_dir, optimizer_name, f"run_{seed}")
        os.makedirs(result_dir, exist_ok=True)
        
        optimizer.save_results(os.path.join(result_dir, 'optimization_history.json'))
        
        # Retrain best model
        print(f"\n🔄 Retraining best model for final evaluation...")
        final_results = self.retrain_best_model(
            best_params, loader, dataset_info, result_dir, seed
        )
        
        return {
            'optimizer': optimizer_name,
            'seed': seed,
            'best_params': best_params,
            'best_val_acc': best_score,
            'final_test_acc': final_results['test_acc'],
            'training_time': final_results['training_time']
        }
    
    def retrain_best_model(self, best_params, loader, dataset_info, save_dir, seed):
        """Retrain best model with more epochs"""
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get data loaders
        train_loader, val_loader, test_loader = loader.get_loaders(
            batch_size=best_params['batch_size'],
            num_workers=self.config['experiment']['num_workers']
        )
        
        # Create model
        model = self.create_model(best_params, dataset_info)
        
        # Training config
        retrain_config = {
            'learning_rate': best_params['learning_rate'],
            'weight_decay': best_params['weight_decay'],
            'optimizer': self.config['training']['optimizer_type'],
            'epochs': self.config['retraining']['max_epochs'],
            'patience': self.config['retraining']['early_stopping_patience'],
            'verbose': True
        }
        
        # Train
        results, trainer = train_model_with_config(
            model, train_loader, val_loader, retrain_config, self.device
        )
        
        # Evaluate on test set
        test_acc = trainer.evaluate(test_loader)
        print(f"✓ Final test accuracy: {test_acc:.2f}%")
        
        # Save model
        if self.config['experiment']['save_models']:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(results['history'], f, indent=2)
        
        return {
            'test_acc': test_acc,
            'val_acc': results['final_val_acc'],
            'training_time': results['training_time'],
            'epochs_trained': results['epochs_trained']
        }
    
    def run_all_experiments(self):
        """Run all configured optimizers with multiple seeds"""
        n_runs = self.config['experiment']['n_runs']
        base_seed = self.config['experiment']['base_seed']
        
        all_results = []
        
        # Get enabled optimizers
        enabled_optimizers = [
            name for name, opts in self.config['optimizers'].items()
            if opts['enabled']
        ]
        
        print(f"\n🚀 Starting experiments")
        print(f"  Optimizers: {enabled_optimizers}")
        print(f"  Runs per optimizer: {n_runs}")
        print(f"  Total experiments: {len(enabled_optimizers) * n_runs}")
        
        # Run each optimizer multiple times
        for optimizer_name in enabled_optimizers:
            for run in range(n_runs):
                seed = base_seed + run
                
                try:
                    result = self.run_optimizer(optimizer_name, seed=seed)
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Error in {optimizer_name} run {run}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save summary
        self.save_summary(all_results)
        
        print(f"\n✅ All experiments completed!")
        print(f"📁 Results saved to: {self.results_dir}")
        
        return all_results
    
    def save_summary(self, all_results):
        """Save experiment summary"""
        summary = {
            'config': self.config,
            'results': all_results
        }
        
        with open(os.path.join(self.results_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        optimizers = list(set(r['optimizer'] for r in all_results))
        
        for opt in optimizers:
            opt_results = [r for r in all_results if r['optimizer'] == opt]
            test_accs = [r['final_test_acc'] for r in opt_results]
            
            print(f"\n{opt.upper()}:")
            print(f"  Test Accuracy: {np.mean(test_accs):.2f}% ± {np.std(test_accs):.2f}%")
            print(f"  Best: {np.max(test_accs):.2f}%")
            print(f"  Worst: {np.min(test_accs):.2f}%")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION RESEARCH")
    print("="*60)
    
    # Create experiment
    experiment = HyperparameterExperiment('config.yaml')
    
    # Run all experiments
    results = experiment.run_all_experiments()