"""
Visualization utilities for experiment results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from glob import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """Visualize optimization results"""
    
    def __init__(self, results_dir):
        """
        Args:
            results_dir: Path to results directory
        """
        self.results_dir = results_dir
        self.load_results()
    
    def load_results(self):
        """Load all results from directory"""
        # Load summary
        summary_path = os.path.join(self.results_dir, 'summary.json')
        with open(summary_path, 'r') as f:
            self.summary = json.load(f)
        
        self.results = self.summary['results']
        self.config = self.summary['config']
        
        # Load optimization histories
        self.histories = {}
        
        for result in self.results:
            opt_name = result['optimizer']
            seed = result['seed']
            
            history_path = os.path.join(
                self.results_dir, opt_name, f"run_{seed}", 
                'optimization_history.json'
            )
            
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                key = f"{opt_name}_seed{seed}"
                self.histories[key] = history
        
        print(f"✓ Loaded {len(self.results)} results")
        print(f"✓ Loaded {len(self.histories)} optimization histories")
    
    def plot_convergence(self, save_path=None):
        """Plot convergence curves for all optimizers"""
        optimizers = list(set(r['optimizer'] for r in self.results))
        
        fig, axes = plt.subplots(1, len(optimizers), figsize=(6*len(optimizers), 5))
        if len(optimizers) == 1:
            axes = [axes]
        
        for idx, opt_name in enumerate(optimizers):
            ax = axes[idx]
            
            # Get all runs for this optimizer
            opt_histories = {
                k: v for k, v in self.histories.items() 
                if k.startswith(opt_name)
            }
            
            # Plot each run
            for key, history in opt_histories.items():
                iterations = [h['iteration'] for h in history['history']]
                scores = [h['score'] for h in history['history']]
                
                # Compute running maximum
                running_max = np.maximum.accumulate(scores)
                
                ax.plot(iterations, running_max, alpha=0.3, color='blue')
            
            # Plot mean
            if len(opt_histories) > 0:
                # Align all runs to same length
                max_iter = max(len(h['history']) for h in opt_histories.values())
                all_scores = []
                
                for history in opt_histories.values():
                    scores = [h['score'] for h in history['history']]
                    running_max = np.maximum.accumulate(scores)
                    # Pad if needed
                    if len(running_max) < max_iter:
                        running_max = np.pad(running_max, 
                                            (0, max_iter - len(running_max)), 
                                            mode='edge')
                    all_scores.append(running_max)
                
                mean_scores = np.mean(all_scores, axis=0)
                std_scores = np.std(all_scores, axis=0)
                iterations = list(range(len(mean_scores)))
                
                ax.plot(iterations, mean_scores, linewidth=2, 
                       color='darkblue', label='Mean')
                ax.fill_between(iterations, 
                               mean_scores - std_scores,
                               mean_scores + std_scores,
                               alpha=0.2, color='blue')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Validation Accuracy (%)')
            ax.set_title(f'{opt_name.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved convergence plot to {save_path}")
        
        plt.show()
    
    def plot_comparison_boxplot(self, save_path=None):
        """Box plot comparing final test accuracies"""
        optimizers = list(set(r['optimizer'] for r in self.results))
        
        data = []
        for opt in optimizers:
            opt_results = [r for r in self.results if r['optimizer'] == opt]
            test_accs = [r['final_test_acc'] for r in opt_results]
            
            for acc in test_accs:
                data.append({'Optimizer': opt.upper(), 'Test Accuracy (%)': acc})
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Optimizer', y='Test Accuracy (%)', palette='Set2')
        sns.swarmplot(data=df, x='Optimizer', y='Test Accuracy (%)', 
                     color='black', alpha=0.5, size=4)
        
        plt.title('Test Accuracy Comparison')
        plt.ylabel('Test Accuracy (%)')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, optimizer_name, seed, save_path=None):
        """Plot training curves for specific run"""
        history_path = os.path.join(
            self.results_dir, optimizer_name, f"run_{seed}",
            'training_history.json'
        )
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved training curves to {save_path}")
        
        plt.show()
    
    def plot_hyperparameter_distributions(self, save_path=None):
        """Plot distributions of explored hyperparameters"""
        optimizers = list(set(r['optimizer'] for r in self.results))
        
        # Get all hyperparameters explored
        all_params = set()
        for history in self.histories.values():
            for h in history['history']:
                all_params.update(h['params'].keys())
        
        all_params = sorted(all_params)
        n_params = len(all_params)
        
        fig, axes = plt.subplots(n_params, len(optimizers), 
                                figsize=(4*len(optimizers), 3*n_params))
        
        if n_params == 1:
            axes = axes.reshape(1, -1)
        
        for col, opt_name in enumerate(optimizers):
            opt_histories = {
                k: v for k, v in self.histories.items() 
                if k.startswith(opt_name)
            }
            
            for row, param_name in enumerate(all_params):
                ax = axes[row, col]
                
                # Collect all values
                values = []
                for history in opt_histories.values():
                    for h in history['history']:
                        if param_name in h['params']:
                            values.append(h['params'][param_name])
                
                if len(values) > 0:
                    ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                    ax.set_ylabel('Frequency')
                    
                    if row == 0:
                        ax.set_title(opt_name.upper())
                    if col == 0:
                        ax.set_ylabel(f'{param_name}\n(count)', fontsize=9)
                    if row == n_params - 1:
                        ax.set_xlabel('Value')
                    
                    ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved hyperparameter distributions to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir=None):
        """Generate complete visual report"""
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, 'visualizations')
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}\n")
        
        # 1. Convergence curves
        self.plot_convergence(
            save_path=os.path.join(output_dir, 'convergence.png')
        )
        
        # 2. Comparison boxplot
        self.plot_comparison_boxplot(
            save_path=os.path.join(output_dir, 'comparison_boxplot.png')
        )
        
        # 3. Hyperparameter distributions
        self.plot_hyperparameter_distributions(
            save_path=os.path.join(output_dir, 'hyperparameter_distributions.png')
        )
        
        # 4. Training curves for best runs
        for result in self.results[:3]:  # Plot first 3 runs
            self.plot_training_curves(
                optimizer_name=result['optimizer'],
                seed=result['seed'],
                save_path=os.path.join(
                    output_dir, 
                    f"training_{result['optimizer']}_seed{result['seed']}.png"
                )
            )
        
        print(f"\n✅ All visualizations saved to {output_dir}")


# Main
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    visualizer = ResultsVisualizer(results_dir)
    visualizer.generate_report()