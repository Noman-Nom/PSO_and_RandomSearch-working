"""
Generate publication-quality visualizations for hyperparameter optimization results
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# Publication-quality settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300  # High resolution
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Color palette for publication
PUBLICATION_COLORS = {
    'PSO': '#2E86AB',  # Blue
    'RANDOM_SEARCH': '#A23B72',  # Purple
    'train': '#FF6B6B',  # Red
    'val': '#4ECDC4',  # Teal
}


def load_results(results_dir):
    """Load all results"""
    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    results = summary['results']
    config = summary['config']
    
    # Load optimization histories
    histories = {}
    for result in results:
        opt_name = result['optimizer']
        seed = result['seed']
        
        history_path = os.path.join(
            results_dir, opt_name, f"run_{seed}", 
            'optimization_history.json'
        )
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            key = f"{opt_name}_seed{seed}"
            histories[key] = history
    
    return results, config, histories


def plot_1_comparison_boxplot(results, save_path):
    """Figure 1: Test Accuracy Comparison (Boxplot)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    data = []
    for result in results:
        opt_name = result['optimizer'].upper()
        if opt_name == 'RANDOM_SEARCH':
            opt_name = 'Random Search'
        data.append({
            'Method': opt_name,
            'Test Accuracy (%)': result['final_test_acc']
        })
    
    df = pd.DataFrame(data)
    
    # Create boxplot
    bp = sns.boxplot(data=df, x='Method', y='Test Accuracy (%)', 
                     palette=[PUBLICATION_COLORS['PSO'], PUBLICATION_COLORS['RANDOM_SEARCH']],
                     width=0.6, ax=ax)
    
    # Add individual points
    sns.stripplot(data=df, x='Method', y='Test Accuracy (%)', 
                 color='black', alpha=0.6, size=6, ax=ax)
    
    # Add mean line
    for method in df['Method'].unique():
        mean_val = df[df['Method'] == method]['Test Accuracy (%)'].mean()
        ax.axhline(mean_val, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Optimization Method', fontweight='bold')
    ax.set_title('Final Test Accuracy Comparison', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    pso_accs = df[df['Method'] == 'PSO']['Test Accuracy (%)'].values
    rs_accs = df[df['Method'] == 'Random Search']['Test Accuracy (%)'].values
    
    stats_text = f'PSO: {np.mean(pso_accs):.2f}% ± {np.std(pso_accs):.2f}%\n'
    stats_text += f'Random Search: {np.mean(rs_accs):.2f}% ± {np.std(rs_accs):.2f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_2_convergence_curves(histories, results, save_path):
    """Figure 2: Convergence Curves"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    optimizers = ['pso', 'random_search']
    opt_labels = ['PSO', 'Random Search']
    colors = [PUBLICATION_COLORS['PSO'], PUBLICATION_COLORS['RANDOM_SEARCH']]
    
    for opt_idx, (opt_name, opt_label, color) in enumerate(zip(optimizers, opt_labels, colors)):
        # Get all runs for this optimizer
        opt_histories = {
            k: v for k, v in histories.items() 
            if k.startswith(opt_name)
        }
        
        # Collect all convergence data
        all_curves = []
        max_iter = 0
        
        for key, history in opt_histories.items():
            iterations = [h['iteration'] for h in history['history']]
            scores = [h['score'] for h in history['history']]
            
            # Compute running maximum
            running_max = np.maximum.accumulate(scores)
            all_curves.append(running_max)
            max_iter = max(max_iter, len(running_max))
        
        # Plot individual runs (transparent)
        for curve in all_curves:
            iterations = list(range(len(curve)))
            ax.plot(iterations, curve, alpha=0.2, color=color, linewidth=1)
        
        # Compute and plot mean ± std
        if len(all_curves) > 0:
            # Pad all curves to same length
            padded_curves = []
            for curve in all_curves:
                if len(curve) < max_iter:
                    padded = np.pad(curve, (0, max_iter - len(curve)), mode='edge')
                else:
                    padded = curve
                padded_curves.append(padded)
            
            mean_curve = np.mean(padded_curves, axis=0)
            std_curve = np.std(padded_curves, axis=0)
            iterations = list(range(len(mean_curve)))
            
            ax.plot(iterations, mean_curve, linewidth=2.5, 
                   color=color, label=opt_label, zorder=5)
            ax.fill_between(iterations, 
                           mean_curve - std_curve,
                           mean_curve + std_curve,
                           alpha=0.15, color=color, zorder=4)
    
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Validation Accuracy (%)', fontweight='bold')
    ax.set_title('Optimization Convergence Curves', fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_3_training_curves_best(results, histories, results_dir, save_path):
    """Figure 3: Training Curves for Best Models"""
    # Find best run for each optimizer
    optimizers = list(set(r['optimizer'] for r in results))
    
    fig, axes = plt.subplots(2, len(optimizers), figsize=(6*len(optimizers), 8))
    if len(optimizers) == 1:
        axes = axes.reshape(2, -1)
    
    for col, opt_name in enumerate(optimizers):
        # Find best run for this optimizer
        opt_results = [r for r in results if r['optimizer'] == opt_name]
        best_result = max(opt_results, key=lambda x: x['final_test_acc'])
        
        # Load training history
        history_path = os.path.join(
            results_dir, opt_name, f"run_{best_result['seed']}",
            'training_history.json'
        )
        
        with open(history_path, 'r') as f:
            train_history = json.load(f)
        
        epochs = list(range(1, len(train_history['train_loss']) + 1))
        
        # Loss plot (top row)
        axes[0, col].plot(epochs, train_history['train_loss'], 
                         label='Train', linewidth=2, color=PUBLICATION_COLORS['train'])
        axes[0, col].plot(epochs, train_history['val_loss'], 
                         label='Validation', linewidth=2, color=PUBLICATION_COLORS['val'])
        axes[0, col].set_xlabel('Epoch', fontweight='bold')
        axes[0, col].set_ylabel('Loss', fontweight='bold')
        axes[0, col].set_title(f'{opt_name.upper()}: Loss Curves', fontweight='bold')
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)
        
        # Accuracy plot (bottom row)
        axes[1, col].plot(epochs, train_history['train_acc'], 
                         label='Train', linewidth=2, color=PUBLICATION_COLORS['train'])
        axes[1, col].plot(epochs, train_history['val_acc'], 
                         label='Validation', linewidth=2, color=PUBLICATION_COLORS['val'])
        axes[1, col].set_xlabel('Epoch', fontweight='bold')
        axes[1, col].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[1, col].set_title(f'{opt_name.upper()}: Accuracy Curves', fontweight='bold')
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)
        
        # Add final test accuracy annotation
        final_test_acc = best_result['final_test_acc']
        axes[1, col].text(0.98, 0.02, f'Test Acc: {final_test_acc:.2f}%',
                         transform=axes[1, col].transAxes,
                         fontsize=10, verticalalignment='bottom',
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_4_hyperparameter_importance(histories, results, save_path):
    """Figure 4: Hyperparameter Importance Analysis"""
    # Collect all hyperparameters and their correlation with performance
    all_data = []
    
    for result in results:
        opt_name = result['optimizer']
        seed = result['seed']
        key = f"{opt_name}_seed{seed}"
        
        if key in histories:
            history = histories[key]
            for h in history['history']:
                row = h['params'].copy()
                row['score'] = h['score']
                row['optimizer'] = opt_name
                all_data.append(row)
    
    df = pd.DataFrame(all_data)
    
    # Compute correlations
    hyperparams = ['learning_rate', 'batch_size', 'hidden_size', 
                   'num_layers', 'dropout', 'weight_decay']
    
    correlations = {}
    for param in hyperparams:
        if param in df.columns:
            corr = df[param].corr(df['score'])
            correlations[param] = abs(corr)
    
    # Sort by importance
    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    params = [p[0].replace('_', ' ').title() for p in sorted_params]
    values = [p[1] for p in sorted_params]
    colors_plt = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
    
    bars = ax.barh(params, values, color=colors_plt, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.01, i, f'{val:.3f}', 
               va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Absolute Correlation with Validation Accuracy', fontweight='bold')
    ax.set_title('Hyperparameter Importance Analysis', fontweight='bold', pad=15)
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_5_performance_table(results, save_path):
    """Figure 5: Performance Summary Table (as image)"""
    optimizers = list(set(r['optimizer'] for r in results))
    
    # Create summary data
    summary_data = []
    for opt in optimizers:
        opt_results = [r for r in results if r['optimizer'] == opt]
        test_accs = [r['final_test_acc'] for r in opt_results]
        val_accs = [r['best_val_acc'] for r in opt_results]
        times = [r['training_time'] for r in opt_results]
        
        summary_data.append({
            'Method': opt.upper().replace('_', ' '),
            'Test Acc\nMean ± Std': f'{np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}',
            'Test Acc\nRange': f'[{np.min(test_accs):.2f}, {np.max(test_accs):.2f}]',
            'Val Acc\nMean ± Std': f'{np.mean(val_accs):.2f} ± {np.std(val_accs):.2f}',
            'Time\n(seconds)': f'{np.mean(times):.0f}',
            'Runs': len(opt_results)
        })
    
    df = pd.DataFrame(summary_data)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(df) + 1):
        if i % 2 == 0:
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Performance Summary', fontweight='bold', fontsize=14, pad=20)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_all_plots(results_dir):
    """Generate all publication-quality plots"""
    print("\n" + "="*70)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*70 + "\n")
    
    results, config, histories = load_results(results_dir)
    
    output_dir = os.path.join(results_dir, 'publication_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Generate all plots
    plot_1_comparison_boxplot(results, 
                             os.path.join(output_dir, 'fig1_comparison_boxplot.png'))
    
    plot_2_convergence_curves(histories, results,
                             os.path.join(output_dir, 'fig2_convergence_curves.png'))
    
    plot_3_training_curves_best(results, histories, results_dir,
                            os.path.join(output_dir, 'fig3_training_curves.png'))
    
    plot_4_hyperparameter_importance(histories, results,
                                     os.path.join(output_dir, 'fig4_hyperparameter_importance.png'))
    
    plot_5_performance_table(results,
                            os.path.join(output_dir, 'fig5_performance_table.png'))
    
    print("\n" + "="*70)
    print("✅ ALL PUBLICATION PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 All plots saved to: {output_dir}")
    print("\nGenerated figures:")
    print("  - fig1_comparison_boxplot.png: Test accuracy comparison")
    print("  - fig2_convergence_curves.png: Optimization convergence")
    print("  - fig3_training_curves.png: Training dynamics")
    print("  - fig4_hyperparameter_importance.png: Hyperparameter analysis")
    print("  - fig5_performance_table.png: Performance summary table")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_publication_plots.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    generate_all_plots(results_dir)

