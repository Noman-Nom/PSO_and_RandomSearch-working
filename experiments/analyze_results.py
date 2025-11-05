"""
Statistical analysis of optimization results
"""
import numpy as np
import pandas as pd
import json
import os
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalAnalyzer:
    """Perform statistical analysis on experiment results"""
    
    def __init__(self, results_dir):
        """Load results"""
        self.results_dir = results_dir
        
        # Load summary
        summary_path = os.path.join(results_dir, 'summary.json')
        with open(summary_path, 'r') as f:
            self.summary = json.load(f)
        
        self.results = self.summary['results']
        self.config = self.summary['config']
        
        print(f"✓ Loaded {len(self.results)} results for analysis")
    
    def compute_statistics(self):
        """Compute comprehensive statistics"""
        optimizers = list(set(r['optimizer'] for r in self.results))
        
        stats_data = []
        
        for opt in optimizers:
            opt_results = [r for r in self.results if r['optimizer'] == opt]
            
            # Extract metrics
            test_accs = [r['final_test_acc'] for r in opt_results]
            val_accs = [r['best_val_acc'] for r in opt_results]
            times = [r['training_time'] for r in opt_results]
            
            stats_data.append({
                'Optimizer': opt.upper(),
                'Test Acc Mean': np.mean(test_accs),
                'Test Acc Std': np.std(test_accs),
                'Test Acc Min': np.min(test_accs),
                'Test Acc Max': np.max(test_accs),
                'Val Acc Mean': np.mean(val_accs),
                'Val Acc Std': np.std(val_accs),
                'Time Mean (s)': np.mean(times),
                'Time Std (s)': np.std(times),
                'N Runs': len(opt_results)
            })
        
        df = pd.DataFrame(stats_data)
        
        return df
    
    def perform_significance_tests(self):
        """Perform statistical significance tests"""
        optimizers = list(set(r['optimizer'] for r in self.results))
        
        # Collect test accuracies for each optimizer
        test_accs = {}
        for opt in optimizers:
            opt_results = [r for r in self.results if r['optimizer'] == opt]
            test_accs[opt] = [r['final_test_acc'] for r in opt_results]
        
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*60)
        
        # Friedman test (if we have paired data)
        if len(optimizers) >= 3:
            # Check if all optimizers have same number of runs
            n_runs = [len(test_accs[opt]) for opt in optimizers]
            
            if len(set(n_runs)) == 1:
                # Perform Friedman test
                samples = [test_accs[opt] for opt in optimizers]
                statistic, p_value = friedmanchisquare(*samples)
                
                print(f"\nFriedman Test:")
                print(f"  Statistic: {statistic:.4f}")
                print(f"  p-value: {p_value:.6f}")
                
                if p_value < 0.05:
                    print("  ✓ Significant difference detected (p < 0.05)")
                else:
                    print("  ✗ No significant difference (p >= 0.05)")
        
        # Pairwise Wilcoxon tests
        print(f"\nPairwise Wilcoxon Signed-Rank Tests:")
        print("  (Testing if differences are significant)")
        
        pairwise_results = []
        
        for i, opt1 in enumerate(optimizers):
            for opt2 in optimizers[i+1:]:
                accs1 = test_accs[opt1]
                accs2 = test_accs[opt2]
                
                # Ensure same length
                min_len = min(len(accs1), len(accs2))
                accs1 = accs1[:min_len]
                accs2 = accs2[:min_len]
                
                if min_len >= 3:  # Need at least 3 samples
                    try:
                        statistic, p_value = wilcoxon(accs1, accs2)
                        
                        mean_diff = np.mean(accs1) - np.mean(accs2)
                        
                        pairwise_results.append({
                            'Comparison': f"{opt1.upper()} vs {opt2.upper()}",
                            'Mean Diff': mean_diff,
                            'p-value': p_value,
                            'Significant': '✓' if p_value < 0.05 else '✗'
                        })
                        
                        print(f"\n  {opt1.upper()} vs {opt2.upper()}:")
                        print(f"    Mean difference: {mean_diff:+.2f}%")
                        print(f"    p-value: {p_value:.6f}")
                        print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'}")
                    
                    except Exception as e:
                        print(f"  ⚠️ Could not perform test for {opt1} vs {opt2}: {e}")
        
        if pairwise_results:
            df_pairwise = pd.DataFrame(pairwise_results)
            return df_pairwise
        
        return None
    
    def analyze_hyperparameter_importance(self):
        """Analyze which hyperparameters matter most"""
        print("\n" + "="*60)
        print("HYPERPARAMETER IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Collect all hyperparameters and scores
        all_data = []
        
        for result in self.results:
            row = result['best_params'].copy()
            row['test_acc'] = result['final_test_acc']
            row['optimizer'] = result['optimizer']
            all_data.append(row)
        
        df = pd.DataFrame(all_data)
        
        # Compute correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'test_acc']
        
        correlations = []
        for col in numeric_cols:
            corr = df[col].corr(df['test_acc'])
            correlations.append({
                'Hyperparameter': col,
                'Correlation': corr,
                'Abs Correlation': abs(corr)
            })
        
        df_corr = pd.DataFrame(correlations)
        df_corr = df_corr.sort_values('Abs Correlation', ascending=False)
        
        print("\nCorrelation with Test Accuracy:")
        print(df_corr.to_string(index=False))
        
        return df_corr
    
    def compute_efficiency_metrics(self):
        """Analyze sample efficiency"""
        print("\n" + "="*60)
        print("EFFICIENCY ANALYSIS")
        print("="*60)
        
        optimizers = list(set(r['optimizer'] for r in self.results))
        
        efficiency_data = []
        
        for opt in optimizers:
            opt_results = [r for r in self.results if r['optimizer'] == opt]
            
            # Average metrics
            avg_test_acc = np.mean([r['final_test_acc'] for r in opt_results])
            avg_time = np.mean([r['training_time'] for r in opt_results])
            
            # Compute "score per minute"
            efficiency = avg_test_acc / (avg_time / 60)
            
            efficiency_data.append({
                'Optimizer': opt.upper(),
                'Avg Test Acc': avg_test_acc,
                'Avg Time (min)': avg_time / 60,
                'Efficiency (Acc/min)': efficiency
            })
        
        df_efficiency = pd.DataFrame(efficiency_data)
        df_efficiency = df_efficiency.sort_values('Efficiency (Acc/min)', ascending=False)
        
        print("\n", df_efficiency.to_string(index=False))
        
        return df_efficiency
    
    def generate_latex_table(self, save_path=None):
        """Generate LaTeX table for paper"""
        df_stats = self.compute_statistics()
        
        # Format for LaTeX
        latex_rows = []
        latex_rows.append("\\begin{table}[h]")
        latex_rows.append("\\centering")
        latex_rows.append("\\caption{Comparison of Hyperparameter Optimization Methods}")
        latex_rows.append("\\label{tab:comparison}")
        latex_rows.append("\\begin{tabular}{lcccc}")
        latex_rows.append("\\toprule")
        latex_rows.append("Method & Test Acc (\\%) & Val Acc (\\%) & Time (s) & N Runs \\\\")
        latex_rows.append("\\midrule")
        
        for _, row in df_stats.iterrows():
            latex_rows.append(
                f"{row['Optimizer']} & "
                f"${row['Test Acc Mean']:.2f} \\pm {row['Test Acc Std']:.2f}$ & "
                f"${row['Val Acc Mean']:.2f} \\pm {row['Val Acc Std']:.2f}$ & "
                f"${row['Time Mean (s)']:.1f} \\pm {row['Time Std (s)']:.1f}$ & "
                f"{row['N Runs']} \\\\"
            )
        
        latex_rows.append("\\bottomrule")
        latex_rows.append("\\end{tabular}")
        latex_rows.append("\\end{table}")
        
        latex_table = "\n".join(latex_rows)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(latex_table)
            print(f"\n✓ LaTeX table saved to {save_path}")
        
        print("\n" + "="*60)
        print("LATEX TABLE")
        print("="*60)
        print(latex_table)
        
        return latex_table
    
    def generate_full_report(self):
        """Generate complete statistical report"""
        print("\n" + "="*70)
        print(" " * 15 + "STATISTICAL ANALYSIS REPORT")
        print("="*70)
        
        # 1. Summary statistics
        print("\n1. SUMMARY STATISTICS")
        print("-" * 70)
        df_stats = self.compute_statistics()
        print(df_stats.to_string(index=False))
        
        # 2. Significance tests
        print("\n2. SIGNIFICANCE TESTS")
        print("-" * 70)
        df_pairwise = self.perform_significance_tests()
        
        # 3. Hyperparameter importance
        print("\n3. HYPERPARAMETER IMPORTANCE")
        print("-" * 70)
        df_corr = self.analyze_hyperparameter_importance()
        
        # 4. Efficiency metrics
        print("\n4. EFFICIENCY METRICS")
        print("-" * 70)
        df_efficiency = self.compute_efficiency_metrics()
        
        # 5. Generate LaTeX table
        print("\n5. LATEX TABLE FOR PAPER")
        print("-" * 70)
        latex_table = self.generate_latex_table(
            save_path=os.path.join(self.results_dir, 'results_table.tex')
        )
        
        # Save all dataframes
        output_dir = os.path.join(self.results_dir, 'statistical_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        df_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
        if df_pairwise is not None:
            df_pairwise.to_csv(os.path.join(output_dir, 'pairwise_tests.csv'), index=False)
        df_corr.to_csv(os.path.join(output_dir, 'hyperparameter_correlations.csv'), index=False)
        df_efficiency.to_csv(os.path.join(output_dir, 'efficiency_metrics.csv'), index=False)
        
        print(f"\n✅ Statistical analysis complete!")
        print(f"📁 Results saved to: {output_dir}")


# Main
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    analyzer = StatisticalAnalyzer(results_dir)
    analyzer.generate_full_report()