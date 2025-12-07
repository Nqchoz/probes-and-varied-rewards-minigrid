"""Visualize concept probe results across models."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_comparison_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison plots for probe results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with probe results.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Shorten model names for better display
    model_name_map = {
        'ppo_door_key_subgoal_final': 'Subgoal',
        'ppo_door_key_subgoal_decay_final': 'Subgoal Decay',
        'ppo_door_key_exploration_final': 'Exploration',
    }
    results_df['model_short'] = results_df['model'].map(model_name_map)
    
    # 1. Binary Concepts - Accuracy Comparison (Split into two plots)
    print("Creating binary concepts accuracy plots...")
    binary_df = results_df[results_df['concept_type'] == 'binary'].copy()
    
    if not binary_df.empty:
        # Pivot for grouped bar chart
        pivot_accuracy = binary_df.pivot(index='concept', columns='model_short', values='accuracy')
        
        # Split concepts into two groups
        concepts = pivot_accuracy.index.tolist()
        mid_point = (len(concepts) + 1) // 2  # Split roughly in half
        
        # Plot 1: First half of concepts
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_accuracy.iloc[:mid_point].plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Binary Concept Probe Accuracy by Model (Part 1)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Concept', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Chance')
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / 'binary_concepts_accuracy_part1.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
        
        # Plot 2: Second half of concepts
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_accuracy.iloc[mid_point:].plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Binary Concept Probe Accuracy by Model (Part 2)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Concept', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Chance')
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / 'binary_concepts_accuracy_part2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
        
        # 2. Binary Concepts - F1 Score Comparison (Split into two plots)
        print("Creating binary concepts F1 score plots...")
        pivot_f1 = binary_df.pivot(index='concept', columns='model_short', values='f1_score')
        
        # Plot 1: First half of concepts
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_f1.iloc[:mid_point].plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Binary Concept Probe F1 Score by Model (Part 1)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Concept', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / 'binary_concepts_f1_part1.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
        
        # Plot 2: Second half of concepts
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_f1.iloc[mid_point:].plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Binary Concept Probe F1 Score by Model (Part 2)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Concept', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / 'binary_concepts_f1_part2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    # 3. Continuous Concepts - R² Comparison
    print("Creating continuous concepts R² plot...")
    continuous_df = results_df[results_df['concept_type'] == 'continuous'].copy()
    
    if not continuous_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        pivot_r2 = continuous_df.pivot(index='concept', columns='model_short', values='r2_score')
        pivot_r2.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Continuous Concept Probe R² Score by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Concept', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Baseline')
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / 'continuous_concepts_r2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
        
        # 4. Continuous Concepts - MAE Comparison
        print("Creating continuous concepts MAE plot...")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        pivot_mae = continuous_df.pivot(index='concept', columns='model_short', values='mae')
        pivot_mae.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Continuous Concept Probe MAE by Model (Lower is Better)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Concept', fontsize=12)
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / 'continuous_concepts_mae.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    # 5. Heatmap - All Metrics
    print("Creating comprehensive heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Binary concepts heatmap
    if not binary_df.empty:
        pivot_accuracy = binary_df.pivot(index='concept', columns='model_short', values='accuracy')
        sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Accuracy'})
        axes[0].set_title('Binary Concept Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Model', fontsize=10)
        axes[0].set_ylabel('Concept', fontsize=10)
    
    # Continuous concepts heatmap
    if not continuous_df.empty:
        pivot_r2 = continuous_df.pivot(index='concept', columns='model_short', values='r2_score')
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'R² Score'})
        axes[1].set_title('Continuous Concept R²', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Model', fontsize=10)
        axes[1].set_ylabel('Concept', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'concepts_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()
    
    # 6. Overall Performance Comparison
    print("Creating overall performance comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate average performance per model
    model_performance = []
    for model in results_df['model_short'].unique():
        model_data = results_df[results_df['model_short'] == model]
        
        # Average binary accuracy
        binary_acc = model_data[model_data['concept_type'] == 'binary']['accuracy'].mean()
        # Average continuous R²
        continuous_r2 = model_data[model_data['concept_type'] == 'continuous']['r2_score'].mean()
        
        model_performance.append({
            'Model': model,
            'Binary Accuracy': binary_acc,
            'Continuous R²': continuous_r2,
        })
    
    perf_df = pd.DataFrame(model_performance)
    
    # Plot grouped bar chart
    x = np.arange(len(perf_df))
    width = 0.35
    
    ax.bar(x - width/2, perf_df['Binary Accuracy'], width, label='Binary Accuracy', alpha=0.8)
    ax.bar(x + width/2, perf_df['Continuous R²'], width, label='Continuous R²', alpha=0.8)
    
    ax.set_title('Overall Model Performance on Concept Probes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(perf_df['Model'])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'overall_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()
    
    # 7. Concept Difficulty Ranking
    print("Creating concept difficulty ranking...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate average accuracy/R² across models for each concept
    concept_difficulty = []
    
    for concept in results_df['concept'].unique():
        concept_data = results_df[results_df['concept'] == concept]
        concept_type = concept_data['concept_type'].iloc[0]
        
        if concept_type == 'binary':
            avg_score = concept_data['accuracy'].mean()
            metric = 'Accuracy'
        else:
            avg_score = concept_data['r2_score'].mean()
            metric = 'R²'
        
        concept_difficulty.append({
            'Concept': concept,
            'Average Score': avg_score,
            'Type': concept_type,
            'Metric': metric,
        })
    
    diff_df = pd.DataFrame(concept_difficulty).sort_values('Average Score', ascending=True)
    
    # Color by type
    colors = ['#ff7f0e' if t == 'continuous' else '#1f77b4' for t in diff_df['Type']]
    
    ax.barh(diff_df['Concept'], diff_df['Average Score'], color=colors, alpha=0.7)
    ax.set_xlabel('Average Score Across Models', fontsize=12)
    ax.set_ylabel('Concept', fontsize=12)
    ax.set_title('Concept Decodability Ranking\n(Lower = Harder to Decode)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='Chance (Binary)')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend for concept types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='Binary'),
        Patch(facecolor='#ff7f0e', alpha=0.7, label='Continuous'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'concept_difficulty.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=str,
        default="probe_results.csv",
        help="Path to probe results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results_df = pd.read_csv(args.results)
    print(f"Loaded {len(results_df)} probe results")
    
    # Create plots
    output_dir = Path(args.output_dir)
    print(f"\nGenerating plots in {output_dir}/...")
    create_comparison_plots(results_df, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ All plots saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
