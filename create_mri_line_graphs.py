#!/usr/bin/env python3
"""
MRI Model Line Graph Comparison
Creates line graphs to visualize and compare MRI model performance metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

def create_mri_line_comparison():
    """Create comprehensive line graph comparisons for MRI models"""
    
    # Read the MRI model comparison results
    try:
        df = pd.read_csv('mri_model_comparison_results.csv')
    except FileNotFoundError:
        print("Error: mri_model_comparison_results.csv not found!")
        print("Please run the MRI model comparison script first.")
        return
    
    # Convert string metrics to float for plotting
    df['Accuracy'] = df['Accuracy'].astype(float)
    df['F1 Score'] = df['F1 Score'].astype(float)
    df['Precision'] = df['Precision'].astype(float)
    df['Recall'] = df['Recall'].astype(float)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MRI Model Performance Comparison - Line Graphs', fontsize=16, fontweight='bold')
    
    # Model names for x-axis
    models = df['Model'].tolist()
    x_pos = range(len(models))
    
    # Define colors for each metric
    colors = {
        'Accuracy': '#1f77b4',
        'F1 Score': '#ff7f0e', 
        'Precision': '#2ca02c',
        'Recall': '#d62728'
    }
    
    # Plot 1: All Metrics Line Graph
    ax1 = axes[0, 0]
    ax1.plot(x_pos, df['Accuracy'], marker='o', linewidth=2.5, markersize=8, 
             label='Accuracy', color=colors['Accuracy'])
    ax1.plot(x_pos, df['F1 Score'], marker='s', linewidth=2.5, markersize=8, 
             label='F1 Score', color=colors['F1 Score'])
    ax1.plot(x_pos, df['Precision'], marker='^', linewidth=2.5, markersize=8, 
             label='Precision', color=colors['Precision'])
    ax1.plot(x_pos, df['Recall'], marker='D', linewidth=2.5, markersize=8, 
             label='Recall', color=colors['Recall'])
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('All Performance Metrics', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Add value annotations
    for i, model in enumerate(models):
        ax1.annotate(f"{df.iloc[i]['Accuracy']:.3f}", 
                    (i, df.iloc[i]['Accuracy']), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 2: F1 Score Focus
    ax2 = axes[0, 1]
    ax2.plot(x_pos, df['F1 Score'], marker='o', linewidth=3, markersize=10, 
             color='#ff7f0e', markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(x_pos, df['F1 Score'], alpha=0.3, color='#ff7f0e')
    
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('F1 Score Comparison (Primary Metric)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Add value annotations for F1 Score
    for i, model in enumerate(models):
        ax2.annotate(f"{df.iloc[i]['F1 Score']:.4f}", 
                    (i, df.iloc[i]['F1 Score']), 
                    textcoords="offset points", xytext=(0,15), ha='center', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Plot 3: Precision vs Recall
    ax3 = axes[1, 0]
    ax3.plot(x_pos, df['Precision'], marker='o', linewidth=2.5, markersize=8, 
             label='Precision', color='#2ca02c')
    ax3.plot(x_pos, df['Recall'], marker='s', linewidth=2.5, markersize=8, 
             label='Recall', color='#d62728')
    
    ax3.set_xlabel('Models', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Precision vs Recall Trade-off', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # Plot 4: Model Ranking by Performance
    ax4 = axes[1, 1]
    
    # Calculate overall performance score (weighted average)
    df['Overall_Score'] = (df['Accuracy'] * 0.3 + df['F1 Score'] * 0.4 + 
                          df['Precision'] * 0.15 + df['Recall'] * 0.15)
    
    # Create ranking
    df_ranked = df.sort_values('Overall_Score', ascending=True)
    ranking_pos = range(len(df_ranked))
    
    bars = ax4.barh(ranking_pos, df_ranked['Overall_Score'], 
                    color=['#ff7f0e' if score >= 0.95 else '#1f77b4' if score >= 0.8 else '#d62728' 
                          for score in df_ranked['Overall_Score']])
    
    ax4.set_xlabel('Overall Performance Score', fontweight='bold')
    ax4.set_ylabel('Models (Ranked)', fontweight='bold')
    ax4.set_title('Model Ranking by Overall Performance', fontweight='bold')
    ax4.set_yticks(ranking_pos)
    ax4.set_yticklabels(df_ranked['Model'])
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 1.05)
    
    # Add value annotations on bars
    for i, (bar, score) in enumerate(zip(bars, df_ranked['Overall_Score'])):
        ax4.annotate(f'{score:.4f}', 
                    (score + 0.01, bar.get_y() + bar.get_height()/2),
                    va='center', fontweight='bold')
    
    # Add performance tier legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', label='Excellent (≥0.95)'),
        Rectangle((0, 0), 1, 1, facecolor='#1f77b4', label='Good (0.8-0.95)'),
        Rectangle((0, 0), 1, 1, facecolor='#d62728', label='Needs Improvement (<0.8)')
    ]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plt.savefig('mri_model_line_comparison.png', dpi=300, bbox_inches='tight')
    print("Line graph saved as: mri_model_line_comparison.png")
    
    # Create a second figure for detailed metrics
    create_detailed_comparison(df)
    
    # Show the plot
    plt.show()

def create_detailed_comparison(df):
    """Create detailed comparison with confusion matrix insights"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Detailed MRI Model Analysis', fontsize=16, fontweight='bold')
    
    models = df['Model'].tolist()
    x_pos = range(len(models))
    
    # Plot 1: Error Analysis
    ax1 = axes[0]
    false_positives = df['FP'].tolist()
    false_negatives = df['FN'].tolist()
    
    width = 0.35
    ax1.bar([x - width/2 for x in x_pos], false_positives, width, 
            label='False Positives', color='#ff7f0e', alpha=0.8)
    ax1.bar([x + width/2 for x in x_pos], false_negatives, width, 
            label='False Negatives', color='#d62728', alpha=0.8)
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Number of Errors', fontweight='bold')
    ax1.set_title('Error Analysis (FP vs FN)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for i, (fp, fn) in enumerate(zip(false_positives, false_negatives)):
        if fp > 0:
            ax1.annotate(str(fp), (i - width/2, fp), ha='center', va='bottom')
        if fn > 0:
            ax1.annotate(str(fn), (i + width/2, fn), ha='center', va='bottom')
    
    # Plot 2: True Predictions
    ax2 = axes[1]
    true_negatives = df['TN'].tolist()
    true_positives = df['TP'].tolist()
    
    ax2.bar([x - width/2 for x in x_pos], true_negatives, width, 
            label='True Negatives (Normal)', color='#2ca02c', alpha=0.8)
    ax2.bar([x + width/2 for x in x_pos], true_positives, width, 
            label='True Positives (Parkinson)', color='#1f77b4', alpha=0.8)
    
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('Number of Correct Predictions', fontweight='bold')
    ax2.set_title('Correct Predictions (TN vs TP)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for i, (tn, tp) in enumerate(zip(true_negatives, true_positives)):
        ax2.annotate(str(tn), (i - width/2, tn), ha='center', va='bottom')
        ax2.annotate(str(tp), (i + width/2, tp), ha='center', va='bottom')
    
    # Plot 3: Performance Radar Chart (simplified as line plot)
    ax3 = axes[2]
    
    # Normalize metrics for radar-like visualization
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    for i, model in enumerate(models):
        values = [df.iloc[i]['Accuracy'], df.iloc[i]['F1 Score'], 
                 df.iloc[i]['Precision'], df.iloc[i]['Recall']]
        
        ax3.plot(metrics, values, marker='o', linewidth=2, markersize=6, 
                label=model, alpha=0.8)
        ax3.fill_between(metrics, values, alpha=0.1)
    
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Performance Profile by Model', fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('mri_model_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved as: mri_model_detailed_analysis.png")

def create_summary_report():
    """Create a text summary of the results"""
    try:
        df = pd.read_csv('mri_model_comparison_results.csv')
    except FileNotFoundError:
        print("Error: Results file not found!")
        return
    
    print("\n" + "="*80)
    print("MRI MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Convert string metrics to float
    df['Accuracy'] = df['Accuracy'].astype(float)
    df['F1 Score'] = df['F1 Score'].astype(float)
    
    # Find best models
    best_accuracy = df.loc[df['Accuracy'].idxmax()]
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    
    print(f"\nBest Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    print(f"Best F1 Score: {best_f1['Model']} ({best_f1['F1 Score']:.4f})")
    
    # Perfect models
    perfect_models = df[df['Accuracy'] == 1.0]['Model'].tolist()
    if perfect_models:
        print(f"\nPerfect Performance Models: {', '.join(perfect_models)}")
    
    # Models with issues
    low_performance = df[df['F1 Score'] < 0.5]['Model'].tolist()
    if low_performance:
        print(f"\nModels Needing Attention: {', '.join(low_performance)}")
    
    print(f"\nTotal Models Tested: {len(df)}")
    print(f"Average Accuracy: {df['Accuracy'].mean():.4f}")
    print(f"Average F1 Score: {df['F1 Score'].mean():.4f}")
    
    print("\nRecommendations:")
    if perfect_models:
        print(f"✓ Use {perfect_models[0]} for clinical deployment (perfect performance)")
    print("✓ CNN models generally outperform traditional ML on this dataset")
    print("✓ Consider ensemble methods for critical applications")

def main():
    """Main function to create all visualizations"""
    print("Creating MRI model line graph comparisons...")
    
    try:
        create_mri_line_comparison()
        create_summary_report()
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("- mri_model_line_comparison.png")
        print("- mri_model_detailed_analysis.png")
        print("\nFiles show comprehensive performance comparison with:")
        print("• Line graphs for all metrics")
        print("• F1 score focus (primary metric)")
        print("• Precision vs Recall trade-offs")
        print("• Model ranking by overall performance")
        print("• Error analysis (FP/FN breakdown)")
        print("• Correct prediction analysis")
        print("• Performance profiles")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        print("Make sure mri_model_comparison_results.csv exists in the current directory.")

if __name__ == '__main__':
    main()