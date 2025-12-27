#!/usr/bin/env python3
"""
Voice Model Confusion Matrix Visualization
Creates comprehensive confusion matrices for voice model comparison results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

def create_voice_confusion_matrices():
    """Create confusion matrices for all voice models"""
    
    # Read the voice model comparison results
    try:
        df = pd.read_csv('voice_model_comparison_results.csv')
    except FileNotFoundError:
        print("Error: voice_model_comparison_results.csv not found!")
        print("Please run the voice model comparison script first.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate number of models and grid layout
    n_models = len(df)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    # Create main figure for individual confusion matrices
    fig1, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig1.suptitle('Voice Model Confusion Matrices - Individual Models', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten() if rows > 1 else axes
    
    # Create confusion matrices for each model
    for i, row in df.iterrows():
        ax = axes_flat[i]
        
        # Extract confusion matrix values
        tn, fp, fn, tp = row['TN'], row['FP'], row['FN'], row['TP']
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Parkinson'],
                   yticklabels=['Normal', 'Parkinson'],
                   cbar_kws={'shrink': 0.8})
        
        # Add title with performance metrics
        title = f"{row['Model']}\nAcc: {row['Accuracy']:.3f} | F1: {row['F1 Score']:.3f}"
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        
        # Color code the title based on performance
        f1_score = row['F1 Score']
        if f1_score >= 0.5:
            ax.title.set_color('#2ca02c')  # Green for good performance
        elif f1_score >= 0.4:
            ax.title.set_color('#ff7f0e')  # Orange for moderate performance
        else:
            ax.title.set_color('#d62728')  # Red for poor performance
    
    # Hide empty subplots
    for i in range(n_models, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('voice_model_confusion_matrices_individual.png', dpi=300, bbox_inches='tight')
    print("Individual confusion matrices saved as: voice_model_confusion_matrices_individual.png")
    
    # Create comparative analysis figure
    create_comparative_analysis(df)
    
    # Create performance summary figure
    create_performance_summary(df)
    
    plt.show()

def create_comparative_analysis(df):
    """Create comparative analysis of confusion matrices"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Voice Model Confusion Matrix Analysis', fontsize=16, fontweight='bold')
    
    models = df['Model'].tolist()
    
    # Plot 1: True Positives vs True Negatives
    ax1 = axes[0, 0]
    true_positives = df['TP'].tolist()
    true_negatives = df['TN'].tolist()
    
    x_pos = range(len(models))
    width = 0.35
    
    bars1 = ax1.bar([x - width/2 for x in x_pos], true_negatives, width, 
                    label='True Negatives (Normal)', color='#2ca02c', alpha=0.8)
    bars2 = ax1.bar([x + width/2 for x in x_pos], true_positives, width, 
                    label='True Positives (Parkinson)', color='#1f77b4', alpha=0.8)
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Number of Correct Predictions', fontweight='bold')
    ax1.set_title('Correct Predictions Analysis', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for i, (tn, tp) in enumerate(zip(true_negatives, true_positives)):
        ax1.annotate(str(tn), (i - width/2, tn), ha='center', va='bottom', fontweight='bold')
        ax1.annotate(str(tp), (i + width/2, tp), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: False Positives vs False Negatives
    ax2 = axes[0, 1]
    false_positives = df['FP'].tolist()
    false_negatives = df['FN'].tolist()
    
    bars1 = ax2.bar([x - width/2 for x in x_pos], false_positives, width, 
                    label='False Positives', color='#ff7f0e', alpha=0.8)
    bars2 = ax2.bar([x + width/2 for x in x_pos], false_negatives, width, 
                    label='False Negatives', color='#d62728', alpha=0.8)
    
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('Number of Errors', fontweight='bold')
    ax2.set_title('Error Analysis (FP vs FN)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for i, (fp, fn) in enumerate(zip(false_positives, false_negatives)):
        if fp > 0:
            ax2.annotate(str(fp), (i - width/2, fp), ha='center', va='bottom', fontweight='bold')
        if fn > 0:
            ax2.annotate(str(fn), (i + width/2, fn), ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Sensitivity vs Specificity
    ax3 = axes[1, 0]
    
    # Calculate Sensitivity (Recall) and Specificity
    sensitivity = df['Recall'].tolist()  # TP / (TP + FN)
    specificity = []
    for _, row in df.iterrows():
        tn, fp = row['TN'], row['FP']
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    
    ax3.plot(x_pos, sensitivity, marker='o', linewidth=2.5, markersize=8, 
             label='Sensitivity (Recall)', color='#d62728')
    ax3.plot(x_pos, specificity, marker='s', linewidth=2.5, markersize=8, 
             label='Specificity', color='#2ca02c')
    
    ax3.set_xlabel('Models', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Sensitivity vs Specificity', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # Plot 4: Confusion Matrix Heatmap Summary
    ax4 = axes[1, 1]
    
    # Create a summary matrix showing average values
    metrics_data = []
    for _, row in df.iterrows():
        total_samples = row['TP'] + row['TN'] + row['FP'] + row['FN']
        metrics_data.append([
            row['TN'] / total_samples,  # True Negative Rate
            row['FP'] / total_samples,  # False Positive Rate
            row['FN'] / total_samples,  # False Negative Rate
            row['TP'] / total_samples   # True Positive Rate
        ])
    
    metrics_matrix = np.array(metrics_data)
    
    im = ax4.imshow(metrics_matrix.T, cmap='RdYlBu_r', aspect='auto')
    
    # Set labels
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(['TN Rate', 'FP Rate', 'FN Rate', 'TP Rate'])
    ax4.set_title('Confusion Matrix Rates Heatmap', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Rate', fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(4):
            text = ax4.text(i, j, f'{metrics_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('voice_model_confusion_analysis.png', dpi=300, bbox_inches='tight')
    print("Comparative analysis saved as: voice_model_confusion_analysis.png")

def create_performance_summary(df):
    """Create performance summary with confusion matrix insights"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Voice Model Performance Summary', fontsize=16, fontweight='bold')
    
    models = df['Model'].tolist()
    
    # Plot 1: Model Ranking by F1 Score
    ax1 = axes[0]
    df_sorted = df.sort_values('F1 Score', ascending=True)
    
    colors = ['#d62728' if f1 < 0.3 else '#ff7f0e' if f1 < 0.5 else '#2ca02c' 
              for f1 in df_sorted['F1 Score']]
    
    bars = ax1.barh(range(len(df_sorted)), df_sorted['F1 Score'], color=colors, alpha=0.8)
    
    ax1.set_xlabel('F1 Score', fontweight='bold')
    ax1.set_ylabel('Models (Ranked by F1 Score)', fontweight='bold')
    ax1.set_title('Model Ranking by F1 Score', fontweight='bold')
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['Model'])
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, max(df_sorted['F1 Score']) + 0.1)
    
    # Add value annotations
    for i, (bar, f1) in enumerate(zip(bars, df_sorted['F1 Score'])):
        ax1.annotate(f'{f1:.4f}', 
                    (f1 + 0.01, bar.get_y() + bar.get_height()/2),
                    va='center', fontweight='bold')
    
    # Add performance tier legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#2ca02c', alpha=0.8, label='Good (≥0.5)'),
        Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', alpha=0.8, label='Fair (0.3-0.5)'),
        Rectangle((0, 0), 1, 1, facecolor='#d62728', alpha=0.8, label='Poor (<0.3)')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Accuracy vs F1 Score Scatter
    ax2 = axes[1]
    
    accuracies = df['Accuracy'].tolist()
    f1_scores = df['F1 Score'].tolist()
    
    scatter = ax2.scatter(accuracies, f1_scores, c=f1_scores, cmap='RdYlGn', 
                         s=100, alpha=0.7, edgecolors='black')
    
    # Add model labels
    for i, model in enumerate(models):
        ax2.annotate(model, (accuracies[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, ha='left')
    
    ax2.set_xlabel('Accuracy', fontweight='bold')
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('Accuracy vs F1 Score Relationship', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(accuracies) + 0.1)
    ax2.set_ylim(0, max(f1_scores) + 0.1)
    
    # Add diagonal line for reference
    max_val = max(max(accuracies), max(f1_scores))
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
    ax2.legend()
    
    plt.colorbar(scatter, ax=ax2, label='F1 Score', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('voice_model_performance_summary.png', dpi=300, bbox_inches='tight')
    print("Performance summary saved as: voice_model_performance_summary.png")

def print_detailed_analysis(df):
    """Print detailed analysis of confusion matrices"""
    
    print("\n" + "="*80)
    print("VOICE MODEL CONFUSION MATRIX DETAILED ANALYSIS")
    print("="*80)
    
    print(f"\nDataset Summary:")
    print(f"Total test samples per model: {df.iloc[0]['TP'] + df.iloc[0]['TN'] + df.iloc[0]['FP'] + df.iloc[0]['FN']}")
    
    print(f"\nConfusion Matrix Analysis:")
    print(f"{'Model':<20} {'TN':<4} {'FP':<4} {'FN':<4} {'TP':<4} {'Sensitivity':<12} {'Specificity':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        tn, fp, fn, tp = row['TN'], row['FP'], row['FN'], row['TP']
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"{row['Model']:<20} {tn:<4} {fp:<4} {fn:<4} {tp:<4} "
              f"{sensitivity:<12.4f} {specificity:<12.4f}")
    
    # Find best models for different criteria
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    best_accuracy = df.loc[df['Accuracy'].idxmax()]
    best_precision = df.loc[df['Precision'].idxmax()]
    best_recall = df.loc[df['Recall'].idxmax()]
    
    print(f"\n🏆 Best Performers:")
    print(f"Best F1 Score: {best_f1['Model']} ({best_f1['F1 Score']:.4f})")
    print(f"Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    print(f"Best Precision: {best_precision['Model']} ({best_precision['Precision']:.4f})")
    print(f"Best Recall: {best_recall['Model']} ({best_recall['Recall']:.4f})")
    
    # Analysis insights
    print(f"\n📊 Key Insights:")
    
    # Models with zero false negatives (didn't miss any Parkinson cases)
    zero_fn = df[df['FN'] == 0]['Model'].tolist()
    if zero_fn:
        print(f"✓ Models with zero false negatives (no missed Parkinson cases): {', '.join(zero_fn)}")
    
    # Models with zero false positives (didn't misclassify any normal cases)
    zero_fp = df[df['FP'] == 0]['Model'].tolist()
    if zero_fp:
        print(f"✓ Models with zero false positives (no misclassified normal cases): {', '.join(zero_fp)}")
    
    # Balanced models (similar FP and FN rates)
    balanced_models = []
    for _, row in df.iterrows():
        total = row['TP'] + row['TN'] + row['FP'] + row['FN']
        fp_rate = row['FP'] / total
        fn_rate = row['FN'] / total
        if abs(fp_rate - fn_rate) < 0.1:  # Within 10% difference
            balanced_models.append(row['Model'])
    
    if balanced_models:
        print(f"✓ Well-balanced models (similar error rates): {', '.join(balanced_models)}")
    
    print(f"\n⚠️  Recommendations:")
    print(f"• Use {best_f1['Model']} for best overall performance (highest F1 score)")
    print(f"• Voice models show moderate performance - consider feature engineering")
    print(f"• Ensemble methods might improve results by combining multiple models")

def main():
    """Main function to create all confusion matrix visualizations"""
    print("Creating voice model confusion matrix visualizations...")
    
    try:
        df = pd.read_csv('voice_model_comparison_results.csv')
        
        create_voice_confusion_matrices()
        print_detailed_analysis(df)
        
        print("\n" + "="*60)
        print("CONFUSION MATRIX ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("- voice_model_confusion_matrices_individual.png")
        print("- voice_model_confusion_analysis.png")
        print("- voice_model_performance_summary.png")
        print("\nVisualization includes:")
        print("• Individual confusion matrices for all 9 models")
        print("• True/False positive and negative analysis")
        print("• Sensitivity vs Specificity comparison")
        print("• Model ranking and performance summary")
        print("• Detailed statistical analysis")
        
    except Exception as e:
        print(f"Error creating confusion matrices: {str(e)}")
        print("Make sure voice_model_comparison_results.csv exists in the current directory.")

if __name__ == '__main__':
    main()