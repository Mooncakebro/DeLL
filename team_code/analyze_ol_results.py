"""
analyze_results.py - Utility to analyze and visualize evaluation results
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_analyze_results(results_dir):
    """Load and analyze saved evaluation results"""
    results_dir = Path(results_dir)
    
    # Load metrics
    with open(results_dir / 'evaluation_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Load detailed predictions
    data = np.load(results_dir / 'detailed_predictions.npz', allow_pickle=True)
    predictions = data['predictions']
    ground_truths = data['ground_truths']
    
    print("="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # 1. Basic statistics
    print(f"\nNumber of evaluated samples: {len(predictions)}")
    
    # 2. Trajectory error analysis
    if 'trajectory' in metrics:
        traj_metrics = metrics['trajectory']
        print("\n--- Trajectory Error Analysis ---")
        
        for key in ['ADE', 'FDE']:
            if key in traj_metrics:
                values = traj_metrics[key]
                print(f"{key}:")
                print(f"  Mean: {np.mean(values):.4f}m")
                print(f"  Std: {np.std(values):.4f}m")
                print(f"  25th percentile: {np.percentile(values, 25):.4f}m")
                print(f"  50th percentile: {np.percentile(values, 50):.4f}m")
                print(f"  75th percentile: {np.percentile(values, 75):.4f}m")
                print(f"  95th percentile: {np.percentile(values, 95):.4f}m")
                print(f"  Max: {np.max(values):.4f}m")
    
    # 3. Error distribution by distance
    print("\n--- Error vs Trajectory Length ---")
    
    # Calculate trajectory lengths
    traj_lengths = []
    for gt in ground_truths:
        points = gt['checkpoints']
        length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        traj_lengths.append(length)
    
    traj_lengths = np.array(traj_lengths)
    
    if 'ADE' in traj_metrics:
        ades = np.array(traj_metrics['ADE'])
        
        # Bin by trajectory length
        bins = np.linspace(traj_lengths.min(), traj_lengths.max(), 6)
        bin_indices = np.digitize(traj_lengths, bins)
        
        print("\nADE by trajectory length:")
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.any():
                bin_ades = ades[mask]
                print(f"  Length {bins[i-1]:.1f}-{bins[i]:.1f}m: "
                      f"{np.mean(bin_ades):.4f}m ± {np.std(bin_ades):.4f}m "
                      f"(n={mask.sum()})")
    
    # 4. Worst case analysis
    if 'ADE' in traj_metrics:
        ades = np.array(traj_metrics['ADE'])
        worst_indices = np.argsort(ades)[-10:]  # Top 10 worst
        
        print("\n--- Worst Performing Samples ---")
        for i, idx in enumerate(worst_indices[::-1]):  # From worst to better
            print(f"{i+1}. Sample {idx}: ADE={ades[idx]:.4f}m, "
                  f"Speed Pred={predictions[idx]['speed_class']}, "
                  f"Speed GT={ground_truths[idx]['speed_class']}")
    
    # 5. Generate comprehensive report
    generate_comprehensive_report(metrics, predictions, ground_truths, results_dir)

def generate_comprehensive_report(metrics, predictions, ground_truths, output_dir):
    """Generate a comprehensive PDF report"""
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = output_dir / "comprehensive_report.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Executive Summary
        fig = plt.figure(figsize=(11, 8.5))
        plt.suptitle('Evaluation Report - Executive Summary', fontsize=16, fontweight='bold')
        
        # Create summary table
        summary_data = []
        
        if 'summary' in metrics:
            summary = metrics['summary']
            
            # Trajectory metrics
            if 'ADE_mean' in summary:
                summary_data.append(['Trajectory ADE (m)', f"{summary['ADE_mean']:.4f} ± {summary['ADE_std']:.4f}"])
                summary_data.append(['Trajectory FDE (m)', f"{summary.get('FDE_mean', 0):.4f} ± {summary.get('FDE_std', 0):.4f}"])
            
            # Speed metrics
            if 'speed_accuracy' in summary:
                summary_data.append(['Speed Accuracy (%)', f"{summary['speed_accuracy']:.2f}"])
            
            # Overall score
            if 'overall_score' in summary:
                summary_data.append(['Overall Score', f"{summary['overall_score']:.4f}"])
        
        # Create table
        ax = plt.subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        if summary_data:
            table = ax.table(cellText=summary_data,
                           colLabels=['Metric', 'Value'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
        
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Detailed trajectory metrics
        if 'trajectory' in metrics:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            axes = axes.flatten()
            
            traj_metrics = metrics['trajectory']
            
            # ADE vs FDE scatter
            if 'ADE' in traj_metrics and 'FDE' in traj_metrics:
                axes[0].scatter(traj_metrics['ADE'], traj_metrics['FDE'], alpha=0.5)
                axes[0].set_xlabel('ADE (m)')
                axes[0].set_ylabel('FDE (m)')
                axes[0].set_title('ADE vs FDE Correlation')
                axes[0].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                correlation = np.corrcoef(traj_metrics['ADE'], traj_metrics['FDE'])[0, 1]
                axes[0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                           transform=axes[0].transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Cumulative distribution of ADE
            if 'ADE' in traj_metrics:
                sorted_ade = np.sort(traj_metrics['ADE'])
                cdf = np.arange(1, len(sorted_ade) + 1) / len(sorted_ade)
                
                axes[1].plot(sorted_ade, cdf, 'b-', linewidth=2)
                axes[1].set_xlabel('ADE (m)')
                axes[1].set_ylabel('Cumulative Probability')
                axes[1].set_title('CDF of ADE')
                axes[1].grid(True, alpha=0.3)
                
                # Add percentile markers
                for percentile in [50, 75, 90, 95]:
                    value = np.percentile(sorted_ade, percentile)
                    axes[1].axvline(value, color='r', linestyle='--', alpha=0.5)
                    axes[1].text(value, 0.5, f'{percentile}%: {value:.2f}m',
                               rotation=90, fontsize=8, verticalalignment='bottom')
            
            # Lateral vs Longitudinal error distribution
            if 'Avg_Lateral_Error' in traj_metrics and 'Avg_Longitudinal_Error' in traj_metrics:
                axes[2].hist2d(traj_metrics['Avg_Lateral_Error'],
                             traj_metrics['Avg_Longitudinal_Error'],
                             bins=30, cmap='viridis')
                axes[2].set_xlabel('Average Lateral Error (m)')
                axes[2].set_ylabel('Average Longitudinal Error (m)')
                axes[2].set_title('Error Distribution')
                plt.colorbar(axes[2].images[0], ax=axes[2])
            
            # Box plot of pointwise errors
            if 'Pointwise_Errors' in traj_metrics:
                pointwise_data = np.array(traj_metrics['Pointwise_errors']).T
                bp = axes[3].boxplot(pointwise_data, showfliers=False)
                axes[3].set_xlabel('Waypoint Index')
                axes[3].set_ylabel('Error (m)')
                axes[3].set_title('Pointwise Error Distribution')
                axes[3].set_xticks(range(1, len(pointwise_data) + 1, 2))
                axes[3].set_xticklabels(range(1, len(pointwise_data) + 1, 2))
                axes[3].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Trajectory Prediction Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # Page 3: Speed classification analysis
        if 'speed' in metrics:
            speed_metrics = metrics['speed']
            
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            
            # Confusion matrix (if available in summary)
            if 'summary' in metrics and 'confusion_matrix' in metrics['summary']:
                cm = np.array(metrics['summary']['confusion_matrix'])
                
                im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                axes[0, 0].set_title('Confusion Matrix')
                plt.colorbar(im, ax=axes[0, 0])
                
                # Add percentages
                total = cm.sum(axis=1)[:, np.newaxis]
                percentage = cm / total * 100
                
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        text = f'{cm[i, j]}\n({percentage[i, j]:.1f}%)'
                        axes[0, 0].text(j, i, text, ha='center', va='center',
                                      color='white' if cm[i, j] > cm.max()/2 else 'black')
            
            # Accuracy by speed class
            if 'Predicted_Classes' in speed_metrics and 'True_Classes' in speed_metrics:
                pred_classes = speed_metrics['Predicted_Classes']
                true_classes = speed_metrics['True_Classes']
                
                # Calculate per-class accuracy
                unique_classes = np.unique(true_classes + pred_classes)
                class_accuracy = []
                
                for cls in unique_classes:
                    mask = true_classes == cls
                    if mask.any():
                        correct = np.sum(np.array(pred_classes)[mask] == cls)
                        total = np.sum(mask)
                        class_accuracy.append(correct / total * 100)
                    else:
                        class_accuracy.append(0)
                
                axes[0, 1].bar(unique_classes, class_accuracy)
                axes[0, 1].set_xlabel('Speed Class')
                axes[0, 1].set_ylabel('Accuracy (%)')
                axes[0, 1].set_title('Per-Class Accuracy')
                axes[0, 1].set_xticks(unique_classes)
                axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Confidence distribution
            if 'Prediction_Confidence' in speed_metrics:
                axes[1, 0].hist(speed_metrics['Prediction_Confidence'], bins=30, alpha=0.7)
                axes[1, 0].set_xlabel('Prediction Confidence')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Confidence Distribution')
                axes[1, 0].axvline(np.mean(speed_metrics['Prediction_Confidence']),
                                 color='r', linestyle='--', label='Mean')
                axes[1, 0].legend()
            
            # Confidence vs Accuracy
            if 'Prediction_Confidence' in speed_metrics and 'Accuracy' in speed_metrics:
                confidences = speed_metrics['Prediction_Confidence']
                accuracy = speed_metrics['Accuracy']
                
                # Bin by confidence
                bins = np.linspace(0, 1, 11)
                bin_accuracies = []
                bin_counts = []
                
                for i in range(len(bins)-1):
                    mask = (confidences >= bins[i]) & (confidences < bins[i+1])
                    if mask.any():
                        bin_accuracies.append(np.mean(np.array(accuracy)[mask]) * 100)
                        bin_counts.append(np.sum(mask))
                    else:
                        bin_accuracies.append(0)
                        bin_counts.append(0)
                
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                axes[1, 1].bar(bin_centers, bin_accuracies, width=0.08)
                axes[1, 1].set_xlabel('Confidence Bin')
                axes[1, 1].set_ylabel('Accuracy (%)')
                axes[1, 1].set_title('Accuracy vs Confidence')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                # Add count labels
                for center, acc, count in zip(bin_centers, bin_accuracies, bin_counts):
                    if count > 0:
                        axes[1, 1].text(center, acc + 1, f'n={count}',
                                      ha='center', fontsize=8)
            
            plt.suptitle('Speed Classification Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"\nComprehensive report saved to: {pdf_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing evaluation results')
    
    args = parser.parse_args()
    
    load_and_analyze_results(args.results_dir)