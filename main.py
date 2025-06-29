import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from Src.train_multinb_rf import MultinomialNBModel
from Src.train_LLM import LLMSentimentClassifier

def main():
    data_path = "data.csv"
    vncorenlp_path = "./VnCoreNLP/VnCoreNLP-1.1.1.jar"
    results = {}
    print("Training Multinomial NB (Preprocessing + Feature Engineering)...")
    nb_model = MultinomialNBModel(vncorenlp_path, use_parallel=True)
    nb_results, _, _, _ = nb_model.train_model(data_path)
    results['Multinomial_NB'] = nb_results
    results['Multinomial_NB']['type'] = 'ML + Feature Engineering'
    
    print("Training PhoBERT (Chỉ Preprocessing với Custom FC Head)...")
    phobert_model = LLMSentimentClassifier("vinai/phobert-base", vncorenlp_path)
    phobert_results = phobert_model.train_model(data_path, epochs=5, batch_size=16, max_length=256)
    results['PhoBERT'] = phobert_results
    results['PhoBERT']['type'] = 'LLM (Custom PyTorch FC Head)'
    
    print("-" * 80)
    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Model': model_name,
            'Type': model_results['type'],
            'Accuracy': f"{model_results['accuracy']:.4f}",
            'F1-Score': f"{model_results['f1']:.4f}",
            'Precision': f"{model_results['precision']:.4f}",
            'Recall': f"{model_results['recall']:.4f}",
            'Training_Time': f"{model_results['training_time']:.2f}s"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nMô hình tốt nhất: {best_f1[0]} (F1: {best_f1[1]['f1']:.4f})")
    comparison_df.to_csv('comparison_results.csv', index=False)
    
    models = list(results.keys())
    f1_scores = [results[m]['f1'] for m in models]
    accuracies = [results[m]['accuracy'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = ['blue', 'red'] 
    ax1.bar(models, f1_scores, color=colors, alpha=0.7)
    ax1.set_title('F1-Score Comparison')
    ax1.set_ylabel('F1-Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax2.bar(models, accuracies, color=colors, alpha=0.7)
    ax2.set_title('Accuracy Comparison')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
