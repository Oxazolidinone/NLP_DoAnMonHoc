import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

def map_label(label):
    label_map = {
        "POS": 0,
        "NEG": 1, 
        "NEU": 2
    }
    return label_map[label]

def get_reverse_label_map():
    """Get reverse label mapping"""
    return {0: 'POS', 1: 'NEG', 2: 'NEU'}

def encode_labels(data):
    data['label'] = data['label'].apply(map_label)
    label_mapping = {"POS": 0, "NEG": 1, "NEU": 2}
    reverse_label_map = get_reverse_label_map()
    
    return data, reverse_label_map

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def print_results(eval_results, model_name="MODEL"):
    """Print training results"""
    print("\n" + "="*50)
    print(f"{model_name} TRAINING RESULTS")
    print("="*50)
    print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Test F1-Score: {eval_results['eval_f1']:.4f}")
    print(f"Test Precision: {eval_results['eval_precision']:.4f}")
    print(f"Test Recall: {eval_results['eval_recall']:.4f}")
