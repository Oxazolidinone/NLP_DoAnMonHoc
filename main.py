import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from train_multinb_rf import train_traditional_model, predict_text as predict_traditional
from train_LLM import train_llm, predict_text
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparison:
    def __init__(self):
        self.results = {}
        self.models = {}
        
    def train_all_models(self, data_path, epochs=3):
        print("STARTING MODEL TRAINING")
        print("="*80)
        
        # Train MultinomialNB
        print("Training MultinomialNB...")
        start_time = time.time()
        trainer_nb, results_nb = train_traditional_model(data_path, model_name='MultinomialNB')
        self.results['MultinomialNB'] = results_nb
        self.models['MultinomialNB'] = trainer_nb
        print(f"MultinomialNB training completed in {time.time() - start_time:.2f}s")

        # Train RandomForest
        print("\nTraining RandomForest...")
        start_time = time.time()
        trainer_rf, results_rf = train_traditional_model(data_path, model_name='RandomForest')
        self.results['RandomForest'] = results_rf
        self.models['RandomForest'] = trainer_rf
        print(f"RandomForest training completed in {time.time() - start_time:.2f}s")
            
        # Train PhoBERT
        print("\nTraining PhoBERT...")
        start_time = time.time()
        trainer_pho, _, results_pho = train_llm(data_path, model_name='vinai/phobert-base', epochs=epochs)
        self.results['PhoBERT'] = {
            'accuracy': results_pho.get('eval_accuracy', 0),
            'precision': results_pho.get('eval_precision', 0),
            'recall': results_pho.get('eval_recall', 0),
            'f1': results_pho.get('eval_f1', 0),
            'training_time': time.time() - start_time
        }
        self.models['PhoBERT'] = trainer_pho
        print(f"PhoBERT training completed in {time.time() - start_time:.2f}s")
            
        # Train XLM-RoBERTa
        print("\nTraining XLM-RoBERTa...")
        start_time = time.time()
        trainer_xlm, _, results_xlm = train_llm(data_path, model_name='xlm-roberta-base', epochs=epochs)
        self.results['XLM-RoBERTa'] = {
            'accuracy': results_xlm.get('eval_accuracy', 0),
            'precision': results_xlm.get('eval_precision', 0),
            'recall': results_xlm.get('eval_recall', 0),
            'f1': results_xlm.get('eval_f1', 0),
            'training_time': time.time() - start_time
        }
        self.models['XLM-RoBERTa'] = trainer_xlm
        print(f"XLM-RoBERTa training completed in {time.time() - start_time:.2f}s")
            
    def print_comparison_table(self):
        print("\n" + "="*100)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*100)
        
        if not self.results:
            print("No results available")
            return
            
        df = pd.DataFrame(self.results).T
        print(df.round(4))
                
    def create_visualizations(self):
        if not self.results:
            print("No results to visualize")
            return
            
        df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[i//2, i%2]
                bars = ax.bar(df.index, df[metric], color=colors[i], alpha=0.7)
                ax.set_title(f'{metric.capitalize()} by Model')
                ax.set_ylabel(metric.capitalize())
                ax.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, df[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Visualization saved as 'model_comparison.png'")
        
    def predict_single_text(self, text):
        print(f"\n Text: '{text}'")
        print("-" * 60)
        
        predictions = {}
        pred, conf = predict_traditional(text, model_name='MultinomialNB')
        predictions['MultinomialNB'] = (pred, conf)
        print(f"MultinomialNB: {pred} (confidence: {conf:.4f})")
            
        pred, conf = predict_traditional(text, model_name='RandomForest')
        predictions['RandomForest'] = (pred, conf)
        print(f"RandomForest: {pred} (confidence: {conf:.4f})")

        pred, conf = predict_text(text, model_path='./PhoBERT_model')
        predictions['PhoBERT'] = (pred, conf)
        print(f"PhoBERT: {pred} (confidence: {conf:.4f})")

        pred, conf = predict_text(text, model_path='./XLM-RoBERTa_model')
        predictions['XLM-RoBERTa'] = (pred, conf)
        print(f"XLM-RoBERTa: {pred} (confidence: {conf:.4f})")
            
        return predictions
        
    def cleanup(self):
        for model_name, model in self.models.items():
            if hasattr(model, 'close'):
                model.close()

def main():
    data_path = "data.csv"
    
    print("COMPREHENSIVE MODEL COMPARISON")
    print("Models: MultinomialNB, RandomForest, PhoBERT, XLM-RoBERTa")
    print("="*80)
    
    comparison = ModelComparison()
    
    # Train all models
    comparison.train_all_models(data_path, epochs=3)
    
    # Print comparison results
    comparison.print_comparison_table()
    
    # Create visualizations
    comparison.create_visualizations()
    
    # Test sample predictions
    test_texts = [
        "Sản phẩm này rất tốt, tôi rất hài lòng!",
        "Chất lượng kém, không đáng tiền",
        "Bình thường, không có gì đặc biệt",
        "Tuyệt vời! Hoàn toàn vượt mong đợi",
        "Dịch vụ tệ, không khuyên dùng"
    ]
    
    print("\n" + "="*80)
    print("TESTING SAMPLE PREDICTIONS")
    print("="*80)
    
    for text in test_texts:
        comparison.predict_single_text(text)
            

if __name__ == "__main__":
    main()
