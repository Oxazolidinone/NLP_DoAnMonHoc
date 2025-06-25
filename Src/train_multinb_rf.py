import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import time
from Src.preprocessing import preprocess_text
from Src.feature_enginerring import VietnameseSentimentFeatureExtractor
import warnings
warnings.filterwarnings('ignore')

class TraditionalMLModels:
    def __init__(self, model_name='MultinomialNB', vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar"):
        self.model_name = model_name
        self.feature_extractor = VietnameseSentimentFeatureExtractor(vncorenlp_path)
        self.label_encoder = LabelEncoder()
        self.model = None 
        self.is_fitted = False

    def train_model(self, data_path, test_size=0.2, random_state=42):
        df = pd.read_csv(data_path)
        processed_texts = preprocess_text(df['text'].tolist())
        features = self.feature_extractor.extract_batch_features(processed_texts, include_ngram=True)
        labels = self.label_encoder.fit_transform(df['label'])
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
        
        start_time = time.time()
        
        if self.model_name == 'RandomForest':
            X_train_processed = X_train
            X_test_processed = X_test
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
            self.model.fit(X_train_processed, y_train)
            y_pred = self.model.predict(X_test_processed)

            
        else:
            X_train_processed = np.maximum(X_train, 0)
            X_test_processed = np.maximum(X_test, 0)
            
            self.model = MultinomialNB(alpha=1.0)
            self.model.fit(X_train_processed, y_train)
            y_pred = self.model.predict(X_test_processed)

        
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time
        }
        
        print(f"{self.model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"\n{self.model_name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        self.is_fitted = True
        return results, X_test, y_test, y_pred    
    
    def predict(self, text):
        processed_text = preprocess_text([text])
        features = self.feature_extractor.extract_batch_features(processed_text, include_ngram=True)
        
        if self.model_name == 'MultinomialNB':
            features = np.maximum(features, 0)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        label = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        return label, confidence
    
    def save_model(self, model_path="./models"):
        joblib.dump(self.model, f"{model_path}/{self.model_name}_model.pkl")
        joblib.dump(self.label_encoder, f"{model_path}/label_encoder.pkl")
    
    def load_model(self, model_path="./models"):
        self.model = joblib.load(f"{model_path}/{self.model_name}_model.pkl")
        self.label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
        self.is_fitted = True
    
    def close(self):
        self.feature_extractor.close()

def train_traditional_model(data_path, model_name='MultinomialNB'):
    trainer = TraditionalMLModels(model_name=model_name)
    results, X_test, y_test, y_pred = trainer.train_model(data_path)
    return trainer, results

def predict_text(text, model_name='MultinomialNB', model_path="./models"):
    trainer = TraditionalMLModels(model_name=model_name)
    trainer.load_model(model_path)
    prediction, confidence = trainer.predict(text)
    trainer.close()
    return prediction, confidence

