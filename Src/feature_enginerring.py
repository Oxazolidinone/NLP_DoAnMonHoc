import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import threading
from tqdm import tqdm
from scipy import sparse

class VietnameseSentimentFeatureExtractor:
    def __init__(self, vncorenlp_path: str = None):
        self.vncorenlp_path = vncorenlp_path  # Keep for compatibility
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,  # Further reduced from 5000 to 3000
            ngram_range=(1, 2),
            min_df=5,  # Increased from 3 to 5
            max_df=0.8,  # Reduced from 0.9 to 0.8
            lowercase=True,
            sublinear_tf=True,
            norm='l2'
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=1500,  # Further reduced from 2000 to 1500
            ngram_range=(1, 2),
            min_df=5,  # Increased from 3 to 5
            max_df=0.8,  # Reduced from 0.9 to 0.8
            lowercase=True
        )

        self.feature_selector = SelectKBest(chi2, k=500) 
        self.pca = PCA(n_components=30)  
        self.svd = TruncatedSVD(n_components=30, random_state=42)  
        
        self.is_fitted = False
    
    def extract_tfidf_features(self, texts: List[str], fit: bool = True):
        if fit:
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            features = self.tfidf_vectorizer.transform(texts)
        return features  # Keep as sparse matrix
    
    def extract_count_features(self, texts: List[str], fit: bool = True):
        """Extract Count features - returns sparse matrix"""
        if fit:
            features = self.count_vectorizer.fit_transform(texts)
        else:
            features = self.count_vectorizer.transform(texts)
        return features  # Keep as sparse matrix
    
    def apply_feature_selection(self, X, y: np.ndarray = None, fit: bool = True):
        """Apply feature selection using chi2 - handles sparse matrices"""
        if fit and y is not None:
            # Ensure non-negative values for chi2
            if sparse.issparse(X):
                X.data = np.maximum(X.data, 0)
            else:
                X = np.maximum(X, 0)
            features = self.feature_selector.fit_transform(X, y)
        else:
            if sparse.issparse(X):
                X.data = np.maximum(X.data, 0)
            else:
                X = np.maximum(X, 0)
            features = self.feature_selector.transform(X)
        return features
    
    def apply_pca(self, X, fit: bool = True):
        """Apply PCA dimensionality reduction - converts to dense"""
        if sparse.issparse(X):
            X = X.toarray()  # PCA requires dense arrays
        if fit:
            features = self.pca.fit_transform(X)
        else:
            features = self.pca.transform(X)
        return features
    
    def apply_svd(self, X, fit: bool = True):
        """Apply SVD dimensionality reduction - works with sparse matrices"""
        if fit:
            features = self.svd.fit_transform(X)  # TruncatedSVD works with sparse matrices
        else:
            features = self.svd.transform(X)
        return features
    
    def extract_batch_features(self, texts: List[str], y: np.ndarray = None, 
                             use_tfidf: bool = True, use_count: bool = True,
                             use_feature_selection: bool = True, use_pca: bool = False,
                             use_svd: bool = True):
        """Extract comprehensive features for batch of texts - memory optimized"""
        if not texts:
            return np.array([])
        
        features_list = []
        
        # Extract TF-IDF features
        if use_tfidf:
            tfidf_features = self.extract_tfidf_features(texts, fit=not self.is_fitted)
            features_list.append(tfidf_features)
        
        # Extract Count features
        if use_count:
            count_features = self.extract_count_features(texts, fit=not self.is_fitted)
            features_list.append(count_features)
        
        # Combine features (horizontally stack sparse matrices)
        if features_list:
            combined_features = sparse.hstack(features_list, format='csr')
        else:
            return np.array([])
        
        # Apply feature selection (works with sparse matrices)
        if use_feature_selection and y is not None:
            combined_features = self.apply_feature_selection(combined_features, y, fit=not self.is_fitted)
        
        # Apply dimensionality reduction (prefer SVD for sparse matrices)
        if use_svd:
            combined_features = self.apply_svd(combined_features, fit=not self.is_fitted)
        elif use_pca:
            combined_features = self.apply_pca(combined_features, fit=not self.is_fitted)
        
        self.is_fitted = True
        
        # Convert to dense only at the very end and only if needed
        if sparse.issparse(combined_features):
            combined_features = combined_features.toarray()
        
        return combined_features


class ParallelVietnameseSentimentFeatureExtractor:
    """Parallel version of Vietnamese sentiment feature extractor"""
    def __init__(self, vncorenlp_path: str = None, n_workers: int = None):
        self.vncorenlp_path = vncorenlp_path  # Keep for compatibility
        self.n_workers = n_workers or min(cpu_count(), 4)
        self.main_extractor = VietnameseSentimentFeatureExtractor(vncorenlp_path)
        self._local = threading.local()
    
    def _get_extractor(self):
        """Get thread-local extractor instance"""
        if not hasattr(self._local, 'extractor'):
            self._local.extractor = VietnameseSentimentFeatureExtractor(self.vncorenlp_path)
        return self._local.extractor
    
    def extract_batch_features_parallel(self, texts: List[str], y: np.ndarray = None,
                                       use_parallel: bool = True, **kwargs) -> np.ndarray:
        """Extract features for batch of texts with optional parallel processing"""
        if not texts:
            return np.array([])
        
        # For feature extraction, we don't parallelize the sklearn operations
        # as they are already optimized. Just use the main extractor.
        return self.main_extractor.extract_batch_features(texts, y, **kwargs)
