import numpy as np
from typing import List, Dict
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from vncorenlp import VnCoreNLP
import re
class VietnameseSentimentFeatureExtractor:
    def __init__(self, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar"):
        self.nlp = VnCoreNLP(vncorenlp_path, annotators="wseg,pos", max_heap_size='-Xmx1g')
        self.tfidf_vectorizer = None
        
        self.positive_words = {
            'tốt', 'hay', 'đẹp', 'thích', 'yêu', 'vui', 'hạnh phúc', 'tuyệt vời', 
            'xuất sắc', 'hoàn hảo', 'tuyệt', 'ổn', 'ok', 'ngon', 'chất lượng',
            'hài lòng', 'thoải mái', 'dễ chịu', 'ấn tượng', 'tích cực'
        }
        
        self.negative_words = {
            'xấu', 'tệ', 'ghét', 'buồn', 'tức giận', 'khó chịu', 'thất vọng',
            'dở', 'kém', 'tồi', 'chán', 'phiền', 'bực', 'giận', 'lo lắng',
            'stress', 'mệt', 'không hài lòng', 'thất bại', 'tiêu cực'
        }
        
        self.intensifiers = {
            'rất', 'cực kỳ', 'vô cùng', 'cực', 'siêu', 'hết sức', 'quá', 
            'thật', 'thực sự', 'hoàn toàn', 'tuyệt đối', 'hơi', 'khá'
        }
        
        self.negation_words = {
            'không', 'chưa', 'chẳng', 'chả', 'đừng', 'đừng có', 'không có'
        }
    
    def extract_sentiment_lexicon_features(self, text: str) -> Dict:
        tokens = self.nlp.tokenize(text)
        words = [word.lower() for sentence in tokens for word in sentence]
        
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        intensifier_count = sum(1 for word in words if word in self.intensifiers)
        negation_count = sum(1 for word in words if word in self.negation_words)
        
        total_words = len(words)
        
        features = {
            'positive_word_count': pos_count,
            'negative_word_count': neg_count,
            'positive_word_ratio': pos_count / total_words if total_words > 0 else 0,
            'negative_word_ratio': neg_count / total_words if total_words > 0 else 0,
            'sentiment_polarity': (pos_count - neg_count) / total_words if total_words > 0 else 0,
            'intensifier_count': intensifier_count,
            'intensifier_ratio': intensifier_count / total_words if total_words > 0 else 0,
            'negation_count': negation_count,
            'negation_ratio': negation_count / total_words if total_words > 0 else 0,
            'sentiment_strength': (pos_count + neg_count) / total_words if total_words > 0 else 0
        }
        return features
    
    def extract_pos_sentiment_features(self, text: str) -> Dict:
        annotated = self.nlp.annotate(text)
        pos_tags = []
        for sentence in annotated['sentences']:
            for token in sentence:
                pos_tags.append(token['posTag'])
        
        pos_counts = Counter(pos_tags)
        total_pos = len(pos_tags)
        
        features = {
            'adj_ratio': pos_counts.get('A', 0) / total_pos if total_pos > 0 else 0,
            'adv_ratio': pos_counts.get('R', 0) / total_pos if total_pos > 0 else 0,
            'verb_ratio': pos_counts.get('V', 0) / total_pos if total_pos > 0 else 0,
            'adj_adv_ratio': (pos_counts.get('A', 0) + pos_counts.get('R', 0)) / total_pos if total_pos > 0 else 0
        }
        return features
    
    def extract_punctuation_features(self, text: str) -> Dict:
        total_chars = len(text)
        
        features = {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'exclamation_ratio': text.count('!') / total_chars if total_chars > 0 else 0,
            'question_ratio': text.count('?') / total_chars if total_chars > 0 else 0,
            'caps_ratio': sum(1 for c in text if c.isupper()) / total_chars if total_chars > 0 else 0,
            'ellipsis_count': len(re.findall(r'\.{3,}', text)),
            'multiple_punct': len(re.findall(r'[!?]{2,}', text))
        }
        return features
    
    def extract_context_features(self, text: str) -> Dict:
        tokens = self.nlp.tokenize(text)
        words = [word.lower() for sentence in tokens for word in sentence]
        
        negated_sentiment = 0
        intensified_sentiment = 0
        
        for i, word in enumerate(words):
            if word in self.positive_words or word in self.negative_words:
                if i > 0 and words[i-1] in self.negation_words:
                    negated_sentiment += 1
                if i > 0 and words[i-1] in self.intensifiers:
                    intensified_sentiment += 1
        
        features = {
            'negated_sentiment_count': negated_sentiment,
            'intensified_sentiment_count': intensified_sentiment,
            'sentence_count': len(tokens),
            'avg_sentence_length': len(words) / len(tokens) if tokens else 0
        }
        return features
    
    def extract_ngram_features(self, texts: List[str], max_features: int = 2000) -> np.ndarray:
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=max_features,
                min_df=2,
                max_df=0.8,
                lowercase=True,
                analyzer='word'
            )
            return self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            return self.tfidf_vectorizer.transform(texts).toarray()
    
    def extract_all_features(self, text: str) -> Dict:
        features = {}
        features.update(self.extract_sentiment_lexicon_features(text))
        features.update(self.extract_pos_sentiment_features(text))
        features.update(self.extract_punctuation_features(text))
        features.update(self.extract_context_features(text))
        return features
    
    def extract_batch_features(self, texts: List[str], include_ngram: bool = True) -> np.ndarray:
        basic_features = []
        for text in texts:
            features = self.extract_all_features(text)
            basic_features.append(list(features.values()))
        
        basic_features = np.array(basic_features)
        
        if include_ngram:
            ngram_features = self.extract_ngram_features(texts)
            basic_features = np.hstack([basic_features, ngram_features])
        
        return basic_features
    
    def close(self):
        self.nlp.close()


def extract_sentiment_features(texts: List[str], vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar") -> np.ndarray:
    extractor = VietnameseSentimentFeatureExtractor(vncorenlp_path)
    features = extractor.extract_batch_features(texts)
    extractor.close()
    return features


def get_sentiment_feature_names() -> List[str]:
    return [
        'positive_word_count', 'negative_word_count', 'positive_word_ratio', 'negative_word_ratio',
        'sentiment_polarity', 'intensifier_count', 'intensifier_ratio', 'negation_count', 
        'negation_ratio', 'sentiment_strength', 'adj_ratio', 'adv_ratio', 'verb_ratio',
        'adj_adv_ratio', 'exclamation_count', 'question_count', 'exclamation_ratio', 
        'question_ratio', 'caps_ratio', 'ellipsis_count', 'multiple_punct',
        'negated_sentiment_count', 'intensified_sentiment_count', 'sentence_count', 'avg_sentence_length'
    ]