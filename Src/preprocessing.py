import regex as re
import string
import json
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import threading
from functools import partial
from tqdm import tqdm
from vncorenlp import VnCoreNLP

# Emoji pattern
emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)

def clean_text(text):
    """Clean text using the provided function"""
    text = text.lower()
    text = re.sub(emoji_pattern, " ", text)
    text = re.sub(r'([a-z]+?)\1+',r'\1', text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])\s*(\w)", r"\1 \2 \3", text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])", r"\1 \2", text)
    text = re.sub(f"([{string.punctuation}])([{string.punctuation}])+",r"\1", text)
    text = text.strip()
    while text.endswith(tuple(string.punctuation+string.whitespace)):
        text = text[:-1]
    while text.startswith(tuple(string.punctuation+string.whitespace)):
        text = text[1:]
    text = re.sub(r"\s+", " ", text)
    return text

def map_label(label):
    """Map labels to integers"""
    label_map = {
        "POS": 0,
        "NEG": 1,
        "NEU": 2
    }
    return label_map[label]

class VietnameseTextPreprocessor:
    def __init__(self, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar"):
        # Initialize VnCoreNLP for word segmentation
        self.nlp = VnCoreNLP(vncorenlp_path, annotators="wseg", max_heap_size='-Xmx500m')
        
        # Abbreviation dictionary for Vietnamese text
        self.abbrev_dict = {
            'ko': 'không', 'k': 'không', 'khong': 'không', 'kg': 'không', 'hok': 'không',
            'dc': 'được', 'đc': 'được', 'đk': 'được', 'r': 'rồi', 'vs': 'với',
            'j': 'gì', 'bt': 'bình thường', 'mk': 'mình', 'mik': 'mình',
            'cx': 'cũng', 'bn': 'bạn', 'b': 'bạn', 'tks': 'cảm ơn', 'thanks': 'cảm ơn',
            'nchung': 'nói chung', 'nhìu': 'nhiều', 'nhieu': 'nhiều',
            'cty': 'công ty', 'nt': 'nhắn tin', 'sp': 'sản phẩm',
            'i': 'giống', 'sz': 'size', 'sdt': 'số điện thoại',
            'ok': 'tốt', 'oke': 'tốt', 'okey': 'tốt', 'okie': 'tốt',
            'ok': 'tốt', 'oke': 'tốt', 'okey': 'tốt', 'okie': 'tốt',
            'okeyy': 'tốt', 'okiee': 'tốt', 'thik': 'thích',
            'thix': 'thích', 'ib': 'nhắn tin', 'ibx': 'nhắn tin',
            'inbox': 'nhắn tin', 'inb': 'nhắn tin', 'inbx': 'nhắn tin',
            'sz': 'cỡ', 'mn': 'mọi người', 'mng': 'mọi người',
            'sr': 'xin lỗi', 'sorry': 'xin lỗi', 'éo': 'không', 'kh': 'không',
            'rep': 'trả lời', 'ship': 'giao hàng', 'h': 'giờ',
            'lm': 'làm', 'rùi': 'rồi', 'tl': 'trả lời',
        }

    def normalize_abbreviation(self, text: str) -> str:
        """Normalize abbreviations"""
        words = text.split()
        normalized_words = [self.abbrev_dict.get(w.lower(), w) for w in words]
        return ' '.join(normalized_words)

    def word_segment(self, text: str) -> str:
        """Perform word segmentation using VnCoreNLP"""
        try:
            annotated = self.nlp.annotate(text)
            segmented_words = []
            for sentence in annotated['sentences']:
                for token in sentence:
                    segmented_words.append(token['form'])
            return ' '.join(segmented_words)
        except:
            # Fallback to original text if segmentation fails
            return text

    def preprocess(self, text: str, use_word_segmentation: bool = True) -> str:
        """Complete preprocessing pipeline"""
        if not isinstance(text, str):
            text = str(text)
        
        # Clean text
        text = clean_text(text)
        
        # Normalize abbreviations
        text = self.normalize_abbreviation(text)
        
        # Word segmentation
        if use_word_segmentation:
            text = self.word_segment(text)
        
        return text

    def close(self):
        """Close VnCoreNLP instance"""
        if hasattr(self, 'nlp'):
            self.nlp.close()


class ParallelVietnameseTextPreprocessor:
    """Parallel version of Vietnamese text preprocessor"""
    def __init__(self, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar", n_workers: int = None):
        self.vncorenlp_path = vncorenlp_path
        self.n_workers = n_workers or min(cpu_count(), 4)
        
        # Initialize one preprocessor for main thread
        self.main_preprocessor = VietnameseTextPreprocessor(vncorenlp_path)
        
        # Thread-local storage for preprocessors
        self._local = threading.local()
    
    def _get_preprocessor(self):
        """Get thread-local preprocessor instance"""
        if not hasattr(self._local, 'preprocessor'):
            self._local.preprocessor = VietnameseTextPreprocessor(self.vncorenlp_path)
        return self._local.preprocessor
    
    def _preprocess_single(self, text: str, use_word_segmentation: bool = True) -> str:
        """Process single text - used by workers"""
        try:
            preprocessor = self._get_preprocessor()
            return preprocessor.preprocess(text, use_word_segmentation)
        except Exception:
            return str(text)  # Return original text if error
    
    def preprocess_batch(self, texts: List[str], use_parallel: bool = True, use_word_segmentation: bool = True) -> List[str]:
        """Process batch of texts with optional parallel processing"""
        if not texts:
            return []
        
        if len(texts) < 10 or not use_parallel:
            # Sequential processing for small batches
            results = []
            with tqdm(texts, desc="Processing texts") as pbar:
                for text in pbar:
                    results.append(self.main_preprocessor.preprocess(text, use_word_segmentation))
            return results
        
        # Parallel processing for larger batches
        try:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                process_func = partial(self._preprocess_single, use_word_segmentation=use_word_segmentation)
                with tqdm(total=len(texts), desc="Processing texts") as pbar:
                    futures = [executor.submit(process_func, text) for text in texts]
                    results = []
                    for future in futures:
                        results.append(future.result())
                        pbar.update(1)
            return results
        except Exception as e:
            # Fallback to sequential
            results = []
            with tqdm(texts, desc="Processing texts (fallback)") as pbar:
                for text in pbar:
                    results.append(self.main_preprocessor.preprocess(text, use_word_segmentation))
            return results

    def close(self):
        """Close all preprocessor instances"""
        if hasattr(self, 'main_preprocessor'):
            self.main_preprocessor.close()
        
        # Close thread-local preprocessors
        if hasattr(self._local, 'preprocessor'):
            self._local.preprocessor.close()


def preprocess_text(texts, use_parallel: bool = True, n_workers: int = None, use_word_segmentation: bool = True, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar"):
    if isinstance(texts, str):
        # Single text - use regular preprocessor
        preprocessor = VietnameseTextPreprocessor(vncorenlp_path)
        result = preprocessor.preprocess(texts, use_word_segmentation)
        preprocessor.close()
        return result
    elif isinstance(texts, list):
        # Multiple texts - use parallel preprocessor
        if len(texts) <= 1:
            preprocessor = VietnameseTextPreprocessor(vncorenlp_path)
            result = [preprocessor.preprocess(text, use_word_segmentation) for text in texts]
            preprocessor.close()
            return result
        
        parallel_preprocessor = ParallelVietnameseTextPreprocessor(vncorenlp_path, n_workers=n_workers)
        result = parallel_preprocessor.preprocess_batch(texts, use_parallel, use_word_segmentation)
        parallel_preprocessor.close()
        return result
    else:
        # Convert to string and process
        preprocessor = VietnameseTextPreprocessor(vncorenlp_path)
        result = preprocessor.preprocess(str(texts), use_word_segmentation)
        preprocessor.close()
        return result


def preprocess_text_for_llm(texts, use_parallel: bool = True, n_workers: int = None, use_word_segmentation: bool = True, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar"):
    return preprocess_text(texts, use_parallel, n_workers, use_word_segmentation, vncorenlp_path)
