import re
import sys
from typing import List, Optional
from vncorenlp import VnCoreNLP

sys.path.insert(0, '.')

class VietnameseTextPreprocessor:
    def __init__(self, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar", stopword_path: str = "./VnCoreNLP/vietnamese-stopwords.txt"):
        self.nlp = VnCoreNLP(vncorenlp_path, annotators="wseg,pos", max_heap_size='-Xmx500m')
        
        try:
            with open(stopword_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            self.stopwords = set()  # Empty set if stopwords file not found

        self.emoji_pattern = re.compile("["
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
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.abbrev_dict = {
            'ko': 'không', 'k': 'không', 'khong': 'không', 'kg': 'không', 'hok': 'không',
            'dc': 'được', 'đc': 'được', 'đk': 'được', 'r': 'rồi', 'vs': 'với',
            'j': 'gì', 'bt': 'bình thường', 'mk': 'mình', 'mik': 'mình',
            'cx': 'cũng', 'bn': 'bạn', 'b': 'bạn', 'tks': 'cảm ơn', 'thanks': 'cảm ơn',
            'nchung': 'nói chung', 'nhìu': 'nhiều', 'nhieu': 'nhiều',
            'cty': 'công ty', 'nt': 'nhắn tin', 'sp': 'sản phẩm',
            'i': 'giống', 'sz': 'size', 'sdt': 'số điện thoại',
            'ok': 'tốt', 'oke': 'tốt', 'okey': 'tốt', 'okie': 'tốt',
            'okeyy': 'tốt', 'okiee': 'tốt', 'thik': 'thích',
            'thix': 'thích', 'ib': 'nhắn tin', 'ibx': 'nhắn tin',
            'inbox': 'nhắn tin', 'inb': 'nhắn tin', 'inbx': 'nhắn tin',
            'sz': 'cỡ', 'mn': 'mọi người', 'mng': 'mọi người',
            'sr': 'xin lỗi', 'sorry': 'xin lỗi', 'éo': 'không', 'kh': 'không',
            'rep': 'trả lời', 'ship': 'giao hàng', 'h': 'giờ',
            'lm': 'làm', 'rùi': 'rồi', 'tl': 'trả lời',
        }

    def remove_emoji(self, text: str) -> str:
        return self.emoji_pattern.sub('', text)

    def remove_url(self, text: str) -> str:
        return self.url_pattern.sub('', text)

    def normalize_punctuation(self, text: str) -> str:
        return re.sub(r'[!?.]{2,}', lambda m: m.group(0)[-1], text)

    def normalize_repetition(self, text: str) -> str:
        vietnamese_chars = r'[a-záàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]'
        return re.sub(f'({vietnamese_chars})\\1{{2,}}', r'\1', text, flags=re.IGNORECASE)

    def remove_special_chars(self, text: str) -> str:
        return re.sub(r'[^\w\s]', ' ', text)

    def normalize_abbreviation(self, text: str) -> str:
        words = text.split()
        normalized_words = [self.abbrev_dict.get(w, w) for w in words]
        return ' '.join(normalized_words)

    def remove_stopwords(self, text: str) -> str:
        return ' '.join([w for w in text.split() if w not in self.stopwords])

    def preprocess_minimal(self, text: str) -> str:
        text = self.remove_emoji(text)
        text = self.remove_url(text)
        text = text.lower()
        text = self.normalize_abbreviation(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_repetition(text)
        return text

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

    def pos_tag(self, text: str) -> List[tuple]:
        annotated = self.nlp.annotate(text)
        pos_tags = []
        for sentence in annotated['sentences']:
            for token in sentence:
                pos_tags.append((token['form'], token['posTag']))
        return pos_tags

    def filter_by_pos(self, text: str, keep_pos: List[str] = None) -> str:
        if keep_pos is None:
            keep_pos = ['N', 'V', 'A', 'R']
        pos_tags = self.pos_tag(text)
        filtered_words = []
        for word, pos in pos_tags:
            if any(pos.startswith(p) for p in keep_pos):
                filtered_words.append(word)
        return ' '.join(filtered_words)

    def preprocess(self, text: str, keep_pos: List[str] = None, use_word_segmentation: bool = True) -> str:
        text = self.preprocess_minimal(text)
        text = self.remove_special_chars(text)
        
        # Add word segmentation step
        if use_word_segmentation:
            text = self.word_segment(text)
            
        text = self.remove_stopwords(text)
        text = self.filter_by_pos(text, keep_pos)
        return text

    def close(self):
        self.nlp.close()

def preprocess_text(texts, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar", keep_pos: List[str] = None, use_word_segmentation: bool = True):
    preprocessor = VietnameseTextPreprocessor(vncorenlp_path)
    if isinstance(texts, str):
        result = preprocessor.preprocess(texts, keep_pos, use_word_segmentation)
    elif isinstance(texts, list):
        result = [preprocessor.preprocess(text, keep_pos, use_word_segmentation) for text in texts]
    else:
        result = preprocessor.preprocess(str(texts), keep_pos, use_word_segmentation)
    
    preprocessor.close()
    return result

def preprocess_text_for_llm(texts, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar", use_word_segmentation: bool = True):
    """
    Lightweight preprocessing specifically for LLM models (PhoBERT, XLM-RoBERTa)
    - Less aggressive preprocessing to preserve context
    - Optional word segmentation for better Vietnamese tokenization
    """
    preprocessor = VietnameseTextPreprocessor(vncorenlp_path)
    
    def preprocess_for_llm(text: str) -> str:
        text = preprocessor.preprocess_minimal(text)  # Remove emoji, URL, normalize text
        if use_word_segmentation:
            text = preprocessor.word_segment(text)
        return text
    
    if isinstance(texts, str):
        result = preprocess_for_llm(texts)
    elif isinstance(texts, list):
        result = [preprocess_for_llm(text) for text in texts]
    else:
        result = preprocess_for_llm(str(texts))
    
    preprocessor.close()
    return result
