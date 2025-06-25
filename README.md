# NLP_UIT_2025
# Vietnamese Sentiment Analysis Project
## Overview
This project implements sentiment analysis for Vietnamese text using multiple machine learning models, including traditional ML and deep learning approaches.

## Models Used
1. **MultinomialNB** - Naive Bayes with TF-IDF features
2. **RandomForest** - Random Forest with feature engineering
3. **PhoBERT** - Pre-trained Vietnamese BERT model
4. **XLM-RoBERTa** - Multilingual transformer model

## Project Structure
```
FinalSemester/
├── data.csv                    # Main dataset
├── demo_all_models.py         # Script to run and compare all models
├── train_LLM.py              # Training for PhoBERT and XLM-RoBERTa
├── train_multinb_rf.py       # Training for MultinomialNB and RandomForest
├── utils.py                  # Utility functions
├── Src/
│   ├── preprocessing.py      # Vietnamese text preprocessing
│   └── feature_engineering.py # Feature extraction
└── VnCoreNLP/               # Vietnamese NLP toolkit
    ├── VnCoreNLP-1.1.1.jar
    └── models/
```

### Dependencies
```bash
pip install pandas numpy scikit-learn torch transformers matplotlib seaborn
pip install vncorenlp
```
## Installation

1. **Clone repository**
```bash
git clone (https://github.com/Oxazolidinone/NLP_UIT_2025.git)
cd FinalSemester
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download VnCoreNLP**
- Ensure VnCoreNLP-1.1.1.jar is in the `VnCoreNLP/` directory
- Download models from [VnCoreNLP GitHub](https://github.com/vncorenlp/VnCoreNLP)

4. **Prepare data**
- Place `data.csv` file in the root directory
- Format: `text,label` (POS/NEG/NEU)

## Usage

### Vietnamese Text Preprocessing
- **Word segmentation** using VnCoreNLP
- **Normalization**: emoji, URL, punctuation, repetition
- **Abbreviation expansion**: Convert abbreviations (ko → không, dc → được)
- **POS filtering**: Keep nouns, verbs, adjectives, adverbs
- **Stopwords removal**

### Feature Engineering
- **Sentiment lexicon features**: Positive/negative words
- **Linguistic features**: Punctuation, capitalization patterns
- **Context features**: Negation, intensification
- **N-gram TF-IDF features**: Unigram, bigram, trigram

### Performance Metrics
Models are evaluated using:
- **Accuracy**: Overall accuracy
- **Precision**: Class-wise precision
- **Recall**: Class-wise recall  
- **F1-Score**: Balanced score

### Main Functions

**preprocess_text(texts, use_word_segmentation=True)**
- Process text for traditional ML models
- Returns: List[str] processed texts

**preprocess_text_for_llm(texts, use_word_segmentation=True)** 
- Process text for LLM models (lightweight)
- Returns: List[str] processed texts

**train_llm(data_path, model_name, epochs=3)**
- Train LLM models (PhoBERT/XLM-RoBERTa)
- Returns: trainer, reverse_label_map, eval_results

**train_traditional_model(data_path, model_name)**
- Train traditional ML models
- Returns: trainer, eval_results

## Example Dataset Format

```csv
text,label
"This product is very good I am satisfied",POS
"Poor quality not worth the money",NEG
"Average product nothing special",NEU
"Sản phẩm rất tốt tôi hài lòng",POS
"Chất lượng kém không đáng tiền",NEG
"Sản phẩm bình thường",NEU
```



