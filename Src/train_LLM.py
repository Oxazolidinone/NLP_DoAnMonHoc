import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from Src.preprocessing import preprocess_text_for_llm
from utils import *
import warnings
warnings.filterwarnings('ignore')

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
def train_llm(data_path, model_name='vinai/phobert-base', max_length=256, epochs=10, batch_size=32):
    df = pd.read_csv(data_path)
    df['text'] = preprocess_text_for_llm(df['text'].tolist())
    data, reverse_label_map = encode_labels(df) # return df (content, label(int)), reverse_label_map
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['text'].tolist(),
        data['label'].tolist(),
        test_size=0.2,
        random_state=42,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        problem_type="single_label_classification"
    )
    
    model.to('cuda')
    
    train_dataset = BaseDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = BaseDataset(test_texts, test_labels, tokenizer, max_length)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir='./phobert_results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42,
        dataloader_pin_memory=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
        
    if model_name == 'xlm-roberta-base': 
        model_save_path = './XLM-RoBERTa_model'
        print_results(eval_results, "XLM-RoBERTa")
    else:
        model_save_path = './PhoBERT_model'
        print_results(eval_results, "PhoBERT")

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    return trainer, reverse_label_map, eval_results

def predict_text(text, model_path='./PhoBERT_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to('cuda')
    processed_text  = preprocess_text_for_llm(text)
    
    inputs = tokenizer(
        processed_text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    label_map = get_reverse_label_map()
    predicted_label = label_map.get(predicted_class, 'UNKNOWN')
    
    return predicted_label, confidence