
import torch
from transformers import BertTokenizer

def preprocess_text(tokenizer, text, max_length):
    tokens = tokenizer.encode(text, truncation=True, padding='max_length', max_length=max_length)
    return tokens
