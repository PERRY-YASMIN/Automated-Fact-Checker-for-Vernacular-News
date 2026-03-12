#!/usr/bin/env python3
"""Test exact NLI scores for sample2."""
import json
import sys
sys.path.insert(0, '.')
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ml.inference.fluff_filter import clean_text

MODEL = 'joeddav/xlm-roberta-large-xnli'
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL)
mdl.eval()

# Sample2 data
text = "पृथ्वी सूर्य से तीसरा ग्रह है — यह बात हर विज्ञान की किताब में लिखी है। 😊"
fact = "पृथ्वी सूर्य से तीसरा ग्रह है।"

# Clean the text as the pipeline does
cleaned = clean_text(text)
print(f"Original: {text}")
print(f"Cleaned:  {cleaned}")
print(f"Fact:     {fact}")
print()

# Test NLI
with torch.no_grad():
    enc = tok([fact.lower()], [cleaned.lower()], truncation=True, padding=True, return_tensors='pt')
    logits = mdl(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    id2label = mdl.config.id2label
    print("NLI scores:")
    for i, p in enumerate(probs):
        label = id2label.get(i, i)
        print(f"  {label}: {p:.4f}")
