#!/usr/bin/env python3
"""
Assignment 2: Neural Models for Human Values Detection
======================================================
Multi-label text classification with 20 human value labels.
Models: Baseline (CountVec+LogReg), BiLSTM, DistilBERT
Metric: F1-Macro

Run: python assignment2_complete.py
"""

import os, time, warnings, json, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# 0. CONFIGURATION & SEED
# ============================================================
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

LABEL_COLS = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation',
    'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources',
    'Face', 'Security: personal', 'Security: societal', 'Tradition',
    'Conformity: rules', 'Conformity: interpersonal', 'Humility',
    'Benevolence: caring', 'Benevolence: dependability',
    'Universalism: concern', 'Universalism: nature',
    'Universalism: tolerance', 'Universalism: objectivity'
]
NUM_LABELS = len(LABEL_COLS)

# ============================================================
# 1. DATA LOADING & SPLITTING (80/10/10)
# ============================================================
print("\n" + "="*60)
print("SECTION 1: DATA LOADING")
print("="*60)

train_args = pd.read_csv(os.path.join(DATA_DIR, "arguments-training.tsv"), sep='\t')
train_labels = pd.read_csv(os.path.join(DATA_DIR, "labels-training.tsv"), sep='\t')
df_train = pd.merge(train_args, train_labels, on="Argument ID")

val_args = pd.read_csv(os.path.join(DATA_DIR, "arguments-validation.tsv"), sep='\t')
val_labels = pd.read_csv(os.path.join(DATA_DIR, "labels-validation.tsv"), sep='\t')
df_val = pd.merge(val_args, val_labels, on="Argument ID")

test_args = pd.read_csv(os.path.join(DATA_DIR, "arguments-test.tsv"), sep='\t')
test_labels = pd.read_csv(os.path.join(DATA_DIR, "labels-test.tsv"), sep='\t')
df_test = pd.merge(test_args, test_labels, on="Argument ID")

all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
all_df['text'] = all_df['Conclusion'] + " " + all_df['Stance'] + " " + all_df['Premise']

# Ensure all label columns exist (handle the 19 vs 20 label issue)
for col in LABEL_COLS:
    if col not in all_df.columns:
        all_df[col] = 0

X_all = all_df['text'].values
y_all = all_df[LABEL_COLS].values

print(f"Total examples: {len(X_all)}, Labels: {y_all.shape[1]}")

# Split: 80% train, 10% val, 10% test
msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for train_idx, temp_idx in msss1.split(X_all, y_all):
    X_train_full, X_temp = X_all[train_idx], X_all[temp_idx]
    y_train_full, y_temp = y_all[train_idx], y_all[temp_idx]

msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
for val_idx, test_idx in msss2.split(X_temp, y_temp):
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]

print(f"Train: {len(X_train_full)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Label prevalence (first 3): Train={np.mean(y_train_full, axis=0)[:3].round(3)}")

# Keep text + arg IDs for error analysis later
all_df_indexed = all_df.copy()
test_texts_for_analysis = X_test.copy()

# ============================================================
# 2. ASSIGNMENT 1 BASELINE: CountVec + LogReg
# ============================================================
print("\n" + "="*60)
print("SECTION 2: BASELINE MODEL (Assignment 1)")
print("="*60)

def train_baseline(X_tr, y_tr, X_te, y_te):
    """Train CountVec + LogReg and return predictions, timing info."""
    pipeline = Pipeline([
        ('vec', CountVectorizer(ngram_range=(1, 3), analyzer='char',
                                lowercase=True, min_df=3, max_features=20000)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear',
                                                        random_state=SEED)))
    ])
    t0 = time.time()
    pipeline.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = pipeline.predict(X_te)
    inference_time = time.time() - t0

    f1 = f1_score(y_te, y_pred, average='macro')
    n_params = sum(c.coef_.size + c.intercept_.size for c in pipeline['clf'].estimators_)
    return y_pred, f1, train_time, inference_time, n_params, pipeline

baseline_pred, baseline_f1, baseline_train_t, baseline_inf_t, baseline_params, baseline_pipe = \
    train_baseline(X_train_full, y_train_full, X_test, y_test)
print(f"Baseline F1-Macro: {baseline_f1:.4f}")
print(f"Baseline params: {baseline_params:,}, train: {baseline_train_t:.2f}s, inference: {baseline_inf_t:.4f}s")


# ============================================================
# 3. MODEL 1: BiLSTM (from scratch)
# ============================================================
print("\n" + "="*60)
print("SECTION 3: BiLSTM MODEL")
print("="*60)

# --- Vocabulary ---
class Vocabulary:
    def __init__(self, texts, max_size=15000):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in word_counts.most_common(max_size):
            self.word2idx[word] = len(self.word2idx)
        self.vocab_size = len(self.word2idx)

    def encode(self, text, max_len=128):
        tokens = text.lower().split()
        indices = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
        indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

vocab = Vocabulary(X_train_full)
print(f"Vocabulary size: {vocab.vocab_size}")

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        x = self.vocab.encode(self.texts[idx], self.max_len)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

# --- LSTM Model ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, n_classes=NUM_LABELS, dropout=0.3,
                 bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        fc_input = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input, n_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        # Global Max Pooling over time steps
        pooled, _ = torch.max(lstm_out, dim=1)
        return self.fc(self.dropout(pooled))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Generic Training Function (for LSTM) ---
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3,
                patience=5, clip_grad=1.0, model_name="model"):
    """
    Train with early stopping, gradient clipping, timing, and memory tracking.
    Returns: history dict, best model state, training time
    """
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_f1': [], 'epoch_time': []}
    best_f1 = 0
    best_state = None
    patience_counter = 0
    total_train_time = 0

    print(f"\nTraining {model_name} | Params: {model.count_parameters():,}")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        epoch_start = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += loss.item()

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_loss = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                outputs = model(batch_x)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        history['train_loss'].append(avg_loss)
        history['val_f1'].append(val_f1)
        history['epoch_time'].append(epoch_time)

        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")

        # --- Early Stopping ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    print(f"  Best Val F1: {best_f1:.4f} | Total time: {total_train_time:.1f}s | GPU: {gpu_mem:.0f}MB")

    return history, model, total_train_time, gpu_mem


def evaluate_model(model, test_loader, is_bert=False):
    """Evaluate model on test set. Returns preds, f1, inference_time."""
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()

    with torch.no_grad():
        for batch in test_loader:
            if is_bert:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels']
                outputs = model(input_ids, mask)
            else:
                batch_x, labels = batch
                batch_x = batch_x.to(DEVICE)
                outputs = model(batch_x)

            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    inference_time = time.time() - t0
    f1 = f1_score(all_labels, all_preds, average='macro')
    return np.array(all_preds), np.array(all_labels), f1, inference_time


# --- Train BiLSTM ---
set_seed()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

train_ds = TextDataset(X_train_full, y_train_full, vocab)
val_ds = TextDataset(X_val, y_val, vocab)
test_ds = TextDataset(X_test, y_test, vocab)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

lstm_model = BiLSTMClassifier(vocab.vocab_size)
lstm_history, lstm_model, lstm_train_time, lstm_gpu_mem = \
    train_model(lstm_model, train_loader, val_loader, epochs=30, lr=1e-3,
                patience=5, model_name="BiLSTM")

lstm_preds, lstm_labels, lstm_f1, lstm_inf_time = evaluate_model(lstm_model, test_loader)
lstm_params = lstm_model.count_parameters()
print(f"\nBiLSTM Test F1-Macro: {lstm_f1:.4f}")

# Save model
torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, "bilstm_best.pt"))

# ============================================================
# 4. MODEL 2: DistilBERT Fine-tuning
# ============================================================
print("\n" + "="*60)
print("SECTION 4: DistilBERT MODEL")
print("="*60)

from transformers import (DistilBertTokenizer, DistilBertModel,
                          get_linear_schedule_with_warmup)

BERT_MODEL_NAME = 'distilbert-base-uncased'
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 16
BERT_LR = 2e-5
BERT_EPOCHS = 10

tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=BERT_MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True,
                             padding='max_length', max_length=self.max_len,
                             return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

class DistilBertClassifier(nn.Module):
    def __init__(self, model_name=BERT_MODEL_NAME, n_classes=NUM_LABELS,
                 freeze_encoder=False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.fc(self.dropout(cls_output))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_bert(model, train_loader, val_loader, epochs=BERT_EPOCHS,
               lr=BERT_LR, patience=3, model_name="DistilBERT"):
    """Train BERT with warmup scheduler and early stopping."""
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {'train_loss': [], 'val_f1': [], 'epoch_time': []}
    best_f1 = 0
    best_state = None
    patience_counter = 0
    total_train_time = 0

    print(f"\nTraining {model_name} | Trainable params: {model.count_parameters():,}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        epoch_start = time.time()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_loss = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids, mask)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        history['train_loss'].append(avg_loss)
        history['val_f1'].append(val_f1)
        history['epoch_time'].append(epoch_time)

        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    print(f"  Best Val F1: {best_f1:.4f} | Total time: {total_train_time:.1f}s | GPU: {gpu_mem:.0f}MB")

    return history, model, total_train_time, gpu_mem

# --- Train DistilBERT ---
set_seed()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

bert_train_ds = BertDataset(X_train_full, y_train_full, tokenizer)
bert_val_ds = BertDataset(X_val, y_val, tokenizer)
bert_test_ds = BertDataset(X_test, y_test, tokenizer)

bert_train_loader = DataLoader(bert_train_ds, batch_size=BERT_BATCH_SIZE, shuffle=True)
bert_val_loader = DataLoader(bert_val_ds, batch_size=BERT_BATCH_SIZE)
bert_test_loader = DataLoader(bert_test_ds, batch_size=BERT_BATCH_SIZE)

bert_model = DistilBertClassifier()
bert_history, bert_model, bert_train_time, bert_gpu_mem = \
    train_bert(bert_model, bert_train_loader, bert_val_loader)

bert_preds, bert_labels, bert_f1, bert_inf_time = \
    evaluate_model(bert_model, bert_test_loader, is_bert=True)
bert_params = bert_model.count_parameters()
print(f"\nDistilBERT Test F1-Macro: {bert_f1:.4f}")

torch.save(bert_model.state_dict(), os.path.join(MODELS_DIR, "distilbert_best.pt"))


# ============================================================
# 5. EXPERIMENT 1: Architecture Comparison
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Architecture Comparison")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Baseline (CountVec+LogReg)', 'BiLSTM', 'DistilBERT'],
    'F1-Macro': [baseline_f1, lstm_f1, bert_f1],
    'Parameters': [baseline_params, lstm_params, bert_params],
    'Train Time (s)': [baseline_train_t, lstm_train_time, bert_train_time],
    'Inference Time (s)': [baseline_inf_t, lstm_inf_time, bert_inf_time],
    'GPU Memory (MB)': [0, lstm_gpu_mem, bert_gpu_mem]
})
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(RESULTS_DIR, 'architecture_comparison.csv'), index=False)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#2196F3', '#FF9800', '#4CAF50']

axes[0].bar(comparison['Model'], comparison['F1-Macro'], color=colors)
axes[0].set_title('F1-Macro Score', fontsize=13, fontweight='bold')
axes[0].set_ylabel('F1-Macro')
for i, v in enumerate(comparison['F1-Macro']):
    axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)

axes[1].bar(comparison['Model'], comparison['Train Time (s)'], color=colors)
axes[1].set_title('Training Time', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Seconds')

axes[2].bar(comparison['Model'], comparison['Parameters'], color=colors)
axes[2].set_title('Parameter Count', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Parameters')
axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

for ax in axes:
    ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'architecture_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/architecture_comparison.png")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(lstm_history['train_loss'], 'o-', label='BiLSTM', color='#FF9800')
axes[0].plot(bert_history['train_loss'], 's-', label='DistilBERT', color='#4CAF50')
axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(lstm_history['val_f1'], 'o-', label='BiLSTM', color='#FF9800')
axes[1].plot(bert_history['val_f1'], 's-', label='DistilBERT', color='#4CAF50')
axes[1].set_title('Validation F1-Macro', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1-Macro')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/training_curves.png")


# ============================================================
# 6. EXPERIMENT 2: Learning Curve Analysis
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 2: Learning Curve Analysis")
print("="*60)

fractions = [0.25, 0.50, 0.75, 1.0]
lc_results = {'fraction': fractions, 'baseline': [], 'lstm': [], 'bert': []}

for frac in fractions:
    print(f"\n--- Training on {int(frac*100)}% of data ---")
    n = int(len(X_train_full) * frac)
    set_seed()

    # Subsample
    if frac < 1.0:
        msss_sub = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1.0-frac, random_state=SEED)
        for sub_idx, _ in msss_sub.split(X_train_full, y_train_full):
            X_sub, y_sub = X_train_full[sub_idx], y_train_full[sub_idx]
    else:
        X_sub, y_sub = X_train_full, y_train_full

    print(f"  Subset size: {len(X_sub)}")

    # Baseline
    _, bl_f1, _, _, _, _ = train_baseline(X_sub, y_sub, X_test, y_test)
    lc_results['baseline'].append(bl_f1)
    print(f"  Baseline F1: {bl_f1:.4f}")

    # LSTM
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sub_train_ds = TextDataset(X_sub, y_sub, vocab)
    sub_train_loader = DataLoader(sub_train_ds, batch_size=64, shuffle=True)
    sub_lstm = BiLSTMClassifier(vocab.vocab_size)
    _, sub_lstm, _, _ = train_model(sub_lstm, sub_train_loader, val_loader,
                                     epochs=30, lr=1e-3, patience=5,
                                     model_name=f"LSTM-{int(frac*100)}%")
    _, _, sub_lstm_f1, _ = evaluate_model(sub_lstm, test_loader)
    lc_results['lstm'].append(sub_lstm_f1)
    print(f"  LSTM F1: {sub_lstm_f1:.4f}")
    del sub_lstm; gc.collect(); torch.cuda.empty_cache()

    # BERT
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sub_bert_ds = BertDataset(X_sub, y_sub, tokenizer)
    sub_bert_loader = DataLoader(sub_bert_ds, batch_size=BERT_BATCH_SIZE, shuffle=True)
    sub_bert = DistilBertClassifier()
    _, sub_bert, _, _ = train_bert(sub_bert, sub_bert_loader, bert_val_loader,
                                    epochs=BERT_EPOCHS, patience=3,
                                    model_name=f"BERT-{int(frac*100)}%")
    _, _, sub_bert_f1, _ = evaluate_model(sub_bert, bert_test_loader, is_bert=True)
    lc_results['bert'].append(sub_bert_f1)
    print(f"  BERT F1: {sub_bert_f1:.4f}")
    del sub_bert; gc.collect(); torch.cuda.empty_cache()

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(fractions, lc_results['baseline'], 'o-', label='Baseline (CountVec+LogReg)',
         color='#2196F3', linewidth=2, markersize=8)
plt.plot(fractions, lc_results['lstm'], 's-', label='BiLSTM',
         color='#FF9800', linewidth=2, markersize=8)
plt.plot(fractions, lc_results['bert'], '^-', label='DistilBERT',
         color='#4CAF50', linewidth=2, markersize=8)
plt.xlabel('Fraction of Training Data', fontsize=12)
plt.ylabel('Test F1-Macro', fontsize=12)
plt.title('Learning Curve Analysis', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(fractions, [f'{int(f*100)}%' for f in fractions])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: results/learning_curves.png")

lc_df = pd.DataFrame(lc_results)
lc_df.to_csv(os.path.join(RESULTS_DIR, 'learning_curves.csv'), index=False)


# ============================================================
# 7. EXPERIMENT 3: Ablation Studies
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Ablation Studies")
print("="*60)

# --- 3A: LSTM Ablations ---
print("\n--- LSTM Ablations ---")
lstm_ablations = []

# Ablation 1: Unidirectional vs Bidirectional
for bidir in [False, True]:
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    name = "Bidirectional" if bidir else "Unidirectional"
    m = BiLSTMClassifier(vocab.vocab_size, bidirectional=bidir)
    _, m, t, mem = train_model(m, train_loader, val_loader, epochs=30,
                                patience=5, model_name=f"LSTM-{name}")
    _, _, f1, _ = evaluate_model(m, test_loader)
    lstm_ablations.append({'Config': name, 'F1-Macro': f1, 'Params': m.count_parameters(),
                           'Train Time': t})
    print(f"  {name}: F1={f1:.4f}, Params={m.count_parameters():,}")
    del m; gc.collect(); torch.cuda.empty_cache()

# Ablation 2: Number of layers
for n_layers in [1, 2, 3]:
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = BiLSTMClassifier(vocab.vocab_size, num_layers=n_layers)
    _, m, t, mem = train_model(m, train_loader, val_loader, epochs=30,
                                patience=5, model_name=f"LSTM-{n_layers}layer")
    _, _, f1, _ = evaluate_model(m, test_loader)
    lstm_ablations.append({'Config': f'{n_layers} Layer(s)', 'F1-Macro': f1,
                           'Params': m.count_parameters(), 'Train Time': t})
    print(f"  {n_layers} layers: F1={f1:.4f}, Params={m.count_parameters():,}")
    del m; gc.collect(); torch.cuda.empty_cache()

# Ablation 3: Hidden dimensions
for hdim in [64, 128, 256]:
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = BiLSTMClassifier(vocab.vocab_size, hidden_dim=hdim)
    _, m, t, mem = train_model(m, train_loader, val_loader, epochs=30,
                                patience=5, model_name=f"LSTM-h{hdim}")
    _, _, f1, _ = evaluate_model(m, test_loader)
    lstm_ablations.append({'Config': f'Hidden={hdim}', 'F1-Macro': f1,
                           'Params': m.count_parameters(), 'Train Time': t})
    print(f"  Hidden={hdim}: F1={f1:.4f}, Params={m.count_parameters():,}")
    del m; gc.collect(); torch.cuda.empty_cache()

lstm_abl_df = pd.DataFrame(lstm_ablations)
lstm_abl_df.to_csv(os.path.join(RESULTS_DIR, 'ablation_lstm.csv'), index=False)
print("\n" + lstm_abl_df.to_string(index=False))

# Plot LSTM ablations
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
groups = [
    ('Direction', lstm_abl_df.iloc[0:2]),
    ('Layers', lstm_abl_df.iloc[2:5]),
    ('Hidden Dim', lstm_abl_df.iloc[5:8])
]
colors_abl = ['#e91e63', '#9c27b0', '#673ab7', '#3f51b5']
for ax, (title, data) in zip(axes, groups):
    bars = ax.bar(data['Config'], data['F1-Macro'], color=colors_abl[:len(data)])
    ax.set_title(f'LSTM: {title}', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Macro')
    for bar, val in zip(bars, data['F1-Macro']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}',
                ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'ablation_lstm.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/ablation_lstm.png")

# --- 3B: BERT Ablations ---
print("\n--- BERT Ablations ---")
bert_ablations = []

# Ablation 1: Frozen vs Unfrozen encoder
for freeze in [True, False]:
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    name = "Frozen" if freeze else "Fine-tuned"
    m = DistilBertClassifier(freeze_encoder=freeze)
    lr_val = 1e-3 if freeze else BERT_LR
    _, m, t, mem = train_bert(m, bert_train_loader, bert_val_loader,
                               lr=lr_val, patience=3,
                               model_name=f"BERT-{name}")
    _, _, f1, _ = evaluate_model(m, bert_test_loader, is_bert=True)
    bert_ablations.append({'Config': f'Encoder {name}', 'F1-Macro': f1,
                           'Params': m.count_parameters(), 'Train Time': t})
    print(f"  {name}: F1={f1:.4f}, Trainable={m.count_parameters():,}")
    del m; gc.collect(); torch.cuda.empty_cache()

# Ablation 2: Learning rates
for lr_val in [1e-5, 2e-5, 5e-5]:
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = DistilBertClassifier()
    _, m, t, mem = train_bert(m, bert_train_loader, bert_val_loader,
                               lr=lr_val, patience=3,
                               model_name=f"BERT-lr{lr_val}")
    _, _, f1, _ = evaluate_model(m, bert_test_loader, is_bert=True)
    bert_ablations.append({'Config': f'LR={lr_val}', 'F1-Macro': f1,
                           'Params': m.count_parameters(), 'Train Time': t})
    print(f"  LR={lr_val}: F1={f1:.4f}")
    del m; gc.collect(); torch.cuda.empty_cache()

bert_abl_df = pd.DataFrame(bert_ablations)
bert_abl_df.to_csv(os.path.join(RESULTS_DIR, 'ablation_bert.csv'), index=False)
print("\n" + bert_abl_df.to_string(index=False))

# Plot BERT ablations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
groups_bert = [
    ('Encoder Strategy', bert_abl_df.iloc[0:2]),
    ('Learning Rate', bert_abl_df.iloc[2:5])
]
for ax, (title, data) in zip(axes, groups_bert):
    bars = ax.bar(data['Config'], data['F1-Macro'], color=['#00bcd4', '#009688', '#4caf50'][:len(data)])
    ax.set_title(f'DistilBERT: {title}', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Macro')
    for bar, val in zip(bars, data['F1-Macro']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}',
                ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'ablation_bert.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/ablation_bert.png")


# ============================================================
# 8. EXPERIMENT 4: Error Analysis
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 4: Error Analysis")
print("="*60)

# Per-label comparison
print("\nPer-label F1 scores:")
print(f"{'Label':<30} {'Baseline':>10} {'LSTM':>10} {'BERT':>10}")
print("-" * 65)

per_label_f1 = {}
for i, label in enumerate(LABEL_COLS):
    bl = f1_score(y_test[:, i], baseline_pred[:, i], zero_division=0)
    ls = f1_score(y_test[:, i], lstm_preds[:, i], zero_division=0)
    bt = f1_score(y_test[:, i], bert_preds[:, i], zero_division=0)
    per_label_f1[label] = {'Baseline': bl, 'LSTM': ls, 'BERT': bt}
    print(f"{label:<30} {bl:>10.4f} {ls:>10.4f} {bt:>10.4f}")

# Find examples where neural models fixed baseline errors
error_analysis_lines = []
error_analysis_lines.append("=" * 80)
error_analysis_lines.append("ERROR ANALYSIS")
error_analysis_lines.append("=" * 80)

error_analysis_lines.append("\n\n--- Examples where Neural Models FIXED Baseline Errors ---\n")
fixed_count = 0
for i in range(len(X_test)):
    if fixed_count >= 7:
        break
    # Check if baseline was wrong but BERT was right (on any label)
    bl_correct = (baseline_pred[i] == y_test[i])
    bt_correct = (bert_preds[i] == y_test[i])
    bl_wrong_mask = ~bl_correct
    bt_fixed_mask = bl_wrong_mask & bt_correct

    if bt_fixed_mask.any():
        fixed_count += 1
        fixed_labels = [LABEL_COLS[j] for j in range(NUM_LABELS) if bt_fixed_mask[j]]
        error_analysis_lines.append(f"Example {fixed_count}:")
        error_analysis_lines.append(f"  Text: {X_test[i][:200]}...")
        error_analysis_lines.append(f"  Ground truth:     {[LABEL_COLS[j] for j in range(NUM_LABELS) if y_test[i][j]==1]}")
        error_analysis_lines.append(f"  Baseline pred:    {[LABEL_COLS[j] for j in range(NUM_LABELS) if baseline_pred[i][j]==1]}")
        error_analysis_lines.append(f"  BERT pred:        {[LABEL_COLS[j] for j in range(NUM_LABELS) if bert_preds[i][j]==1]}")
        error_analysis_lines.append(f"  Labels BERT fixed: {fixed_labels}")
        error_analysis_lines.append("")

error_analysis_lines.append("\n--- Examples where Neural Models INTRODUCED New Errors ---\n")
new_error_count = 0
for i in range(len(X_test)):
    if new_error_count >= 7:
        break
    bl_correct = (baseline_pred[i] == y_test[i])
    bt_correct = (bert_preds[i] == y_test[i])
    bl_right_mask = bl_correct
    bt_wrong_mask = ~bt_correct
    introduced_mask = bl_right_mask & bt_wrong_mask

    if introduced_mask.any():
        new_error_count += 1
        err_labels = [LABEL_COLS[j] for j in range(NUM_LABELS) if introduced_mask[j]]
        error_analysis_lines.append(f"Example {new_error_count}:")
        error_analysis_lines.append(f"  Text: {X_test[i][:200]}...")
        error_analysis_lines.append(f"  Ground truth:          {[LABEL_COLS[j] for j in range(NUM_LABELS) if y_test[i][j]==1]}")
        error_analysis_lines.append(f"  Baseline pred:         {[LABEL_COLS[j] for j in range(NUM_LABELS) if baseline_pred[i][j]==1]}")
        error_analysis_lines.append(f"  BERT pred:             {[LABEL_COLS[j] for j in range(NUM_LABELS) if bert_preds[i][j]==1]}")
        error_analysis_lines.append(f"  Labels BERT got wrong: {err_labels}")
        error_analysis_lines.append("")

error_text = "\n".join(error_analysis_lines)
with open(os.path.join(RESULTS_DIR, 'error_analysis.txt'), 'w') as f:
    f.write(error_text)
print(error_text[:2000])
print("\nSaved: results/error_analysis.txt")

# --- Attention Visualization ---
print("\nGenerating attention visualization...")

class DistilBertWithAttention(nn.Module):
    """Wrapper to extract attention weights."""
    def __init__(self, model_name=BERT_MODEL_NAME, n_classes=NUM_LABELS):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
        logits = self.fc(self.dropout(cls_output))
        return logits, attentions

# Load weights into attention model
attn_model = DistilBertWithAttention()
# Copy weights from trained model (excluding attention output differences)
trained_state = bert_model.state_dict()
attn_state = attn_model.state_dict()
# Map compatible keys
for key in trained_state:
    if key in attn_state:
        attn_state[key] = trained_state[key]
attn_model.load_state_dict(attn_state, strict=False)
attn_model = attn_model.to(DEVICE)
attn_model.eval()

# Pick 3 examples for attention visualization
n_examples = 3
fig, axes = plt.subplots(n_examples, 1, figsize=(14, 4 * n_examples))
if n_examples == 1:
    axes = [axes]

for ex_idx in range(n_examples):
    text = X_test[ex_idx]
    enc = tokenizer(text, truncation=True, max_length=64, return_tensors='pt')
    input_ids = enc['input_ids'].to(DEVICE)
    mask = enc['attention_mask'].to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])

    with torch.no_grad():
        logits, attentions = attn_model(input_ids, mask)

    # Average attention from last layer across all heads, from [CLS] token
    last_layer_attn = attentions[-1][0].mean(dim=0)  # (seq, seq)
    cls_attn = last_layer_attn[0].cpu().numpy()  # attention from [CLS]

    # Trim to actual tokens
    n_tokens = mask.sum().item()
    cls_attn = cls_attn[:n_tokens]
    display_tokens = tokens[:n_tokens]

    ax = axes[ex_idx]
    ax.barh(range(len(display_tokens)), cls_attn, color='#3f51b5', alpha=0.7)
    ax.set_yticks(range(len(display_tokens)))
    ax.set_yticklabels(display_tokens, fontsize=8)
    ax.invert_yaxis()
    true_labels = [LABEL_COLS[j] for j in range(NUM_LABELS) if y_test[ex_idx][j] == 1]
    ax.set_title(f'Attention Weights (Example {ex_idx+1}) | True: {", ".join(true_labels[:3])}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Attention Weight')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'attention_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/attention_visualization.png")

del attn_model; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# 9. EXPERIMENT 5: Computational Cost Analysis
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 5: Computational Cost Analysis")
print("="*60)

cost_data = {
    'Model': ['Baseline', 'BiLSTM', 'DistilBERT'],
    'Parameters': [f'{baseline_params:,}', f'{lstm_params:,}', f'{bert_params:,}'],
    'Training Time (s)': [f'{baseline_train_t:.2f}', f'{lstm_train_time:.2f}', f'{bert_train_time:.2f}'],
    'Inference Time (s)': [f'{baseline_inf_t:.4f}', f'{lstm_inf_time:.4f}', f'{bert_inf_time:.4f}'],
    'Inference per sample (ms)': [
        f'{baseline_inf_t/len(X_test)*1000:.3f}',
        f'{lstm_inf_time/len(X_test)*1000:.3f}',
        f'{bert_inf_time/len(X_test)*1000:.3f}'
    ],
    'GPU Memory (MB)': ['0', f'{lstm_gpu_mem:.0f}', f'{bert_gpu_mem:.0f}'],
    'F1-Macro': [f'{baseline_f1:.4f}', f'{lstm_f1:.4f}', f'{bert_f1:.4f}'],
    'Requires GPU': ['No', 'Recommended', 'Yes']
}
cost_df = pd.DataFrame(cost_data)
print(cost_df.to_string(index=False))
cost_df.to_csv(os.path.join(RESULTS_DIR, 'computational_cost.csv'), index=False)

# Deployment discussion
deployment_text = """
DEPLOYMENT CONSIDERATIONS
=========================
1. Baseline (CountVec + LogReg):
   - Fastest training and inference, no GPU needed
   - Easy to deploy in resource-constrained environments
   - Best choice when computational budget is limited

2. BiLSTM:
   - Moderate complexity, benefits from GPU but can run on CPU
   - Good balance between performance and cost
   - Suitable for batch processing pipelines

3. DistilBERT:
   - Highest performance but requires GPU for practical training
   - Inference can run on CPU with acceptable latency
   - Best for applications where accuracy is critical
   - Consider model distillation or quantization for production
"""
print(deployment_text)

with open(os.path.join(RESULTS_DIR, 'deployment_considerations.txt'), 'w') as f:
    f.write(deployment_text)


# ============================================================
# 10. GENERATE PER-LABEL CONFUSION MATRIX (Bonus)
# ============================================================
print("\n" + "="*60)
print("BONUS: Per-label Performance Heatmap")
print("="*60)

label_f1_data = []
for label in LABEL_COLS:
    label_f1_data.append(per_label_f1[label])

f1_df = pd.DataFrame(label_f1_data, index=LABEL_COLS)
f1_df.index = [l.replace('Universalism: ', 'Univ: ').replace('Self-direction: ', 'SD: ')
               .replace('Benevolence: ', 'Ben: ').replace('Conformity: ', 'Conf: ')
               .replace('Security: ', 'Sec: ').replace('Power: ', 'Pow: ')
               for l in LABEL_COLS]

plt.figure(figsize=(10, 10))
sns.heatmap(f1_df, annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'F1 Score'})
plt.title('Per-Label F1 Scores by Model', fontsize=14, fontweight='bold')
plt.xlabel('Model')
plt.ylabel('Human Value Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'per_label_f1_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/per_label_f1_heatmap.png")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"\n{'Model':<30} {'F1-Macro':>10} {'Params':>12} {'Train Time':>12}")
print("-" * 65)
print(f"{'Baseline (CountVec+LogReg)':<30} {baseline_f1:>10.4f} {baseline_params:>12,} {baseline_train_t:>11.2f}s")
print(f"{'BiLSTM':<30} {lstm_f1:>10.4f} {lstm_params:>12,} {lstm_train_time:>11.2f}s")
print(f"{'DistilBERT':<30} {bert_f1:>10.4f} {bert_params:>12,} {bert_train_time:>11.2f}s")

print(f"\nAll results saved to: {RESULTS_DIR}/")
print("Files generated:")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  - {f}")

print("\nDone! ✅")
