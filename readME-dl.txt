# 📰 Fake News Detection using RoBERTa

This project implements a binary text classification system using `roberta-base` to detect fake news. It uses HuggingFace Transformers, PyTorch, and scikit-learn, and includes full evaluation (Accuracy, Precision, Recall, F1, AUC, and EER).

---

## 🚀 Features

- Preprocessing and data loading with HuggingFace Datasets
- Fine-tuning `roberta-base` for fake news detection
- Metrics: Accuracy, Precision, Recall, F1, AUC, and Equal Error Rate (EER)
- Visualization of training/validation loss and ROC curves
- Classification reports and confusion matrices

---

## 📦 Installation

git clone https://github.com/yourusername/fake-news-roberta.git
cd fake-news-roberta
pip install -r requirements.txt

Dataset
Use any labeled dataset for fake news detection (e.g., LIAR dataset, Kaggle Fake News, or a custom one).

The CSV format must have:
text: the input news article/text
label: 0 (real) or 1 (fake)

---

# Usage
1. Data Preprocessing
Update your data_loader or script to:
Tokenize using RobertaTokenizer
Split into train/val/test sets
Format with torch.utils.data.Dataset or HuggingFace datasets.Dataset

2. Training
python roberta_fakenews.py --epochs 3 --batch_size 16 --lr 2e-5

3. Evaluation Output
Accuracy, Precision, Recall, F1
Classification report and confusion matrix
ROC Curve and AUC
Equal Error Rate (EER) and threshold

---

# Sample Results
Evaluation Metrics for RoBERTa Model:
Accuracy: 0.9978
Precision: 0.9984
Recall: 0.9970
F1 Score: 0.9977
AUC: 0.9986
EER: 0.0143 at threshold: 0.4821

---

# Notes
1.The model is based on roberta-base. You can switch to roberta-large with minimal changes.
2.Ensure GPU is available for efficient training.
3.Early stopping or model checkpointing can improve performance.

