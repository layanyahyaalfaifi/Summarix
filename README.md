# Summarix — Automatic Text Summarization

Fine-tuning **Facebook's BART-large-CNN** on the **CNN/DailyMail** dataset for abstractive text summarization.

## Tech Stack

- **Model:** `facebook/bart-large-cnn` (HuggingFace Transformers)
- **Dataset:** CNN/DailyMail 3.0.0
- **Evaluation:** ROUGE-1, ROUGE-2, ROUGE-L, BERTScore
- **Framework:** PyTorch + HuggingFace

## Getting Started

```bash
pip install datasets transformers nltk rouge-score torch scikit-learn evaluate bert-score
jupyter notebook NLP_project.ipynb
```

> GPU recommended for training.

## Training Configuration

| Parameter | Value |
|---|---|
| Model | facebook/bart-large-cnn |
| Learning Rate | 5e-5 |
| Batch Size | 16 |
| Epochs | 10 (Early Stopping) |
| Max Input Length | 512 tokens |
| Max Summary Length | 150 tokens |
| Precision | FP16 |

## Approach

Partial fine-tuning — first 12 encoder/decoder layers are frozen to preserve pretrained knowledge while adapting to the target domain.
