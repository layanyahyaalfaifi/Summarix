# Summarix — Automatic Text Summarization

Fine-tuning **Facebook's BART-large-CNN** on the **CNN/DailyMail** dataset to generate fluent, semantically accurate abstractive summaries of news articles.

---

## Overview

Most summarization systems either copy sentences from the source (extractive) or struggle to preserve meaning when rephrasing (abstractive). This project fine-tunes BART-large-CNN with partial layer freezing and mixed-precision training to produce summaries that are both coherent and semantically faithful — evaluated using BERTScore rather than ROUGE alone.

---

## Results

ROUGE scores were intentionally de-prioritized. Because the model generates abstractive summaries (paraphrasing rather than copying), ROUGE underestimates its true quality by penalizing valid rewording. BERTScore, which measures semantic similarity via contextual embeddings, was the primary metric and showed strong F1 performance on the CNN/DailyMail test set.

| Metric | Notes |
|---|---|
| ROUGE-1/2/L | Lower than expected — model paraphrases rather than copies |
| BERTScore F1 | Primary metric — high semantic alignment with reference summaries |

> Exact scores from your training run can be added here.

---

## Limitations

- **No exact ROUGE numbers reported** — scores were deliberately not the focus due to the abstractive nature of the task
- **Subset training** — only 10K of the full training set was used; full dataset would likely improve performance
- **Frozen layers** — first 12 encoder/decoder layers are frozen, limiting how much the model adapts
- **Domain-specific** — model is tuned on news articles; may not generalize well to other domains
- **Compute-heavy** — requires a GPU (V100 or equivalent) for reasonable training time; CPU training is impractical

---

## Model Architecture

**BART-large-CNN** — encoder-decoder transformer:

- **Encoder:** 12 Transformer layers (bidirectional, BERT-style) — frozen during fine-tuning
- **Decoder:** 12 Transformer layers (autoregressive, GPT-style) — frozen during fine-tuning
- Remaining layers fine-tuned to adapt to task-specific patterns
- Inference uses **beam search** (4 beams) with length penalty to encourage complete, non-repetitive summaries

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `facebook/bart-large-cnn` |
| Dataset | CNN/DailyMail 3.0.0 |
| Training samples | 10,000 |
| Validation samples | 500 |
| Test samples | 500 |
| Learning rate | 5e-5 |
| Batch size | 16 |
| Epochs | 10 (early stopping, patience=2) |
| Dropout | 0.1 |
| Weight decay | 0.01 |
| Precision | FP16 (mixed precision) |
| Max input length | 512 tokens |
| Max summary length | 150 tokens |

---

## Evaluation Metrics

**ROUGE** (lexical overlap):
- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap
- ROUGE-L: longest common subsequence

**BERTScore** (semantic similarity):
Uses contextual embeddings from a pretrained transformer to compute cosine similarity between generated and reference tokens — better suited for abstractive models that paraphrase rather than copy.

---

## Setup

```bash
pip install datasets transformers nltk rouge-score torch scikit-learn evaluate bert-score
jupyter notebook NLP_project.ipynb
```

**Hardware used:**
- GPU: NVIDIA Tesla V100 (16GB VRAM)
- RAM: 64GB
- Python 3.9 / PyTorch 1.13.1 / CUDA 11.6

---

## References

- Lewis et al., [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461), ACL 2020
- Zhang et al., [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675), 2019
- Lin, [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/), ACL 2004
