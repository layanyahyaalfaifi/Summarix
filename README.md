# 🧠 NLP Project — Automatic Text Summarization

**Course:** CAI350: Natural Language Processing  
**Phase:** 5 — Final Report  
**Topic:** Automatic Text Summarization using BART  

## 👥 Group Members

| ID | Name |
|---|---|
| 444009021 | Atheer Alshehri |
| 444009016 | Layan Yahya Alfaifi |
| 444008980 | Sarah Bader Alharbi |
| 443007677 | Norah Basheer Alqahtani |
| 444008990 | Farah Saud Alhowidi |

---

## 📌 Project Overview

This project fine-tunes **Facebook's BART-large-CNN** model on the **CNN/DailyMail (v3.0.0)** dataset to perform **abstractive text summarization**.

## 📂 Project Structure

```
NLP_project/
├── NLP_project.ipynb       # Main notebook with all code
├── Final_reportNLP.docx    # Full project report
└── README.md               # Project documentation
```

## ⚙️ Tech Stack

- **Model:** `facebook/bart-large-cnn` (HuggingFace Transformers)
- **Dataset:** CNN/DailyMail 3.0.0
- **Evaluation Metrics:** ROUGE-1, ROUGE-2, ROUGE-L, BERTScore
- **Framework:** PyTorch + HuggingFace

## 🚀 How to Run

1. Install dependencies:
```bash
pip install datasets transformers nltk rouge-score torch scikit-learn evaluate bert-score
```

2. Open and run the notebook:
```bash
jupyter notebook NLP_project.ipynb
```

> ⚠️ GPU recommended for training. The model uses partial fine-tuning (first 12 encoder/decoder layers are frozen).

## 📊 Model Training Details

| Parameter | Value |
|---|---|
| Learning Rate | 5e-5 |
| Batch Size | 16 |
| Epochs | 10 (with Early Stopping) |
| Max Input Length | 512 tokens |
| Max Summary Length | 150 tokens |
| Precision | FP16 (mixed precision) |

## 📖 References

See `Final_reportNLP.docx` for full references and detailed analysis.
