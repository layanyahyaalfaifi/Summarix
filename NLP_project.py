# Installing all required libraries for text summarization:
# - datasets: for loading benchmark datasets like CNN/DailyMail
# - transformers: for using pre-trained models like BART
# - nltk: for text preprocessing tasks (e.g., tokenization)
# - rouge-score: for evaluating summaries using ROUGE metric
# - torch: PyTorch framework
# - scikit-learn: for  data processing
# - evaluate: unified library for evaluation metrics
# - bert-score: for semantic evaluation of summaries
!pip install datasets transformers nltk rouge-score torch scikit-learn evaluate bert-score

# Importing the dataset utility to load datasets
from datasets import load_dataset
from datasets import Dataset  # Allows creating a Dataset object from custom data (like  dictionaries)

# Loading evaluation metrics from Hugging Face's 'evaluate' library
from evaluate import load
bertscore = load("bertscore")  # BERTScore: semantic similarity metric for summaries
rouge = load("rouge")          # ROUGE: standard metric for summarization quality

# Regular expressions and text tokenization utilities
import re
import nltk
from nltk.tokenize import sent_tokenize  # For splitting long texts into sentences

# Importing pre-trained model, tokenizer, and training tools from Hugging Face
from transformers import (
    AutoTokenizer,                     # Tokenizer for the BART model
    BartForConditionalGeneration,     # Pre-trained BART model for summarization
    Seq2SeqTrainer,                   # Trainer specifically for sequence-to-sequence tasks
    Seq2SeqTrainingArguments,         # Arguments/settings for training process
    EarlyStoppingCallback             # Stops training early if performance stops improving
)

# Scikit-learn for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# PyTorch backend for tensor operations and model training
import torch

# NumPy for handling numerical arrays and metrics
import numpy as np

# Downloading sentence tokenizer for English
nltk.download("punkt_tab")


# Check if a GPU is available for model training
print("GPU Available:", torch.cuda.is_available())

# If a GPU is detected, print its name
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    # If no GPU is detected, warn the user that training may be slower
    print("No GPU detected, training might be slow.")


#Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Define a function to clean text by removing newlines, extra spaces, and trimming whitespace
def clean_text(text):
    text = re.sub(r"\n", " ", text)        # Replace newline characters with space
    text = re.sub(r"\s+", " ", text)       # Replace multiple spaces with a single space
    text = text.strip()                    # Remove leading and trailing whitespace
    return text

# Apply the cleaning function to both the 'article' and 'summary' fields of the dataset
dataset = dataset.map(lambda x: {
    "article": clean_text(x["article"]),
    "summary": clean_text(x["highlights"])
})


# Convert the training portion of the dataset into a Python list of dictionaries
train_data_list = dataset["train"].to_list()

# Split the data into training and temporary sets (90% train, 10% temp)
train_data, temp_data = train_test_split(train_data_list, test_size=0.1, random_state=42)

# Further split the temporary set into validation and test sets (each 5% of the original)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)



# Load the pre-trained tokenizer for the BART-large CNN model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Define a function to tokenize input articles and summaries
def tokenize_data(dataset):
    input_ids, attention_masks, labels = [], [], []  # Initialize lists to store tokenized outputs

    for item in dataset:
        # Tokenize the article (input)
        inputs = tokenizer(
            item["article"],
            padding="max_length",       # Pad sequences to max length
            truncation=True,            # Truncate sequences that exceed max_length
            max_length=512,             # BART's max input size
            return_tensors="pt"         # Return PyTorch tensors
        )

        # Tokenize the summary (target)
        targets = tokenizer(
            item["summary"],
            padding="max_length",       # Pad summaries to max length
            truncation=True,            # Truncate if too long
            max_length=150,             # Max length for summaries
            return_tensors="pt"
        )

        # Append tokenized tensors to the respective lists
        input_ids.append(inputs["input_ids"].squeeze(0))           # Remove extra dimension
        attention_masks.append(inputs["attention_mask"].squeeze(0))
        labels.append(targets["input_ids"].squeeze(0))

    # Return all inputs as a dictionary of stacked tensors (ready for model training)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels)
    }

# Tokenize subsets of the data for training, validation, and testing
train_tokens = tokenize_data(train_data[:10000])  # Tokenize first 10K samples
val_tokens = tokenize_data(val_data[:500])        # Tokenize first 500 samples for validation
test_tokens = tokenize_data(test_data[:500])      # Tokenize first 500 samples for testing




# Set the device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained BART model for conditional text generation (summarization)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
model.to(device)  # Move the model to the selected device (GPU or CPU)

# Load the corresponding tokenizer again (optional if already loaded earlier)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Freeze the first 12 encoder layers to prevent them from being updated during training
# This is part of Partial Fine-Tuning – useful when using pre-trained knowledge on new data
for param in model.model.encoder.layers[:12].parameters():
    param.requires_grad = False

# Freeze the first 12 decoder layers
for param in model.model.decoder.layers[:12].parameters():
    param.requires_grad = False


# Loop through all model parameters by name
# Print the name of each parameter that is still set to be trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# Define training arguments for the Seq2SeqTrainer
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",               # Directory to save model checkpoints and logs
    eval_strategy="epoch",                # Run evaluation at the end of every epoch
    learning_rate=5e-5,                   # Learning rate for the optimizer
    per_device_train_batch_size=16,       # Batch size per GPU/CPU during training
    per_device_eval_batch_size=16,        # Batch size per GPU/CPU during evaluation
    weight_decay=0.01,                    # Apply L2 weight regularization to prevent overfitting
    save_total_limit=2,                   # Keep only the 2 most recent checkpoints
    num_train_epochs=10,                  # Total number of training epochs
    predict_with_generate=True,           # Enables generation during evaluation
    fp16=True,                            # Use 16-bit floating point precision if GPU supports it
    load_best_model_at_end=True,          # Load the best model based on eval_loss after training ends
    metric_for_best_model="eval_loss",    # Use evaluation loss to select the best model
    greater_is_better=False,              # Lower eval_loss is better
    save_strategy="epoch",                # Save a checkpoint at the end of every epoch
    evaluation_strategy="epoch"           # Evaluate after every epoch
)

# Define the custom evaluation metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predicted and target sequences from token IDs to strings
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores (n-gram based text overlap metrics)
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    # Compute BERTScore (semantic similarity metric)
    bert_scores = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    # Return evaluation metrics
    return {
        "rouge1": rouge_scores["rouge1"],     # ROUGE-1: unigrams
        "rouge2": rouge_scores["rouge2"],     # ROUGE-2: bigrams
        "rougeL": rouge_scores["rougeL"],     # ROUGE-L: longest common subsequence
        "bert_score": np.mean(bert_scores["f1"])  # Average BERTScore F1 across all samples
    }

#  adjust dropout rate for regularization
model.config.dropout = 0.1


# Convert tokenized training data into a Hugging Face Dataset object
train_dataset = Dataset.from_dict(train_tokens)

# Convert tokenized validation data into a Hugging Face Dataset
val_dataset = Dataset.from_dict(val_tokens)

# Convert tokenized test data into a Hugging Face Dataset
test_dataset = Dataset.from_dict(test_tokens)


# Initialize the Hugging Face Seq2SeqTrainer with all training components
trainer = Seq2SeqTrainer(
    model=model,                          # The BART model to fine-tune
    args=training_args,                   # Training configurations (epochs, batch size, etc.)
    train_dataset=train_dataset,          # Tokenized training dataset
    eval_dataset=val_dataset,            # Tokenized validation dataset
    tokenizer=tokenizer,                  # Tokenizer used for input/output processing
    compute_metrics=compute_metrics,      # Custom function to compute ROUGE & BERTScore during evaluation
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    # Early stopping: if validation doesn't improve for 2 consecutive epochs, stop training
)


# Start the fine-tuning process using the configured trainer
trainer.train()

# Evaluate the fine-tuned model on the validation set
eval_results = trainer.evaluate()

# Print the evaluation metrics (e.g., ROUGE-1, ROUGE-2, ROUGE-L, BERTScore)
print(eval_results)


# Define a function that generates a summary for a given input text
def generate_summary(text):
    device = next(model.parameters()).device  # Ensure input tensor is on the same device as the model (GPU)

    # Tokenize the input text and move it to the appropriate device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    # Generate a summary using beam search decoding
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,          # Maximum length of the generated summary
        min_length=40,           # Minimum length to avoid very short outputs
        length_penalty=2.0,      # Controls output length (higher = shorter summaries)
        num_beams=4              # Beam search with 4 beams for better quality
    )

    # Decode the generated token IDs back into a readable summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Example: Generate a summary for the first test article
sample_text = dataset["test"][0]["article"]
print("Original Text:\n", sample_text)
print("\nGenerated Summary:\n", generate_summary(sample_text))

