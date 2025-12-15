import os
import numpy as np
import torch
import logging

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed
)
import evaluate

# ------------------ CONFIG ------------------
MODEL_CHECKPOINT = "google/mt5-small"
DATA_PATH = "/content/transliteration_data_ready"   # your HuggingFace saved dataset
OUTPUT_DIR = "/content/mt5_translit_model"

MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-5
SEED = 42
FP16 = False          # safer off
LOGGING_STEPS = 50
# -------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_seed(SEED)

# Load metrics ONCE
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


# ------------------ SAFE TARGET TOKENIZATION ------------------
def safe_tokenize_target(tokenizer, targets):
    """Handles both new & old HF tokenizers."""
    try:
        return tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )
    except TypeError:
        with tokenizer.as_target_tokenizer():
            return tokenizer(
                targets,
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding="max_length",
            )


# ------------------ TOKENIZATION FUNCTION ------------------
def tokenize_function(batch):
    model_inputs = tokenizer(
        batch["source_text"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )

    labels = safe_tokenize_target(tokenizer, batch["target_text"])

    # Replace pad_token_id → -100
    labels_ids = labels["input_ids"]
    label_ids_fixed = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels_ids
    ]

    model_inputs["labels"] = label_ids_fixed

    return model_inputs


# ------------------ METRICS (OverflowError FIXED) ------------------
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Clip to avoid HF "out of range" decode issues
    preds = np.where((preds < 0) | (preds >= tokenizer.vocab_size), tokenizer.pad_token_id, preds)

    # Decode predictions safely
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Convert -100 → pad_token for decoding
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"cer": cer, "wer": wer}


# ------------------ MAIN TRAINING ------------------
logger.info(f"Loading dataset from {DATA_PATH}")
dataset = load_from_disk(DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

logger.info("Tokenizing dataset...")
cols = [c for c in dataset["train"].column_names if c not in ["source_text", "target_text"]]

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=cols,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")


MAX_INPUT_LEN = 32
MAX_TARGET_LEN = 32
BATCH_SIZE = 4
EPOCHS = 5
FP16 = True

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    predict_with_generate=False,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     predict_with_generate=True,
#     learning_rate=LR,
#     load_best_model_at_end=True,
#     metric_for_best_model="cer",
#     greater_is_better=False,
#     fp16=FP16,
#     logging_steps=LOGGING_STEPS,
# )

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logger.info("Starting training...")
trainer.train()

logger.info("Evaluating on test set...")
results = trainer.evaluate(tokenized["test"])
print("TEST RESULTS:", results)

# Save final best model
save_path = os.path.join(OUTPUT_DIR, "final_best_model")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

logger.info(f"Model saved at: {save_path}")
