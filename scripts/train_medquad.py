from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import os
import json

# Step 1: Download and truncate SQuAD dataset to 100 examples
print("🔽 Downloading SQuAD and selecting 100 examples...")
dataset = load_dataset("squad", split="train[:100]")  # Using 100 samples for quick training

# Optional: Save locally
os.makedirs("data/filtered", exist_ok=True)
dataset.to_json("data/filtered/squad_100.json", orient="records", lines=False)
print("✅ Saved dataset to data/filtered/squad_100.json")

# Step 2: Load tiny GPT2 model & tokenizer
model_name = "sshleifer/tiny-gpt2"
print(f"🔽 Loading tokenizer and model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Add a PAD token (required for GPT2)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Update model with new tokenizer size

# Step 3: Preprocess dataset
print("🧹 Preprocessing dataset...")

def preprocess(example):
    full_text = f"question: {example['question']} context: {example['context']}"
    encoding = tokenizer(full_text, truncation=True, padding="max_length", max_length=128)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Step 4: Training arguments
training_args = TrainingArguments(
    output_dir="models/tiny-gpt2-medquad",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_dir="./logs",
    save_steps=10,
    logging_steps=5,
    save_total_limit=1,
    report_to="none"
)

# Step 5: Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 6: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Step 7: Train the model
print("🚀 Training...")
trainer.train()

# Step 8: Save model and tokenizer
print("💾 Saving model and tokenizer...")
model.save_pretrained("models/tiny-gpt2-medquad")
tokenizer.save_pretrained("models/tiny-gpt2-medquad")
print("✅ Training complete.")
