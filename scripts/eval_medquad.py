import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 🧠 Load model and tokenizer from local directory
model_dir = r"C:\Users\manis\Downloads\medical-llm-project\models\tiny-gpt2-medquad"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

# 🧪 Load evaluation dataset (100 examples saved locally)
dataset = load_dataset("json", data_files=r"C:\Users\manis\Downloads\medical-llm-project\data\filtered\squad_100.json", split="train")

# 📊 Load evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# 📝 Store predictions and references
predictions = []
references = []

# For accuracy/precision/recall calculation
binary_preds = []
binary_refs = []

# 🧹 Generate predictions
for example in tqdm(dataset, desc="Evaluating"):
    input_text = f"context: {example['context']} question: {example['question']}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id  # Ensure no padding error
        )
    
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Keep only the generated answer (remove the prompt)
    if "question:" in pred:
        pred = pred.split("question:")[-1]
    if "answer:" in pred:
        pred = pred.split("answer:")[-1].strip()

    ground_truth = example["answers"]["text"][0]

    predictions.append(pred)
    references.append(ground_truth)

    # Exact match logic (case-insensitive)
    binary_preds.append(pred.strip().lower() == ground_truth.strip().lower())
    binary_refs.append(1)  # ground truth is always 1 (correct)

# 🧾 Evaluate ROUGE and BLEU
rouge_result = rouge.compute(predictions=predictions, references=references)
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# 🎯 Calculate Accuracy, Precision, Recall, F1
accuracy = accuracy_score(binary_refs, binary_preds)
precision = precision_score(binary_refs, binary_preds)
recall = recall_score(binary_refs, binary_preds)
f1 = f1_score(binary_refs, binary_preds)

# 📢 Print scores
print("\n🟩 Evaluation Results:")
print("ROUGE:", rouge_result)
print("BLEU:", bleu_result)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
