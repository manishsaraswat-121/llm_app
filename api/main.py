from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ Define model path
model_dir = r"C:\Users\manis\Downloads\medical-llm-project\models\tiny-gpt2-medquad"

# ✅ Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

# ✅ Initialize FastAPI
app = FastAPI(title="Medical QA API", description="Ask medical questions", version="1.0")

# ✅ Define input schema
class Query(BaseModel):
    context: str
    question: str

@app.post("/generate")
def generate_answer(query: Query):
    try:
        input_text = f"context: {query.context} question: {query.question}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
        if "answer:" in response:
            response = response.split("answer:")[-1].strip()

        return {"answer": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
