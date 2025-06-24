from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = FastAPI()
model_path = "models/medquad-tinyllama"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    input_text = f"Medical Question: {q.question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_new_tokens=100)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"answer": answer}
