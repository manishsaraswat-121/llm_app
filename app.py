import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer locally
@st.cache_resource  # Cache model and tokenizer for faster reloads
def load_model():
    model_dir = r"C:\Users\manis\Downloads\medical-llm-project\models\tiny-gpt2-medquad"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# App UI
st.title("🩺 Medical Q&A Chatbot")
st.markdown("Ask me medical questions and I'll try to answer using a fine-tuned model!")

context = st.text_area("📚 Context", height=200, help="Provide a medical paragraph or background.")
question = st.text_input("❓ Your Question", help="Ask a question related to the context.")

if st.button("🧠 Generate Answer"):
    if not context.strip() or not question.strip():
        st.warning("Please fill in both context and question.")
    else:
        input_text = f"context: {context} question: {question}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only generated answer
        if "question:" in pred:
            pred = pred.split("question:")[-1]
        if "answer:" in pred:
            pred = pred.split("answer:")[-1].strip()

        st.success("🧾 Model's Answer:")
        st.write(pred)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Transformers")
