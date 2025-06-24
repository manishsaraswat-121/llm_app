# CPU-Based Medical LLM Fine-tuning Project

## Model
Uses TinyLlama 1.1B for question answering on MedQuAD-like data.

## Setup

```bash
pip install -r requirements.txt
```

## Training

Put your MedQuAD-style dataset as `data/medquad.json` with fields `question` and `answer`.

```bash
python scripts/train_medquad.py
```

## Inference

```bash
uvicorn inference.app_vllm:app --reload
```