# reranker/bge_reranker.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 預設用這個，支援中文
MODEL_NAME = "BAAI/bge-reranker-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def rerank(query: str, docs, top_n=4):
    pairs = [[query, doc.page_content] for doc in docs]

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        scores = model(**inputs).logits.squeeze(-1)

    # 加入 score 並排序
    reranked = list(zip(scores.tolist(), docs))
    reranked.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in reranked[:top_n]]
