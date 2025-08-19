from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from loaders.pdf_loader import load_pdf

# Tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")
docs = load_pdf("data/g6-final.pdf")

# 你的原始文件（list of Document）
raw_texts = [doc.page_content for doc in docs]

# 計算每段 token 數
lengths = [len(tokenizer.encode(text)) for text in raw_texts]

# 查看統計結果
import numpy as np
print("平均 token 數:", np.mean(lengths))
print("最大 token 數:", np.max(lengths))
print("前幾筆:", lengths[:10])
